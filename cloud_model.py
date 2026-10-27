"""
cloud_model.py -- WHAT the farm keeps in its cloud database, and how events become records.   (No AWS in here.)

THE IDEA
    One DynamoDB table holds everything the farm knows about each pod, keyed by the pod's identity -- the same number
    printed on the pod's tag and read by the robot camera ("R1-L-F1"). Images are NOT stored: they stay on the computer
    (see `assert_no_images`, which refuses them in code, and the `image_name` / `image_sha256` fingerprint that lets a
    cloud record be matched to the local photo without the photo ever leaving).

ONE TABLE, FOUR KINDS OF ITEM         (partition key `pk` = POD#<pod_id>, sort key `sk`)

    sk = PROFILE                          the pod RIGHT NOW: crop type, growth stage, last inspection, disease status,
                                          treatment history summary, harvest readiness, tag/identity state, and a single
                                          `status` for dashboards.        (one item per pod, updated in place)
    sk = INSPECTION#<time>#<visit>        one item per inspection: what the models saw and what the farm decided.
    sk = ORDER#<created>#<order id>       one item per robot work order (TREAT / HARVEST / REPLANT / RECAPTURE / VERIFY /
                                          REVIEW): this is the treatment + harvest history. Updated as it progresses.
    sk = TAGREAD#<time>                   one item per time a robot camera read the pod's tag (and what it concluded).

    Two sparse indexes, on PROFILE items only:
        gsi_crop    gsi1pk = CROP#<crop>      "every lettuce pod"
        gsi_status  gsi2pk = STATUS#<status>  "every pod that is harvest_ready / disease_flagged / ..."

WHY THIS SHAPE
    The questions the farm asks are "everything about pod X" (one Query on the partition) and "which pods need me?" (one
    Query on a sparse index). It never needs joins or ad-hoc scans, which is exactly what DynamoDB is good at, and the
    on-demand billing mode means an idle demo farm costs ~nothing.

SINGLE WRITER
    The profile is maintained by ONE writer process (cloud_sync.CloudSync keeps a mirror of each pod's profile, applies
    each event to it with `apply_event`, and writes the changed attributes). Two apps writing the same pods at once would
    race on counters. That is fine for one farm controller; it would need conditional writes for several.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from decimal import Decimal

SCHEMA_VERSION = 1
PK, SK = "pk", "sk"
GSI_CROP, GSI_STATUS = "gsi_crop", "gsi_status"

#: The table definition (used by cloud_cli.create-table and mirrored by aws/farm_table.yaml; a test keeps them equal).
TABLE_DEFINITION = {
    "KeySchema": [{"AttributeName": PK, "KeyType": "HASH"}, {"AttributeName": SK, "KeyType": "RANGE"}],
    "AttributeDefinitions": [{"AttributeName": a, "AttributeType": "S"}
                             for a in (PK, SK, "gsi1pk", "gsi1sk", "gsi2pk", "gsi2sk")],
    "GlobalSecondaryIndexes": [
        {"IndexName": GSI_CROP, "Projection": {"ProjectionType": "ALL"},
         "KeySchema": [{"AttributeName": "gsi1pk", "KeyType": "HASH"}, {"AttributeName": "gsi1sk", "KeyType": "RANGE"}]},
        {"IndexName": GSI_STATUS, "Projection": {"ProjectionType": "ALL"},
         "KeySchema": [{"AttributeName": "gsi2pk", "KeyType": "HASH"}, {"AttributeName": "gsi2sk", "KeyType": "RANGE"}]},
    ],
    "BillingMode": "PAY_PER_REQUEST",
}

#: every value `status` on a PROFILE can take, most urgent first
STATUSES = ("identity_unconfirmed", "treatment_in_progress", "disease_flagged", "verifying_treatment",
            "harvest_in_progress", "harvest_ready", "harvested", "growing")

OPEN_ORDER = {"awaiting_approval", "planning", "approved", "scheduled", "dispatched", "in_progress"}
STATION_LETTERS = "LMR"


# ======================================================================================================== privacy
class ImageDataRejected(ValueError):
    """Raised when something that looks like image data is about to be written to the cloud."""


_FORBIDDEN_KEYS = {"jpg", "jpeg", "png", "image", "images", "image_bytes", "image_b64", "frame", "frames", "sheet_png",
                   "evidence", "evidence_data", "thumbnail", "pixels", "b64", "base64", "rgba", "bgr"}
_ALLOWED_IMAGE_KEYS = {"image_name", "image_sha256"}          # a file NAME and a hash: neither is an image
_MAX_TEXT = 1200                                              # no legitimate field here is longer; an encoded image is
_IMAGE_PREFIXES = ("data:image", "/9j/", "iVBOR", "R0lGOD", "UklGR")   # data-URI, base64 JPEG / PNG / GIF / WEBP


def assert_no_images(obj, _path="item"):
    """Refuse anything that is, or could carry, an image. Called on EVERY item before it is written, so 'the images stay
    on the computer' is enforced by the code and not just by what the callers happen to send."""
    if isinstance(obj, (bytes, bytearray, memoryview)):
        raise ImageDataRejected(f"{_path}: binary data is never written to the cloud")
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):                                    # a numpy array / image
        raise ImageDataRejected(f"{_path}: arrays (images) are never written to the cloud")
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in _FORBIDDEN_KEYS and key not in _ALLOWED_IMAGE_KEYS:
                raise ImageDataRejected(f"{_path}.{k}: image-like field names are never written to the cloud")
            assert_no_images(v, f"{_path}.{k}")
    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            assert_no_images(v, f"{_path}[{i}]")
    elif isinstance(obj, str):
        if len(obj) > _MAX_TEXT or obj.startswith(_IMAGE_PREFIXES):
            raise ImageDataRejected(f"{_path}: text that long / image-encoded is never written to the cloud")


def fingerprint_image(path):
    """(file name, sha256) of a LOCAL image: lets a cloud record be matched to the photo on the computer without the
    photo (or its path) leaving it. Returns (None, None) if the file cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return os.path.basename(path), h.hexdigest()
    except OSError:
        return None, None


# ======================================================================================================== keys
def pod_pk(pod_id):
    return f"POD#{pod_id}"


def inspection_sk(ts, visit):
    return f"INSPECTION#{ts}#{int(visit or 0):04d}"


def order_sk(created, order_id):
    # order ids restart at WO-0001 every run, so the creation time is part of the key: two runs never collide
    return f"ORDER#{created}#{order_id}"


def tagread_sk(ts):
    return f"TAGREAD#{ts}"


def parse_pod_id(pod_id):
    """'R1-L-F2' -> {rail: 1, station: 0, station_letter: 'L', side: 'front', slot: 2, trough_id: 'R1-L-F'}"""
    m = re.fullmatch(r"R([1-3])-([LMR])-([FB])([1-4])", str(pod_id))      # fullmatch: "$" would let a trailing newline through
    if not m:
        return {}
    rail, letter, fb, slot = m.groups()
    return {"rail": int(rail), "station": STATION_LETTERS.index(letter), "station_letter": letter,
            "side": "front" if fb == "F" else "back", "slot": int(slot), "trough_id": f"R{rail}-{letter}-{fb}"}


# ======================================================================================================== DynamoDB types
def to_dynamo(obj):
    """Python -> what boto3's DynamoDB resource accepts: floats become Decimals, None / NaN are dropped, sets are lists."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return Decimal(str(round(obj, 6)))
    if isinstance(obj, Decimal):
        return obj
    if isinstance(obj, dict):
        return {str(k): v2 for k, v in obj.items() if (v2 := to_dynamo(v)) is not None}
    if isinstance(obj, (list, tuple, set)):
        return [v2 for v in obj if (v2 := to_dynamo(v)) is not None]
    return str(obj)


def from_dynamo(obj):
    """DynamoDB -> plain Python (Decimals become int / float)."""
    if isinstance(obj, Decimal):
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: from_dynamo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_dynamo(v) for v in obj]
    return obj


def plain(obj):
    """Anything json-able: numpy scalars -> python numbers; used to make events safe to queue, spool and replay."""
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _r(x, n=3):
    try:
        return None if x is None else round(float(x), n)
    except (TypeError, ValueError):
        return None


def _short(text, n=300):
    return None if text is None else str(text)[:n]


# ======================================================================================================== events
def pod_meta(pod):
    """The identity facts of a pod, from a pod dict (pod_registry) -- the tag number is the pod_id."""
    pid = pod["pod_id"]
    meta = parse_pod_id(pid)
    return {"pod_id": pid, "crop_type": pod.get("crop_type"), **{k: meta.get(k) for k in
            ("rail", "station", "station_letter", "side", "slot", "trough_id")},
            **({"side": pod["side"]} if pod.get("side") else {})}


def inspection_event(pod, classification, decision, ts, visit, trend="", cycle=None, ground_truth_stage=None,
                     image_path=None, kind="scan"):
    """An inspection of a pod: what the models read, what the farm decided, and the pod's crop clock.
    `image_path` is only fingerprinted (name + sha256); the image itself is never part of the event."""
    name, digest = fingerprint_image(image_path) if image_path else (None, None)
    c = classification or {}
    return plain({
        "type": "inspection", "kind": kind, "ts": ts, "visit": visit, "pod": pod_meta(pod),
        "classification": {k: c.get(k) for k in ("growth_stage", "growth_confidence", "disease_flag", "disease_name",
                                                  "disease_confidence")},
        "n_disease_detections": len(c.get("disease_detections") or []),
        "decision": {"action": (decision or {}).get("action"), "reason": _short((decision or {}).get("reason"))},
        "trend": _short(trend), "cycle": {k: (cycle or {}).get(k) for k in
                                          ("age", "harvest_age", "stage", "waiting_for_harvest", "cycle")} if cycle else {},
        "ground_truth_stage": ground_truth_stage, "image_name": name, "image_sha256": digest})


def order_event(event, order, extra=None):
    """A work-order lifecycle step (from OrderBook._trace). `order` is the live WorkOrder: only small fields are copied."""
    extra = extra or {}
    outcome = (extra.get("outcome") or getattr(order, "outcome", None) or {})
    created = order.created or "undated"       # part of the item's KEY, so it must be stable (real orders always have it)
    updated = order.updated or order.created or "undated"
    return plain({
        "type": "order", "event": event, "ts": updated,
        "order": {"order_id": order.order_id, "task": order.task, "status": order.status, "agent": order.agent,
                  "pod_id": order.pod_id, "crop_type": order.crop_type, "title": _short(order.title, 200),
                  "created": created, "updated": updated, "follow_up_of": order.follow_up_of or None,
                  "progress": order.progress, "steps": len(order.steps or []), "priority": order.priority,
                  "requires_approval": order.requires_approval, "note": _short(extra.get("note"), 200),
                  "outcome_result": outcome.get("result"), "outcome_headline": _short(outcome.get("headline"), 300)}})


def tag_read_event(read, ts):
    """The robot camera's reading of a pod's printed tag (what bridge.tag_read() holds, minus every picture)."""
    return plain({"type": "tag_read", "ts": ts, "pod_id": read.get("expected") or read.get("label"),
                  "state": read.get("state"), "label": read.get("label"), "expected": read.get("expected"),
                  "confidence": _r(read.get("confidence")), "agreeing_views": read.get("agreeing_views"),
                  "views_total": read.get("views_total"), "crop_type_from_tag": read.get("crop_type"),
                  "dispatched_crop": read.get("dispatched_crop"), "reason": _short(read.get("reason")),
                  "location": read.get("location")})


# ======================================================================================================== the profile
def new_profile(meta):
    p = {"entity": "PROFILE", "schema_version": SCHEMA_VERSION, "pod_id": meta["pod_id"], "tag": meta["pod_id"],
         "crop_type": meta.get("crop_type"), "rail": meta.get("rail"), "station_letter": meta.get("station_letter"),
         "side": meta.get("side"), "slot": meta.get("slot"), "trough_id": meta.get("trough_id"),
         # identity (the tag the robot camera read)
         "tag_state": "unread",
         # growth / inspection
         "visit_count": 0, "growth_stage": None, "growth_confidence": None, "crop_age": None, "harvest_age": None,
         "crop_cycle": 1, "last_inspection_at": None, "last_action": None, "last_reason": None, "last_trend_note": None,
         # disease
         "disease_status": "healthy", "disease_flag": False, "disease_name": None, "disease_confidence": None,
         "disease_streak": 0,
         # treatment history (summary; every order is its own ORDER item)
         "treatments_completed": 0, "treatments_failed": 0, "open_treatment": None,
         # harvest
         "harvest_readiness": "not_ready", "harvest_progress": None, "harvests_completed": 0, "open_harvest": None,
         "x_ripe": False}
    return p


def _derive(p, now):
    tag_failed = p.get("tag_state") == "failed"
    ds, hr = p["disease_status"], p["harvest_readiness"]
    if tag_failed:
        status = "identity_unconfirmed"
    elif ds == "treating":
        status = "treatment_in_progress"
    elif ds == "flagged":
        status = "disease_flagged"
    elif ds == "verifying":
        status = "verifying_treatment"
    elif hr in ("ordered", "harvesting"):
        status = "harvest_in_progress"
    elif hr == "ready":
        status = "harvest_ready"
    elif hr == "harvested":
        status = "harvested"
    else:
        status = "growing"
    p["status"] = status
    p["updated_at"] = now
    p["gsi1pk"], p["gsi1sk"] = f"CROP#{p.get('crop_type') or 'unknown'}", pod_pk(p["pod_id"])
    p["gsi2pk"], p["gsi2sk"] = f"STATUS#{status}", f"{now}#{p['pod_id']}"
    return p


def _apply_inspection(p, ev):
    c, d, cy = ev["classification"], ev["decision"], ev.get("cycle") or {}
    p.update(growth_stage=c.get("growth_stage"), growth_confidence=_r(c.get("growth_confidence")),
             last_inspection_at=ev["ts"], last_action=d.get("action"), last_reason=d.get("reason"),
             last_trend_note=ev.get("trend"))
    if ev.get("kind") == "scan":
        p["visit_count"] += 1
    if cy:
        p.update(crop_age=cy.get("age"), harvest_age=cy.get("harvest_age"), crop_cycle=cy.get("cycle") or p["crop_cycle"])
    # --- disease
    flag = bool(c.get("disease_flag"))
    p.update(disease_flag=flag, disease_confidence=_r(c.get("disease_confidence")),
             disease_name=c.get("disease_name") if flag else None)
    if flag:
        p["disease_streak"] += 1
        p["disease_status"] = "treating" if p.get("open_treatment") else "flagged"
        if not p.get("disease_first_flagged_at"):
            p["disease_first_flagged_at"] = ev["ts"]
    else:
        p["disease_streak"] = 0
        p["disease_first_flagged_at"] = None
        if p.get("open_treatment"):
            p["disease_status"] = "treating"
        elif p["disease_status"] in ("flagged", "verifying"):
            p["disease_status"] = "cleared"
    # --- harvest readiness
    ripe = bool(cy.get("waiting_for_harvest")) if cy else d.get("action") == "flag_for_harvest"
    p["x_ripe"] = ripe
    if p["harvest_readiness"] == "harvested" and cy and (cy.get("cycle") or 0) > (p.get("x_harvested_cycle") or 0):
        p["harvest_readiness"] = "not_ready"                      # the pod holds a NEW crop now
    if p["harvest_readiness"] not in ("ordered", "harvesting", "harvested"):
        p["harvest_readiness"] = "ready" if ripe else "not_ready"
    if p.get("crop_age") is not None and p.get("harvest_age"):
        p["harvest_progress"] = round(min(1.0, p["crop_age"] / p["harvest_age"]), 2)


def _apply_order(p, ev):
    o, event = ev["order"], ev.get("event")
    task, st = o["task"], o["status"]
    is_open = st in OPEN_ORDER
    if task == "TREAT":
        if is_open:
            p["open_treatment"], p["disease_status"] = st, "treating"
        else:
            p["open_treatment"] = None
            if st == "completed":
                p.update(treatments_completed=p["treatments_completed"] + 1, last_treatment_at=o["updated"],
                         last_treatment_order=o["order_id"], last_treatment_result="applied", disease_status="verifying")
            else:
                if st == "failed":
                    p.update(treatments_failed=p["treatments_failed"] + 1, last_treatment_result="failed")
                p["disease_status"] = "flagged" if p["disease_flag"] else ("cleared" if p["disease_status"] == "treating" else p["disease_status"])
    elif task == "VERIFY" and event == "loop_closed":
        result = o.get("outcome_result")
        if result == "treatment_worked":
            p.update(disease_status="cleared", last_treatment_result="worked")
        elif result == "treatment_not_resolved":
            p.update(disease_status="flagged", last_treatment_result="not_resolved")
    elif task == "HARVEST":
        if is_open:
            p["open_harvest"], p["harvest_readiness"] = st, ("harvesting" if st == "in_progress" else "ordered")
        else:
            p["open_harvest"] = None
            if st == "completed":
                p.update(harvests_completed=p["harvests_completed"] + 1, last_harvest_at=o["updated"],
                         harvest_readiness="harvested", x_harvested_cycle=p.get("crop_cycle") or 1)
            else:
                p["harvest_readiness"] = "ready" if p.get("x_ripe") else "not_ready"
    elif task == "REPLANT" and st == "completed":
        p.update(harvest_readiness="not_ready", crop_age=0, crop_cycle=(p.get("x_harvested_cycle") or p["crop_cycle"]) + 1,
                 growth_stage=None, growth_confidence=None, x_ripe=False, harvest_progress=0.0, replanted_at=o["updated"],
                 disease_status="healthy", disease_flag=False, disease_name=None, disease_streak=0)


def _apply_tag_read(p, ev):
    ok = ev.get("state") == "confirmed"
    p.update(tag_state="confirmed" if ok else "failed", tag_last_read_at=ev["ts"], tag_confidence=ev.get("confidence"),
             tag_views=(f"{ev.get('agreeing_views')}/{ev.get('views_total')}"
                        if ev.get("views_total") else None),
             tag_failure_reason=None if ok else ev.get("reason"))
    if ok and ev.get("crop_type_from_tag") and not p.get("crop_type"):
        p["crop_type"] = ev["crop_type_from_tag"]


def apply_event(profile, ev, now):
    """Apply one event to a COPY of a pod's profile and return it (with `status` and index keys recomputed)."""
    p = copy.deepcopy(profile)
    {"inspection": _apply_inspection, "order": _apply_order, "tag_read": _apply_tag_read}[ev["type"]](p, ev)
    return _derive(p, now)


def event_pod_id(ev):
    return (ev.get("pod") or {}).get("pod_id") or (ev.get("order") or {}).get("pod_id") or ev.get("pod_id")


def event_pod_meta(ev):
    if ev["type"] == "inspection":
        return ev["pod"]
    pid = event_pod_id(ev)
    meta = {"pod_id": pid, **parse_pod_id(pid)}
    if ev["type"] == "order":
        meta["crop_type"] = ev["order"].get("crop_type")
    return meta


# ======================================================================================================== items
def inspection_item(ev):
    c, d, cy = ev["classification"], ev["decision"], ev.get("cycle") or {}
    pid = ev["pod"]["pod_id"]
    return {PK: pod_pk(pid), SK: inspection_sk(ev["ts"], ev.get("visit")), "entity": "INSPECTION", "pod_id": pid,
            "crop_type": ev["pod"].get("crop_type"), "ts": ev["ts"], "visit": ev.get("visit"), "kind": ev.get("kind"),
            "growth_stage": c.get("growth_stage"), "growth_confidence": _r(c.get("growth_confidence")),
            "disease_flag": bool(c.get("disease_flag")), "disease_name": c.get("disease_name"),
            "disease_confidence": _r(c.get("disease_confidence")), "n_disease_detections": ev.get("n_disease_detections"),
            "action": d.get("action"), "reason": d.get("reason"), "trend": ev.get("trend"),
            "crop_age": cy.get("age"), "harvest_age": cy.get("harvest_age"), "waiting_for_harvest": cy.get("waiting_for_harvest"),
            "crop_cycle": cy.get("cycle"), "ground_truth_stage": ev.get("ground_truth_stage"),
            "image_name": ev.get("image_name"), "image_sha256": ev.get("image_sha256")}


def order_item(ev):
    o = ev["order"]
    return {PK: pod_pk(o["pod_id"]), SK: order_sk(o["created"], o["order_id"]), "entity": "ORDER", "pod_id": o["pod_id"],
            "crop_type": o.get("crop_type"), "order_id": o["order_id"], "task": o["task"], "status": o["status"],
            "agent": o["agent"], "title": o["title"], "created": o["created"], "updated": o["updated"],
            "follow_up_of": o.get("follow_up_of"), "progress": o.get("progress"), "steps": o.get("steps"),
            "priority": o.get("priority"), "requires_approval": o.get("requires_approval"), "note": o.get("note"),
            "outcome_result": o.get("outcome_result"), "outcome_headline": o.get("outcome_headline")}


def tag_item(ev):
    pid = ev["pod_id"]
    return {PK: pod_pk(pid), SK: tagread_sk(ev["ts"]), "entity": "TAGREAD", "pod_id": pid, "ts": ev["ts"],
            "state": ev.get("state"), "tag_read": ev.get("label"), "dispatched_pod": ev.get("expected"),
            "confidence": ev.get("confidence"), "agreeing_views": ev.get("agreeing_views"),
            "views_total": ev.get("views_total"), "crop_type_from_tag": ev.get("crop_type_from_tag"),
            "dispatched_crop": ev.get("dispatched_crop"), "reason": ev.get("reason"), "location": ev.get("location")}


def event_item(ev):
    """The history item (INSPECTION / ORDER / TAGREAD) an event produces, or None for events that only move the profile."""
    if ev["type"] == "inspection":
        return inspection_item(ev)
    if ev["type"] == "order":
        return order_item(ev)
    return tag_item(ev)
