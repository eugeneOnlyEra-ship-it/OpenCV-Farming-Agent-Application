"""
dev/check_cloud.py -- the cloud database layer: what is stored, that images never are, and that the farm never depends
on the network.

Run from the project root:   python dev/check_cloud.py        (~2 min; needs boto3 + moto, models/ and pybullet)

NOTHING HERE TOUCHES REAL AWS: DynamoDB is replaced by `moto`, a mock of the service (fake credentials are set so a real
account could not be used even by mistake). It proves the code speaks DynamoDB correctly; it cannot prove your AWS account,
region or IAM permissions are set up -- `python cloud_cli.py ping` does that against the real table.

  [1] model        a pod's whole life (disease -> treated -> verified -> ripe -> harvested -> replanted) changes its profile
                   the right way, and applying an event twice never double-counts
  [2] privacy      images are refused in every form; events carry only a file NAME and a SHA-256 of the local photo
  [3] schema       the table definition, the CloudFormation template and the IAM policy agree
  [4] parity       the local-file backend and DynamoDB give identical results
  [5] resilience   a slow or unreachable database never slows the farm; nothing is lost; nothing is counted twice
  [6] engine       the real application run end to end fills the database correctly
  [7] identity     what the robot camera read off the pod's tag is linked to the pod's record (needs pybullet)
"""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dev"))

# fake credentials: even a bug could not reach a real account from this test
for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_SESSION_TOKEN": "testing",
             "AWS_DEFAULT_REGION": "us-east-1"}.items():
    os.environ[k] = v
for k in ("FARM_CLOUD", "FARM_DDB_TABLE", "FARM_DDB_ENDPOINT", "AWS_PROFILE"):
    os.environ.pop(k, None)

import numpy as np                                                  # noqa: E402
from moto import mock_aws                                           # noqa: E402

import cloud_model as M                                             # noqa: E402
from cloud_backends import DynamoBackend, LocalBackend              # noqa: E402
from cloud_sync import (CloudConfig, CloudSync, FarmCloud, FarmCloudStore, config_from_env)   # noqa: E402
from worker_planner import WorkOrder                                # noqa: E402

FAILS = []
TABLE = "FarmPods"


def ok(cond, name, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


# ------------------------------------------------------------------------------------------------ helpers
POD = {"pod_id": "R1-L-F1", "crop_type": "lettuce", "side": "front"}


def cls(stage, flag=False, conf=0.9):
    return {"growth_stage": stage, "growth_confidence": conf, "disease_flag": flag,
            "disease_name": "Septoria" if flag else None, "disease_confidence": 0.6 if flag else 0.1,
            "disease_detections": [1] if flag else []}


def insp(ts, visit, stage, flag=False, action="log_healthy", cycle=None, kind="scan", image_path=None):
    return M.inspection_event(POD, cls(stage, flag), {"action": action, "reason": "because"}, ts, visit, "trend", cycle,
                              stage, image_path, kind)


def order(task, status, oid="WO-0001", created="2026-10-25T10:00:00+00:00", updated="2026-10-25T10:00:05+00:00",
          outcome=None, pod_id="R1-L-F1"):
    return WorkOrder(order_id=oid, task=task, role="worker", agent="worker_front", pod_id=pod_id, crop_type="lettuce",
                     trough_id="R1-L-F", side="front", location="x", priority=1, title=f"{task} it", rationale="r",
                     evidence={}, targets=[], steps=[1, 2, 3], requires_approval=False, status=status, created=created,
                     updated=updated, outcome=outcome or {})


def cycle(age, ripe=False, n=1):
    return {"age": age, "harvest_age": 12, "stage": "s", "waiting_for_harvest": ripe, "cycle": n}


def lifecycle():
    """A pod's whole life as a list of events."""
    oe = M.order_event
    return [
        insp("2026-10-25T10:00:00+00:00", 1, "stage 02- Leafy growth", True, "flag_for_treatment", cycle(5)),
        oe("status", order("TREAT", "dispatched"), None), oe("status", order("TREAT", "in_progress"), None),
        oe("status", order("TREAT", "completed", updated="2026-10-25T10:01:00+00:00"), None),
        oe("loop_closed", order("VERIFY", "completed", oid="WO-0002", created="2026-10-25T10:01:00+00:00"),
           {"outcome": {"result": "treatment_worked", "headline": "ok"}}),
        insp("2026-10-25T10:02:00+00:00", 2, "stage 04- Harvest stage", False, "flag_for_harvest", cycle(12, True)),
        oe("status", order("HARVEST", "dispatched", oid="WO-0003", created="2026-10-25T10:03:00+00:00"), None),
        oe("status", order("HARVEST", "completed", oid="WO-0003", created="2026-10-25T10:03:00+00:00",
                           updated="2026-10-25T10:04:00+00:00"), None),
        oe("status", order("REPLANT", "completed", oid="WO-0004", created="2026-10-25T10:04:00+00:00",
                           updated="2026-10-25T10:05:00+00:00"), None),
    ]


def states(profile):
    return profile["status"], profile["disease_status"], profile["harvest_readiness"]


def all_items(backend):
    if isinstance(backend, LocalBackend):
        return backend.all_items()
    out, kw = [], {}
    while True:
        r = backend._table().scan(**kw)
        out += [M.from_dynamo(i) for i in r["Items"]]
        if "LastEvaluatedKey" not in r:
            return out
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]


def new_ddb():
    b = DynamoBackend(TABLE, "us-east-1")
    b.create_table()
    return b


# ================================================================================================ [1]
def check_model():
    print("\n[1] model: a pod's life changes its profile the right way")
    store = FarmCloudStore(LocalBackend(tempfile.mktemp(suffix=".json")))
    seen = []
    for ev in lifecycle():
        store.apply(ev)
        seen.append(states(store.profile("R1-L-F1")))
    want = [("disease_flagged", "flagged", "not_ready"), ("treatment_in_progress", "treating", "not_ready"),
            ("treatment_in_progress", "treating", "not_ready"), ("verifying_treatment", "verifying", "not_ready"),
            ("growing", "cleared", "not_ready"), ("harvest_ready", "cleared", "ready"),
            ("harvest_in_progress", "cleared", "ordered"), ("harvested", "cleared", "harvested"),
            ("growing", "healthy", "not_ready")]
    ok(seen == want, "status follows the pod: flagged > treating > verifying > cleared > ripe > harvested > replanted", str(seen))
    p = store.profile("R1-L-F1")
    ok(p["crop_type"] == "lettuce" and p["tag"] == "R1-L-F1" and p["rail"] == 1 and p["station_letter"] == "L" and
       p["side"] == "front" and p["slot"] == 1, "the profile carries the pod's identity and its crop type")
    ok(p["treatments_completed"] == 1 and p["last_treatment_result"] == "worked" and p["harvests_completed"] == 1 and
       p["crop_cycle"] == 2 and p["crop_age"] == 0 and p.get("growth_stage") is None,
       "treatment and harvest history are summarised; the replanted pod starts a new crop cycle")
    ok(len(store.inspections("R1-L-F1")) == 2 and len(store.treatments("R1-L-F1")) == 1 and len(store.harvests("R1-L-F1")) == 1
       and len(store.orders("R1-L-F1")) == 4, "every inspection and every order is its own history item")
    ok(store.inspections("R1-L-F1")[0]["visit"] == 2, "history comes back newest first")
    n = len(store.backend.query(M.pod_pk("R1-L-F1")))
    store.apply(lifecycle()[0])
    ok(len(store.backend.query(M.pod_pk("R1-L-F1"))) == n and store.profile("R1-L-F1")["visit_count"] == 2,
       "applying the same event twice changes nothing (no double counting)")

    s2 = FarmCloudStore(LocalBackend(tempfile.mktemp(suffix=".json")))
    s2.apply(insp("2026-10-25T10:00:00+00:00", 1, "stage 04- Harvest stage", False, "flag_for_harvest", cycle(12, True)))
    ok(states(s2.profile("R1-L-F1"))[2] == "ready" and s2.profile("R1-L-F1")["harvest_progress"] == 1.0, "a ripe crop is harvest_ready")
    s2.apply(M.order_event("status", order("HARVEST", "dispatched", oid="WO-9"), None))
    s2.apply(M.order_event("status", order("HARVEST", "failed", oid="WO-9", updated="2026-10-25T10:09:00+00:00"), None))
    ok(s2.profile("R1-L-F1")["harvest_readiness"] == "ready" and s2.profile("R1-L-F1")["harvests_completed"] == 0,
       "a failed harvest leaves the pod ready, not harvested")
    s2.apply(M.tag_read_event({"state": "failed", "label": None, "expected": "R1-L-F1", "reason": "no tag"}, "2026-10-25T11:00:00+00:00"))
    ok(s2.profile("R1-L-F1")["status"] == "identity_unconfirmed" and s2.profile("R1-L-F1")["tag_failure_reason"] == "no tag",
       "a tag the camera could not confirm makes the pod identity_unconfirmed")
    s2.apply(M.tag_read_event({"state": "confirmed", "label": "R1-L-F1", "expected": "R1-L-F1", "confidence": .9,
                               "agreeing_views": 3, "views_total": 3}, "2026-10-25T11:05:00+00:00"))
    ok(s2.profile("R1-L-F1")["status"] == "harvest_ready" and "tag_failure_reason" not in s2.profile("R1-L-F1"),
       "a later confirmed read clears it again")
    ok([i["pod_id"] for i in s2.pods_with_status("harvest_ready")] == ["R1-L-F1"] and
       [i["pod_id"] for i in s2.pods_of_crop("lettuce")] == ["R1-L-F1"], "pods can be found by status and by crop type")


# ================================================================================================ [2]
def check_privacy():
    print("\n[2] privacy: images never reach the cloud")
    refused = {"raw bytes": {"x": b"\xff\xd8\xff"}, "a numpy image": {"x": np.zeros((4, 4, 3), np.uint8)},
               "an 'image' field": {"image": "abc"}, "a 'frame' field": {"frame": 1}, "a jpg field": {"jpg": "x"},
               "a base64 jpeg": {"note": "/9j/4AAQSkZJRgABAQ" + "A" * 50}, "a base64 png": {"note": "iVBORw0KGgo" + "A" * 50},
               "a data URI": {"note": "data:image/png;base64,AAAA"}, "an oversized string": {"note": "x" * 5000},
               "nested bytes": {"a": [{"b": bytearray(b"abc")}]}}
    for name, item in refused.items():
        try:
            M.assert_no_images(item)
            ok(False, f"refused: {name}")
        except M.ImageDataRejected:
            ok(True, f"refused: {name}")
    try:
        M.assert_no_images({"image_name": "a.jpg", "image_sha256": "ab" * 32, "title": "Treat Septoria"})
        ok(True, "a file NAME and a hash are allowed (they are not images)")
    except M.ImageDataRejected:
        ok(False, "a file NAME and a hash are allowed")
    for backend in (LocalBackend(tempfile.mktemp(suffix=".json")),):
        for op in (lambda: backend.put({"pk": "POD#X", "sk": "A", "frame": "x"}),
                   lambda: backend.update("POD#X", "PROFILE", {"jpg": "x"})):
            try:
                op()
                ok(False, "the backend itself refuses image fields")
            except M.ImageDataRejected:
                ok(True, "the backend itself refuses image fields (second line of defence)")
    with mock_aws():
        b = new_ddb()
        try:
            b.put({"pk": "POD#X", "sk": "A", "evidence": {"views": []}})
            ok(False, "DynamoDB backend refuses image fields")
        except M.ImageDataRejected:
            ok(True, "DynamoDB backend refuses image fields")
    import pod_registry
    pod = next(p for p in pod_registry.PODS if p["pod_id"] == "R2-M-F4")
    path = pod["image_path"]
    ev = M.inspection_event(pod, cls("stage 04- Harvest stage"), {"action": "log_healthy", "reason": "r"},
                            "2026-10-25T10:00:00+00:00", 1, "", cycle(12, True), None, path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    ok(ev["image_sha256"] == digest and ev["image_name"] == os.path.basename(path),
       "an inspection carries the local photo's file name and SHA-256, matching the file on this computer")
    ok(path not in json.dumps(ev) and len(json.dumps(ev)) < 2500 and "/" not in ev["image_name"],
       "the local path is not sent, and the whole event is tiny (a few hundred bytes, not an image)")
    sync = CloudSync(FarmCloudStore(LocalBackend(tempfile.mktemp(suffix=".json"))), None)
    ok(sync.submit({"type": "inspection", "jpg": b"xx"}) is False and sync.status()["dropped"] == 1,
       "the sync queue refuses an image-carrying event before it is even queued")


# ================================================================================================ [3]
def check_schema():
    print("\n[3] schema: table definition, CloudFormation template and IAM policy agree")
    import yaml
    path = os.path.join(ROOT, "aws", "farm_table.yaml")

    class Loader(yaml.SafeLoader):
        pass
    Loader.add_multi_constructor("!", lambda loader, suffix, node: loader.construct_scalar(node) if isinstance(node, yaml.ScalarNode)
                                 else (loader.construct_sequence(node) if isinstance(node, yaml.SequenceNode) else loader.construct_mapping(node)))
    tpl = yaml.load(open(path, encoding="utf-8"), Loader=Loader)
    props = tpl["Resources"]["FarmTable"]["Properties"]
    d = M.TABLE_DEFINITION
    ok(props["KeySchema"] == d["KeySchema"], "CloudFormation key schema == TABLE_DEFINITION")
    ok(sorted((a["AttributeName"], a["AttributeType"]) for a in props["AttributeDefinitions"]) ==
       sorted((a["AttributeName"], a["AttributeType"]) for a in d["AttributeDefinitions"]), "attribute definitions match")
    ok({g["IndexName"]: g["KeySchema"] for g in props["GlobalSecondaryIndexes"]} ==
       {g["IndexName"]: g["KeySchema"] for g in d["GlobalSecondaryIndexes"]}, "indexes match")
    ok(props["BillingMode"] == d["BillingMode"] == "PAY_PER_REQUEST", "on-demand billing (an idle farm costs ~nothing)")
    ok(props["PointInTimeRecoverySpecification"]["PointInTimeRecoveryEnabled"] in (True, "true"), "point-in-time recovery is on")
    pol = json.load(open(os.path.join(ROOT, "aws", "iam_policy.json")))
    actions = {a for s in pol["Statement"] for a in (s["Action"] if isinstance(s["Action"], list) else [s["Action"]])}
    ok(actions == {"dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:GetItem", "dynamodb:Query", "dynamodb:DescribeTable"},
       "the IAM policy grants only the 5 calls the application makes (no scan, no delete, no table admin)", str(actions))
    import subprocess
    try:
        import shutil
        r = subprocess.run([shutil.which("cfn-lint") or "cfn-lint", path], capture_output=True, text=True, timeout=120)
        ok(r.returncode == 0, "cfn-lint finds no problems in the template", (r.stdout + r.stderr)[-400:])
    except Exception as exc:                                        # noqa: BLE001
        print(f"  skip  cfn-lint not available ({exc})")
    with mock_aws():
        b = new_ddb()
        d2 = b._table().meta.client.describe_table(TableName=TABLE)["Table"]
        ok({g["IndexName"] for g in d2["GlobalSecondaryIndexes"]} == {"gsi_crop", "gsi_status"}, "the table is created from the definition")
    ok(config_from_env({}, "x").backend == "local" and config_from_env({"FARM_DDB_TABLE": "T"}, "x").backend == "aws"
       and config_from_env({"FARM_CLOUD": "off"}, "x").backend == "off", "config: local by default, aws when a table is named, off on request")


# ================================================================================================ [4]
def check_parity():
    print("\n[4] parity: the local file and DynamoDB behave the same")
    results = []
    with mock_aws():
        for backend in (LocalBackend(tempfile.mktemp(suffix=".json")), new_ddb()):
            st = FarmCloudStore(backend)
            for ev in lifecycle():
                st.apply(ev)
            p = st.profile("R1-L-F1")
            for k in ("updated_at", "gsi2sk"):
                p.pop(k, None)
            results.append((p, [i["sk"] for i in st.backend.query(M.pod_pk("R1-L-F1"))],
                            [i["pod_id"] for i in st.pods_with_status("growing")], [i["pod_id"] for i in st.pods_of_crop("lettuce")]))
    ok(results[0] == results[1], "profile, history keys and index queries are identical on both backends")
    with mock_aws():
        b = new_ddb()
        s1 = FarmCloudStore(b)
        for ev in lifecycle()[:3]:
            s1.apply(ev)
        s2 = FarmCloudStore(b)                                      # a restart: a fresh process continues from the cloud
        s2.apply(lifecycle()[3])
        ok(s2.profile("R1-L-F1")["treatments_completed"] == 1 and s2.profile("R1-L-F1")["visit_count"] == 1,
           "a restarted application carries on from what the cloud already holds")


# ================================================================================================ [5]
class Flaky(LocalBackend):
    """A backend that can be slow, broken, or break half way through an event."""
    def __init__(self, path):
        super().__init__(path)
        self.delay, self.down, self.fail_update_once, self.calls = 0.0, False, False, 0

    def _gate(self):
        self.calls += 1
        if self.down:
            raise ConnectionError("network is down")
        time.sleep(self.delay)

    def put(self, item):
        self._gate()
        super().put(item)

    def get(self, pk, sk):
        self._gate()
        return super().get(pk, sk)

    def update(self, pk, sk, values, remove=()):
        self._gate()
        if self.fail_update_once:
            self.fail_update_once = False
            raise ConnectionError("dropped mid-event")
        super().update(pk, sk, values, remove)

    def ping(self):
        return (not self.down), "flaky"


def scans(n):
    return [insp(f"2026-10-25T10:{i:02d}:00+00:00", i + 1, "stage 02- Leafy growth", False, "log_healthy", cycle(i)) for i in range(n)]


def check_resilience():
    print("\n[5] resilience: the farm never waits on, or loses data to, the network")
    tmp = tempfile.mkdtemp()
    b = Flaky(os.path.join(tmp, "a.json"))
    b.delay = 0.3
    sync = CloudSync(FarmCloudStore(b), os.path.join(tmp, "out.jsonl"), backoff=(0.05, 0.05))
    sync.start()
    t0 = time.time()
    for ev in scans(10):
        sync.submit(ev)
    took = time.time() - t0
    ok(took < 0.2, "submitting 10 events to a SLOW database takes <0.2 s (the farm does not wait)", f"{took:.3f}s")
    ok(sync.join(30) and sync.status()["sent"] == 10 and sync.status()["online"], "...and they are all delivered in the background")
    ok(FarmCloudStore(b).profile("R1-L-F1")["visit_count"] == 10, "exactly once each")
    sync.close()

    b = Flaky(os.path.join(tmp, "b.json"))
    sync = CloudSync(FarmCloudStore(b), os.path.join(tmp, "out_b.jsonl"), backoff=(0.02, 0.02))
    sync.start()
    sync.submit(scans(1)[0])
    sync.join(10)
    b.down = True
    t0 = time.time()
    for ev in scans(6)[1:]:
        sync.submit(ev)
    sync.join(30)
    st = sync.status()
    ok(st["online"] is False and st["outbox"] == 5 and st["last_error"], "network down: events are kept on disk, status says offline",
       str(st))
    ok(time.time() - t0 < 10, "...without stalling the queue behind every failed event")
    b.down = False
    sync._offline_until = 0
    sent = sync.replay_outbox()
    st = sync.status()
    ok(sent == 5 and st["outbox"] == 0 and st["online"], "network back: the waiting events are sent, oldest first")
    p = FarmCloudStore(b).profile("R1-L-F1")
    ok(p["visit_count"] == 6 and len(FarmCloudStore(b).inspections("R1-L-F1", limit=50)) == 6,
       "nothing lost, nothing counted twice", f"visit_count={p['visit_count']}")
    sync.close()

    b = Flaky(os.path.join(tmp, "c.json"))
    sync = CloudSync(FarmCloudStore(b), os.path.join(tmp, "out_c.jsonl"), backoff=(0.02, 0.02))
    sync.start()
    sync.submit(scans(1)[0])
    sync.join(10)
    b.fail_update_once = True                                       # the history item is written, then the profile update fails
    sync.submit(scans(2)[1])
    sync.join(10)
    ok(FarmCloudStore(b).profile("R1-L-F1")["visit_count"] == 2, "an event that fails half way is retried without double counting")
    sync.close()

    b = Flaky(os.path.join(tmp, "d.json"))
    b.delay = 0.2
    sync = CloudSync(FarmCloudStore(b), os.path.join(tmp, "out_d.jsonl"), max_queue=2)
    sync.start()
    for ev in scans(8):
        sync.submit(ev)
    ok(sync.status()["spooled"] >= 4, "a full queue spills to disk instead of blocking or dropping", str(sync.status()))
    sync.join(30)
    sync.replay_outbox()
    ok(FarmCloudStore(b).profile("R1-L-F1")["visit_count"] == 8, "...and everything still arrives")
    ok(sync.submit({"type": "inspection"}) and sync.join(5) and sync._thread.is_alive() and sync.status()["dropped"] >= 1,
       "a malformed event is dropped and the worker carries on")
    sync.close()
    ok(not sync._thread.is_alive(), "close() stops the worker")

    cloud = FarmCloud(CloudConfig("local", local_path=os.path.join(tmp, "e.json"), outbox_path=os.path.join(tmp, "o.jsonl")),
                      LocalBackend(os.path.join(tmp, "e.json")))
    ok(cloud.record_inspection(None, None, None, None, None) is False, "record_* never raises, even on garbage")
    cloud.close()
    off = FarmCloud(CloudConfig("off"), None)
    ok(off.record_inspection(POD, cls("x"), {"action": "a"}, "t", 1) is False and off.status()["backend"] == "off",
       "cloud 'off' is a clean no-op")


# ================================================================================================ [6]
def check_engine():
    print("\n[6] engine: the real application fills the database")
    import regression as R
    from farm_engine import FarmEngine
    import pod_registry
    ids = ["R2-M-F4", "R1-L-F1", "R1-R-F1", "R1-L-B1"]
    tmp = tempfile.mkdtemp()
    with mock_aws():
        backend = new_ddb()
        cfg = CloudConfig("aws", TABLE, "us-east-1", outbox_path=os.path.join(tmp, "outbox.jsonl"))
        cloud = FarmCloud(cfg, backend)
        e = FarmEngine([p for p in pod_registry.PODS if p["pod_id"] in ids], n_slots=2, log_dir=os.path.join(tmp, "logs"),
                       history_path=os.path.join(tmp, "h.json"), orders_dir=os.path.join(tmp, "orders"), cloud=cloud)
        e.settings.delay_ms = 100
        e.orders.policy.sim_step_seconds = 0.12
        e.orders.policy.auto_verify_delay_s = 1.0
        ok(e.farm_cloud is cloud and e.orders.on_event is not None, "the engine is connected to the cloud layer")
        e.start()
        R.wait_finished(e, 200)
        t = time.time()
        while e.orders.open_count() and time.time() - t < 90:
            time.sleep(0.5)
        e.close()
        ok(cloud.status()["queued"] == 0 and cloud.status()["outbox"] == 0, "everything was delivered by shutdown", str(cloud.status()))
        st = cloud.store
        by_id = {p["pod_id"]: p for p in pod_registry.PODS}
        for pid in ids:
            prof = st.profile(pid)
            ok(prof is not None and prof["crop_type"] == by_id[pid]["crop_type"] and prof["tag"] == pid and
               prof["status"] in M.STATUSES, f"{pid}: profile exists with its crop type ({by_id[pid]['crop_type']}) and a valid status")
            local = e.history.query(pid)
            insps = [i for i in st.inspections(pid, limit=100) if i["kind"] == "scan"]
            ok(len(insps) == len(local) == prof["visit_count"], f"{pid}: cloud inspections == local history == visit_count ({len(local)})",
               f"{len(insps)} vs {len(local)} vs {prof['visit_count']}")
            ok(prof.get("growth_stage") is not None or prof["tag"] == "R1-L-B1", f"{pid}: growth stage recorded")
            ok(prof.get("last_inspection_at") is not None and prof.get("last_action") is not None, f"{pid}: last inspection and decision recorded")
        orders_local = e.orders.list_orders()
        all_cloud = [o for pid in ids for o in st.orders(pid, limit=500)]
        ok(len(all_cloud) == len(orders_local), "every work order is in the cloud", f"{len(all_cloud)} vs {len(orders_local)}")
        ok(all(c["status"] == next(o for o in orders_local if o.order_id == c["order_id"]).status for c in all_cloud),
           "each order's final status matches the application's")
        treats = [o for o in orders_local if o.task == "TREAT" and o.status == "completed"]
        for o in treats:
            ok(st.profile(o.pod_id)["treatments_completed"] >= 1 and st.profile(o.pod_id).get("last_treatment_order"),
               f"treatment {o.order_id} on {o.pod_id} is in that pod's treatment history")
        done_harvest = [o for o in orders_local if o.task == "HARVEST" and o.status == "completed"]
        for o in done_harvest:
            ok(st.profile(o.pod_id)["harvests_completed"] >= 1, f"harvest {o.order_id} on {o.pod_id} is recorded")
        ok(bool(treats) or bool(done_harvest), "the run produced treatment or harvest work to record",
           f"orders: {[(o.task, o.status) for o in orders_local]}")
        items = all_items(backend)
        bad = []
        for it in items:
            try:
                M.assert_no_images(it)
            except M.ImageDataRejected as exc:
                bad.append(str(exc))
        ok(not bad and len(items) > 10, f"{len(items)} items stored; none contains image data", str(bad[:2]))
        total = sum(len(json.dumps(M.from_dynamo(i), default=str)) for i in items)
        ok(total < 400_000, f"the whole database is {total / 1024:.0f} KB of text (no photos)")
        hashed = checked = 0
        for pid in ids:
            for i in st.inspections(pid, limit=100):
                if i.get("image_name"):
                    f = os.path.join(by_id[pid]["folder"], i["image_name"])
                    if os.path.exists(f):
                        checked += 1
                        hashed += hashlib.sha256(open(f, "rb").read()).hexdigest() == i["image_sha256"]
        ok(checked >= 8 and hashed == checked, f"every cloud inspection matches the photo on this computer by SHA-256 ({hashed}/{checked})")
        counts = st.status_counts()
        ok(sum(counts.values()) == len(ids), "the status index lists every pod exactly once", str(counts))
        crops = {c: len(st.pods_of_crop(c)) for c in ("lettuce", "mushroom", "cabbage")}
        ok(sum(crops.values()) == len(ids), "the crop index lists every pod exactly once", str(crops))


# ================================================================================================ [7]
def check_identity():
    print("\n[7] identity: the tag the robot camera read is linked to the pod's record")
    import pybullet as p
    from check_pod_tags import _bridge_with_real_robot, add_order, wait_for
    from pod_registry import PODS
    with mock_aws():
        backend = new_ddb()
        cloud = FarmCloud(CloudConfig("aws", TABLE, "us-east-1", outbox_path=os.path.join(tempfile.mkdtemp(), "o.jsonl")), backend)
        bridge, book, robots, logs = _bridge_with_real_robot()
        bridge.on_tag_read = cloud.record_tag_read
        book.on_event = cloud.record_order_event
        bridge.start(timeout=120)
        robot = robots[0]
        p.removeBody(robot.pod_label_ids["R0C0-L0-B"], physicsClientId=robot.client)       # tear R1-L-B1's tag off
        pod = next(x for x in PODS if x["pod_id"] == "R1-L-F1")
        bridge.enqueue_visit(pod, {"action": "log_healthy"}, {"growth_stage": "S4_5"}, 1)
        o = add_order(bridge, book, "WO-CLOUD", "TREAT", "R1-L-F2", "front")
        ok(wait_for(lambda: bridge.wait_until_idle(0.05) and o.status in ("completed", "failed"), 120), "the robot visits complete")
        cloud.sync.join(15)
        st = cloud.store
        prof = st.profile("R1-L-F1")
        ok(prof is not None and prof["tag_state"] == "confirmed" and prof["tag_views"] == "3/3" and prof["tag_confidence"] > 0.8,
           "R1-L-F1: the camera's confirmed tag read is on the pod's profile", str(prof and {k: prof.get(k) for k in ("tag_state", "tag_views")}))
        tr = st.tag_reads("R1-L-F1")
        ok(len(tr) == 1 and tr[0]["tag_read"] == "R1-L-F1" and tr[0]["crop_type_from_tag"] == "cabbage" and tr[0]["location"],
           "the tag read item links the number to the crop type found in that pod's record", str(tr))
        ok(prof["crop_type"] == "cabbage", "the profile's crop type came from the tag's pod record", str(prof.get("crop_type")))
        bad = st.profile("R1-L-B1")
        ok(bad["tag_state"] == "failed" and bad["status"] == "identity_unconfirmed" and bad["tag_failure_reason"],
           "R1-L-B1 (tag torn off): identity_unconfirmed, with the reason", str(bad and bad.get("tag_failure_reason")))
        ok("R1-L-B1" in [i["pod_id"] for i in st.pods_with_status("identity_unconfirmed")],
           "...and it shows up in the identity_unconfirmed queue")
        treat = st.treatments("R1-L-F2")
        ok(len(treat) == 1 and treat[0]["status"] == o.status, "the robot's treatment order is in the pod's treatment history", str(treat))
        everything = all_items(backend)
        ok(not any("sheet_png" in i or "evidence" in i or "jpg" in i for i in everything), "no picture from the camera reached the database")
        bridge.request_stop()
        bridge.join(10)
        cloud.close()


if __name__ == "__main__":
    t0 = time.time()
    only = set(sys.argv[1:])                      # e.g.  python dev/check_cloud.py 5 7   runs just sections 5 and 7
    for n, fn in enumerate((check_model, check_privacy, check_schema, check_parity, check_resilience, check_engine,
                            check_identity), 1):
        if only and str(n) not in only:
            continue
        try:
            fn()
        except ImportError as exc:
            print(f"\n[{n}] skipped: {exc}")
    print("\n" + ("ALL CLOUD CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}") + f"   ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAILS else 0)
