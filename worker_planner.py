"""
worker_planner.py  --  from "what the vision model saw" to "what the robots do"

The robot system has four agents, two per side of the pod troughs:

    camera_front   worker_front        (pods on the FRONT side)
    camera_back    worker_back         (pods on the BACK side)

The camera agents collect the images. This module is the missing half: after
the OpenCV/ONNX analysis of an image, it writes the *instructions* for those
agents. One analysed image becomes zero, one or two WORK ORDERS:

    decision from agent.py            order(s) produced
    --------------------------------  ---------------------------------------------
    flag_for_treatment                worker  TREAT      (needs human approval)
                                      -> when done: camera VERIFY follow-up
    flag_for_harvest                  worker  HARVEST    (approval by default)
    schedule_frequent_monitoring      camera  RECAPTURE  (retake the shot, tuned to
                                                          what was unclear)
    log_healthy                       nothing

The visual evidence drives the order, not just labels it:
  * the detection boxes become the worker's TARGET REGIONS (normalised image
    coordinates, so any arm / camera calibration can map them);
  * the disease the model named selects the instruction set (remove leaves
    vs. remove plant vs. isolate-and-review ...);
  * how much was detected (localised vs widespread) and how confident the
    model was change the steps and the approval note;
  * a TREAT order supersedes an open RECAPTURE for the same pod; a later,
    contradicting reading marks an unapproved order STALE for the human;
  * completing a TREAT spawns a camera VERIFY visit (perception -> action ->
    re-perception).

Safety / responsible use: anything that removes plant material or applies a
treatment waits for a HUMAN to approve it. The playbook is deliberately
generic (no product names, no doses) -- it is a starting point for an
agronomist to review, not agronomic advice.

Integration surface for the real robots (replace the simulator with these):
    * every dispatched order is written as JSON to
      work_orders/<run>/outbox/<agent>/<order_id>.json  (swap FileOutbox for
      MQTT / AWS IoT Core / SQS / ROS 2 by providing any object with .send(dict))
    * robots report back with OrderBook.begin / advance / complete / report
    * work_orders/<run>/trace.jsonl logs every perception -> decision -> order
      -> status event (useful as the "agent trace" evidence).

No UI code here; the UI (app_ui.py) only reads and drives this.
"""

import copy
import heapq
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

import cv2

ALL_AGENTS = ["camera_front", "worker_front", "camera_back", "worker_back"]
OPEN_STATUSES = {"planning", "awaiting_approval", "approved", "dispatched", "in_progress", "scheduled"}
PRIORITY = {"REVIEW": 1, "TREAT": 1, "HARVEST": 2, "REPLANT": 2, "RECAPTURE": 3, "VERIFY": 3}
# A pod's NEXT image waits while any of these are still open for it (REVIEW = a human, VERIFY = a later visit).
BLOCKING_STATUSES = {"planning", "awaiting_approval", "approved", "dispatched", "in_progress"}
BLOCKING_TASKS = {"TREAT", "HARVEST", "REPLANT", "RECAPTURE"}
CONSULT_TASKS = ("TREAT", "HARVEST")     # tasks the AI planner is asked about (it can only soften them)
PRIORITY_LABEL = {1: "HIGH", 2: "NORMAL", 3: "LOW"}
MAX_TARGETS = 8
WIDESPREAD_DETECTIONS = 10     # this many disease boxes = "widespread", flagged to the human


# ---------------------------------------------------------------------------
# Policy + data
# ---------------------------------------------------------------------------
@dataclass
class OrderPolicy:
    auto_approve_all: bool = True         # MASTER SWITCH: approve + dispatch every robot order automatically.
                                          # Off = every order waits for a human. (REVIEW escalations always wait.)
    auto_verify_delay_s: float = 8.0      # auto mode: how soon after a treatment the camera follow-up runs
                                          # (demo timing; real deployments use verify_after_hours)
    verify_after_hours: int = 24          # camera re-check after a treatment
    recapture_revisit_hours: int = 4      # re-inspect window written into RECAPTURE orders
    simulate_robots: bool = True          # built-in fleet simulator executes dispatched orders
    sim_step_seconds: float = 0.8
    close_the_loop: bool = True           # re-analyse a NEW frame when a RECAPTURE / VERIFY order finishes


@dataclass
class WorkOrder:
    order_id: str
    task: str                 # TREAT | HARVEST | REPLANT | RECAPTURE | VERIFY | REVIEW
    role: str                 # worker | camera | human
    agent: str                # e.g. worker_front
    pod_id: str
    crop_type: str
    trough_id: str
    side: str
    location: str
    priority: int
    title: str
    rationale: str
    evidence: dict
    targets: list
    steps: list
    requires_approval: bool
    approval_note: str = ""
    human_notes: list = field(default_factory=list)
    status: str = "awaiting_approval"
    progress: int = 0
    occurrences: int = 1
    stale: bool = False
    follow_up_of: str = ""
    not_before: str = ""
    created: str = ""
    updated: str = ""
    history: list = field(default_factory=list)
    ai: dict = field(default_factory=dict)          # AI planner verdict (if one was consulted)
    outcome: dict = field(default_factory=dict)     # result of the closed loop (before/after)
    aspect: str = ""                                # "disease" | "growth": which score panel shows this order's progress

    def to_dict(self):
        return asdict(self)


def _json_default(o):
    """Model outputs are numpy scalars; make them plain JSON numbers."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _nice(name):
    n = (name or "").replace("_", " ").replace("Cabbage-", "").replace(" on lettuce", "")
    return n.strip()


# ---------------------------------------------------------------------------
# Playbook: disease -> instruction category  (generic, for expert review)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "foliar_fungal": dict(
        label="foliar fungal / mildew disease", remove="affected leaves", chemical="fungicide",
        human=["Choose a fungicide approved for this crop and follow its label (product and dose are the human's call).",
               "Check airflow and leaf wetness around this pod."]),
    "bacterial": dict(
        label="bacterial disease", remove="affected leaves and tissue", chemical="bactericide",
        human=["Choose a bactericide approved for this crop only if you decide treatment is warranted.",
               "Avoid overhead watering; inspect neighbouring pods."]),
    "soilborne": dict(
        label="root / stem rot or wilt", remove="the whole plant and its root zone", chemical=None,
        human=["Inspect neighbouring pods and the shared water / substrate.",
               "Do not reuse the removed substrate."]),
    "viral": dict(
        label="viral disease (no cure)", remove="the whole plant", chemical=None,
        human=["Look for insect vectors (aphids, thrips) on neighbouring plants."]),
    "mushroom_bacterial": dict(
        label="bacterial blotch", remove="the affected mushrooms", chemical=None,
        human=["Reduce surface moisture / increase ventilation (room settings are the human's call)."]),
    "mushroom_fungal": dict(
        label="fungal contamination", remove="the affected mushrooms and the contaminated patch", chemical=None,
        human=["Check hygiene and air handling in this growing room."]),
    "review": dict(
        label="cause unclear", remove=None, chemical=None,
        human=["The model's class is ambiguous: a human should inspect this pod before any treatment."]),
}

DISEASE_CATEGORY = {
    ("cabbage", "Cabbage-Alternaria_leaf_spot"): "foliar_fungal",
    ("cabbage", "Cabbage-Bacterial_leaf_spot"): "bacterial",
    ("cabbage", "Cabbage-Black_rot"): "bacterial",
    ("cabbage", "Cabbage-Downy_mildew"): "foliar_fungal",
    ("cabbage", "Cabbage-Fusarium_wilt"): "soilborne",
    ("cabbage", "Cabbage-Powdery_mildew"): "foliar_fungal",
    ("cabbage", "Cabbage-Ringspot"): "foliar_fungal",
    ("cabbage", "Cabbage-Sclerotinia_rot"): "soilborne",
    ("lettuce", "Bacterial"): "bacterial",
    ("lettuce", "Downy_mildew_on_lettuce"): "foliar_fungal",
    ("lettuce", "Powdery_mildew_on_lettuce"): "foliar_fungal",
    ("lettuce", "Septoria_Blight_on_lettuce"): "foliar_fungal",
    ("lettuce", "Viral"): "viral",
    ("lettuce", "Wilt_and_leaf_blight_on_lettuce"): "review",
    ("mushroom", "Bacterial Blotch"): "mushroom_bacterial",
    ("mushroom", "Dry Bubble"): "mushroom_fungal",
    ("mushroom", "Trichoderma"): "mushroom_fungal",
    ("mushroom", "Wilt"): "review",
}
HARVEST_MODE = {"cabbage": "cut the head at the base", "lettuce": "cut the head at the base",
                "mushroom": "twist-pick the mature caps"}


def category_for(crop, disease):
    return DISEASE_CATEGORY.get((crop, disease), "review")


# ---------------------------------------------------------------------------
# Order construction (pure functions of the vision result)
# ---------------------------------------------------------------------------
def _add(steps, action, text, **params):
    steps.append({"seq": len(steps) + 1, "action": action, "text": text, "params": params})


def _targets(dets, size, only_class=None, exclude_class=None, limit=MAX_TARGETS):
    w, h = size
    sel = [d for d in dets
           if (only_class is None or d["class_name"] == only_class)
           and (exclude_class is None or d["class_name"] != exclude_class)]
    sel.sort(key=lambda d: -d["confidence"])
    out = []
    for d in sel[:limit]:
        x, y, bw, bh = d["box"]
        out.append({"class": d["class_name"], "confidence": round(float(d["confidence"]), 3),
                    "box_norm": [round(float(x) / w, 4), round(float(y) / h, 4),
                                 round(float(bw) / w, 4), round(float(bh) / h, 4)],
                    "center_norm": [round((float(x) + float(bw) / 2) / w, 4),
                                    round((float(y) + float(bh) / 2) / h, 4)]})
    return out


def _evidence(pod, result, size):
    c, d = result.classification, result.decision
    return {
        "visit": result.visit_num, "step": result.step_index, "n_steps": result.n_steps,
        "image": result.image_path, "image_size": list(size),
        "growth_stage": c["growth_stage"], "growth_confidence": round(float(c["growth_confidence"]), 3),
        "disease_flag": c["disease_flag"], "disease_name": c["disease_name"],
        "disease_confidence": round(float(c["disease_confidence"]), 3),
        "n_disease_detections": len(c["disease_detections"]),
        "n_growth_detections": len(c["growth_detections"]),
        "action": d["action"], "reason": d["reason"], "trend": result.trend_note,
        "model_sensitivity": result.conf_threshold, "scan_time": result.timestamp,
    }


def _nav(steps, pod, role):
    _add(steps, "NAVIGATE", f"Move the {role} agent to {pod['trough_id']} and align with pod {pod['pod_id']}.",
         trough=pod["trough_id"], pod=pod["pod_id"], side=pod["side"], rail=pod["rail"])


def build_treat(pod, result, size, policy):
    c = result.classification
    disease = c["disease_name"]
    cat = CATEGORIES[category_for(pod["crop_type"], disease)]
    from perception import MODEL_REGISTRY
    healthy = MODEL_REGISTRY[pod["crop_type"]]["disease"]["healthy_class"]
    targets = _targets(c["disease_detections"], size, only_class=disease) or \
        _targets(c["disease_detections"], size, exclude_class=healthy)
    n_det = sum(1 for d in c["disease_detections"] if d["class_name"] == disease)
    widespread = n_det >= WIDESPREAD_DETECTIONS
    steps = []
    _nav(steps, pod, "worker")
    if cat["remove"] is None:
        _add(steps, "MARK_POD", f"Flag {pod['pod_id']} for human inspection and keep other tools away from it.",
             pod=pod["pod_id"])
    else:
        if targets:
            _add(steps, "LOCATE", f"Align the tool with the {len(targets)} target region(s) from the vision analysis "
                                  f"(strongest: {targets[0]['confidence']:.2f}).", n_targets=len(targets))
        whole = cat["remove"].startswith(("the whole", "the affected mushrooms and"))
        _add(steps, "REMOVE_TISSUE", f"Remove {cat['remove']}" + ("." if whole else " at the target regions."),
             mode="whole_plant" if cat["remove"].startswith("the whole") else "targets")
        _add(steps, "BAG_AND_SEAL", "Bag and seal everything removed; nothing is left in the trough.")
        if cat["chemical"]:
            _add(steps, "APPLY_TREATMENT",
                 f"Apply the human-selected {cat['chemical']} to the treated area only (product chosen at approval).",
                 product="HUMAN_SELECTED", area="targets")
        _add(steps, "SANITIZE_TOOL", "Sanitize the tool before moving to another pod.")
    _add(steps, "REPORT", "Report completion, what was removed, and anything unexpected.")
    notes = list(cat["human"])
    if widespread:
        notes.insert(0, f"Widespread symptoms ({n_det} detections): decide whether to treat or remove the whole plant.")
    if c["disease_confidence"] < 0.65:
        notes.append(f"Moderate model confidence ({c['disease_confidence']:.2f}): glance at the photo before approving.")
    return dict(
        task="TREAT", role="worker", aspect="disease",
        title=f"Treat {_nice(disease)} on {pod['pod_id']}",
        rationale=f"{_nice(disease)} detected at {c['disease_confidence']:.2f} confidence "
                  f"({n_det} detection{'s' if n_det != 1 else ''}) -> {cat['label']} playbook.",
        targets=targets, steps=steps, requires_approval=True, human_notes=notes,
        approval_note=("Only flags the pod for inspection - approval requested so a human stays in the loop."
                       if cat["remove"] is None else
                       "Removes plant material" + (f" and applies a {cat['chemical']}" if cat["chemical"] else "")
                       + " - a human must approve."))


def build_harvest(pod, result, size, policy):
    c = result.classification
    stage = c["growth_stage"]
    targets = _targets(c["growth_detections"], size, only_class=stage)
    steps = []
    _nav(steps, pod, "worker")
    if targets:
        _add(steps, "LOCATE", f"Align the tool with the {len(targets)} harvest-stage target region(s).",
             n_targets=len(targets))
    _add(steps, "HARVEST", f"Harvest: {HARVEST_MODE.get(pod['crop_type'], 'harvest gently')}.", crop=pod["crop_type"])
    _add(steps, "PLACE_IN_CRATE", f"Place in the crate and label it {pod['pod_id']} with the time.", label=pod["pod_id"])
    _add(steps, "SANITIZE_TOOL", "Sanitize the tool before moving to another pod.")
    _add(steps, "REPORT", "Report completion and yield; pod is ready for replanting.")
    return dict(
        task="HARVEST", role="worker", aspect="growth", title=f"Harvest {pod['crop_type']} at {pod['pod_id']}",
        rationale=f"{pod['crop_type']} at harvest stage ({_nice(stage)}, {c['growth_confidence']:.2f} confidence), "
                  f"no confident disease reading.",
        targets=targets, steps=steps, requires_approval=True,
        human_notes=["After replanting, reset this pod's growth history so it restarts at the first stage."],
        approval_note="Harvesting is irreversible - approval required while auto-approve is off.")


def build_recapture(pod, result, size, policy):
    c = result.classification
    from perception import MODEL_REGISTRY
    healthy = MODEL_REGISTRY[pod["crop_type"]]["disease"]["healthy_class"]
    focus = _targets(c["disease_detections"], size, exclude_class=healthy, limit=3)
    if c["growth_stage"] is None and not c["disease_detections"]:
        why = "nothing readable in frame"
        adjust = "Re-frame so the whole pod fills the view, move closer, and raise the lighting."
    elif c["disease_flag"] and focus:
        why = f"suspected {_nice(c['disease_name'])} ({c['disease_confidence']:.2f}) is below the action threshold"
        adjust = "Zoom to the suspected region(s) and capture two angles."
    else:
        why = (f"growth stage unclear ({_nice(c['growth_stage']) if c['growth_stage'] else 'none'}, "
               f"{c['growth_confidence']:.2f})")
        adjust = "Capture the whole plant from the side and from above."
    steps = []
    _nav(steps, pod, "camera")
    _add(steps, "ADJUST_VIEW", adjust, focus_regions=len(focus))
    _add(steps, "CAPTURE", "Capture 3 frames (bracket exposure if the camera supports it).", frames=3)
    _add(steps, "UPLOAD", "Send the frames to the vision pipeline tagged with this order id.")
    _add(steps, "REQUEUE", f"Pod is re-inspected within {policy.recapture_revisit_hours} h.",
         revisit_hours=policy.recapture_revisit_hours)
    return dict(
        task="RECAPTURE", role="camera", aspect="disease" if (c["disease_flag"] and focus) else "growth",
        title=f"Re-capture {pod['pod_id']}",
        rationale=f"The last reading was too uncertain to act on: {why}.",
        targets=focus, steps=steps, requires_approval=False, human_notes=[], approval_note="")


def build_verify(treat):
    steps = []
    pod = {"trough_id": treat.trough_id, "pod_id": treat.pod_id, "side": treat.side,
           "rail": treat.evidence.get("rail", 0)}
    _nav(steps, pod, "camera")
    _add(steps, "CAPTURE", "Close-up of the treated area / target regions for this pod.", frames=3)
    _add(steps, "UPLOAD", "Send frames to the vision pipeline tagged with this order id.")
    _add(steps, "COMPARE", "Pipeline compares the new disease reading with the visit that triggered treatment; "
                           "if it is still above threshold the case is escalated to a human.")
    return steps


def build_replant(harvest):
    """Growth-side worker job after a harvest: the pod is empty, start the next crop."""
    pod = {"trough_id": harvest.trough_id, "pod_id": harvest.pod_id, "side": harvest.side,
           "rail": harvest.evidence.get("rail", 0)}
    start = ("Install a fresh spawn block / substrate" if harvest.crop_type == "mushroom"
             else f"Plant a new {harvest.crop_type} seedling")
    steps = []
    _nav(steps, pod, "worker")
    _add(steps, "CLEAR_POD", "Remove remaining roots, stalks and old substrate; bag and seal them.")
    _add(steps, "SANITIZE_TOOL", "Sanitize the tool and the pod before planting.")
    _add(steps, "PLANT", f"{start} in {harvest.pod_id}.", crop=harvest.crop_type)
    _add(steps, "WATER_FEED", "Water / feed with the pod's standard recipe.")
    _add(steps, "REPORT", "Report completion; the pod restarts at the first growth stage.")
    return steps


def build_review(pod, result, size, policy, reason, notes):
    """Escalation to a human: used when the automatic loop could not settle a case."""
    from perception import MODEL_REGISTRY
    c = result.classification
    healthy = MODEL_REGISTRY[pod["crop_type"]]["disease"]["healthy_class"]
    steps = []
    _add(steps, "HUMAN_INSPECT", f"Look at pod {pod['pod_id']} (in person or on the attached photo).", pod=pod["pod_id"])
    _add(steps, "HUMAN_DECIDE", "Decide: treat, harvest, re-capture again, or leave it. Create the order yourself if needed.")
    return dict(
        task="REVIEW", role="human", title=f"Human review needed: {pod['pod_id']}", rationale=reason,
        targets=_targets(c["disease_detections"], size, exclude_class=healthy), steps=steps,
        requires_approval=True, human_notes=list(notes),
        approval_note="The automatic loop could not settle this case - a human decision is needed.")


BUILDERS = {"flag_for_treatment": build_treat, "flag_for_harvest": build_harvest,
            "schedule_frequent_monitoring": build_recapture}
TASK_FOR_ACTION = {"flag_for_treatment": "TREAT", "flag_for_harvest": "HARVEST",
                   "schedule_frequent_monitoring": "RECAPTURE", "log_healthy": None}


# ---------------------------------------------------------------------------
# Outbox (swap for MQTT / IoT Core / SQS / ROS 2)
# ---------------------------------------------------------------------------
class FileOutbox:
    def __init__(self, root):
        self.root = root

    def send(self, order):
        d = os.path.join(self.root, order["agent"])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, order["order_id"] + ".json"), "w") as f:
            json.dump(order, f, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------
class OrderBook:
    def __init__(self, out_dir, policy=None, on_change=None, transport=None):
        self.policy = policy or OrderPolicy()
        self.out_dir = out_dir
        self.on_change = on_change or (lambda: None)
        self.transport = transport or FileOutbox(os.path.join(out_dir, "outbox"))
        self._lock = threading.RLock()
        self._orders = {}
        self._counter = 0
        self.version = 0
        self.fleet = None
        self._written = False
        self.advisor = None                       # optional LlmPlanner (see llm_planner.py)
        self._planner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-planner")
        self._llm_downgraded = {}                 # pod_id -> did the last AI consult ask for a recapture?
        self._timers = []
        self._loop_pending = {}                   # RECAPTURE order id -> pod, until its follow-up frame is analysed
        self.on_complete = None                   # engine hook: called when a RECAPTURE / VERIFY order finishes

    # ----- planning --------------------------------------------------------
    @staticmethod
    def _image_size(path):
        img = cv2.imread(path)
        return (img.shape[1], img.shape[0]) if img is not None else (1, 1)

    def plan_from_result(self, pod, result, history_rows=None, follow_up_of="", frame_note=""):
        """Called by the engine for every LOGGED scan, and by close_loop() for follow-up frames.
        Returns [(order_id, 'created'|'updated'|'superseded'), ...]."""
        if result.kind not in ("scan", "followup"):
            return []
        action = result.decision["action"]
        events = []
        size = self._image_size(result.image_path)
        with self._lock:
            self._observe(pod, action, result)
            builder = BUILDERS.get(action)
            if builder is None:
                return events
            spec = builder(pod, result, size, self.policy)
            ev = _evidence(pod, result, size)
            ev["rail"] = pod["rail"]
            if follow_up_of:
                ev["frame_source"] = frame_note or "follow-up frame"
            order, kind = self._upsert(pod, spec, ev)
            if follow_up_of and kind == "created":
                order.follow_up_of = follow_up_of
            events.append((order.order_id, kind))
            if spec["task"] == "TREAT":
                for o in self._orders.values():
                    if o.pod_id == pod["pod_id"] and o.task == "RECAPTURE" and o.status in OPEN_STATUSES:
                        self._set(o, "cancelled", "superseded by a treatment order for the same pod")
                        events.append((o.order_id, "superseded"))
            if kind == "created":
                self._trace("order_created", order, extra={"decision": result.decision["action"], "evidence": ev})
                if self.advisor is not None and spec["task"] in CONSULT_TASKS:
                    self._set(order, "planning", "asking the AI planner")
                    self._planner_pool.submit(self._consult, order.order_id, pod, result, history_rows or [])
                else:
                    self._open_for_work(order)
        self._changed()
        return events

    def _open_for_work(self, order):
        """Gatekeeper for every order. Auto-approve ON: robot orders are approved and dispatched at once.
        Auto-approve OFF (manual mode): EVERY robot order waits for a human, including camera re-captures.
        Escalations to a human (REVIEW) always wait."""
        if order.task == "REVIEW":
            self._set(order, "awaiting_approval", "waiting for a human decision")
        elif self.policy.auto_approve_all:
            self._set(order, "approved", "auto-approved (auto-approve is ON)" if order.requires_approval
                      else "no approval required by policy")
            self._dispatch(order)
        else:
            if not order.approval_note:
                order.approval_note = "Manual approval mode is on - approve to send this to the robot."
            self._set(order, "awaiting_approval", "waiting for human approval (manual mode)")

    # ----- AI planner ------------------------------------------------------
    def set_advisor(self, advisor):
        """advisor: an llm_planner.LlmPlanner, or None for rules only. Orders still 'planning' are released to the rules."""
        self.advisor = advisor
        if advisor is None:
            with self._lock:
                for o in list(self._orders.values()):
                    if o.status == "planning":
                        o.ai = {"backend": "-", "verdict": "fallback", "why": "AI planner switched off: rules kept",
                                "proposal": None}
                        self._open_for_work(o)
        self._changed()

    def _consult(self, order_id, pod, result, history_rows):
        """Runs on the single planner thread (an LLM call may take seconds; scanning never waits for it)."""
        from perception import MODEL_REGISTRY
        adv, res = self.advisor, None
        if adv is not None:
            prior = self._llm_downgraded.get(pod["pod_id"], False)
            try:
                _, res = adv.plan(pod, result.classification, result.decision, history_rows, result.step_index,
                                  result.n_steps, MODEL_REGISTRY[pod["crop_type"]]["growth"]["harvest_stage"],
                                  image_path=result.image_path, prior_downgrade=prior)
            except Exception:
                res = None
        with self._lock:
            o = self._orders.get(order_id)
            if o is None or o.status != "planning":
                return
            self._finish_planning(o, pod, result, res)
        self._changed()

    @staticmethod
    def _ai_dict(res):
        p = res.proposal
        return {"backend": res.source, "verdict": res.verdict, "why": res.why, "rule_task": res.rule_task,
                "final_task": res.final_task,
                "proposal": None if p is None else {"task": p.task, "reason": p.reason, "urgency": p.urgency,
                                                     "notes": list(p.notes), "ignored_keys": list(p.ignored_keys)}}

    def _finish_planning(self, o, pod, result, res):
        o.ai = self._ai_dict(res) if res is not None else {
            "backend": "?", "verdict": "fallback", "why": "AI planner failed: rules kept", "proposal": None}
        self._trace("ai_consulted", o, extra={"ai": o.ai})
        self._llm_downgraded[o.pod_id] = (o.ai["verdict"] == "downgraded")
        prop = o.ai.get("proposal")
        if o.ai["verdict"] == "downgraded":
            reason = prop["reason"] if prop else ""
            self._set(o, "cancelled", f"AI planner asked for a better look first: {reason}")
            spec = build_recapture(pod, result, tuple(o.evidence["image_size"]), self.policy)
            new, kind = self._upsert(pod, spec, copy.deepcopy(o.evidence))
            if kind == "created":
                new.ai = copy.deepcopy(o.ai)
                new.rationale = f"AI planner asked for a better look before {o.task.lower()}: {reason} | " + new.rationale
                self._trace("order_created", new, extra={"from": o.order_id, "by": "ai_planner"})
                self._open_for_work(new)
            return
        if prop:
            if prop["urgency"] == "high":                 # a 'low' urgency is never allowed to delay anything
                o.priority = max(1, o.priority - 1)
            o.human_notes = [f"AI: {n}" for n in prop["notes"]] + o.human_notes
        self._open_for_work(o)

    def _observe(self, pod, action, result):
        """A newer reading that contradicts an order still waiting for a human marks it stale."""
        expected = TASK_FOR_ACTION.get(action)
        for o in self._orders.values():
            if o.pod_id == pod["pod_id"] and o.status == "awaiting_approval" and o.task in ("TREAT", "HARVEST"):
                stale = expected != o.task
                if stale != o.stale:
                    o.stale = stale
                    self._note(o, (f"visit {result.visit_num}" if result.visit_num else "follow-up frame") + ": latest reading says "
                                  f"{action} - " + ("review before approving" if stale else "supports this order again"))

    def _upsert(self, pod, spec, ev):
        for o in self._orders.values():
            if o.pod_id == pod["pod_id"] and o.task == spec["task"] and o.status in OPEN_STATUSES:
                o.occurrences += 1
                if o.status in ("awaiting_approval", "scheduled"):     # not started: refresh with newest evidence
                    o.evidence, o.targets, o.steps, o.rationale = ev, spec["targets"], spec["steps"], spec["rationale"]
                    o.stale = False
                self._note(o, f"seen again at visit {ev['visit'] or 'follow-up'} (x{o.occurrences})")
                return o, "updated"
        self._counter += 1
        side = pod["side"]
        o = WorkOrder(
            order_id=f"WO-{self._counter:04d}", task=spec["task"], role=spec["role"],
            agent="human" if spec["role"] == "human" else f"{spec['role']}_{side}", pod_id=pod["pod_id"], crop_type=pod["crop_type"],
            trough_id=pod["trough_id"], side=side,
            location=f"rail {pod['rail']} · trough {pod['trough_id']} · {side}",
            priority=PRIORITY[spec["task"]], title=spec["title"], rationale=spec["rationale"],
            evidence=ev, targets=spec["targets"], steps=spec["steps"],
            requires_approval=spec["requires_approval"], approval_note=spec["approval_note"],
            human_notes=spec["human_notes"], aspect=spec.get("aspect", ""), created=_now(), updated=_now())
        self._orders[o.order_id] = o
        return o, "created"

    # ----- human / robot actions ------------------------------------------
    def approve(self, order_id, by="human"):
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o.status not in ("awaiting_approval", "scheduled"):
                return False
            if o.task == "REVIEW":                       # a human decision, not a robot job
                self._set(o, "approved", f"acknowledged by {by}")
                self._set(o, "completed", "reviewed by human")
            else:
                self._set(o, "approved", f"approved by {by}")
                self._dispatch(o)
        self._changed()
        return True

    def approve_all_pending(self):
        """Bulk-approves robot jobs. Escalations to a human (REVIEW) are never bulk-dismissed."""
        with self._lock:
            ids = [o.order_id for o in self._orders.values() if o.status == "awaiting_approval" and o.task != "REVIEW"]
        return sum(self.approve(i) for i in ids)

    def reject(self, order_id):
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o.status not in ("awaiting_approval", "scheduled"):
                return False
            self._set(o, "rejected", "rejected by human")
        self._changed()
        return True

    def begin(self, order_id, agent=None):
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o.status != "dispatched":
                return None
            self._set(o, "in_progress", f"{agent or o.agent} started")
            n = len(o.steps)
        self._changed()
        return n

    def advance(self, order_id):
        with self._lock:
            o = self._orders.get(order_id)
            if o and o.status == "in_progress":
                o.progress = min(len(o.steps), o.progress + 1)
                o.updated = _now()
                self._write(o)
        self._changed()

    def complete(self, order_id, note="completed"):
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o.status not in ("dispatched", "in_progress", "approved"):
                return False
            o.progress = len(o.steps)
            self._set(o, "completed", note)
            if o.task == "TREAT" and not any(
                    x.follow_up_of == o.order_id for x in self._orders.values()):
                self._spawn_verify(o)
            if o.task == "HARVEST" and not any(x.follow_up_of == o.order_id and x.task == "REPLANT"
                                               for x in self._orders.values()):
                self._spawn_replant(o)
            trigger = o.task in ("RECAPTURE", "VERIFY")
            if o.task == "RECAPTURE" and self.policy.close_the_loop and self.on_complete is not None:
                self._loop_pending[order_id] = o.pod_id      # the pod stays 'busy' until the follow-up frame is analysed
        self._changed()
        if trigger and self.policy.close_the_loop and self.on_complete is not None:
            self.on_complete(order_id)            # engine takes a NEW frame and calls close_loop()
        return True

    report = complete          # robots can call book.report(order_id, note)

    def _spawn_replant(self, harvest):
        self._counter += 1
        r = WorkOrder(
            order_id=f"WO-{self._counter:04d}", task="REPLANT", role="worker", agent=f"worker_{harvest.side}",
            pod_id=harvest.pod_id, crop_type=harvest.crop_type, trough_id=harvest.trough_id, side=harvest.side,
            location=harvest.location, priority=PRIORITY["REPLANT"], title=f"Replant {harvest.crop_type} at {harvest.pod_id}",
            rationale=f"Harvest {harvest.order_id} finished: the pod is empty, so start the next crop.",
            evidence=copy.deepcopy(harvest.evidence), targets=[], steps=build_replant(harvest),
            requires_approval=False, human_notes=["This pod's growth history is NOT reset automatically - reset it "
                                                  "when the new crop is in (see README)."],
            follow_up_of=harvest.order_id, aspect="growth", created=_now(), updated=_now())
        self._orders[r.order_id] = r
        self._trace("order_created", r, extra={"follow_up_of": harvest.order_id})
        self._open_for_work(r)

    def _spawn_verify(self, treat):
        self._counter += 1
        auto = self.policy.auto_approve_all
        delta = (timedelta(seconds=self.policy.auto_verify_delay_s) if auto
                 else timedelta(hours=self.policy.verify_after_hours))
        due = (datetime.now(timezone.utc) + delta).isoformat(timespec="seconds")
        v = WorkOrder(
            order_id=f"WO-{self._counter:04d}", task="VERIFY", role="camera", agent=f"camera_{treat.side}",
            pod_id=treat.pod_id, crop_type=treat.crop_type, trough_id=treat.trough_id, side=treat.side,
            location=treat.location, priority=PRIORITY["VERIFY"], title=f"Verify treatment on {treat.pod_id}",
            rationale=f"Treatment {treat.order_id} finished: re-inspect "
                      + (f"in {self.policy.auto_verify_delay_s:.0f} s (auto mode, demo timing)" if auto
                         else f"after {self.policy.verify_after_hours} h") + " to confirm it worked.",
            evidence=copy.deepcopy(treat.evidence), targets=copy.deepcopy(treat.targets),
            steps=build_verify(treat), requires_approval=False, follow_up_of=treat.order_id,
            not_before=due, aspect="disease", created=_now(), updated=_now())
        self._orders[v.order_id] = v
        self._trace("order_created", v, extra={"follow_up_of": treat.order_id})
        self._set(v, "scheduled", f"due {due}" + (" (auto mode releases it)" if auto else " (use 'Run now' to release early)"))
        if auto:
            self._schedule_auto_release(v.order_id, self.policy.auto_verify_delay_s)

    # ----- closing the loop --------------------------------------------------
    @staticmethod
    def _summary(c, action=None):
        out = {"growth_stage": c["growth_stage"], "growth_confidence": round(float(c["growth_confidence"]), 3),
               "disease_flag": bool(c["disease_flag"]), "disease_name": c["disease_name"],
               "disease_confidence": round(float(c["disease_confidence"]), 3),
               "n_disease_detections": len(c["disease_detections"])}
        if action:
            out["action"] = action
        return out

    def close_loop(self, order_id, pod, result, frame_note, simulated=True, history_rows=None):
        """A RECAPTURE / VERIFY order finished and a NEW frame was analysed (`result`, kind='followup').
        Compare with what the order was based on and decide what happens next:

            RECAPTURE  better frame says treat / harvest   -> create that order (linked to this one)
                       better frame says healthy           -> resolved, nothing more to do
                       still uncertain                     -> escalate to a human (REVIEW)
            VERIFY     disease no longer above threshold   -> treatment verified
                       still flagged                       -> escalate to a human (REVIEW)
        """
        with self._lock:
            o = self._orders.get(order_id)
            if not o or o.task not in ("RECAPTURE", "VERIFY"):
                return None
            c, action = result.classification, result.decision["action"]
            size = self._image_size(result.image_path)
            b = {k: o.evidence[k] for k in ("growth_stage", "growth_confidence", "disease_flag", "disease_name",
                                            "disease_confidence", "n_disease_detections", "action")}
            a = self._summary(c, action)
            out = {"result": "", "headline": "", "before": b, "after": a,
                   "after_boxes": _targets(c["disease_detections"], size, limit=12)
                                  + _targets(c["growth_detections"], size, limit=4),
                   "after_size": list(size), "frame": result.image_path, "frame_note": frame_note,
                   "simulated": simulated, "next_order": ""}
            nice = _nice(c["disease_name"]) if c["disease_name"] else ""
            if o.task == "RECAPTURE":
                if action in ("flag_for_treatment", "flag_for_harvest"):
                    out["result"] = "confirmed_disease" if action == "flag_for_treatment" else "confirmed_harvest"
                    evs = self.plan_from_result(pod, result, history_rows=history_rows, follow_up_of=order_id,
                                                frame_note=frame_note)
                    made = [e[0] for e in evs if e[1] in ("created", "updated")]
                    out["next_order"] = made[0] if made else ""
                    what = (f"{nice} ({a['disease_confidence']:.2f})" if action == "flag_for_treatment"
                            else f"harvest stage ({_nice(a['growth_stage'])}, {a['growth_confidence']:.2f})")
                    out["headline"] = f"Better frame confirms {what} -> {'treatment' if action == 'flag_for_treatment' else 'harvest'} order {out['next_order']}"
                elif action == "log_healthy":
                    out["result"] = "resolved_healthy"
                    out["headline"] = "Better frame reads healthy -> uncertainty resolved, no action needed"
                else:
                    out["result"] = "still_uncertain"
                    rv = self._escalate(pod, result, size, o, history_rows,
                                        "The re-capture did not settle the reading (still low confidence).",
                                        ["Two looks were not enough for the model: please inspect this pod yourself."])
                    out["next_order"] = rv
                    out["headline"] = f"Still uncertain after the re-capture -> escalated to a human ({rv})"
            else:   # VERIFY
                if action == "flag_for_treatment":
                    out["result"] = "treatment_not_resolved"
                    rv = self._escalate(pod, result, size, o, history_rows,
                                        f"Still flagged after treatment: {nice} {a['disease_confidence']:.2f} "
                                        f"(was {b['disease_confidence']:.2f}; {a['n_disease_detections']} vs "
                                        f"{b['n_disease_detections']} boxes).",
                                        ["The treatment did not clear it (or only part of it was reached)."])
                    out["next_order"] = rv
                    out["headline"] = (f"Still flagged after treatment ({b['disease_confidence']:.2f} -> "
                                       f"{a['disease_confidence']:.2f}) -> escalated to a human ({rv})")
                else:
                    out["result"] = "treatment_worked"
                    out["headline"] = (f"Treatment verified: disease reading {b['disease_confidence']:.2f} -> "
                                       f"{a['disease_confidence']:.2f} ({b['n_disease_detections']} -> "
                                       f"{a['n_disease_detections']} boxes), below the action threshold")
            o.outcome = out
            self._note(o, "loop closed: " + out["headline"])
            self._trace("loop_closed", o, extra={"outcome": {k: out[k] for k in ("result", "headline", "next_order",
                                                                                  "simulated")}})
        self._changed()
        return out

    def _escalate(self, pod, result, size, origin, history_rows, reason, notes):
        spec = build_review(pod, result, size, self.policy, reason, notes)
        ev = _evidence(pod, result, size)
        ev["rail"] = pod["rail"]
        ev["frame_source"] = "follow-up frame"
        order, kind = self._upsert(pod, spec, ev)
        if kind == "created":
            order.follow_up_of = origin.order_id
            self._trace("order_created", order, extra={"escalated_from": origin.order_id})
            self._open_for_work(order)
        return order.order_id

    # ----- internals -------------------------------------------------------
    def _dispatch(self, o):
        try:
            self.transport.send(o.to_dict())
        except Exception as exc:
            self._note(o, f"outbox error: {exc!r}")
        self._set(o, "dispatched", f"sent to {o.agent}")
        if self.fleet is not None:
            self.fleet.submit(o)

    def _note(self, o, text):
        o.history.append([_now(), o.status, text])
        o.updated = _now()
        self._write(o)

    def _set(self, o, status, note=""):
        o.status = status
        self._note(o, note)
        self._trace("status", o, extra={"note": note})

    def _write(self, o):
        try:
            d = os.path.join(self.out_dir, "orders")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, o.order_id + ".json"), "w") as f:
                json.dump(o.to_dict(), f, indent=2, default=_json_default)
        except (OSError, TypeError, ValueError):
            pass

    def _trace(self, event, o, extra=None):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            rec = {"ts": _now(), "event": event, "order_id": o.order_id, "task": o.task, "agent": o.agent,
                   "pod_id": o.pod_id, "status": o.status}
            rec.update(extra or {})
            with open(os.path.join(self.out_dir, "trace.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=_json_default) + "\n")
        except (OSError, TypeError, ValueError):
            pass

    def _changed(self):
        self.version += 1
        self.on_change()

    # ----- queries (all return copies) ------------------------------------
    def list_orders(self):
        with self._lock:
            rank = lambda o: (0 if o.status in OPEN_STATUSES else 1, o.priority, o.order_id)  # noqa: E731
            return [copy.deepcopy(o) for o in sorted(self._orders.values(), key=rank)]

    def get(self, order_id):
        with self._lock:
            o = self._orders.get(order_id)
            return copy.deepcopy(o) if o else None

    def for_pod(self, pod_id):
        """Light tuples for the pod card:
        (order_id, agent, task, status, progress, n_steps, stale, priority, current_step_action)."""
        with self._lock:
            rows = [(o.order_id, o.agent, o.task, o.status, o.progress, len(o.steps), o.stale, o.priority,
                     o.steps[min(o.progress, len(o.steps) - 1)]["action"] if o.steps else "")
                    for o in self._orders.values() if o.pod_id == pod_id and o.status in OPEN_STATUSES]
        return sorted(rows, key=lambda r: (r[7], r[0]))

    def loop_done(self, order_id):
        """Engine calls this when the follow-up frame for a RECAPTURE has been analysed (or failed)."""
        with self._lock:
            self._loop_pending.pop(order_id, None)
        self._changed()

    def pod_busy(self, pod_id):
        """True while robots still have work on this pod: the pod's next image must wait."""
        with self._lock:
            if pod_id in self._loop_pending.values():
                return True
            return any(o.pod_id == pod_id and o.task in BLOCKING_TASKS and o.status in BLOCKING_STATUSES
                       for o in self._orders.values())

    def pod_panel_orders(self, pod_id, recent_s=3.0):
        """For the pod card: the order to show in the DISEASE panel and in the GROWTH panel (copies), or None.
        An open order wins (in progress > sent > waiting for approval > planning); a just-finished one is kept
        for `recent_s` seconds so you see it complete."""
        now = datetime.now(timezone.utc)
        rank = {"in_progress": 0, "dispatched": 1, "approved": 1, "awaiting_approval": 2, "planning": 3}
        best = {"disease": None, "growth": None}
        with self._lock:
            for o in self._orders.values():
                if o.pod_id != pod_id or o.aspect not in best:
                    continue
                if o.status in rank:
                    key = (rank[o.status], -self._seq(o))
                elif o.status in ("completed", "rejected", "cancelled"):
                    try:
                        age = (now - datetime.fromisoformat(o.updated)).total_seconds()
                    except ValueError:
                        continue
                    if age > recent_s:
                        continue
                    key = (4, -self._seq(o))
                else:
                    continue
                if best[o.aspect] is None or key < best[o.aspect][0]:
                    best[o.aspect] = (key, o)
            return {k: (copy.deepcopy(v[1]) if v else None) for k, v in best.items()}

    @staticmethod
    def _seq(o):
        return int(o.order_id.split("-")[1])

    def fleet_status(self):
        """One snapshot per agent for the always-visible strip in the main window: what it is doing right now,
        what is queued for it, what is waiting for the human, and what it finished last. Works the same for the
        built-in simulator and for real robots that report progress through begin/advance/complete."""
        with self._lock:
            res = {a: {"active": None, "queued": 0, "awaiting": 0, "last_done": None} for a in ALL_AGENTS}
            for o in self._orders.values():
                r = res.get(o.agent)
                if r is None:                       # e.g. REVIEW orders belong to the human, not a robot
                    continue
                n = len(o.steps)
                info = {"order_id": o.order_id, "task": o.task, "pod": o.pod_id, "n": n, "progress": o.progress,
                        "step": min(o.progress + 1, n), "updated": o.updated,
                        "action": o.steps[min(o.progress, n - 1)]["action"] if n else ""}
                if o.status == "in_progress":
                    r["active"] = info
                elif o.status in ("approved", "dispatched"):
                    r["queued"] += 1
                elif o.status == "awaiting_approval":
                    r["awaiting"] += 1
                elif o.status == "completed" and (r["last_done"] is None or o.updated >= r["last_done"]["updated"]):
                    r["last_done"] = info
            return res

    def pending_count(self):
        """Everything waiting on a human (robot jobs to approve + escalations to review)."""
        with self._lock:
            return sum(1 for o in self._orders.values() if o.status == "awaiting_approval")

    def pending_robot_count(self):
        """Only robot jobs: what 'Approve all' acts on (REVIEW escalations are never bulk-dismissed)."""
        with self._lock:
            return sum(1 for o in self._orders.values() if o.status == "awaiting_approval" and o.task != "REVIEW")

    def open_count(self):
        with self._lock:
            return sum(1 for o in self._orders.values() if o.status in OPEN_STATUSES)

    def total(self):
        with self._lock:
            return len(self._orders)

    # ----- policy / simulator ---------------------------------------------
    def set_auto_approve_all(self, on):
        """Flip the master switch. Turning it ON also releases robot orders already waiting for approval and
        schedules due follow-ups; turning it OFF affects only orders created from now on."""
        self.policy.auto_approve_all = on
        self._trace_policy(on)
        if on:
            with self._lock:
                waiting = [o.order_id for o in self._orders.values()
                           if o.status == "awaiting_approval" and o.task != "REVIEW"]
                scheduled = [o.order_id for o in self._orders.values() if o.status == "scheduled"]
            for oid in waiting:
                self.approve(oid, by="auto-approve switched on")
            for oid in scheduled:
                self._schedule_auto_release(oid, 2.0)
        self._changed()

    def _trace_policy(self, on):
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "trace.jsonl"), "a") as f:
                f.write(json.dumps({"ts": _now(), "event": "policy_changed", "auto_approve_all": on}) + "\n")
        except OSError:
            pass

    def _schedule_auto_release(self, order_id, delay):
        t = threading.Timer(delay, self._auto_release, args=(order_id,))
        t.daemon = True
        self._timers.append(t)
        t.start()

    def _auto_release(self, order_id):
        with self._lock:
            o = self._orders.get(order_id)
            ok = o is not None and o.status == "scheduled" and self.policy.auto_approve_all
        if ok:
            self.approve(order_id, by="auto mode (follow-up due)")

    def set_simulation(self, on):
        self.policy.simulate_robots = on
        if on and self.fleet is None:
            self.fleet = SimulatedFleet(self)
            with self._lock:      # pick up anything already dispatched
                for o in self._orders.values():
                    if o.status == "dispatched":
                        self.fleet.submit(o)
        elif not on and self.fleet is not None:
            self.fleet.stop()
            self.fleet = None
            self._revert_in_progress()
        self._changed()

    def _revert_in_progress(self):
        with self._lock:
            for o in self._orders.values():
                if o.status == "in_progress":
                    o.progress = 0
                    self._set(o, "dispatched", "robot simulation switched off - waiting for a real agent")

    def shutdown(self):
        self.on_complete = None
        self._loop_pending.clear()
        for t in self._timers:
            t.cancel()
        self._planner_pool.shutdown(wait=False, cancel_futures=True)
        if self.fleet is not None:
            self.fleet.stop()
            self.fleet = None
        self._revert_in_progress()
        with self._lock:
            for o in list(self._orders.values()):
                if o.status == "planning":
                    o.ai = {"backend": "-", "verdict": "fallback", "why": "run ended before the AI answered: rules kept",
                            "proposal": None}
                    self._open_for_work(o)


# ---------------------------------------------------------------------------
# Built-in fleet simulator (stands in for the 4 physical agents)
# ---------------------------------------------------------------------------
class SimulatedFleet:
    def __init__(self, book):
        self.book = book
        self._stop = threading.Event()
        self._cv = threading.Condition()
        self._heaps = {a: [] for a in ALL_AGENTS}
        self._seq = 0
        self.state = {a: {"status": "idle", "order": None, "step": 0, "n": 0} for a in ALL_AGENTS}
        self._threads = [threading.Thread(target=self._loop, args=(a,), daemon=True, name=f"sim-{a}")
                         for a in ALL_AGENTS]
        for t in self._threads:
            t.start()

    def submit(self, order):
        with self._cv:
            self._seq += 1
            heapq.heappush(self._heaps[order.agent], (order.priority, self._seq, order.order_id))
            self._cv.notify_all()

    def queue_len(self, agent):
        with self._cv:
            return len(self._heaps[agent])

    def stop(self):
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        for t in self._threads:
            t.join(timeout=3)

    def _loop(self, agent):
        while not self._stop.is_set():
            with self._cv:
                while not self._heaps[agent] and not self._stop.is_set():
                    self._cv.wait(0.3)
                if self._stop.is_set():
                    return
                _, _, oid = heapq.heappop(self._heaps[agent])
            n = self.book.begin(oid, agent)
            if not n:
                continue
            for i in range(n):
                self.state[agent] = {"status": "busy", "order": oid, "step": i, "n": n}
                if self._stop.wait(self.book.policy.sim_step_seconds):
                    return
                self.book.advance(oid)
            self.book.complete(oid, note=f"completed by simulated {agent}")
            self.state[agent] = {"status": "idle", "order": None, "step": 0, "n": 0}
