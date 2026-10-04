"""
farm_engine.py

The run engine behind the interactive app (app_ui.py). It contains NO
drawing code at all -- it owns the worker threads, the pod queue, and
every "interact with it" control, and tells whoever is listening what
happened through a thread-safe event queue.

It reuses the existing pipeline exactly as concurrent_monitor.py does
(camera.py -> cloud_pipeline.py -> perception.py / agent.py, then
dynamo_client.py, trend.py, run_logger.py). Nothing about detection or
decision-making is reimplemented here.

What's new compared to concurrent_monitor.py is control:

    pause() / resume()      halt / continue the whole run (in-flight
                            inference finishes first, then nothing new
                            starts)
    step_once()             while paused, advance every un-held slot by
                            exactly one image
    stop()                  end the run gracefully, flush logs + history
    toggle_hold(slot)       freeze ONE slot on its pod, others carry on
    skip_pod(slot)          abandon the slot's current pod, move on
    rescan(slot)            re-run the model on the image on screen (e.g.
                            after changing the sensitivity). Not logged,
                            does not advance the pod's growth history.
    set_view(slot, idx)     browse that pod's earlier images (review mode)
    run_next(pod_id)        move a queued pod to the front of the queue
    skip_queued(pod_id)     drop a queued pod from this run
    settings                live-editable: delay, sensitivity, thresholds

Events put on `engine.events` (consumed by the UI on its own thread):
    ("slot", slot_id)   ("queue", None)   ("state", state_str)
    ("log", (time_str, level, text))      ("stopped", info_dict)
    ("finished", info_dict)
"""

import collections
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import perception
from agent import DISEASE_ACTION_THRESHOLD, GROWTH_ACTION_THRESHOLD, decide_action
from camera import PodCamera
from cloud_pipeline import LocalCloudClient
from dynamo_client import LocalDynamoTable
from frame_source import SimulatedCamera
from llm_planner import RULE_TASK, LlmPlanner, OllamaBackend, ScriptedBackend
from perception import CONF_THRESHOLD, MODEL_REGISTRY
from run_logger import RunLogger
from trend import summarize_trend
from worker_planner import OrderBook, OrderPolicy

FEEDBACK_TIMEOUT_S = 15.0
ROBOT_DONE_DWELL_S = 3.0   # after waiting for robots, keep the finished job on screen this long
MIN_DWELL_S = 1.2          # how long a finished pod stays on screen before the slot moves on
ACTIONS = ["log_healthy", "flag_for_harvest", "schedule_frequent_monitoring", "flag_for_treatment"]


@dataclass
class Settings:
    """Live-editable. Workers read these fresh on every step, so a slider
    moved in the UI changes the very next scan -- no restart needed."""
    delay_ms: int = 700                                    # pause between images in a slot
    conf_threshold: float = CONF_THRESHOLD                 # model sensitivity (lower = more boxes)
    disease_threshold: float = DISEASE_ACTION_THRESHOLD    # confidence needed to act on a disease
    growth_threshold: float = GROWTH_ACTION_THRESHOLD      # confidence needed to trust a growth stage


@dataclass(frozen=True)
class StepResult:
    """One scan of one image. Immutable so the UI can hold on to it freely."""
    pod_id: str
    crop_type: str
    step_index: int          # index into the pod's image sequence (0-based)
    n_steps: int             # length of that sequence
    visit_num: int           # persisted visit number (0 for un-logged re-scans)
    image_path: str
    classification: dict
    decision: dict
    trend_note: str
    latency_s: float
    worker: str
    timestamp: str
    kind: str                # "scan" or "rescan"
    logged: bool
    conf_threshold: float
    verified: dict           # {"aspect": "growth"|"disease"|None, "expected": str, "ok": bool|None}


# ---------------------------------------------------------------------------
# Pure helpers (also used by the UI)
# ---------------------------------------------------------------------------

def class_scores(classification, kind):
    """Per-class best confidence for one model, as [(class_name, best_conf,
    n_detections)] in the model's class order. Every class the model can
    output is listed (0.0 when it didn't detect it) so the UI can draw a
    full score table, not just the winner. Derived from the real
    detections classify_pod() returned -- nothing is invented."""
    classes = MODEL_REGISTRY[classification["crop_type"]][kind]["classes"]
    dets = classification["growth_detections" if kind == "growth" else "disease_detections"]
    best = {name: 0.0 for name in classes.values()}
    count = {name: 0 for name in classes.values()}
    for d in dets:
        best[d["class_name"]] = max(best[d["class_name"]], d["confidence"])
        count[d["class_name"]] += 1
    return [(classes[i], best[classes[i]], count[classes[i]]) for i in sorted(classes)]


def healthy_class(crop_type):
    return MODEL_REGISTRY[crop_type]["disease"]["healthy_class"]


def image_for_step(pod, step):
    traj = pod.get("trajectory")
    if not traj:
        return pod["image_path"], pod["ground_truth"]["growth_stage"]
    step = min(step, len(traj) - 1)
    return traj[step]["image_path"], traj[step]["growth_stage"]


def n_steps_for(pod):
    traj = pod.get("trajectory")
    return len(traj) if traj else 1


def check_against_truth(pod, gt_stage, classification):
    """Only ONE aspect per pod has verified ground truth (see
    pod_registry.py), so this scores just that one -- same rule as
    run_logger.summary()."""
    gt = pod["ground_truth"]
    if gt["verified_aspect"] == "growth" and gt_stage is not None:
        return {"aspect": "growth", "expected": gt_stage, "ok": classification["growth_stage"] == gt_stage}
    if gt["verified_aspect"] == "disease":
        expected = f"diseased ({gt['disease_name']})" if gt["diseased"] else "healthy"
        return {"aspect": "disease", "expected": expected, "ok": classification["disease_flag"] == gt["diseased"]}
    return {"aspect": None, "expected": "", "ok": None}


# ---------------------------------------------------------------------------
# Slot = one visual lane / worker thread
# ---------------------------------------------------------------------------

class Slot:
    def __init__(self, slot_id):
        self.id = slot_id
        self.lock = threading.RLock()
        self.pod = None
        self.results = []          # every StepResult shown in this slot for the current pod
        self.view_index = None     # None = follow live, int = reviewing an earlier result
        self.first_step = 0
        self.next_step = 0
        self.n_steps = 0
        self.done = False
        self.status = "idle"       # idle | inspecting | showing | done | skipped
        self.hold = False
        self.skip = False
        self.ticket = False        # one-shot permission to run while the engine is paused
        self.rescan_req = False
        self.busy = False
        self.not_before = 0.0
        self.pods_completed = 0

    def view(self):
        with self.lock:
            return {
                "slot_id": self.id, "pod": self.pod, "results": list(self.results),
                "view_index": self.view_index, "first_step": self.first_step, "n_steps": self.n_steps,
                "next_step": self.next_step, "done": self.done, "status": self.status,
                "hold": self.hold, "pods_completed": self.pods_completed,
            }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FarmEngine:
    def __init__(self, pods, n_slots=2, settings=None, reset_history=False,
                 log_dir="logs", history_path="dynamo_table.json"):
        self.pods = list(pods)
        self.n_slots = n_slots
        self.settings = settings or Settings()
        self.events = queue.Queue()

        if reset_history and os.path.exists(history_path):
            os.remove(history_path)
        self.history = LocalDynamoTable(path=history_path)
        self.history_path = history_path
        self.logger = RunLogger(out_dir=log_dir)
        self.cloud = LocalCloudClient(max_concurrency=n_slots)
        self.camera = PodCamera(self.cloud)

        # worker/camera agent instructions generated from each analysed image
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.orders = OrderBook(os.path.join("work_orders", f"run_{stamp}"), OrderPolicy(),
                                on_change=lambda: self.events.put(("orders", None)))
        self.orders.on_complete = self._on_order_complete
        self.pods_by_id = {p["pod_id"]: p for p in self.pods}
        self.frame_source = SimulatedCamera()          # swap for a real camera source (see frame_source.py)
        self._followup_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="loop")
        self.planner_name = "rules only"

        self._data_lock = threading.Lock()     # history + logger + counters
        self._queue_lock = threading.Lock()
        self._queue = collections.deque(self.pods)
        self.pod_status = {p["pod_id"]: "queued" for p in self.pods}
        self.pod_slot = {}

        self.slots = [Slot(i) for i in range(n_slots)]
        self.action_counts = {a: 0 for a in ACTIONS}
        self.steps_done = 0
        self.pods_done = 0
        self.total = len(self.pods)
        self.growth_ok = self.growth_total = 0
        self.disease_ok = self.disease_total = 0
        self.latency_sum = 0.0

        self.state = "idle"
        self._paused = False
        self._completed = False
        self._stop = threading.Event()
        self._stopped = threading.Event()
        self._threads = []
        self._t0 = None
        self._paused_total = 0.0
        self._paused_at = None
        self._t_end = None
        self.log_paths = None

    # ----- lifecycle -----------------------------------------------------

    def start(self):
        self._t0 = time.time()
        self._set_state("running")
        if self.orders.policy.simulate_robots:
            self.orders.set_simulation(True)
        self._log("run started: %d pods, %d slot(s)" % (self.total, self.n_slots))
        for slot in self.slots:
            t = threading.Thread(target=self._worker, args=(slot,), daemon=True, name=f"slot-{slot.id}")
            t.start()
            self._threads.append(t)

    def pause(self):
        if self.state != "running":
            return
        self._paused = True
        self._paused_at = time.time()
        self._set_state("paused")
        self._log("paused - in-flight scans finish, nothing new starts", "warn")

    def resume(self):
        if self.state != "paused":
            return
        self._paused_total += time.time() - self._paused_at
        self._paused_at = None
        self._paused = False
        for s in self.slots:
            s.ticket = False
        self._set_state("running")
        self._log("resumed")

    def step_once(self):
        """While paused: let every slot that isn't held advance one image."""
        if self.state != "paused":
            return
        for s in self.slots:
            if not s.hold:
                s.ticket = True

    def stop(self):
        """Non-blocking graceful stop: waits for in-flight scans, flushes the
        CSV/JSON log, then emits ("stopped", info)."""
        if self.state in ("stopping", "stopped"):
            return
        self._set_state("stopping")
        threading.Thread(target=self._finalize, daemon=True).start()

    def close(self, timeout=20.0):
        """Blocking version of stop() for window-close / starting a new run."""
        self.stop()
        self._stopped.wait(timeout)

    def _finalize(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=FEEDBACK_TIMEOUT_S + 2)
        self._followup_pool.shutdown(wait=True, cancel_futures=True)
        try:
            self.cloud.shutdown()
        except Exception:
            pass
        self.orders.shutdown()
        info = self._flush()
        self._t_end = self._t_end or time.time()
        self.state = "stopped"
        self._stopped.set()
        self.events.put(("stopped", info))

    def _flush(self):
        info = {"summary": self.logger.summary(), "csv": None, "json": None,
                "history": f"{self.history_path} ({len(self.history.all_pod_ids())} pods tracked)"}
        if self.logger.rows:
            with self._data_lock:
                info["csv"], info["json"] = self.logger.flush()
            self.log_paths = (info["csv"], info["json"])
        return info

    # ----- controls ------------------------------------------------------

    def toggle_hold(self, slot_id):
        s = self.slots[slot_id]
        s.hold = not s.hold
        self._log(f"slot {slot_id + 1} {'held on ' + s.pod['pod_id'] if s.hold and s.pod else 'released'}"
                  if s.pod else f"slot {slot_id + 1} {'held' if s.hold else 'released'}")
        self._emit_slot(slot_id)

    def skip_pod(self, slot_id):
        s = self.slots[slot_id]
        s.skip = True
        s.hold = False
        self._emit_slot(slot_id)

    def rescan(self, slot_id):
        s = self.slots[slot_id]
        if s.results:
            s.rescan_req = True

    def set_view(self, slot_id, index):
        s = self.slots[slot_id]
        with s.lock:
            if index is None or not s.results:
                s.view_index = None
            else:
                s.view_index = max(0, min(index, len(s.results) - 1))
                if s.view_index == len(s.results) - 1:
                    s.view_index = None
        self._emit_slot(slot_id)

    def run_next(self, pod_id):
        with self._queue_lock:
            for p in self._queue:
                if p["pod_id"] == pod_id:
                    self._queue.remove(p)
                    self._queue.appendleft(p)
                    self._log(f"{pod_id} moved to the front of the queue")
                    break
            else:
                return False
        self.events.put(("queue", None))
        return True

    def skip_queued(self, pod_id):
        with self._queue_lock:
            for p in self._queue:
                if p["pod_id"] == pod_id:
                    self._queue.remove(p)
                    break
            else:
                return False
        self.pod_status[pod_id] = "skipped"
        self._count_pod_done()
        self._log(f"{pod_id} removed from this run", "warn")
        self.events.put(("queue", None))
        return True

    def queue_position(self, pod_id):
        with self._queue_lock:
            for i, p in enumerate(self._queue):
                if p["pod_id"] == pod_id:
                    return i + 1
        return None

    def redecide(self, classification):
        """Re-evaluates a stored reading under the CURRENT thresholds (cheap,
        pure) so the UI can show what the decision would be right now."""
        return decide_action(classification, self.settings.disease_threshold, self.settings.growth_threshold)

    # ----- AI planner (optional) ------------------------------------------

    def set_planner(self, kind="rules", model="llama3.2:3b", use_image=False):
        """kind: 'rules' (no AI), 'scripted' (hand-written stand-in, NOT an LLM), 'ollama' (a real local LLM).
        Can be changed while a run is in progress."""
        if kind == "scripted":
            backend = ScriptedBackend()
        elif kind == "ollama":
            backend = OllamaBackend(model, use_image=use_image, timeout=90)
        else:
            backend = None
        self.orders.set_advisor(LlmPlanner(backend) if backend else None)
        self.planner_name = backend.name if backend else "rules only"
        self._log(f"planner: {self.planner_name}" + ("  (consulted for TREAT / HARVEST orders; it can only ask for a "
                                                      "re-capture, never create or cancel work)" if backend else ""))

    @staticmethod
    def _summarize_history(rows):
        return [{"visit": r.get("visit"), "stage": r.get("growth_stage"),
                 "disease": r.get("disease_name") if r.get("disease_flag") else None,
                 "disease_conf": round(float(r.get("disease_confidence", 0.0)), 2),
                 "task_taken": RULE_TASK.get(r.get("action"), "NONE")} for r in rows[-3:]]

    # ----- closing the loop ------------------------------------------------

    def _on_order_complete(self, order_id):
        try:
            self._followup_pool.submit(self._run_followup, order_id)
        except RuntimeError:        # pool already shut down
            self.orders.loop_done(order_id)

    def _run_followup(self, order_id):
        """A camera order (RECAPTURE / VERIFY) just finished: take the NEW frame, run the same OpenCV analysis,
        and let the order book decide what that changes."""
        try:
            self._run_followup_inner(order_id)
        finally:
            self.orders.loop_done(order_id)         # the pod is released only after the follow-up is analysed

    def _run_followup_inner(self, order_id):
        if self._stop.is_set():
            return
        o = self.orders.get(order_id)
        pod = self.pods_by_id.get(o.pod_id) if o else None
        if not o or not pod:
            return
        try:
            frame = self.frame_source.capture(o, os.path.join(self.orders.out_dir, "frames"))
            st = self.settings
            rid = self.camera.capture_and_submit(o.pod_id, o.crop_type, frame["path"], conf_threshold=st.conf_threshold)
            record = self.cloud.get_feedback(rid, block=True, timeout=FEEDBACK_TIMEOUT_S)
            if record["status"] != "done":
                raise RuntimeError("no feedback from the analysis pipeline")
            cls = record["classification"]
            dec = decide_action(cls, st.disease_threshold, st.growth_threshold)
            res = StepResult(
                pod_id=o.pod_id, crop_type=o.crop_type, step_index=o.evidence.get("step", 0),
                n_steps=o.evidence.get("n_steps", 1), visit_num=0, image_path=frame["path"], classification=cls,
                decision=dec, trend_note="", latency_s=record["latency_seconds"], worker=record["worker_thread"],
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"), kind="followup", logged=False,
                conf_threshold=st.conf_threshold, verified={"aspect": None, "expected": "", "ok": None})
            tag = "[SIMULATED FRAME] " if frame["simulated"] else ""
            self._log(f"loop: {order_id} ({o.task}) done -> new frame {tag}{frame['note']} -> reads: {dec['action']}")
            out = self.orders.close_loop(order_id, pod, res, frame["note"], frame["simulated"],
                                         history_rows=self._summarize_history(self.history.query(o.pod_id)))
            if out:
                bad = out["result"] in ("still_uncertain", "treatment_not_resolved")
                self._log("  -> loop closed: " + out["headline"], "warn" if bad else "ok")
        except Exception as exc:
            self._log(f"loop for {order_id} failed: {exc!r}", "error")

    # ----- stats ---------------------------------------------------------

    def elapsed(self):
        if self._t0 is None:
            return 0.0
        end = self._t_end or time.time()
        paused = self._paused_total + ((time.time() - self._paused_at) if self._paused_at else 0.0)
        return max(0.0, end - self._t0 - paused)

    def stats(self):
        with self._data_lock:
            return {
                "pods_done": self.pods_done, "total": self.total, "steps": self.steps_done,
                "actions": dict(self.action_counts), "elapsed": self.elapsed(),
                "avg_latency": (self.latency_sum / self.steps_done) if self.steps_done else 0.0,
                "growth": (self.growth_ok, self.growth_total),
                "disease": (self.disease_ok, self.disease_total),
            }

    def slot_view(self, slot_id):
        return self.slots[slot_id].view()

    # ----- internals -----------------------------------------------------

    def _set_state(self, state):
        self.state = state
        self.events.put(("state", state))

    def _log(self, text, level="info"):
        self.events.put(("log", (time.strftime("%H:%M:%S"), level, text)))

    def _emit_slot(self, slot_id):
        self.events.put(("slot", slot_id))

    def _count_pod_done(self):
        with self._data_lock:
            self.pods_done += 1
            all_done = self.pods_done >= self.total and not self._completed
            if all_done:
                self._completed = True
        if all_done:
            self._t_end = time.time()
            info = self._flush()
            self._set_state("finished")
            self._log("run complete - log written, you can still review / re-scan any slot", "ok")
            self.events.put(("finished", info))

    def _may_proceed(self, slot):
        if slot.hold:
            return False
        if self._paused:
            if slot.ticket:
                slot.ticket = False
                return True
            return False
        return True

    def _pull_pod(self):
        with self._queue_lock:
            return self._queue.popleft() if self._queue else None

    def _worker(self, slot):
        while not self._stop.is_set():
            try:
                if slot.rescan_req:
                    slot.rescan_req = False
                    self._do_rescan(slot)
                    continue
                if slot.skip and not slot.busy:
                    slot.skip = False
                    if slot.pod is not None and not slot.done:
                        self._finish_pod(slot, skipped=True)
                    else:
                        slot.not_before = 0.0
                    continue
                if time.time() < slot.not_before:
                    self._stop.wait(0.05)
                    continue
                if not self._may_proceed(slot):
                    self._stop.wait(0.05)
                    continue
                self._advance(slot)
            except Exception as exc:  # keep the lane alive, tell the user
                self._log(f"slot {slot.id + 1} error: {exc!r}", "error")
                self._stop.wait(0.5)

    def _advance(self, slot):
        if slot.pod is None or slot.done:
            pod = self._pull_pod()
            if pod is None:
                if slot.status != "idle":
                    slot.status = "idle"
                    self._emit_slot(slot.id)
                self._stop.wait(0.2)
                return
            self._begin_pod(slot, pod)

        self._run_step(slot)
        waited = self._await_orders(slot)          # the next image only after the robots are done with this one
        hold = ROBOT_DONE_DWELL_S if waited else 0.0   # let the finished robot job stay visible for a moment

        with slot.lock:
            finished = slot.next_step >= slot.n_steps
            skipped = slot.skip and not finished
            slot.skip = False if skipped else slot.skip
        if finished:
            self._finish_pod(slot, skipped=False, extra_dwell=hold)
        elif skipped:
            self._finish_pod(slot, skipped=True)
        else:
            slot.not_before = time.time() + max(self.settings.delay_ms / 1000.0, hold)

    def _await_orders(self, slot):
        """Block this lane while the robots still have work on its pod (worker jobs, camera re-captures and the
        follow-up frame that closes the loop; NOT human reviews or later verification visits). Returns True if it
        had to wait. 'Skip pod' releases it. In manual-approval mode this is where the lane waits for you."""
        pod_id = slot.pod["pod_id"]
        waited = False
        while not self._stop.is_set():
            if slot.skip or not self.orders.pod_busy(pod_id):
                break
            if not waited:
                waited = True
                with slot.lock:
                    slot.status = "robot"
                self._log(f"slot {slot.id + 1}: {pod_id} waits for the robots before its next image")
                self._emit_slot(slot.id)
            self._stop.wait(0.15)
        if waited:
            with slot.lock:
                slot.status = "showing"
            self._emit_slot(slot.id)
        return waited

    def _begin_pod(self, slot, pod):
        n = n_steps_for(pod)
        first = min(self.history.visit_count(pod["pod_id"]), n - 1)   # resume where history left off
        with slot.lock:
            slot.pod, slot.results, slot.view_index = pod, [], None
            slot.first_step, slot.next_step, slot.n_steps = first, first, n
            slot.done, slot.skip, slot.status = False, False, "inspecting"
        self.pod_status[pod["pod_id"]] = "running"
        self.pod_slot[pod["pod_id"]] = slot.id
        self._log(f"slot {slot.id + 1} picked up {pod['pod_id']} ({pod['crop_type']}), "
                  f"images {first + 1}-{n} of {n}")
        self._emit_slot(slot.id)
        self.events.put(("queue", None))

    def _finish_pod(self, slot, skipped, extra_dwell=0.0):
        with slot.lock:
            slot.done = True
            slot.status = "skipped" if skipped else "done"
            slot.pods_completed += 1
            slot.not_before = 0.0 if skipped else time.time() + max(MIN_DWELL_S, self.settings.delay_ms / 1000.0,
                                                                     extra_dwell)
            pod_id = slot.pod["pod_id"]
        self.pod_status[pod_id] = "skipped" if skipped else "done"
        if skipped:
            self._log(f"slot {slot.id + 1}: skipped the rest of {pod_id}", "warn")
        self._emit_slot(slot.id)
        self.events.put(("queue", None))
        self._count_pod_done()

    def _run_step(self, slot):
        slot.busy = True
        try:
            with slot.lock:
                pod, step = slot.pod, slot.next_step
                slot.status = "inspecting"
            self._emit_slot(slot.id)
            result = self._scan(pod, step, slot, rescan=False)
            with slot.lock:
                slot.next_step += 1
                if result is not None:
                    slot.results.append(result)
                    slot.view_index = None if slot.view_index is None else slot.view_index
                slot.status = "showing"
            if result is None:
                self._log(f"{pod['pod_id']}: no feedback within {FEEDBACK_TIMEOUT_S:.0f}s, step skipped", "error")
            self._emit_slot(slot.id)
        finally:
            slot.busy = False

    def _do_rescan(self, slot):
        slot.busy = True
        try:
            with slot.lock:
                if not slot.results or slot.pod is None:
                    return
                idx = slot.view_index if slot.view_index is not None else len(slot.results) - 1
                src = slot.results[idx]
                pod = slot.pod
                slot.status = "inspecting"
            self._emit_slot(slot.id)
            result = self._scan(pod, src.step_index, slot, rescan=True)
            with slot.lock:
                if result is not None:
                    slot.results.append(result)
                    slot.view_index = None       # jump to the fresh reading
                slot.status = "done" if slot.done else "showing"
            self._emit_slot(slot.id)
        finally:
            slot.busy = False

    def _scan(self, pod, step, slot, rescan):
        """Capture -> cloud classify+decide -> (history, trend, log). The
        only difference for a re-scan is that it is NOT written to the
        history/log, so re-scanning never advances a pod's growth."""
        image_path, gt_stage = image_for_step(pod, step)
        st = self.settings
        try:
            rid = self.camera.capture_and_submit(pod["pod_id"], pod["crop_type"], image_path,
                                                 conf_threshold=st.conf_threshold)
        except RuntimeError:        # pool already shut down (run is ending)
            return None
        record = self.cloud.get_feedback(rid, block=True, timeout=FEEDBACK_TIMEOUT_S)
        if record["status"] != "done":
            return None

        classification = record["classification"]
        decision = decide_action(classification, st.disease_threshold, st.growth_threshold)
        verified = check_against_truth(pod, gt_stage, classification)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

        trend_note, visit_num, hist_rows = "", 0, []
        if not rescan:
            with self._data_lock:
                prior = self.history.query(pod["pod_id"])
                hist_rows = self._summarize_history(prior)
                trend_note = summarize_trend(prior, classification)
                visit_num = len(prior) + 1
                self.history.put_item({
                    "pod_id": pod["pod_id"], "timestamp": ts, "visit": visit_num,
                    "crop_type": pod["crop_type"],
                    "growth_stage": classification["growth_stage"],
                    "growth_confidence": classification["growth_confidence"],
                    "disease_flag": classification["disease_flag"],
                    "disease_name": classification["disease_name"],
                    "disease_confidence": classification["disease_confidence"],
                    "action": decision["action"], "ground_truth_stage_this_visit": gt_stage,
                })
                pod_for_log = dict(pod)
                pod_for_log["ground_truth"] = {
                    **pod["ground_truth"],
                    "growth_stage": gt_stage if pod["side"] == "front" else pod["ground_truth"]["growth_stage"],
                }
                row = self.logger.log(pod_for_log, classification, decision)
                row["visit"], row["trend"] = visit_num, trend_note
                self.action_counts[decision["action"]] += 1
                self.steps_done += 1
                self.latency_sum += record["latency_seconds"]
                if verified["aspect"] == "growth":
                    self.growth_total += 1
                    self.growth_ok += bool(verified["ok"])
                elif verified["aspect"] == "disease":
                    self.disease_total += 1
                    self.disease_ok += bool(verified["ok"])

        result = StepResult(
            pod_id=pod["pod_id"], crop_type=pod["crop_type"], step_index=step, n_steps=n_steps_for(pod),
            visit_num=visit_num, image_path=image_path, classification=classification, decision=decision,
            trend_note=trend_note, latency_s=record["latency_seconds"], worker=record["worker_thread"],
            timestamp=ts, kind="rescan" if rescan else "scan", logged=not rescan,
            conf_threshold=st.conf_threshold, verified=verified,
        )

        if not rescan:
            try:      # a planner problem must never break the scanning lane
                planned = self.orders.plan_from_result(pod, result, history_rows=hist_rows)
            except Exception as exc:
                planned = []
                self._log(f"{pod['pod_id']}: could not plan robot orders: {exc!r}", "error")
            for oid, kind in planned:
                o = self.orders.get(oid)
                if o and kind == "created":
                    gate = {"awaiting_approval": "needs human approval", "planning": "asking the AI planner"}.get(
                        o.status, "dispatched")
                    self._log(f"  -> {o.agent} order {oid} {o.task}: {o.title} ({gate})",
                              "warn" if o.task == "TREAT" else "info")
                elif o and kind == "superseded":
                    self._log(f"  -> order {oid} {o.task} superseded by a treatment order", "info")

        disease_bit = classification["disease_name"] if classification["disease_flag"] else "no disease"
        tag = "re-scan (not logged)" if rescan else f"image {step + 1}/{n_steps_for(pod)}"
        self._log(f"{pod['pod_id']} {tag}: stage {classification['growth_stage']} "
                  f"({classification['growth_confidence']:.2f}), {disease_bit} -> {decision['action']}",
                  "warn" if decision["action"] == "flag_for_treatment" else "info")
        return result
