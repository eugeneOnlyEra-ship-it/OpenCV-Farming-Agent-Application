"""Threaded PyBullet executor used by the Tkinter/OpenCV application.

The application remains responsible for perception, decisions, history and
work-order approval. This module turns its scan visits and dispatched work
orders into physical gantry movements. All PyBullet calls stay on ONE thread.

HOW WORK IS SCHEDULED (v2 - replaces the "wait for a partner order" design)
---------------------------------------------------------------------------
Every piece of physical work is a SIDE JOB: one pod, one side (front or back).
  * a *visit*  = the camera agent localises the pod from several rendered views;
  * an *order* = the camera agent localises it, then the worker agent carries out
                 the order (treat / harvest / replant ...).
The gantry serves one STATION (row, column) at a time, with a front agent pair and
a back agent pair that can work at the same time. While it is at a station:
  * as soon as a side's agents are idle, any queued job for that side at this
    station STARTS IMMEDIATELY (nothing waits for a "partner" job);
  * if the other side's job arrives while this side is busy, it simply starts the
    moment its own agents are free - front and back progress independently and
    simultaneously, and a job on one side never delays the other side;
  * the session ends when both sides are idle and nothing more is queued here.
The old design held a lone order for 20-60 s hoping an opposite-side order would
appear; the application's lane waits for its order, so the whole app looked frozen.

PROGRESS is live: the gantry's own events (plant_scan, assess, camera_left,
service_start, service_contact, service_done ...) advance the order's step counter
as the machine moves, so the UI shows front and back agents progressing together.

FAILURES never leave the application waiting: an order whose physical job fails
is marked failed (the application's lane is released), and if the simulator cannot
start or dies, every queued order is handed back (see abandon()).
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque

from . import pod_registry as sim_registry

# Orders carried out by the worker agent (after the camera has localised the pod). RECAPTURE/VERIFY are camera-only.
WORKER_TASKS = {"TREAT", "HARVEST", "REPLANT"}
DECISION_MARK = {
    "log_healthy": "log_and_continue", "flag_for_treatment": "flag_for_treatment",
    "flag_for_harvest": "flag_for_harvest", "schedule_frequent_monitoring": "schedule_monitoring",
}
TASK_DECISION = {"TREAT": "flag_for_treatment", "HARVEST": "flag_for_harvest"}
SERVICE_ACTION = {"TREAT": "apply_treatment", "HARVEST": "harvest", "REPLANT": "replant",
                  "RECAPTURE": None, "VERIFY": None}


class PyBulletBridge:
    """Live scheduler between the application and the PyBullet gantry (see module docstring)."""

    def __init__(self, gui=True, on_log=None, speed=2.0, robot_factory=None, max_session_seconds=1200.0):
        self.gui = gui
        self.speed = speed
        self.on_log = on_log or (lambda message, level="info": None)
        self._robot_factory = robot_factory          # tests inject a fake robot here
        self.max_session_seconds = max_session_seconds
        self._condition = threading.Condition()
        self._paused = False
        self._step_tickets = 0
        self._stopping = False
        self._abandoned = False                      # startup failed -> the built-in fleet took over
        self._dead = False                           # the PyBullet thread ended
        self._started = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._robot = None
        self._orders = None
        self._startup_error = None
        self._last_error = None
        self._pending = deque()                      # side jobs waiting for their side's agents
        self._running = {}                           # sim pod_id -> side job the gantry is working on now
        self._updates = deque()                      # late application scores for an already-captured pod
        self._outstanding = 0                        # pending + running + queued updates
        self._station = None                         # (row, col) being served, or None
        self._scheduled_visit_pairs = set()
        self._sim_pods = sim_registry.service_visit_order()
        self._sim_by_position = {
            (p["row"], p["col"], p["layer"], p["side"]): p
            for p in self._sim_pods
        }

    # ------------------------------------------------------------------ setup / lifecycle
    @staticmethod
    def dependency_available():
        try:
            import pybullet  # noqa: F401
            import pybullet_data  # noqa: F401
            return True, ""
        except Exception as exc:
            return False, str(exc)

    def _dependencies_ok(self):
        """An injected robot (tests) does not need the real PyBullet package."""
        if self._robot_factory is not None:
            return True, ""
        return self.dependency_available()

    def attach_orders(self, order_book):
        self._orders = order_book

    def wrap_transport(self, existing_transport):
        bridge = self

        class CombinedTransport:
            def send(self, order):
                # Preserve the application's JSON outbox for audit/replay.
                existing_transport.send(order)
                bridge.enqueue_order(order)

        return CombinedTransport()

    def start(self, timeout=90.0):
        """Blocking start (kept for scripts/tests). The GUI should use start_async() so the window never freezes."""
        ok, why = self._dependencies_ok()
        if not ok:
            return False, f"PyBullet is unavailable: {why}"
        if self._thread and self._thread.is_alive():
            return True, "already running"
        self._spawn_thread()
        if not self._ready.wait(timeout):
            self.request_stop()
            return False, "PyBullet simulator did not initialise in time"
        if self._startup_error:
            return False, f"PyBullet simulator failed to initialise: {self._startup_error}"
        self._started.set()
        return True, "PyBullet robot simulation connected"

    def start_async(self, on_ready=None, on_failed=None, timeout=180.0):
        """Start PyBullet WITHOUT blocking the caller (building the 3-D scene takes seconds; doing that on the Tk
        thread froze the whole window). Returns immediately; exactly one of on_ready() / on_failed(reason) is called
        later from a helper thread. Work enqueued meanwhile is simply held until the gantry is ready."""
        ok, why = self._dependencies_ok()
        if not ok:
            self._startup_error = why
            self._call_later(on_failed, f"PyBullet is unavailable: {why}")
            return False
        if self._thread and self._thread.is_alive():
            self._call_later(on_ready)
            return True
        self._spawn_thread()

        def watch():
            if not self._ready.wait(timeout):
                self.request_stop()
                self._call(on_failed, "PyBullet simulator did not initialise in time")
            elif self._startup_error:
                self._call(on_failed, f"PyBullet simulator failed to initialise: {self._startup_error}")
            else:
                self._started.set()
                self._call(on_ready)
        threading.Thread(target=watch, daemon=True, name="pybullet-start-watch").start()
        return True

    def _spawn_thread(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name="pybullet-gantry")
        self._thread.start()

    @staticmethod
    def _call(fn, *args):
        if fn is not None:
            try:
                fn(*args)
            except Exception:
                traceback.print_exc()

    def _call_later(self, fn, *args):
        if fn is not None:
            threading.Thread(target=self._call, args=(fn, *args), daemon=True).start()

    def abandon(self, reason=""):
        """The simulator could not start. Drop everything queued WITHOUT cancelling the orders (they stay
        'dispatched' so the application's built-in fleet can take them over) and ignore any new work."""
        with self._condition:
            self._abandoned = True
            self._pending.clear()
            self._updates.clear()
            self._outstanding = 0
            self._condition.notify_all()
        if reason:
            self.on_log(f"PyBullet abandoned: {reason}", "warn")

    # ------------------------------------------------------------------ enqueue (called by the application)
    @staticmethod
    def _opposite_app_pod_id(pod_id):
        """Return the paired front/back application pod ID (e.g. R1-L-F2 -> R1-L-B2)."""
        try:
            prefix, suffix = str(pod_id).rsplit("-", 1)
            if len(suffix) < 2 or suffix[0] not in ("F", "B"):
                return None
            other = "B" if suffix[0] == "F" else "F"
            return f"{prefix}-{other}{suffix[1:]}"
        except ValueError:
            return None

    @staticmethod
    def _station_of(pod):
        return (pod["row"], pod["col"])

    def enqueue_visit(self, app_pod, decision, classification=None, visit_num=1):
        """Schedule one physical station-layer visit for BOTH sides (a front pod and its back counterpart are
        photographed together). The first application result for the pair creates the two side jobs; the later
        result for the counterpart only attaches its model scores to the existing pose record."""
        sim_pod = self._map_pod(app_pod)
        if sim_pod is None:
            self.on_log(f"No PyBullet pod mapping for {app_pod.get('pod_id')}", "error")
            return
        app_pod_id = app_pod.get("pod_id", sim_pod["pod_id"])
        partner_app_id = self._opposite_app_pod_id(app_pod_id)
        other_side = "back" if sim_pod["side"] == "front" else "front"
        partner = self._sim_by_position.get((sim_pod["row"], sim_pod["col"], sim_pod["layer"], other_side))
        pair_key = (sim_pod["row"], sim_pod["col"], sim_pod["layer"], int(visit_num or 1))
        with self._condition:
            if self._stopping or self._abandoned or self._dead:
                return
            if pair_key in self._scheduled_visit_pairs:
                self._outstanding += 1
                self._updates.append({
                    "kind": "classification_update", "pod": sim_pod,
                    "classification": dict(classification or {}), "decision": dict(decision or {}),
                    "app_pod_id": app_pod_id, "visit_num": int(visit_num or 1), "_t": time.monotonic(),
                })
                self._condition.notify_all()
                return
            self._scheduled_visit_pairs.add(pair_key)
            for pod in (sim_pod, partner):
                if pod is None:
                    continue
                current = pod["pod_id"] == sim_pod["pod_id"]
                self._pending.append({
                    "kind": "visit", "pod": pod, "station": self._station_of(pod),
                    "decision": dict(decision or {}) if current else {},
                    "classification": dict(classification or {}) if current else {},
                    "app_pod_id": app_pod_id if current else (partner_app_id or pod["pod_id"]),
                    "visit_num": int(visit_num or 1), "pair_key": pair_key,
                })
                self._outstanding += 1
            self._condition.notify_all()

    def enqueue_order(self, order):
        order_id = order.get("order_id")
        sim_pod = self._map_pod({
            "pod_id": order.get("pod_id"), "rail": order.get("evidence", {}).get("rail"),
            "station": order.get("evidence", {}).get("station"), "side": order.get("side"),
        })
        if sim_pod is None:
            self.on_log(f"Work order {order_id} has no PyBullet pod mapping; left in outbox", "error")
            return
        with self._condition:
            if self._abandoned or self._stopping:
                return
            if not self._dead:
                self._pending.append({"kind": "order", "order": dict(order), "pod": sim_pod,
                                      "station": self._station_of(sim_pod)})
                self._outstanding += 1
                self._condition.notify_all()
                return
        # The PyBullet thread is gone: do not leave the application's lane waiting for an order nobody will run.
        self._release_order(order_id, False, "the PyBullet gantry is not running")

    def _map_pod(self, pod):
        # App IDs are R1-L-F1; the geometry model IDs are R0C0-L0-F.
        pod_id = pod.get("pod_id", "")
        try:
            side = pod.get("side") or ("front" if "-F" in pod_id else "back")
            rail = int(pod.get("rail") or pod_id.split("-")[0][1:])
            station = pod.get("station")
            if station is None:
                station_letter = pod_id.split("-")[1]
                station = {"L": 0, "M": 1, "R": 2}[station_letter]
            suffix = pod_id.rsplit("-", 1)[-1]
            layer = int(suffix[1:]) - 1
            return self._sim_by_position.get((rail - 1, int(station), layer, side))
        except (ValueError, IndexError, KeyError, TypeError):
            return None

    # ------------------------------------------------------------------ run control (job-safe boundaries)
    def pause(self):
        with self._condition:
            self._paused = True
            self._condition.notify_all()

    def resume(self):
        with self._condition:
            self._paused = False
            self._step_tickets = 0
            self._condition.notify_all()

    def step_once(self):
        with self._condition:
            if self._paused and not self._stopping:
                self._step_tickets += 1
                self._condition.notify_all()

    def request_stop(self):
        """Stop starting new work. Jobs the machine is already doing finish normally; everything still queued is
        cancelled (its order is cancelled so the application's lane is released)."""
        with self._condition:
            self._stopping = True
            self._paused = False
            self._step_tickets = 0
            pending = list(self._pending)
            self._pending.clear()
            self._updates.clear()
            self._condition.notify_all()
        for job in pending:
            self._cancel_job(job, "run stopped before the robot started this job")
            self._dec_outstanding()

    def join(self, timeout=30.0):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def wait_until_idle(self, timeout=None):
        """Wait until queued and running robot jobs have drained (or the simulator stopped). True if idle."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._outstanding > 0 and not self._stopping and not self._abandoned and not self._dead:
                remaining = 0.25 if deadline is None else min(0.25, deadline - time.monotonic())
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            return self._outstanding == 0

    def _dec_outstanding(self, n=1):
        with self._condition:
            self._outstanding = max(0, self._outstanding - n)
            self._condition.notify_all()

    # ------------------------------------------------------------------ status for the UI
    def fleet_status(self, base_status):
        """Overlay what the gantry is doing right now on the order book's per-agent status. Orders already report
        their own live progress (see _on_robot_event); this adds the camera agents' work for plain visits and the
        camera phase of worker orders, so BOTH sides show up at the same time."""
        result = {agent: dict(info) for agent, info in base_status.items()}
        with self._condition:
            running = list(self._running.values())
        for job in running:
            side = job["pod"]["side"]
            views = int(job.get("_views", 0))
            camera = f"camera_{side}"
            if camera not in result:
                continue
            camera_busy = job.get("_phase", "camera") == "camera"
            if job["kind"] == "visit" or (camera_busy and job["kind"] == "order"):
                if result[camera].get("active") is None:
                    result[camera]["active"] = {
                        "order_id": job["order"].get("order_id") if job["kind"] == "order" else None,
                        "task": "VISIT" if job["kind"] == "visit" else "LOCALISE",
                        "pod": job.get("app_pod_id") or job["order"].get("pod_id", job["pod"]["pod_id"]),
                        "n": 3, "progress": views, "step": min(3, views + 1), "updated": "",
                        "action": f"camera view {min(3, views + 1)}/3",
                    }
        return result

    # ------------------------------------------------------------------ the physics thread
    def _make_robot(self):
        if self._robot_factory is not None:
            return self._robot_factory(self)
        from .gantry_robot import GantryRobot
        return GantryRobot(pods=self._sim_pods, gui=self.gui, speed_scale=self.speed)

    def _run(self):
        robot = None
        try:
            from .pod_registry import TOP_DOCK_Y, BOTTOM_DOCK_Y
            robot = self._make_robot()
            self._robot = robot
            robot.on_event = self._on_robot_event
            robot.park_at_dock(TOP_DOCK_Y)
            if hasattr(robot, "prewarm_inference"):
                robot.prewarm_inference()            # load the ONNX models in the background thread
            self._ready.set()
            self.on_log("PyBullet gantry connected; front and back agents work independently at each station", "ok")
            while True:
                station = self._wait_for_station()
                if station is None:
                    break
                if station == ("updates",):          # only late score updates are queued: no gantry motion needed
                    self._apply_pending_updates(robot)
                    with self._condition:
                        unresolved = bool(self._updates) and not self._pending
                    if unresolved:                   # their pose record is not ready yet: wait instead of spinning
                        time.sleep(0.2)
                    continue
                self._serve_station(robot, station)
                self._apply_pending_updates(robot)
            try:
                robot.park_at_dock(BOTTOM_DOCK_Y)
            except Exception as exc:
                self.on_log(f"Could not park gantry during shutdown: {exc!r}", "warn")
        except Exception as exc:
            if not self._ready.is_set():
                self._startup_error = repr(exc)
                self._ready.set()
            self._last_error = repr(exc)
            self.on_log(f"PyBullet simulator error: {exc!r}", "error")
            self.on_log(traceback.format_exc(), "error")
        finally:
            self._shutdown_cleanup(robot)

    def _shutdown_cleanup(self, robot):
        """However the thread ends, nothing may stay queued or 'in progress': the application's lanes wait on it."""
        with self._condition:
            self._dead = True
            running = list(self._running.values())
            pending = list(self._pending)
            self._pending.clear()
            self._updates.clear()
            self._condition.notify_all()
        for job in running:
            self._finish_job(job, False, "PyBullet gantry stopped while this job was running")
        for job in pending:
            self._cancel_job(job, "PyBullet gantry stopped before it could run this job")
            self._dec_outstanding()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        self._robot = None
        with self._condition:
            self._outstanding = 0
            self._condition.notify_all()
        self.on_log("PyBullet gantry disconnected", "info")

    def _wait_for_station(self):
        """Block until there is pending work this machine may start; return its station (row, col), or None to stop."""
        with self._condition:
            while True:
                if self._stopping:
                    return None
                runnable = (not self._paused) or self._step_tickets > 0
                if self._pending and runnable:
                    return self._pending[0]["station"]
                if self._updates and not self._pending:
                    return ("updates",)              # handled without moving the gantry
                self._condition.wait(0.2)

    def _serve_station(self, robot, station):
        row, col = station
        with self._condition:
            self._station = station
        try:
            self.on_log(f"PyBullet serving R{row}C{col}", "info")
            robot.service_station_live(row, col, self._assess, lambda idle: self._poll_new_work(station, idle),
                                       max_wall_seconds=self.max_session_seconds)
        except Exception as exc:
            self._last_error = repr(exc)
            self.on_log(f"PyBullet job failed: {exc!r}", "error")
            self.on_log(traceback.format_exc(), "error")
            with self._condition:
                running = list(self._running.values())
            for job in running:                      # release the application's lanes instead of hanging them
                self._finish_job(job, False, f"PyBullet error: {exc!r}")
            if hasattr(robot, "recover_after_failure") and not robot.recover_after_failure():
                self.on_log("Gantry could not be stowed after the error; remaining jobs will fail", "error")
        finally:
            with self._condition:
                leftovers = list(self._running.values())
                self._station = None
            for job in leftovers:                    # a session that ended normally leaves nothing behind
                self._finish_job(job, True, "executed by PyBullet gantry")

    # -- work selection (runs on the physics thread, inside the gantry's step loop) ------------
    def _poll_new_work(self, station, idle_sides):
        """Called by the gantry for every step while a side is idle. Returns {side: [pods]} to start NOW."""
        with self._condition:
            if self._stopping or not self._pending:
                return {}
            if self._paused and self._step_tickets <= 0:
                return {}
            taken = {side: [] for side in idle_sides}
            busy_pods = set(self._running)
            chosen = []
            for job in self._pending:
                if job["station"] != station:
                    continue
                side = job["pod"]["side"]
                pod_id = job["pod"]["pod_id"]
                if side not in taken or pod_id in busy_pods:
                    continue                          # one job per pod at a time; its turn comes later
                busy_pods.add(pod_id)
                taken[side].append(job)
                chosen.append(job)
            if not chosen:
                return {}
            for job in chosen:
                self._pending.remove(job)
                self._running[job["pod"]["pod_id"]] = job
            if self._paused:
                self._step_tickets -= 1
        # Order-book calls happen OUTSIDE our lock (the order book calls enqueue_order() while holding its own).
        robot = self._robot
        started = {}
        for side, jobs in taken.items():
            ok_jobs = [job for job in jobs if self._start_job(robot, job)]
            ok_jobs.sort(key=lambda job: -job["pod"]["layer"])          # top layer first, like the camera sweep
            if ok_jobs:
                started[side] = [job["pod"] for job in ok_jobs]
        return started

    def _start_job(self, robot, job):
        pod = job["pod"]
        if job["kind"] == "order":
            book = self._orders
            order = job["order"]
            if book is None:
                self.on_log(f"No order book attached for {order.get('order_id')}", "error")
                self._drop_running(job)
                return False
            n_steps = book.begin(order.get("order_id"), agent="pybullet_gantry")
            if n_steps is None:                       # cancelled, stale or already handled
                self._drop_running(job)
                return False
            job["_n_steps"], job["_adv"] = int(n_steps), 0
            decision = DECISION_MARK.get(TASK_DECISION.get(order.get("task"), "schedule_frequent_monitoring"))
            self.on_log(f"PyBullet starting {order.get('order_id')} {order.get('task')} at {order.get('pod_id')}", "info")
        else:
            decision = DECISION_MARK.get((job.get("decision") or {}).get("action"))
        job.update(_views=0, _phase="camera", _frac=0.0, _contacts=0, _done=False)
        try:
            robot.mark_pod_decision(pod["pod_id"], decision)
        except Exception:
            pass
        return True

    def _drop_running(self, job):
        with self._condition:
            self._running.pop(job["pod"]["pod_id"], None)
        self._dec_outstanding()

    def _assess(self, pod):
        """The gantry's camera agent calls this when it is in position at a pod."""
        with self._condition:
            job = self._running.get(pod["pod_id"])
        if job is None:
            return {"service_action": None, "camera_dwell": 0.0}
        if job["kind"] == "visit":
            classification = dict(job.get("classification") or {})
            return {
                "service_action": None, "camera_dwell": 0.0,
                "model_scores": classification, "classification": classification,
                "decision": dict(job.get("decision") or {}),
                "visit_num": int(job.get("visit_num", 1) or 1),
                "application_pod_id": job.get("app_pod_id"),
            }
        order = job.get("order", {})
        task = order.get("task")
        whole_plant_removal = task == "TREAT" and any(
            str(step.get("action", "")).upper() == "REMOVE_TISSUE"
            and (step.get("params") or {}).get("mode") == "whole_plant"
            for step in (order.get("steps") or []) if isinstance(step, dict))
        action = "uproot" if whole_plant_removal else SERVICE_ACTION.get(task)
        plant_indices = None
        if whole_plant_removal:
            centers = [t.get("center_norm", [0.5, 0.5])[0]
                       for t in (order.get("targets") or []) if isinstance(t, dict)]
            plant_indices = sorted({max(0, min(2, int(float(x) * 3))) for x in centers}) if centers else [1]
        return {
            "service_action": action, "camera_dwell": 0.0,
            "model_scores": dict(order.get("evidence") or {}),
            "classification": dict(order.get("evidence") or {}),
            "decision": {"action": task, "order_id": order.get("order_id")},
            "plant_indices": plant_indices,
        }

    # -- live progress and completion (events arrive on the physics thread, inside the step loop) ----
    def _on_robot_event(self, event):
        pod_id = event.get("pod_id")
        if not pod_id:
            return
        with self._condition:
            job = self._running.get(pod_id)
        if job is None or job.get("_done"):
            return
        kind = event.get("kind")
        worker = job["kind"] == "order" and job["order"].get("task") in WORKER_TASKS
        frac = job.get("_frac", 0.0)
        if kind == "plant_scan":
            job["_views"] = int(job.get("_views", 0)) + 1
            frac = max(frac, (0.30 if worker else 0.80) * min(1.0, job["_views"] / 3.0))
        elif kind == "assess":
            frac = max(frac, 0.33 if worker else 0.85)
        elif kind == "camera_left":
            job["_phase"] = "service"
            frac = max(frac, 0.35 if worker else 0.95)
        elif kind == "service_start":
            frac = max(frac, 0.45)
        elif kind == "service_contact":
            job["_contacts"] = int(job.get("_contacts", 0)) + 1
            frac = max(frac, 0.45 + 0.50 * min(1.0, job["_contacts"] / 3.0))
        elif kind in ("service_done", "service_skip"):
            if kind == "service_skip" and worker and job.get("_phase") != "service":
                return                                  # a skip before the camera finished is not the end of the job
            self._finish_job(job, True, "executed by PyBullet gantry")
            return
        job["_frac"] = frac
        self._push_progress(job)

    def _push_progress(self, job):
        """Advance the order's step counter to match how far the physical work has got."""
        if job["kind"] != "order" or self._orders is None:
            return
        n = int(job.get("_n_steps", 0))
        target = min(max(0, n - 1), int(job.get("_frac", 0.0) * n))
        order_id = job["order"].get("order_id")
        while job.get("_adv", 0) < target:
            job["_adv"] = job.get("_adv", 0) + 1
            self._orders.advance(order_id)

    def _finish_job(self, job, ok, note):
        """Idempotent: complete (or fail) the order and free the pod. Never called while holding our lock."""
        with self._condition:
            if job.get("_done"):
                return
            job["_done"] = True
            self._running.pop(job["pod"]["pod_id"], None)
        if job["kind"] == "order":
            order_id = job["order"].get("order_id")
            task = job["order"].get("task")
            self._release_order(order_id, ok, note, job)
            self.on_log(f"PyBullet {'completed' if ok else 'FAILED'} {order_id} {task} at "
                        f"{job['order'].get('pod_id')}" + ("" if ok else f" ({note})"), "ok" if ok else "error")
        self._dec_outstanding()

    def _release_order(self, order_id, ok, note, job=None):
        book = self._orders
        if book is None or not order_id:
            return
        try:
            if ok:
                if job is not None:
                    n = int(job.get("_n_steps", 0))
                    while job.get("_adv", 0) < n:
                        job["_adv"] = job.get("_adv", 0) + 1
                        book.advance(order_id)
                book.complete(order_id, note=note)
            else:
                if not book.fail(order_id, note):
                    book.cancel_dispatched(order_id, note)
        except Exception:
            self.on_log(f"Could not update order {order_id}: {traceback.format_exc()}", "error")

    def _cancel_job(self, job, note):
        if job.get("kind") == "order" and self._orders is not None:
            self._orders.cancel_dispatched(job["order"].get("order_id"), note)

    def _apply_pending_updates(self, robot):
        """Attach later application model results to pods whose pose has been captured. An update whose pose is
        not ready yet stays queued for a short while (its visit may still be running); then it is dropped."""
        keep = deque()
        with self._condition:
            items = list(self._updates)
            self._updates.clear()
        for job in items:
            record = getattr(robot, "localization_records", {}).get(job["pod"]["pod_id"])
            if record is None:
                if time.monotonic() - job.get("_t", 0) < 120.0:
                    keep.append(job)
                    continue
                self.on_log(f"Localization record not ready for {job['app_pod_id']}; scores retained in app log", "warn")
                self._dec_outstanding()
                continue
            record["model_scores"] = dict(job.get("classification") or {})
            record["decision"] = dict(job.get("decision") or {})
            record["application_pod_id"] = job.get("app_pod_id")
            record["visit_num"] = job.get("visit_num", 1)
            self.on_log(f"Attached application model scores to {job['app_pod_id']} multi-view pose", "info")
            self._dec_outstanding()
        if keep:
            with self._condition:
                self._updates.extendleft(reversed(keep))
                self._condition.notify_all()
