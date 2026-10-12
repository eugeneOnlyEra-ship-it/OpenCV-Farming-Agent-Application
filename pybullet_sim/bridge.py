"""Threaded PyBullet executor used by the Tkinter/OpenCV application.

The application remains responsible for perception, decisions, history and
work-order approval. This module turns its scan visits and dispatched work
orders into physical gantry movements. All PyBullet calls stay on one thread.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback

from . import pod_registry as sim_registry


class PyBulletBridge:
    """Serial work queue with pause/resume/step/stop at job-safe boundaries."""

    def __init__(self, gui=True, on_log=None, speed=2.0):
        self.gui = gui
        self.speed = speed
        self.on_log = on_log or (lambda message, level="info": None)
        self.jobs = queue.Queue()
        self._condition = threading.Condition()
        self._paused = False
        self._step_tickets = 0
        self._stopping = False
        self._started = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._robot = None
        self._orders = None
        self._startup_error = None
        self._last_error = None
        self._active_jobs = 0
        self._outstanding_jobs = 0
        self._current_job = None
        self._sim_pods = sim_registry.service_visit_order()
        self._sim_by_position = {
            (p["row"], p["col"], p["layer"], p["side"]): p
            for p in self._sim_pods
        }

    @staticmethod
    def dependency_available():
        try:
            import pybullet  # noqa: F401
            import pybullet_data  # noqa: F401
            return True, ""
        except Exception as exc:
            return False, str(exc)

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

    def start(self, timeout=20.0):
        ok, why = self.dependency_available()
        if not ok:
            return False, f"PyBullet is unavailable: {why}"
        if self._thread and self._thread.is_alive():
            return True, "already running"
        self._thread = threading.Thread(target=self._run, daemon=True, name="pybullet-gantry")
        self._thread.start()
        if not self._ready.wait(timeout):
            self.request_stop()
            return False, "PyBullet simulator did not initialise in time"
        if self._startup_error:
            return False, f"PyBullet simulator failed to initialise: {self._startup_error}"
        self._started.set()
        return True, "PyBullet robot simulation connected"

    def enqueue_visit(self, app_pod, decision, classification=None):
        """Queue the robot camera visit with the app's OpenCV model scores."""
        sim_pod = self._map_pod(app_pod)
        if sim_pod is None:
            self.on_log(f"No PyBullet pod mapping for {app_pod.get('pod_id')}", "error")
            return
        with self._condition:
            if self._stopping:
                return
            self._outstanding_jobs += 1
            self.jobs.put({"kind": "visit", "pod": sim_pod, "decision": decision,
                           "classification": dict(classification or {}),
                           "app_pod_id": app_pod.get("pod_id", sim_pod["pod_id"])})
            self._condition.notify_all()

    def enqueue_order(self, order):
        sim_pod = self._map_pod({
            "pod_id": order.get("pod_id"), "rail": order.get("evidence", {}).get("rail"),
            "station": order.get("evidence", {}).get("station"), "side": order.get("side"),
        })
        if sim_pod is None:
            self.on_log(f"Work order {order.get('order_id')} has no PyBullet pod mapping; left in outbox", "error")
            return
        with self._condition:
            if self._stopping:
                return
            self._outstanding_jobs += 1
            self.jobs.put({"kind": "order", "order": dict(order), "pod": sim_pod})
            self._condition.notify_all()

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
        with self._condition:
            self._stopping = True
            self._paused = False
            self._step_tickets = 0
            self._condition.notify_all()
        # Don't start queued actions after Stop; let the currently executing
        # physical job finish and then park the gantry.
        try:
            while True:
                job = self.jobs.get_nowait()
                self.jobs.task_done()
                self._cancel_job(job, "run stopped before the robot started this job")
                with self._condition:
                    self._outstanding_jobs = max(0, self._outstanding_jobs - 1)
                    self._condition.notify_all()
        except queue.Empty:
            pass

    def join(self, timeout=30.0):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def wait_until_idle(self, timeout=None):
        """Wait until queued and currently executing robot jobs have drained.

        If the user has paused the run, this intentionally waits for Resume or
        Stop rather than marking the combined run finished prematurely.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._stopping and self._outstanding_jobs > 0:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._condition.wait(min(0.2, remaining))
                else:
                    self._condition.wait(0.2)
            return True

    def fleet_status(self, base_status):
        """Merge live camera-visit state with the order-book's worker statuses."""
        result = {agent: dict(info) for agent, info in (base_status or {}).items()}
        with self._condition:
            job = self._current_job
        if job and job.get("kind") == "visit":
            side = job["pod"]["side"]
            agent = f"camera_{side}"
            if agent in result:
                result[agent]["active"] = {
                    "order_id": None, "task": "SCAN",
                    "pod": job.get("app_pod_id", job["pod"]["pod_id"]),
                    "n": 1, "step": 0, "action": "camera visit",
                }
                result[agent]["queued"] = max(0, result[agent].get("queued", 0))
        return result

    def _wait_for_permission(self):
        with self._condition:
            while self._paused and not self._stopping and self._step_tickets <= 0:
                self._condition.wait(0.2)
            if self._stopping:
                return False
            if self._paused and self._step_tickets > 0:
                self._step_tickets -= 1
            return True

    def _run(self):
        robot = None
        try:
            from .gantry_robot import GantryRobot
            from .pod_registry import TOP_DOCK_Y, BOTTOM_DOCK_Y
            robot = GantryRobot(pods=self._sim_pods, gui=self.gui, speed_scale=self.speed)
            self._robot = robot
            robot.park_at_dock(TOP_DOCK_Y)
            self._ready.set()
            self.on_log("PyBullet gantry connected; waiting for app scans/work orders", "ok")
            while True:
                try:
                    job = self.jobs.get(timeout=0.2)
                except queue.Empty:
                    if self._stopping:
                        break
                    continue
                with self._condition:
                    self._active_jobs += 1
                    self._condition.notify_all()
                try:
                    if not self._wait_for_permission():
                        self._cancel_job(job, "run stopped before the robot started this job")
                        break
                    if self._stopping:
                        self._cancel_job(job, "run stopped before the robot started this job")
                        continue
                    with self._condition:
                        self._current_job = job
                        self._condition.notify_all()
                    if job["kind"] == "visit":
                        self._execute_visit(robot, job["pod"], job.get("decision"), None,
                                            classification=job.get("classification"))
                    else:
                        self._execute_order(robot, job)
                except Exception as exc:
                    self._last_error = repr(exc)
                    self.on_log(f"PyBullet job failed: {exc!r}", "error")
                    self.on_log(traceback.format_exc(), "error")
                finally:
                    self.jobs.task_done()
                    with self._condition:
                        self._current_job = None
                        self._active_jobs = max(0, self._active_jobs - 1)
                        self._outstanding_jobs = max(0, self._outstanding_jobs - 1)
                        self._condition.notify_all()
            # Stop is graceful: finish the active job, then park before disconnecting.
            try:
                robot.park_at_dock(BOTTOM_DOCK_Y)
            except Exception as exc:
                self.on_log(f"Could not park gantry during shutdown: {exc!r}", "warn")
        except Exception as exc:
            self._startup_error = repr(exc)
            self._ready.set()
            self.on_log(f"PyBullet simulator error: {exc!r}", "error")
        finally:
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:
                    pass
            self._robot = None
            self.on_log("PyBullet gantry disconnected", "info")

    def _cancel_job(self, job, note):
        if job.get("kind") == "order" and self._orders is not None:
            self._orders.cancel_dispatched(job["order"].get("order_id"), note)

    def _execute_visit(self, robot, sim_pod, decision, service_action, classification=None, plant_indices=None):
        action = (decision or {}).get("action")
        decision_map = {
            "log_healthy": "log_and_continue",
            "flag_for_treatment": "flag_for_treatment",
            "flag_for_harvest": "flag_for_harvest",
            "schedule_frequent_monitoring": "schedule_monitoring",
        }
        robot.mark_pod_decision(sim_pod["pod_id"], decision_map.get(action))
        station = {
            "row": sim_pod["row"], "col": sim_pod["col"],
            "front": [sim_pod] if sim_pod["side"] == "front" else [],
            "back": [sim_pod] if sim_pod["side"] == "back" else [],
        }
        def assess(_pod):
            return {
                "service_action": service_action,
                "camera_dwell": 0.0,
                "model_scores": dict(classification or {}),
                "classification": dict(classification or {}),
                "decision": dict(decision or {}),
                "plant_indices": plant_indices,
            }
        robot.service_station(station, assess)

    def _execute_order(self, robot, job):
        order = job["order"]
        order_id = order.get("order_id")
        book = self._orders
        if book is None:
            self.on_log(f"No order book attached for {order_id}", "error")
            return
        n_steps = book.begin(order_id, agent="pybullet_gantry")
        if n_steps is None:
            return  # cancelled, stale, or already handled
        task = order.get("task")
        steps = order.get("steps") or []
        whole_plant_removal = task == "TREAT" and any(
            str(step.get("action", "")).upper() == "REMOVE_TISSUE"
            and (step.get("params") or {}).get("mode") == "whole_plant"
            for step in steps if isinstance(step, dict))
        service_action = {
            "TREAT": "uproot" if whole_plant_removal else "apply_treatment",
            "HARVEST": "harvest",
            "REPLANT": "replant",
            "RECAPTURE": None,
            "VERIFY": None,
        }.get(task)
        plant_indices = None
        if whole_plant_removal:
            # The app model's normalized detection centres select the nearest of
            # the three simulated plants along the trough. If there are no boxes,
            # remove one central plant rather than clearing the entire pod.
            targets = order.get("targets") or []
            centers = [t.get("center_norm", [0.5, 0.5])[0] for t in targets if isinstance(t, dict)]
            if centers:
                plant_indices = sorted({max(0, min(2, int(float(x) * 3))) for x in centers})
            else:
                plant_indices = [1]
        self.on_log(f"PyBullet executing {order_id} {task} at {order.get('pod_id')}"
                    + (" (visible uprooting)" if whole_plant_removal else ""), "info")
        self._execute_visit(robot, job["pod"], {"action": "flag_for_treatment" if task == "TREAT" else
                                                  "flag_for_harvest" if task == "HARVEST" else
                                                  "schedule_frequent_monitoring"}, service_action,
                            classification=order.get("evidence", {}), plant_indices=plant_indices)
        for _ in range(n_steps):
            book.advance(order_id)
        book.complete(order_id, note="executed by PyBullet gantry simulation")
        self.on_log(f"PyBullet completed {order_id} {task} at {order.get('pod_id')}", "ok")
