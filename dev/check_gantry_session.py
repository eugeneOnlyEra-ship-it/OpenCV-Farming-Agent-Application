"""
dev/check_gantry_session.py -- tests the REAL GantryRobot.service_station_live() loop without PyBullet.

PyBullet is replaced by a stub module and the camera/service task generators by tiny scripted ones; what is under
test is the new session loop itself: a side that is idle picks up new work at any moment (even while the other side is
busy), the per-side camera counter is reset for each new task pair, the session ends only when both sides are idle and
poll() has nothing more, and the wall-clock / step guards raise instead of hanging.
(It does NOT test the physics or rendering: those need real PyBullet - see dev/integration_smoke.py.)
Run from the project root:  python dev/check_gantry_session.py
"""
import os
import sys
import time
import types
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


class Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock(name=name)


for name in ("pybullet", "pybullet_data"):
    sys.modules.setdefault(name, Stub(name))
from pybullet_sim import gantry_robot as gr           # noqa: E402

FAILS = []


def ok(cond, name, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def make_robot(ticks=20):
    r = object.__new__(gr.GantryRobot)
    r.log = []
    r.step_count = 0
    r.go_to_station = lambda row, col, step_sleep=0.0: 0
    r.boxes_engaged = lambda: True
    r._emit = lambda kind, pod=None, **kw: r.log.append((r.step_count, kind, (pod or {}).get("pod_id"), kw))
    r._step = lambda step_sleep=0.0: setattr(r, "step_count", r.step_count + 1)

    def g_camera(side, pods, assess, shared):
        for pod in pods:
            for _ in range(ticks):
                yield
            shared["assessed"][pod["pod_id"]] = assess(pod)
            r._emit("camera_done", pod)
            shared["camera_left"][side] += 1
            r._emit("camera_left", pod)

    def g_service(side, pods, shared):
        for i, pod in enumerate(pods):
            while shared["camera_left"][side] <= i:
                yield
            r._emit("service_start", pod)
            for _ in range(ticks):
                yield
            r._emit("service_done", pod)
    r._g_camera, r._g_service = g_camera, g_service
    return r


def pod(side, layer=0):
    return {"pod_id": f"R0C0-L{layer}-{side[0].upper()}", "side": side, "layer": layer}


def first_step(r, kind, pod_id):
    return next(s for s, k, p, _ in r.log if k == kind and p == pod_id)


def first_index(r, kind, pod_id):
    """Position of the event in the log: events inside the same physics step still have a defined order."""
    return next(i for i, (_, k, p, _) in enumerate(r.log) if k == kind and p == pod_id)


def check_dynamic_join():
    print("\n[1] a side that is idle starts new work while the OTHER side is still busy")
    r = make_robot(ticks=40)
    jobs = {"front": [[pod("front", 0)]], "back": []}
    calls = {"n": 0}

    def poll(idle):
        calls["n"] += 1
        out = {}
        # the back job only "arrives" once the front agents have been working for a while
        if "back" in idle and r.step_count >= 30 and not jobs.get("back_sent"):
            jobs["back_sent"] = True
            out["back"] = [pod("back", 0)]
        if "front" in idle and jobs["front"]:
            out["front"] = jobs["front"].pop()
        return out
    steps = r.service_station_live(0, 0, lambda p: {}, poll)
    f, b = pod("front")["pod_id"], pod("back")["pod_id"]
    ok(first_step(r, "service_start", f) >= 40, "front worker waited for ITS camera (counter reset works)")
    ok(first_step(r, "camera_done", b) >= 30, "back job joined mid-session")
    overlap = first_step(r, "camera_done", b) < first_step(r, "service_done", f)
    ok(overlap, "back camera ran while the front worker was still busy")
    ok(steps > 0 and calls["n"] > 0, "session ended by itself when both sides were idle", f"steps={steps}")


def check_reuse_side_after_finish():
    print("\n[2] a side that finished (stowed) can take another job; its camera counter is fresh")
    r = make_robot(ticks=10)
    plan = [("front", [pod("front", 1)]), ("back", [pod("back", 0)]), ("front", [pod("front", 0)])]
    state = {"i": 0}

    def poll(idle):
        out = {}
        for side in idle:
            for idx, (s, pods) in enumerate(plan):
                if s == side and state["i"] >= idx and idx not in state.setdefault("done", set()):
                    state["done"].add(idx)
                    out[side] = pods
                    break
        state["i"] += 1
        return out
    r.service_station_live(0, 0, lambda p: {}, poll)
    second = pod("front", 0)["pod_id"]
    ok(first_index(r, "camera_done", second) < first_index(r, "service_start", second),
       "the second front job's worker waited for its own camera (no stale counter)")
    done = sorted({p for _, k, p, _ in r.log if k == "service_done"})
    ok(len(done) == 3, "all three jobs ran to completion", str(done))


def check_two_pods_one_side():
    print("\n[3] several pods on one side run in order inside one task pair")
    r = make_robot(ticks=8)
    pods = [pod("front", 2), pod("front", 0)]
    sent = {"v": False}

    def poll(idle):
        if "front" in idle and not sent["v"]:
            sent["v"] = True
            return {"front": pods}
        return {}
    r.service_station_live(0, 0, lambda p: {}, poll)
    order = [p for _, k, p, _ in r.log if k == "service_done"]
    ok(order == [pods[0]["pod_id"], pods[1]["pod_id"]], "pods are served in the given order", str(order))


def check_guards():
    print("\n[4] guards raise instead of hanging")
    r = make_robot(ticks=10 ** 9)
    r._step = lambda step_sleep=0.0: (time.sleep(0.002), setattr(r, "step_count", r.step_count + 1))
    sent = {"v": False}

    def poll(idle):
        if "front" in idle and not sent["v"]:
            sent["v"] = True
            return {"front": [pod("front")]}
        return {}
    t = time.monotonic()
    try:
        r.service_station_live(0, 0, lambda p: {}, poll, max_wall_seconds=0.3)
        ok(False, "wall-clock guard raises TimeoutError")
    except TimeoutError:
        ok(time.monotonic() - t < 2.0, "wall-clock guard raises TimeoutError promptly")
    gr_limit = gr.MAX_RUN_STEPS
    gr.MAX_RUN_STEPS = 50
    r2 = make_robot(ticks=10 ** 9)
    sent["v"] = False
    try:
        r2.service_station_live(0, 0, lambda p: {}, poll)
        ok(False, "step guard raises RuntimeError")
    except RuntimeError:
        ok(True, "step guard raises RuntimeError")
    finally:
        gr.MAX_RUN_STEPS = gr_limit
    r3 = make_robot()
    res = r3.service_station_live(0, 0, lambda p: {}, lambda idle: {})
    ok(res == 0, "an empty session returns immediately")


def check_inference_helpers():
    print("\n[5] off-thread scoring helpers")
    import numpy as np
    r = object.__new__(gr.GantryRobot)
    fake = lambda pod_id, crop, bgr: {"growth_stage": "S9", "growth_confidence": 0.7, "disease_flag": False,   # noqa: E731
                                      "disease_name": None, "disease_confidence": 0.1,
                                      "growth_detections": [1, 2], "disease_detections": []}
    out = r._score_rendered_views(fake, {"pod_id": "p", "crop_type": "cabbage"}, [(0, 0, np.zeros((4, 4, 3))), (2, 1, None)])
    ok(out[0]["growth_stage"] == "S9" and out[0]["growth_detection_count"] == 2, "scores have the original keys")

    def boom(*a):
        raise RuntimeError("model missing")
    out = r._score_rendered_views(boom, {"pod_id": "p", "crop_type": "cabbage"}, [(1, 2, None)])
    ok("error" in out[1] and out[1]["view_index"] == 2, "a failing model gives an error entry instead of crashing")
    pool = r._inference_pool()
    ok(pool is r._inference_pool(), "one dedicated inference thread is reused")
    fut = pool.submit(lambda: threading_name())
    ok(fut.result(5).startswith("sim-infer"), "inference runs on its own thread")
    pool.shutdown(wait=False)


def threading_name():
    import threading
    return threading.current_thread().name


if __name__ == "__main__":
    for fn in (check_dynamic_join, check_reuse_side_after_finish, check_two_pods_one_side, check_guards,
               check_inference_helpers):
        fn()
    print("\n" + ("ALL GANTRY-SESSION CHECKS PASSED" if not FAILS else "FAILED: " + "; ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
