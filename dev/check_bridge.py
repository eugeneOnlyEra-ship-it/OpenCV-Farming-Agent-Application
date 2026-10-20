"""
dev/check_bridge.py -- tests for the PyBullet bridge's SCHEDULING, with a fake robot (no PyBullet needed).

Run from the project root:   python dev/check_bridge.py
It plugs a scripted fake gantry into the real PyBulletBridge and the real OrderBook and checks the behaviours
that used to freeze the application: no waiting for a "partner" order, front and back running at the same time,
live progress (not 0 -> done), failures releasing the lane, non-blocking start, and shutdown/abandon handling.
Exit code 0 = all passed.
"""
import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from pybullet_sim.bridge import PyBulletBridge          # noqa: E402
from worker_planner import FileOutbox, OrderBook, OrderPolicy, WorkOrder   # noqa: E402

FAILS = []


def ok(cond, name, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


class FakeRobot:
    """Same interface the bridge uses on GantryRobot; work is scripted with small sleeps."""
    TICK = 0.004

    def __init__(self, build_delay=0.0, fail_pods=(), ticks=12):
        time.sleep(build_delay)
        self.on_event = None
        self.localization_records = {}
        self.fail_pods = set(fail_pods)
        self.ticks = ticks
        self.sessions = []
        self.marks = []
        self.disconnected = False
        self.timeline = []                       # (monotonic time, kind, pod_id) for every event

    # --- interface used by the bridge
    def park_at_dock(self, y, step_sleep=0.0):
        pass

    def prewarm_inference(self):
        pass

    def mark_pod_decision(self, pod_id, decision):
        self.marks.append((pod_id, decision))

    def recover_after_failure(self):
        return True

    def disconnect(self):
        self.disconnected = True

    def _emit(self, kind, pod):
        self.timeline.append((time.monotonic(), kind, pod["pod_id"]))
        if self.on_event:
            self.on_event({"kind": kind, "pod_id": pod["pod_id"], "pod": pod})

    def _camera(self, side, pods, assess, shared):
        for pod in pods:
            for _ in range(3):
                for _ in range(self.ticks):
                    yield
                self._emit("plant_scan", pod)
            if pod["pod_id"] in self.fail_pods:
                raise RuntimeError("simulated camera failure")
            shared["assessed"][pod["pod_id"]] = assess(pod)
            self._emit("assess", pod)
            self.localization_records[pod["pod_id"]] = {"pod": pod}
            shared["camera_left"][side] += 1
            self._emit("camera_left", pod)

    def _service(self, side, pods, shared):
        for i, pod in enumerate(pods):
            while shared["camera_left"][side] <= i:
                yield
            action = shared["assessed"][pod["pod_id"]].get("service_action")
            if action is None:
                self._emit("service_skip", pod)
                continue
            self._emit("service_start", pod)
            for _ in range(3):
                for _ in range(self.ticks):
                    yield
                self._emit("service_contact", pod)
            self._emit("service_done", pod)

    def service_station_live(self, row, col, assess, poll, step_sleep=0.0, max_wall_seconds=900.0):
        self.sessions.append((row, col, time.monotonic()))
        shared = {"assessed": {}, "camera_left": {"front": 0, "back": 0}}
        live = {"front": [], "back": []}
        t0 = time.monotonic()
        while True:
            idle = tuple(s for s in ("front", "back") if not live[s])
            if idle:
                for side, pods in (poll(idle) or {}).items():
                    if side in idle and pods:
                        shared["camera_left"][side] = 0
                        live[side] = [self._camera(side, pods, assess, shared), self._service(side, pods, shared)]
            if not any(live.values()):
                break
            for side in live:
                for t in list(live[side]):
                    try:
                        next(t)
                    except StopIteration:
                        live[side].remove(t)
            time.sleep(self.TICK)
            if time.monotonic() - t0 > max_wall_seconds:
                raise TimeoutError("session too long")


def make_bridge(build_delay=0.0, fail_pods=(), ticks=12):
    tmp = tempfile.mkdtemp()
    book = OrderBook(tmp, OrderPolicy(auto_approve_all=True, simulate_robots=False))
    robots = []

    def factory(bridge):
        r = FakeRobot(build_delay, fail_pods, ticks)
        robots.append(r)
        return r
    bridge = PyBulletBridge(gui=False, robot_factory=factory)
    bridge.attach_orders(book)
    return bridge, book, robots


def add_order(bridge, book, oid, task, pod_id, side, n_steps=7):
    steps = [{"seq": i + 1, "action": f"STEP{i + 1}", "text": "", "params": {}} for i in range(n_steps)]
    o = WorkOrder(order_id=oid, task=task, role="worker" if task != "RECAPTURE" else "camera",
                  agent=f"{'worker' if task != 'RECAPTURE' else 'camera'}_{side}", pod_id=pod_id, crop_type="cabbage",
                  trough_id=pod_id.rsplit("-", 1)[0], side=side, location="", priority=1, title=oid, rationale="",
                  evidence={"rail": int(pod_id[1])}, targets=[], steps=steps, requires_approval=False,
                  status="dispatched", aspect="disease")
    book._orders[oid] = o
    bridge.enqueue_order(o.to_dict())
    return o


def wait_for(cond, timeout=10.0, step=0.01):
    t = time.monotonic()
    while time.monotonic() - t < timeout:
        if cond():
            return True
        time.sleep(step)
    return False


def check_no_partner_wait():
    print("\n[1] a lone order starts at once (the old design held it 20-60 s waiting for a partner)")
    bridge, book, robots = make_bridge()
    ok(bridge.start()[0], "bridge starts")
    t0 = time.monotonic()
    o = add_order(bridge, book, "WO-1", "TREAT", "R1-L-F1", "front")
    started = wait_for(lambda: o.status == "in_progress", 3.0)
    ok(started and time.monotonic() - t0 < 1.0, "lone front order is running within 1 s", f"{time.monotonic() - t0:.2f}s")
    ok(wait_for(lambda: o.status == "completed", 10), "and it completes", o.status)
    ok(not book.pod_busy("R1-L-F1"), "the application's lane is released")
    bridge.request_stop(); bridge.join(5)


def check_simultaneous_and_live():
    print("\n[2] front and back run at the same time; progress is live")
    bridge, book, robots = make_bridge(ticks=30)
    bridge.start()
    f = add_order(bridge, book, "WO-F", "TREAT", "R1-L-F1", "front")
    wait_for(lambda: f.status == "in_progress")
    time.sleep(0.15)
    b = add_order(bridge, book, "WO-B", "TREAT", "R1-L-B1", "back")     # arrives while the front agent is busy
    seen_f, seen_b, overlap = set(), set(), False
    t = time.monotonic()
    while time.monotonic() - t < 15 and not (f.status == "completed" and b.status == "completed"):
        seen_f.add(f.progress); seen_b.add(b.progress)
        if f.status == "in_progress" and b.status == "in_progress":
            overlap = True
        time.sleep(0.005)
    ok(overlap, "both orders were in progress at the same moment")
    ok(f.status == "completed" and b.status == "completed", "both complete", f"{f.status}/{b.status}")
    ok(len(seen_f - {0, 7}) >= 2 and len(seen_b - {0, 7}) >= 2,
       "progress advanced step by step (not 0 -> done)", f"front saw {sorted(seen_f)}, back saw {sorted(seen_b)}")
    starts = [(k, p) for _, k, p in robots[0].timeline if k == "service_start"]
    ok(len(robots[0].sessions) == 1, "one station session served both sides", str(len(robots[0].sessions)))
    bridge.request_stop(); bridge.join(5)


def check_other_station_and_pairs():
    print("\n[3] other stations wait their turn without deadlock; paired visits run both sides together")
    bridge, book, robots = make_bridge()
    bridge.start()
    a = add_order(bridge, book, "WO-A", "HARVEST", "R1-L-F1", "front")
    c = add_order(bridge, book, "WO-C", "HARVEST", "R2-M-F1", "front")       # different station
    ok(wait_for(lambda: a.status == "completed" and c.status == "completed", 15), "orders at two stations both finish")
    ok(len(robots[0].sessions) == 2, "two separate station sessions", str(len(robots[0].sessions)))
    from pod_registry import PODS
    pod = next(p for p in PODS if p["pod_id"] == "R3-R-F2")
    bridge.enqueue_visit(pod, {"action": "log_healthy"}, {"growth_stage": "S9"}, 1)
    ok(wait_for(lambda: bridge.wait_until_idle(0.05), 10), "a visit drains")
    scans = [(t, p) for t, k, p in robots[0].timeline if k == "plant_scan" and "R2C2" in p]
    sides = {p.split("-")[-1] for _, p in scans}
    ok(sides == {"F", "B"}, "the visit photographed BOTH the front and the back pod", str(sides))
    bridge.request_stop(); bridge.join(5)


def check_failure_releases_lane():
    print("\n[4] a failing job releases the lane instead of hanging it")
    import pybullet_sim.pod_registry as sr
    bridge0, _, _ = make_bridge()
    bad = bridge0._map_pod({"pod_id": "R1-L-F2", "rail": 1, "station": 0, "side": "front"})["pod_id"]
    bridge, book, robots = make_bridge(fail_pods=(bad,))
    bridge.start()
    o = add_order(bridge, book, "WO-X", "TREAT", "R1-L-F2", "front")
    ok(wait_for(lambda: o.status in ("failed", "cancelled"), 8), "the order is failed/cancelled, not stuck in progress", o.status)
    ok(not book.pod_busy("R1-L-F2"), "the application's lane is released")
    good = add_order(bridge, book, "WO-OK", "TREAT", "R1-L-F3", "front")
    ok(wait_for(lambda: good.status == "completed", 10), "the gantry carries on with the next job", good.status)
    bridge.request_stop(); bridge.join(5)


def check_start_and_shutdown():
    print("\n[5] non-blocking start; fallback; shutdown")
    bridge, book, robots = make_bridge(build_delay=1.0)
    ready = threading.Event()
    t0 = time.monotonic()
    bridge.start_async(on_ready=ready.set)
    ok(time.monotonic() - t0 < 0.3, "start_async returns immediately while the scene builds", f"{time.monotonic() - t0:.2f}s")
    early = add_order(bridge, book, "WO-E", "TREAT", "R1-L-F1", "front")          # queued before it is ready
    ok(ready.wait(5), "on_ready fires once the gantry is up")
    ok(wait_for(lambda: early.status == "completed", 10), "work queued during start-up runs afterwards", early.status)
    bridge.request_stop(); bridge.join(5)
    ok(robots[0].disconnected, "the simulator disconnects on shutdown")

    def broken_factory(b):
        raise RuntimeError("no display")
    b2 = PyBulletBridge(gui=False, robot_factory=broken_factory)
    book2 = OrderBook(tempfile.mkdtemp(), OrderPolicy(auto_approve_all=True, simulate_robots=False))
    b2.attach_orders(book2)
    failed = []
    ev = threading.Event()
    b2.start_async(on_failed=lambda why: (failed.append(why), ev.set()))
    o = add_order(b2, book2, "WO-Z", "TREAT", "R1-L-F1", "front")
    ok(ev.wait(5) and "no display" in failed[0], "a failed start reports why, without blocking", str(failed))
    ok(o.status != "in_progress", "no order is left 'in progress' after a failed start", o.status)
    # abandon keeps orders dispatched so the built-in fleet can take over
    book3 = OrderBook(tempfile.mkdtemp(), OrderPolicy(auto_approve_all=True, simulate_robots=False))
    b3 = PyBulletBridge(gui=False, robot_factory=lambda b: FakeRobot(build_delay=5))
    b3.attach_orders(book3)
    o3 = add_order(b3, book3, "WO-Q", "TREAT", "R1-L-F1", "front")
    b3.abandon("test")
    ok(o3.status == "dispatched", "abandon() leaves orders dispatched for the fallback fleet", o3.status)
    # stop cancels what has not started
    bridge4, book4, _ = make_bridge()
    bridge4.start()
    for i, pod in enumerate(("R1-L-F1", "R1-L-F2", "R1-L-F3", "R1-L-F4")):
        add_order(bridge4, book4, f"WO-S{i}", "TREAT", pod, "front")
    time.sleep(0.05)
    bridge4.request_stop(); bridge4.join(8)
    left = [o.status for o in book4._orders.values() if o.status in ("dispatched", "in_progress")]
    ok(not left, "after stop, no order is left dispatched / in progress", str(left))


def check_late_updates():
    print("\n[6] late application scores attach to the pose record, and never busy-spin the CPU")
    from pod_registry import PODS
    bridge, book, robots = make_bridge()
    bridge.start()
    front = next(p for p in PODS if p["pod_id"] == "R2-R-F3")
    back = next(p for p in PODS if p["pod_id"] == "R2-R-B3")
    bridge.enqueue_visit(front, {"action": "log_healthy"}, {"growth_stage": "S9"}, 1)
    bridge.enqueue_visit(back, {"action": "flag_for_treatment"}, {"disease_name": "X"}, 1)    # same pair -> an update
    ok(wait_for(lambda: bridge.wait_until_idle(0.05), 10), "a visit and its counterpart's update drain")
    rec = [r for k, r in robots[0].localization_records.items() if k.endswith("-B") and "R1C2-L2" in k]
    ok(bool(rec) and rec[0].get("application_pod_id") == "R2-R-B3", "the counterpart's scores were attached to its pose record",
       str(rec))
    # an update whose record never appears must wait quietly (it used to spin the loop at 100% CPU)
    sim_back = bridge._map_pod(back)
    with bridge._condition:
        bridge._outstanding += 1
        bridge._updates.append({"kind": "classification_update", "pod": dict(sim_back, pod_id="NO-RECORD"),
                                "classification": {}, "decision": {}, "app_pod_id": "X", "visit_num": 9,
                                "_t": time.monotonic()})
        bridge._condition.notify_all()
    c0, t0 = time.process_time(), time.monotonic()
    time.sleep(1.5)
    cpu = time.process_time() - c0
    ok(cpu < 0.4, "waiting for an unresolved update uses almost no CPU", f"{cpu:.2f}s of CPU in 1.5 s")
    bridge.request_stop(); bridge.join(5)


if __name__ == "__main__":
    t = time.time()
    for fn in (check_no_partner_wait, check_simultaneous_and_live, check_other_station_and_pairs,
               check_failure_releases_lane, check_start_and_shutdown, check_late_updates):
        fn()
    print(f"\n{'ALL BRIDGE CHECKS PASSED' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}   ({time.time() - t:.0f}s)")
    sys.exit(1 if FAILS else 0)
