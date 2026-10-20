"""
dev/ui_live_check.py -- run the REAL application against a scripted fake gantry on a virtual display and measure:
  * the longest time the window was unresponsive (Tk event-loop gap),
  * whether front and back agents were busy at the same moment,
  * how long orders sat 'dispatched' before the robot started them.
Optionally saves screenshots.   Needs xvfb + ImageMagick (see dev/ui_screenshots.py).

  xvfb-run -a -s "-screen 0 1500x960x24" python dev/ui_live_check.py [--seconds 40] [--shots dev/shots] [--build-delay 3]
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dev"))

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=40)
ap.add_argument("--build-delay", type=float, default=3.0, help="seconds the fake PyBullet scene takes to build")
ap.add_argument("--ticks", type=int, default=100, help="job length (bigger = slower robot)")
ap.add_argument("--shots", default="", help="folder for screenshots (omit for none)")
ap.add_argument("--fallback", action="store_true",
                help="use the REAL bridge (PyBullet missing here) to test the fall-back to the built-in fleet")
ap.add_argument("--profile", action="store_true", help="time the UI's own update functions and list the slowest")
ap.add_argument("--pods", nargs="*", default=["R1-L-F1", "R1-L-B1", "R1-L-F2", "R1-L-B2"])
a = ap.parse_args()

import app_ui                                          # noqa: E402
import check_bridge                                    # noqa: E402  (FakeRobot)
from pod_registry import PODS                          # noqa: E402
from pybullet_sim.bridge import PyBulletBridge         # noqa: E402

robots = []


def fake_bridge(gui=True, on_log=None):
    def factory(b):
        r = check_bridge.FakeRobot(build_delay=a.build_delay, ticks=a.ticks)
        robots.append(r)
        return r
    return PyBulletBridge(gui=False, on_log=on_log, robot_factory=factory)


if not a.fallback:
    app_ui.PyBulletBridge = fake_bridge
slow = []
if a.profile:
    import functools

    def timed(owner, name):
        fn = getattr(owner, name)

        @functools.wraps(fn)
        def wrapper(*args, **kw):
            t = time.monotonic()
            try:
                return fn(*args, **kw)
            finally:
                d = time.monotonic() - t
                if d > 0.04:
                    slow.append((round(d, 3), f"{owner.__name__}.{name}", round(time.monotonic() - T0, 1)))
        setattr(owner, name, wrapper)
    T0 = time.monotonic()
    for owner, names in ((app_ui.PodCard, ("render", "refresh_worker_panels", "refresh_orders")),
                         (app_ui.ImageView, ("redraw",)), (app_ui.ScorePanel, ("redraw",)),
                         (app_ui.WorkerPanel, ("show",)), (app_ui.FleetStrip, ("update_status",)),
                         (app_ui.App, ("_pump_once", "_tick_once", "_build_cards", "_do_start", "_log_line")),
                         (app_ui.Timeline, ("draw",))):
        for n in names:
            timed(owner, n)
app = app_ui.App()
app.geometry("1500x940+0+0")
chosen = [p for p in PODS if p["pod_id"] in a.pods]
app._selected_pods = lambda: app._pair_sides(chosen)
app.var_delay.set(300)

gaps, last = [], [time.monotonic()]
measuring = [True]


stalls = []
T0 = time.monotonic()


def beat():
    now = time.monotonic()
    if measuring[0]:
        gaps.append(now - last[0])
        if now - last[0] > 0.15:
            stalls.append((round(now - T0, 1), round(now - last[0], 2)))
    last[0] = now
    app.after(20, beat)


samples = {"both_cameras": 0, "both_workers": 0, "any": 0, "n": 0}


def sample():
    try:
        eng = app.engine
        if eng and eng.simulation_bridge is not None:
            fs = eng.simulation_bridge.fleet_status(eng.orders.fleet_status())
            act = {k: v["active"] is not None for k, v in fs.items()}
            samples["n"] += 1
            samples["both_cameras"] += act["camera_front"] and act["camera_back"]
            samples["both_workers"] += act["worker_front"] and act["worker_back"]
            samples["any"] += any(act.values())
    except Exception as exc:
        print("sample error", exc)
    app.after(100, sample)


t_start = {}


def go():
    t = time.monotonic()
    app.start()
    t_start["blocked"] = time.monotonic() - t          # how long start() held the window
    print(f"app.start() returned after {t_start['blocked'] * 1000:.0f} ms", flush=True)


def shot(i):
    if not a.shots:
        return
    measuring[0] = False                               # screenshots themselves block the window briefly
    os.makedirs(a.shots, exist_ok=True)
    app.update_idletasks(); app.update()
    subprocess.run(["import", "-window", "root", os.path.join(a.shots, f"live_{i:02d}.png")])
    last[0] = time.monotonic(); measuring[0] = True


def finish():
    eng = app.engine
    orders = eng.orders.list_orders() if eng else []
    by = {}
    for o in orders:
        by[o.status] = by.get(o.status, 0) + 1
    print(f"longest window stall: {max(gaps) * 1000:.0f} ms   (median {sorted(gaps)[len(gaps) // 2] * 1000:.0f} ms, {len(gaps)} beats)")
    print(f"samples with BOTH cameras busy: {samples['both_cameras']}/{samples['n']},  BOTH workers busy: {samples['both_workers']}/{samples['n']}")
    print("orders:", by, "| pods done:", eng.stats()["pods_done"] if eng else None)
    print("robot sessions:", [len(r.sessions) for r in robots])
    if a.fallback:
        sim = eng.orders.fleet is not None
        print("fall-back fleet running:", sim, "| orders still 'dispatched' with nobody to run them:",
              sum(1 for o in orders if o.status == "dispatched"))
    errs = [l for l in app.log.get("1.0", "end").splitlines() if "internal UI error" in l]
    print("internal UI errors:", len(errs))
    print("stalls > 150 ms  (seconds since launch, length s):", stalls)
    if a.profile:
        print("slowest UI-thread functions (seconds, function, at t):")
        for row in sorted(slow, reverse=True)[:14]:
            print("   ", row)
    if eng:
        eng.stop()
    app.after(1500, app.destroy)


app.after(50, beat)
app.after(100, sample)
app.after(500, go)
if a.shots:
    for i in range(int(a.seconds // 4)):
        app.after(int((6 + i * 4) * 1000), lambda i=i: shot(i))
app.after(int(a.seconds * 1000), finish)
app.mainloop()
