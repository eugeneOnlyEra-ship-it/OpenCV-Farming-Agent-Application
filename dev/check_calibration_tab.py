"""check_calibration_tab.py -- the Calibration tab, live, without PyBullet.

Feeds the real window the same two streams it gets in a run: (1) the calibration process's JSON events (here produced
by rig_calibration.run on software-rendered chessboard frames, paced like a live process) and (2) pod samples
published by the robot's own publisher (gantry_robot._publish_rig_sample, fed by the real tag estimator on
software-rendered pod views). Checks the tab fills in (frames, calculations, parameters, charts), that the window
never stalls while it does, and that Start calibrates first when no calibration is saved. Needs a display
(on a headless box:  xvfb-run python dev/check_calibration_tab.py); prints SKIP without one.

  python dev/check_calibration_tab.py
"""
import importlib.util, json, os, shutil, sys, tempfile, threading, time
from unittest.mock import MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import tkinter as tk
try:
    tk.Tk().destroy()
except tk.TclError:
    print("SKIP: no display available"); sys.exit(0)
import numpy as np
os.environ["SS"] = "8"
sys.modules.setdefault('pybullet_data', MagicMock())
if importlib.util.find_spec("pybullet") is None:
    sys.modules['pybullet'] = MagicMock()
import app_ui
from app_ui import stereo_rig, sim_options
from dev import check_pod_markers as cpm
from dev.check_pod_markers import look_at, projection
from dev.check_rig_stream import frames as board_frames
from pybullet_sim import rig_calibration as rc, pod_markers as pm
from pybullet_sim.pod_registry import PODS, TROUGH_HALF_EXTENTS_M, ROW_SPACING_M

app = app_ui.App()
app.update()
saved = os.path.join(app_ui.HERE, stereo_rig.CALIBRATION_FILE)
backup = saved + ".testbackup"
if os.path.exists(saved):
    shutil.move(saved, backup)
try:
    app.notebook.select(app._cal_tab_index); app.update()
    app._cal_reload()
    assert app.cal_chip.cget("text") == "NOT CALIBRATED", app.cal_chip.cget("text")

    # heartbeat: the longest gap between two 10 ms ticks of the Tk loop is the longest stall the user would feel
    gaps, last = [], [time.perf_counter()]
    def beat():
        now = time.perf_counter(); gaps.append(now - last[0]); last[0] = now; app.after(10, beat)
    app.after(10, beat)

    # ---- 1. a calibration streaming in like a live process --------------------------------------------------------
    tmp = tempfile.mkdtemp()
    def producer():
        def emit(e):
            app._cal_q.put(("line", json.dumps(e) + "\n")); time.sleep(0.08)
        rc.run(board_frames(20), 20, emit, out_dir=os.path.join(tmp, "f"), save_path=os.path.join(tmp, "rig.json"))
    app._cal_proc = object()                      # as if the process were running (so its events are accepted)
    app._cal_mode = "calibration"
    t = threading.Thread(target=producer); t.start()
    seen_busy = False
    while t.is_alive() or not app._cal_q.empty():
        app.update(); time.sleep(0.01)
        seen_busy = seen_busy or app.cal_chip.cget("text") == "CALIBRATING"
    for _ in range(30): app.update(); time.sleep(0.01)
    text = app.cal_text.get("1.0", "end")
    assert seen_busy, "the chip should show CALIBRATING while frames arrive"
    assert app.cal_chip.cget("text") == "CALIBRATED" and app._cal_proc is None
    assert text.count("pose") >= 20 and "solve with" in text and "result" in text
    assert app.cal_vars["pairs"].get() == "20 of 20", app.cal_vars["pairs"].get()
    assert app.cal_vars["baseline"].get().startswith("39.9") or app.cal_vars["baseline"].get().startswith("40.0")
    assert len(app._cal_baselines) >= 14 and len(app.cal_chart_base.find_all()) > 12
    assert app._cal_photo.get("a") is not None and app._cal_photo.get("b") is not None
    assert "corners" in app.cal_caption["a"].cget("text")

    # ---- 2. pod samples from the robot's own publisher -------------------------------------------------------------
    from pybullet_sim import gantry_robot as gr
    rig = json.load(open(os.path.join(tmp, "rig.json")))
    robot = gr.GantryRobot.__new__(gr.GantryRobot)
    P = projection(); half_y = TROUGH_HALF_EXTENTS_M[1]
    for idx, pod in enumerate(PODS[:3]):
        sign = 1 if pod["side"] == "front" else -1
        views = []
        for vi, plant in enumerate(pod["plants"]):
            eye = np.array([plant["x"], pod["row"] * ROW_SPACING_M + sign * 0.505,
                            pod["position_m"]["z"] + TROUGH_HALF_EXTENTS_M[2] + 0.22])
            tgt = [pod["position_m"]["x"], pod["position_m"]["y"], pod["position_m"]["z"] + TROUGH_HALF_EXTENTS_M[2]]
            (ea, ta), (eb, tb) = stereo_rig.rig_cameras(eye, tgt)
            Va, Vb = look_at(ea, ta), look_at(eb, tb)
            views.append({"view_index": vi, "camera_position_world": ea, "view_matrix": Va.flatten(order="F").tolist(),
                          "rgba": cpm.render(pod, pm.marker_id_for_pod(idx), np.array(ea), Va, P, half_y),
                          "rgba_b": cpm.render(pod, pm.marker_id_for_pod(idx), np.array(eb), Vb, P, half_y)})
        vp = pm.estimate_pod_pose_from_markers(views, pod, pm.marker_id_for_pod(idx), TROUGH_HALF_EXTENTS_M, rig=rig)
        pose = dict(vp); pose["fallback_used"] = False
        robot._publish_rig_sample(pod, views, vp, pose, [pod["position_m"][k] for k in "xyz"])
    end = time.time() + 3.0
    while time.time() < end:
        app.update(); time.sleep(0.01)
    text = app.cal_text.get("1.0", "end")
    assert all(p["pod_id"] in text for p in PODS[:3]) and "fused" in text and "true position" in text and "tag found" in text
    assert len(app._cal_errors) == 3 and max(app._cal_errors) < 8.0, app._cal_errors
    assert len(app.cal_chart_err.find_all()) > 8
    assert PODS[0]["pod_id"] in app.cal_caption["a"].cget("text") or PODS[2]["pod_id"] in app.cal_caption["a"].cget("text")
    stall = max(gaps[5:]) if len(gaps) > 5 else 0
    print(f"longest window stall while the tab was busy: {stall * 1000:.0f} ms over {len(gaps)} beats")
    assert stall < 0.45, f"window stalled for {stall:.2f} s"

    # ---- 3. Start calibrates first when nothing is saved --------------------------------------------------------------
    app.pybullet_var.set(True); sim_options.OPTIONS["stereo_rig"] = True
    app_ui.importlib.util.find_spec = lambda name: object()          # pretend PyBullet is installed
    calls = []
    app.calibrate_rig = lambda then=None: calls.append(then)
    app._selected_pods = lambda: []                                 # so the gated Start stops before a real run
    app._do_start()
    assert len(calls) == 1 and callable(calls[0]), "Start should calibrate first when nothing is saved"
    calls[0]()                                                       # the calibration finished -> start continues
    assert len(calls) == 1, "the continued start must not calibrate again"
    shutil.copy(os.path.join(tmp, "rig.json"), saved)               # now one is saved: no calibration needed
    app._do_start()
    assert len(calls) == 1, "no calibration when one is saved"
    print("OK: Calibration tab fills in live; Start calibrates first when needed")
finally:
    if os.path.exists(saved):
        os.remove(saved)
    if os.path.exists(backup):
        shutil.move(backup, saved)
    app.destroy()
