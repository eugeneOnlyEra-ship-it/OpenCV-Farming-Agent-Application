"""check_rig_stream.py -- the streaming camera-pair calibration, without PyBullet.

Drives pybullet_sim.rig_calibration.run() (the code behind `python -m pybullet_sim.rig_calibration --stream`) with
software-rendered chessboard frames and checks the event stream the Calibration tab reads: a start event, one frame
event per board pose carrying the corners and (from the 6th usable pair on) a re-solved estimate, a final event with the
result; that the estimate converges to the true baseline, the result is saved, and stereo_rig.load_saved accepts it.

  python dev/check_rig_stream.py
"""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dev.check_pod_markers import look_at, projection, W, H
from dev.check_stereo_rig import render_board
from pybullet_sim import rig_calibration as rc, stereo_rig as sr

def frames(n):
    P = projection()
    (ea, ta), (eb, tb) = sr.rig_cameras(rc.RIG_EYE, rc.RIG_TARGET)
    Va, Vb = look_at(ea, ta), look_at(eb, tb)
    for pos, R in rc.board_poses(n):
        ga, gb = render_board(pos, R, Va, P, 4), render_board(pos, R, Vb, P, 4)
        yield (np.dstack([ga, ga, ga, np.full_like(ga, 255)]), np.dstack([gb, gb, gb, np.full_like(gb, 255)]))

tmp = tempfile.mkdtemp()
save_path = os.path.join(tmp, "logs", "stereo_rig_calibration.json")
events = []
rig = rc.run(frames(20), 20, lambda e: events.append(json.loads(json.dumps(e))), out_dir=os.path.join(tmp, "frames"),
             save_path=save_path)
kinds = [e["event"] for e in events]
assert kinds == ["start"] + ["frame"] * 20 + ["done"], kinds
frame_events = [e for e in events if e["event"] == "frame"]
assert all(os.path.exists(e["path_a"]) and os.path.exists(e["path_b"]) for e in frame_events)
assert all(e["corners_a"] and len(e["corners_a"]) == sr.BOARD_COLS * sr.BOARD_ROWS for e in frame_events if e["pair_ok"])
assert frame_events[4]["estimate"] is None and frame_events[5]["estimate"] is not None, "first estimate at the 6th pair"
est = [e["estimate"] for e in frame_events if e["estimate"]]
assert abs(est[-1]["baseline_mm"] - sr.BASELINE_M * 1000) < 1.0, est[-1]
t = rig["truth_comparison"]
assert t["translation_error_mm"] < 2.0 and t["rotation_error_deg"] < 0.3, t
loaded, why = sr.load_saved(tmp, (W, H))
assert why == "ok" and loaded is not None and abs(loaded["baseline_m"] - rig["baseline_m"]) < 1e-12
assert sr.load_saved(tmp, (640, 480))[1] == "stale" and sr.load_saved(os.path.join(tmp, "nothing"), (W, H))[1] == "missing"
print(f"OK: {len(frame_events)} frame events, baseline estimate {est[0]['baseline_mm']:.2f} -> {est[-1]['baseline_mm']:.2f} mm "
      f"(true {sr.BASELINE_M * 1000:.0f}), translation error {t['translation_error_mm']:.2f} mm")
