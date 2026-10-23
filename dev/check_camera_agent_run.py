"""check_camera_agent_run.py -- run a camera agent's WHOLE visit to one pod without PyBullet.

Drives gantry_robot.GantryRobot._g_camera (the real generator: load the saved camera-pair calibration, capture three
views with both lenses, save the frames, localize the pod from its ArUco tag, publish the live sample for the
Calibration tab, emit the events, move on) against a fake PyBullet whose getCameraImage renders the pod's tag in
software. Two cases: with a saved calibration (stereo rig used) and without (single camera, and nothing may block).
A mistake in this generator stops EVERY agent at once (the service agents wait on the camera), so it is worth running.

  python dev/check_camera_agent_run.py

It does NOT check PyBullet itself (rendering, the GUI, physics).
"""
import os, sys, tempfile, types
import numpy as np
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ["SS"] = "8"
from dev.check_pod_markers import look_at, projection, render
from dev import check_stereo_rig as cs
from pybullet_sim.pod_registry import PODS, TROUGH_HALF_EXTENTS_M


class FakePyBullet:
    """Just enough of pybullet for the camera agent; unknown names (constants) come back as harmless mocks."""
    ER_TINY_RENDERER = 1; ER_NO_SEGMENTATION_MASK = 2

    def __init__(self, pod, marker_id):
        self.pod, self.marker_id = pod, marker_id

    def __getattr__(self, name):
        return MagicMock()

    def computeProjectionMatrixFOV(self, **k): return projection().flatten(order="F").tolist()
    def computeViewMatrix(self, eye, tgt, up): return look_at(eye, tgt).flatten(order="F").tolist()
    def getBasePositionAndOrientation(self, b, **k):
        pos = self.pod["position_m"]; return ((pos["x"], pos["y"], pos["z"]), (0, 0, 0, 1))
    def getCameraImage(self, w, h, viewMatrix=None, **k):
        V = np.asarray(viewMatrix, float).reshape(4, 4, order="F")
        rgba = render(self.pod, self.marker_id, np.linalg.inv(V)[:3, 3], V, projection(), TROUGH_HALF_EXTENTS_M[1])
        return (w, h, rgba, np.ones((h, w)), np.zeros((h, w), dtype=np.int64))


def run(with_calibration):
    pod, marker_id = PODS[0], 0
    fake = FakePyBullet(pod, marker_id)
    sys.modules['pybullet'] = fake
    sys.modules['pybullet_data'] = MagicMock()
    sys.modules['perception'] = types.ModuleType('perception')      # no classify_pod -> the model step is skipped
    for m in [m for m in sys.modules if m.startswith('pybullet_sim.gantry_robot')]:
        del sys.modules[m]
    from pybullet_sim import gantry_robot as gr, stereo_rig as sr, rig_feed
    gr.p = fake
    tmp = tempfile.mkdtemp()
    gr.__file__ = os.path.join(tmp, "pybullet_sim", "gantry_robot.py")  # logs/ and the saved calibration live in tmp
    if with_calibration:
        rig, _ = cs.calibrate(8)
        sr.save_rig(rig, os.path.join(tmp, sr.CALIBRATION_FILE))
    for name, value in (("CAMERA_PLANT_SECONDS", 0.005), ("CAMERA_CAPTURE_SECONDS", 0.005), ("CAMERA_HOLD_SECONDS", 0.005)):
        setattr(gr, name, value)

    cam = gr.Agent("front", "camera"); cam.stage_joints = (3, 4); cam.joint = 5
    r = gr.GantryRobot.__new__(gr.GantryRobot)
    r.client = 0; r.robot_id = 1; r.rig = None; r._rig_checked = False
    r.events = []; r.sim_steps = 0; r.on_event = None; r.on_localization = None; r.localization_records = {}
    r.pod_marker_ids = {pod["pod_id"]: 7}; r.aruco_ids = {pod["pod_id"]: marker_id}
    r.agent = lambda side, role: cam
    state = {"k": 0}
    def stage_target_for_plant(c, p_, k): state["k"] = k; return 0.0
    r.stage_target_for_plant = stage_target_for_plant
    r.stage_cmd = lambda c, t: {}
    def settle(targets): yield
    r._g_settle = settle
    r.agent_head_position = lambda c: (pod["plants"][state["k"]]["x"], pod["row"] * gr.ROW_SPACING_M + 0.505,
                                       pod["position_m"]["z"] + TROUGH_HALF_EXTENTS_M[2] + 0.22)
    rig_feed.clear()
    gen = r._g_camera("front", [pod], lambda p_: {}, {"assessed": {}, "camera_left": {}})
    steps = 0
    for _ in gen:
        steps += 1
        assert steps < 200000, "camera generator did not finish"
    return gr, r, tmp, rig_feed.drain(), steps


if __name__ == "__main__":
    for with_cal in (True, False):
        gr, r, tmp, samples, steps = run(with_cal)
        rec = r.localization_records[PODS[0]["pod_id"]]
        events = {e["kind"]: e for e in r.events}
        err = np.linalg.norm(rec["vision_error_m"]) * 1000
        b_frames = [f for f in os.listdir(os.path.join(tmp, "logs", "multiview")) if f.endswith("_B.png")]
        print(f"[saved calibration: {with_cal}] {steps} steps, method {rec['method']}, valid {rec['valid']}, "
              f"fallback {rec['fallback_used']}, error {err:.2f} mm, B frames {len(b_frames)}, "
              f"stereo_rig event: calibrated={events['stereo_rig']['calibrated']} reason={events['stereo_rig']['reason']}")
        assert rec["valid"] and not rec["fallback_used"] and err < 8.0
        assert len(samples) == 1, "exactly one live sample per pod"
        s = samples[0]
        assert s["pod_id"] == PODS[0]["pod_id"] and len(s["views"]) == 3 and s["views"][0]["frame_a"] is not None
        assert s["views"][0]["corners_a"] is not None and s["error_mm"] is not None
        for need in ("stereo_rig", "plant_scan", "assess", "pod_vision_pose", "pod_localized", "camera_left"):
            assert need in events, f"missing event {need}"
        if with_cal:
            assert rec["method"].endswith("_stereo_rig") and len(b_frames) == 3
            assert events["stereo_rig"]["calibrated"] and events["stereo_rig"]["reason"] == "ok"
            assert s["views"][0]["frame_b"] is not None and s["views"][0]["corners_b"] is not None and s["rig"]
        else:
            assert not events["stereo_rig"]["calibrated"] and events["stereo_rig"]["reason"] == "missing"
            assert s["views"][0]["frame_b"] is None and not s["rig"] and not rec["method"].endswith("_stereo_rig")
    print("OK")
