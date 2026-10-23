"""check_stereo_rig.py -- test the two-camera head rig WITHOUT PyBullet.

1. Calibration: software-renders a chessboard seen simultaneously by both lenses at 20 poses and runs
   stereo_rig.calibrate_rig (cv2.calibrateMultiviewExtended); compares with the rig's true geometry.
2. Localization: renders all 72 pods' ArUco tags through both lenses, places camera B in the world from the
   CALIBRATED extrinsics, and compares the single-camera pose with the stereo-rig pose against the truth.

  python dev/check_stereo_rig.py

Like check_pod_markers.py this checks conventions and maths, not PyBullet's own renderer.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dev import check_pod_markers as cpm
from dev.check_pod_markers import look_at, projection, W, H
from pybullet_sim import pod_markers as pm, stereo_rig as sr, rig_calibration as rc
from pybullet_sim.pod_registry import PODS, TROUGH_HALF_EXTENTS_M, ROW_SPACING_M

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def render_board(pos, R, V, P, SS):
    import cv2
    plate, squares = sr.board_shapes()
    img = np.full((H * SS, W * SS), 90, np.uint8)
    for (c, h), shade in [(plate, 220)] + [(s, 25) for s in squares]:
        zf = c[2] - h[2]
        corners = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], zf] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        uv, w = cpm.project(P, V, pos + corners @ R.T)
        if (w <= 0).any():
            continue
        cv2.fillConvexPoly(img, np.round(uv * SS - 0.5).astype(np.int32), shade, lineType=cv2.LINE_8)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if SS > 1 else img


def calibrate(SS):
    P = projection()
    (ea, ta), (eb, tb) = sr.rig_cameras(rc.RIG_EYE, rc.RIG_TARGET)
    Va, Vb = look_at(ea, ta), look_at(eb, tb)
    fa, fb = [], []
    for pos, R in rc.board_poses(20):
        fa.append(render_board(pos, R, Va, P, SS)); fb.append(render_board(pos, R, Vb, P, SS))
    rig = sr.calibrate_rig(fa, fb, (W, H))
    return rig, rc.truth_errors(rig)


def localize(rig, SS):
    os.environ["SS"] = str(SS)
    P = projection(); half_y = TROUGH_HALF_EXTENTS_M[1]
    out = {"single": [], "rig": []}; valid = {"single": 0, "rig": 0}; gaps = []
    for idx, pod in enumerate(PODS):
        sign = 1 if pod["side"] == "front" else -1
        row_y = pod["row"] * ROW_SPACING_M
        views = []
        for vi, plant in enumerate(pod["plants"]):
            eye = np.array([plant["x"], row_y + sign * 0.505, pod["position_m"]["z"] + TROUGH_HALF_EXTENTS_M[2] + 0.22])
            target = [pod["position_m"]["x"], pod["position_m"]["y"], pod["position_m"]["z"] + TROUGH_HALF_EXTENTS_M[2]]
            (ea, ta), (eb, tb) = sr.rig_cameras(eye, target)
            Va, Vb = look_at(ea, ta), look_at(eb, tb)
            views.append({"view_index": vi, "camera_position_world": ea, "view_matrix": Va.flatten(order="F").tolist(),
                          "rgba": cpm.render(pod, pm.marker_id_for_pod(idx), np.array(ea), Va, P, half_y),
                          "rgba_b": cpm.render(pod, pm.marker_id_for_pod(idx), np.array(eb), Vb, P, half_y)})
        nominal = np.array([pod["position_m"][k] for k in "xyz"])
        for name, r in (("single", None), ("rig", rig)):
            pose = pm.estimate_pod_pose_from_markers(views, pod, pm.marker_id_for_pod(idx), TROUGH_HALF_EXTENTS_M, rig=r)
            if pose["valid"]:
                valid[name] += 1
                out[name].append(np.linalg.norm(np.array(pose["center_world"]) - nominal) * 1000)
            if name == "rig" and pose.get("stereo_vs_pnp_mm_mean") is not None:
                gaps.append(pose["stereo_vs_pnp_mm_mean"])
    return {k: (valid[k], np.array(v)) for k, v in out.items()}, gaps


if __name__ == "__main__":
    for SS, label in ((8, "accurate rendering"), (1, "harsh rendering (aliased)")):
        rig, t = calibrate(SS)
        print(f"[{label}] calibration: {rig['frames_used']} frames, rms {rig['rms_px']:.3f} px, "
              f"baseline {rig['baseline_m']*1000:.2f} mm (true {sr.BASELINE_M*1000:.0f}), "
              f"rotation error {t['rotation_error_deg']:.3f} deg, translation error {t['translation_error_mm']:.2f} mm")
        res, gaps = localize(rig, SS)
        for name in ("single", "rig"):
            n, e = res[name]
            print(f"[{label}] localization {name:6s}: {n}/{len(PODS)} pods, error mm mean {e.mean():.2f} max {e.max():.2f}")
        print(f"[{label}] stereo-triangulation vs PnP disagreement: mean {np.mean(gaps):.2f} mm")
        if SS == 8:
            assert t["rotation_error_deg"] < 0.3 and t["translation_error_mm"] < 2.0, "calibration off"
            assert res["rig"][0] == len(PODS) and res["rig"][1].max() < 5.0, "rig localization off"
        else:
            assert res["rig"][0] >= 0.9 * len(PODS) and res["rig"][1].max() < 30.0
    print("OK")
