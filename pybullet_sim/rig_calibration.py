"""Calibrate the two-camera head rig in a scratch PyBullet world, streaming its progress.

    python -m pybullet_sim.rig_calibration            # from the project root; prints a summary
    python -m pybullet_sim.rig_calibration --stream   # the same, but one JSON event per line (what the Calibration tab reads)

It opens its own headless PyBullet connection, holds a chessboard in front of the rig at many poses, renders both
cameras at the same instant each time, and runs cv2.calibrateMultiviewExtended on the pairs -- re-solving after every
new pair so you can watch the estimate converge. The result goes to logs/stereo_rig_calibration.json together with
the error against the rig's TRUE geometry (known here because it is a simulation); the frames go to
logs/calibration_frames/. It runs as a separate process, so it can never stall the application window or the robot.
The robot only ever LOADS the saved result.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from . import stereo_rig as sr
from .multiview_localization import IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_FOV_DEG, CAMERA_NEAR, CAMERA_FAR
from .pod_markers import _matrix_to_quaternion

RIG_EYE = (0.0, 0.0, 0.5)
RIG_TARGET = (0.0, 1.0, 0.5)             # the rig looks along +y
N_POSES = 20
FRAMES_DIR = os.path.join("logs", "calibration_frames")


def build_board(p, client):
    """The calibration board as one PyBullet body (white plate + black squares as fixed links)."""
    plate, squares = sr.board_shapes()
    plate_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=list(plate[1]), visualFramePosition=list(plate[0]),
                                    rgbaColor=[1, 1, 1, 1], physicsClientId=client)
    black = p.createVisualShape(p.GEOM_BOX, halfExtents=list(squares[0][1]), rgbaColor=[0, 0, 0, 1],
                                physicsClientId=client)
    n = len(squares)
    return p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=plate_vis,
        basePosition=[0, 1, 0.5], baseOrientation=[0, 0, 0, 1],
        linkMasses=[0.0] * n, linkCollisionShapeIndices=[-1] * n, linkVisualShapeIndices=[black] * n,
        linkPositions=[list(s[0]) for s in squares], linkOrientations=[[0, 0, 0, 1]] * n,
        linkInertialFramePositions=[[0, 0, 0]] * n, linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=[0] * n, linkJointTypes=[p.JOINT_FIXED] * n, linkJointAxis=[[0, 0, 1]] * n,
        physicsClientId=client)


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def board_poses(n, seed=7):
    """n board poses (position, rotation matrix) in front of the rig, tilted so both cameras see it."""
    rng = np.random.default_rng(seed)
    facing = np.column_stack(([1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]))   # x right, y down, z away from the rig
    poses = []
    for _ in range(n):
        d = rng.uniform(0.32, 0.50)
        pos = np.array([rng.uniform(-0.05, 0.05), RIG_EYE[1] + d, RIG_EYE[2] + rng.uniform(-0.04, 0.04)])
        R = _rot("z", rng.uniform(-28, 28)) @ _rot("x", rng.uniform(-24, 24)) @ _rot("y", rng.uniform(-14, 14)) @ facing
        poses.append((pos, R))
    return poses


def iter_frames(n_poses=N_POSES):
    """Generator of (frame_a, frame_b) RGBA pairs, one per board pose, rendered in a scratch PyBullet world."""
    import pybullet as p
    client = p.connect(p.DIRECT)
    try:
        board = build_board(p, client)
        (eye_a, tgt_a), (eye_b, tgt_b) = sr.rig_cameras(RIG_EYE, RIG_TARGET)
        proj = p.computeProjectionMatrixFOV(fov=CAMERA_FOV_DEG, aspect=IMAGE_WIDTH / IMAGE_HEIGHT,
                                            nearVal=CAMERA_NEAR, farVal=CAMERA_FAR)
        view_a = p.computeViewMatrix(eye_a, tgt_a, [0, 0, 1])
        view_b = p.computeViewMatrix(eye_b, tgt_b, [0, 0, 1])
        for pos, R in board_poses(n_poses):
            p.resetBasePositionAndOrientation(board, [float(v) for v in pos], _matrix_to_quaternion(R),
                                              physicsClientId=client)
            imgs = []
            for vm in (view_a, view_b):                        # both cameras, the same instant
                img = p.getCameraImage(IMAGE_WIDTH, IMAGE_HEIGHT, viewMatrix=vm, projectionMatrix=proj,
                                       renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK,
                                       physicsClientId=client)
                imgs.append(np.asarray(img[2], dtype=np.uint8).reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 4))
            yield imgs[0], imgs[1]
    finally:
        p.disconnect(client)


def truth_errors(rig):
    """Compare the calibration with the rig's true geometry (parallel cameras, B one baseline to A's right)."""
    R_true, T_true = np.eye(3), np.array([-sr.BASELINE_M, 0.0, 0.0])
    R = np.array(rig["R_ba"])
    angle = math.degrees(np.linalg.norm(cv2.Rodrigues(R @ R_true.T)[0]))
    fy = (IMAGE_HEIGHT / 2.0) / math.tan(math.radians(CAMERA_FOV_DEG) / 2.0)
    return {"rotation_error_deg": float(angle),
            "translation_error_mm": float(np.linalg.norm(np.array(rig["T_ba"]) - T_true) * 1000),
            "baseline_error_mm": float((rig["baseline_m"] - sr.BASELINE_M) * 1000),
            "true_fy_px": float(fy), "fy_a_px": float(rig["K_a"][1][1]), "fy_b_px": float(rig["K_b"][1][1])}


def summarize(rig):
    """The numbers worth showing while the estimate converges."""
    Ka, Kb = rig["K_a"], rig["K_b"]
    return {"pairs": rig["frames_used"], "rms_px": rig["rms_px"], "baseline_mm": rig["baseline_m"] * 1000,
            "fx_a": Ka[0][0], "fy_a": Ka[1][1], "cx_a": Ka[0][2], "cy_a": Ka[1][2],
            "fx_b": Kb[0][0], "fy_b": Kb[1][1], "cx_b": Kb[0][2], "cy_b": Kb[1][2],
            "rot_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(np.array(rig["R_ba"]))[0]))),
            "T_mm": [float(v) * 1000 for v in rig["T_ba"]]}


def run(frames, n_poses, emit, out_dir=FRAMES_DIR, save_path=sr.CALIBRATION_FILE):
    """Drive the whole calibration from an iterator of (frame_a, frame_b) pairs, reporting through emit(dict).
    Separate from the PyBullet rendering so it can be tested without PyBullet."""
    size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "cal_*.png")):
        os.remove(old)
    emit({"event": "start", "poses": n_poses, "image_size": list(size),
          "designed_baseline_mm": sr.BASELINE_M * 1000,
          "board": {"cols": sr.BOARD_COLS, "rows": sr.BOARD_ROWS, "square_mm": sr.SQUARE_M * 1000}})
    pairs = []
    for i, (fa, fb) in enumerate(frames):
        path_a = os.path.join(out_dir, f"cal_{i + 1:02d}_A.png")
        path_b = os.path.join(out_dir, f"cal_{i + 1:02d}_B.png")
        cv2.imwrite(path_a, cv2.cvtColor(np.ascontiguousarray(fa[:, :, :3]), cv2.COLOR_RGB2BGR))
        cv2.imwrite(path_b, cv2.cvtColor(np.ascontiguousarray(fb[:, :, :3]), cv2.COLOR_RGB2BGR))
        det = sr.detect_pair(fa, fb)
        if det is not None:
            pairs.append(det)
        estimate = None
        if len(pairs) >= 6:                                    # re-solve with everything seen so far
            try:
                estimate = summarize(sr.solve_rig(pairs, size))
            except Exception:                                  # noqa: BLE001 - keep streaming
                estimate = None
        emit({"event": "frame", "i": i + 1, "n": n_poses, "path_a": path_a, "path_b": path_b,
              "pair_ok": det is not None, "pairs": len(pairs),
              "corners_a": None if det is None else det[0].round(2).tolist(),
              "corners_b": None if det is None else det[1].round(2).tolist(),
              "estimate": estimate})
    rig = sr.solve_rig(pairs, size)
    rig["frames_total"] = n_poses
    rig["truth_comparison"] = truth_errors(rig)
    rig["calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sr.save_rig(rig, save_path)
    emit({"event": "done", "rig": rig})
    return rig


def main():
    stream = "--stream" in sys.argv

    def emit(event):
        if stream:
            print(json.dumps(event), flush=True)
    try:
        rig = run(iter_frames(N_POSES), N_POSES, emit)
    except Exception as exc:                                    # noqa: BLE001
        if stream:
            print(json.dumps({"event": "error", "message": f"{type(exc).__name__}: {exc}"}), flush=True)
        else:
            raise
        return 1
    if not stream:
        t = rig["truth_comparison"]
        print(f"stereo rig calibrated from {rig['frames_used']} frames: reprojection {rig['rms_px']:.3f} px, "
              f"baseline {rig['baseline_m'] * 1000:.1f} mm (designed {sr.BASELINE_M * 1000:.0f}), "
              f"rotation error {t['rotation_error_deg']:.2f} deg, translation error {t['translation_error_mm']:.1f} mm")
        print("saved", os.path.abspath(sr.CALIBRATION_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
