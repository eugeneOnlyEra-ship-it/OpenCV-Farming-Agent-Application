"""The two-camera rig on each camera agent's head, and its calibration with OpenCV 5's
calibrateMultiviewExtended.

Each camera agent's head carries TWO cameras rigidly mounted side by side (A = left, B = right),
BASELINE_M apart and looking the same way. Four cameras in total: two per camera agent (front and back).
They fit inside the existing 0.06 m wide head, so the robot's collision volumes are unchanged.

calibrateMultiviewExtended needs >1 camera that see the same pattern at the same moment with a fixed
relative pose -- exactly this rig looking at a chessboard. It returns each camera's intrinsics and the
rig's extrinsics (B relative to A). pod_markers.py then uses the calibrated intrinsics and extrinsics
instead of hard-coded values.

This module has no PyBullet dependency (the PyBullet side is rig_calibration.py), so the calibration maths
can be tested without a simulator.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

BASELINE_M = 0.04                       # lens-to-lens distance; two lenses fit inside the 0.06 m wide camera head
BOARD_COLS, BOARD_ROWS = 9, 6           # inner corners (a 10 x 7 square board; 9 odd / 6 even = unambiguous)
SQUARE_M = 0.03
BOARD_MARGIN_SQUARES = 1                # white quiet zone around the squares
BOARD_THICKNESS_M = 0.004
BOARD_CELL_RAISE_M = 0.001
CALIBRATION_FILE = os.path.join("logs", "stereo_rig_calibration.json")


def rig_cameras(eye, target, baseline=BASELINE_M, up=(0.0, 0.0, 1.0)):
    """Eyes/targets of the two lenses for a head at `eye` aimed at `target`. Both lenses share the same
    viewing direction (a rigid rig), A sits half a baseline to the left of the head centre, B to the right.
    Returns ((eye_a, target_a), (eye_b, target_b))."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    off = right * (baseline / 2.0)
    return ((eye - off).tolist(), (target - off).tolist()), ((eye + off).tolist(), (target + off).tolist())


def board_object_points():
    """Inner-corner coordinates in the board frame (x right, y down, z into the board), row-major."""
    pts = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    for j in range(BOARD_ROWS):
        for i in range(BOARD_COLS):
            pts[j * BOARD_COLS + i] = ((i - (BOARD_COLS - 1) / 2) * SQUARE_M,
                                       (j - (BOARD_ROWS - 1) / 2) * SQUARE_M, 0.0)
    return pts


def board_shapes():
    """The calibration board as boxes, in the board frame (visible side = -z): a white plate and the black
    squares. Returns (plate, squares); each box is (centre_xyz, half_extents_xyz). Shared by the PyBullet
    builder and the software-render test so both use identical geometry."""
    ncol, nrow = BOARD_COLS + 1, BOARD_ROWS + 1
    pw = (ncol + 2 * BOARD_MARGIN_SQUARES) * SQUARE_M
    ph = (nrow + 2 * BOARD_MARGIN_SQUARES) * SQUARE_M
    plate = ((0.0, 0.0, -BOARD_THICKNESS_M / 2), (pw / 2, ph / 2, BOARD_THICKNESS_M / 2))
    squares = []
    x0, y0 = -ncol * SQUARE_M / 2, -nrow * SQUARE_M / 2
    for j in range(nrow):
        for i in range(ncol):
            if (i + j) % 2 == 0:
                squares.append(((x0 + (i + 0.5) * SQUARE_M, y0 + (j + 0.5) * SQUARE_M,
                                 -BOARD_THICKNESS_M - BOARD_CELL_RAISE_M / 2),
                                (SQUARE_M / 2, SQUARE_M / 2, BOARD_CELL_RAISE_M / 2)))
    return plate, squares


def detect_board(gray):
    """Inner chessboard corners as (N, 2) float32, or None."""
    ok, corners = cv2.findChessboardCornersSB(
        gray, (BOARD_COLS, BOARD_ROWS), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return corners.reshape(-1, 2).astype(np.float32) if ok else None


def detect_pair(frame_a, frame_b):
    """Chessboard corners in both cameras' frames (RGBA or gray) taken at the same instant, or None if either
    camera cannot see the whole board or the corner numbering disagrees between them."""
    def gray(f):
        return f if f.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(f[:, :, :3]), cv2.COLOR_RGB2GRAY)
    ca, cb = detect_board(gray(frame_a)), detect_board(gray(frame_b))
    if ca is None or cb is None or np.linalg.norm(ca[0] - cb[0]) > 60:
        return None
    return ca, cb


def solve_rig(pairs, image_size):
    """cv2.calibrateMultiviewExtended on detected chessboard pairs [(corners_a, corners_b), ...]. Each pair is one
    simultaneous view of the board by both cameras. Returns the calibration dict."""
    n = len(pairs)
    if n < 6:
        raise RuntimeError(f"only {n} frames had the board visible to both cameras (need >= 6)")
    obj = [board_object_points() for _ in pairs]
    pts_a = [p[0] for p in pairs]
    pts_b = [p[1] for p in pairs]
    mask = np.ones((2, n), np.uint8)
    models = np.full((2, 1), cv2.CALIB_MODEL_PINHOLE, np.uint8)
    # the simulator renders an ideal pinhole, so fit focal length + principal point only
    flags = np.full((2, 1), cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
                    np.int32)
    out = cv2.calibrateMultiviewExtended(obj, [pts_a, pts_b], [tuple(image_size)] * 2, mask, models,
                                         None, None, None, None, None, None, None, None, flags)
    rms, Ks, Ds, Rs, Ts = out[:5]
    R_ba, _ = cv2.Rodrigues(np.asarray(Rs[1], dtype=float).reshape(3, 1))   # camera-A coords -> camera-B coords
    T_ba = np.asarray(Ts[1], dtype=float).reshape(3)
    return {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "frames_used": n,
        "rms_px": float(rms),
        "K_a": np.asarray(Ks[0], dtype=float).reshape(3, 3).tolist(),
        "K_b": np.asarray(Ks[1], dtype=float).reshape(3, 3).tolist(),
        "dist_a": np.asarray(Ds[0], dtype=float).reshape(-1).tolist(),
        "dist_b": np.asarray(Ds[1], dtype=float).reshape(-1).tolist(),
        "R_ba": R_ba.tolist(),
        "T_ba": T_ba.tolist(),
        "baseline_m": float(np.linalg.norm(T_ba)),
        "designed_baseline_m": BASELINE_M,
    }


def calibrate_rig(frames_a, frames_b, image_size):
    """Calibrate the two-camera rig from simultaneous frames of the board (lists, frame i of A and of B taken at
    the same instant)."""
    pairs = [p for p in (detect_pair(fa, fb) for fa, fb in zip(frames_a, frames_b)) if p is not None]
    return solve_rig(pairs, image_size)


def save_rig(rig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rig, fh, indent=2)


def load_rig(path):
    with open(path) as fh:
        return json.load(fh)


def calibration_path(project_root):
    return os.path.join(project_root, CALIBRATION_FILE)


def load_saved(project_root, image_size):
    """The saved calibration the robot uses, WITHOUT ever running anything: (rig, reason). reason is "ok",
    "missing" or "stale" (made for another baseline / image size). The calibration itself is made from the
    application's Calibration tab (or `python -m pybullet_sim.rig_calibration`), never inside the robot thread."""
    path = calibration_path(project_root)
    if not os.path.exists(path):
        return None, "missing"
    try:
        rig = load_rig(path)
    except Exception:                                       # noqa: BLE001
        return None, "unreadable"
    if (rig.get("image_size") == [int(image_size[0]), int(image_size[1])]
            and abs(rig.get("designed_baseline_m", -1) - BASELINE_M) < 1e-9):
        return rig, "ok"
    return None, "stale"
