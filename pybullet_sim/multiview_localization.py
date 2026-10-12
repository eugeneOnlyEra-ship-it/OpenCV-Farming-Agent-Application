"""Multi-view RGB-D pod localization for the PyBullet camera agent.

The simulated camera intrinsics and per-shot extrinsics are known from the
renderer and camera-agent joint state. The pod segmentation mask is provided by
PyBullet so this module can validate the geometry/localization pipeline before
replacing that simulator-only mask with a real OpenCV detector.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math

import cv2
import numpy as np


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CAMERA_FOV_DEG = 58.0
CAMERA_NEAR = 0.01
CAMERA_FAR = 3.0


def camera_intrinsics(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fov_deg=CAMERA_FOV_DEG):
    """Return the pinhole intrinsic matrix for PyBullet's vertical FOV renderer."""
    fy = (height / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fx = fy  # square pixels
    return [[float(fx), 0.0, width / 2.0],
            [0.0, float(fy), height / 2.0],
            [0.0, 0.0, 1.0]]


def _matrix4(flat):
    # PyBullet exposes OpenGL matrices in column-major order.
    return np.asarray(flat, dtype=np.float64).reshape((4, 4), order="F")


def _unproject(depth_buffer, pixel_mask, view_matrix, projection_matrix):
    """Convert masked OpenGL depth pixels into world-space XYZ points."""
    height, width = depth_buffer.shape
    ys, xs = np.nonzero(pixel_mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)

    depth = depth_buffer[ys, xs].astype(np.float64)
    keep = np.isfinite(depth) & (depth > 0.0) & (depth < 1.0)
    xs, ys, depth = xs[keep], ys[keep], depth[keep]
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float64)

    ndc = np.column_stack((
        (xs.astype(np.float64) + 0.5) * (2.0 / width) - 1.0,
        1.0 - (ys.astype(np.float64) + 0.5) * (2.0 / height),
        depth * 2.0 - 1.0,
        np.ones(len(xs), dtype=np.float64),
    ))
    view = _matrix4(view_matrix)
    projection = _matrix4(projection_matrix)
    inverse = np.linalg.inv(projection @ view)
    world_h = (inverse @ ndc.T).T
    valid_w = np.abs(world_h[:, 3]) > 1e-10
    world_h = world_h[valid_w]
    if len(world_h) == 0:
        return np.empty((0, 3), dtype=np.float64)
    return world_h[:, :3] / world_h[:, 3:4]


def _yaw_quaternion(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def estimate_pod_pose(views, pod_id, body_id, side, half_extents):
    """Estimate a pod pose from multiple segmented depth views.

    Each view contains depth/segmentation arrays and its camera matrices. The
    returned pose is in the PyBullet world frame. A pose is valid only when at
    least two distinct views contribute enough target pixels.
    """
    point_sets = []
    per_view = []
    for view in views:
        seg = np.asarray(view["segmentation"], dtype=np.int64)
        depth = np.asarray(view["depth"], dtype=np.float64)
        # PyBullet encodes body ID in the low 24 bits and link ID above it.
        mask = ((seg >= 0) & ((seg & ((1 << 24) - 1)) == int(body_id))).astype(np.uint8)
        # OpenCV cleanup prevents isolated segmentation pixels from distorting the
        # multi-view cloud bounds. The target mask itself is still simulator-only.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == largest).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox = None
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = [int(x), int(y), int(w), int(h)]
        points = _unproject(depth, mask.astype(bool), view["view_matrix"], view["projection_matrix"])
        if len(points):
            point_sets.append(points)
        per_view.append({
            "view_index": int(view["view_index"]),
            "camera_position_world": [float(v) for v in view["camera_position_world"]],
            "target_pixels": int(mask.sum()),
            "image_bbox_xywh": bbox,
            "reconstructed_points": int(len(points)),
        })

    points = np.vstack(point_sets) if point_sets else np.empty((0, 3))
    # Distinct views must see enough of the pod to produce a useful estimate.
    contributing_views = sum(v["reconstructed_points"] >= 12 for v in per_view)
    hx, hy, hz = [float(v) for v in half_extents]
    valid = len(points) >= 60 and contributing_views >= 2
    center = None
    quaternion = None
    yaw = None
    dimensions = None
    residual_m = None
    if len(points) >= 12:
        # Quantile bounds suppress a few edge pixels and depth-buffer outliers.
        lo = np.quantile(points, 0.02, axis=0)
        hi = np.quantile(points, 0.98, axis=0)
        center_arr = (lo + hi) / 2.0
        # The camera is elevated and views the trough top. The combined point
        # cloud should cover the top footprint, so use the midpoint of its robust
        # XY bounds rather than the old aisle-facing-wall correction.
        center_arr[1] = (lo[1] + hi[1]) / 2.0
        # Most visible trough pixels lie on its top surface; recover the centre Z
        # from that top plane and the known trough half-height.
        top_z = float(np.quantile(points[:, 2], 0.95))
        center_arr[2] = top_z - hz

        # Multi-view point-cloud PCA estimates the long-axis yaw. Pod orientation
        # is symmetric by 180 degrees, so normalize to [-90, +90] degrees.
        xy = points[:, :2] - np.mean(points[:, :2], axis=0)
        covariance = np.cov(xy.T) if len(xy) > 2 else np.eye(2)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        estimated_yaw = math.atan2(float(axis[1]), float(axis[0]))
        while estimated_yaw > math.pi / 2:
            estimated_yaw -= math.pi
        while estimated_yaw < -math.pi / 2:
            estimated_yaw += math.pi
        # This gantry's service stages move along world X and cannot compensate
        # for large yaw. Preserve the measured yaw but flag excessive misalignment.
        yaw = float(estimated_yaw)
        quaternion = _yaw_quaternion(yaw)
        center = [float(v) for v in center_arr]
        dimensions = [float(v) for v in (hi - lo)]
        # Diagnostic residual: point spread relative to the nominal trough box.
        expected = np.asarray([2 * hx, 2 * hy, 2 * hz])
        residual_m = float(np.linalg.norm(np.sort(np.maximum(hi - lo, 0))[::-1] -
                                          np.sort(expected)[::-1]))
        if abs(yaw) > math.radians(8):
            valid = False

    view_coverage = contributing_views / max(1, len(views))
    point_coverage = min(1.0, len(points) / 240.0)
    confidence_score = float(round(point_coverage * view_coverage, 3))
    return {
        "pod_id": pod_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "pybullet_world",
        "method": "multi_view_rgbd_depth_with_simulator_segmentation",
        "valid": bool(valid),
        "confidence_score": confidence_score,
        "confidence_score_type": "heuristic_not_probability",
        "center_world": center,
        "orientation_quaternion_xyzw": quaternion,
        "orientation_yaw_rad": yaw,
        "orientation_yaw_deg": None if yaw is None else float(math.degrees(yaw)),
        "estimated_visible_dimensions_m": dimensions,
        "geometry_residual_m": residual_m,
        "point_count": int(len(points)),
        "view_count": len(views),
        "contributing_view_count": int(contributing_views),
        "camera_matrix": camera_intrinsics(),
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "views": per_view,
    }
