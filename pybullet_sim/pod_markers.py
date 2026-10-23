"""ArUco fiducial markers for the pods, and the vision-based pod localization that reads them.

Why this exists
---------------
multiview_localization.estimate_pod_pose() picks the pod out of the image with PyBullet's own
segmentation mask -- i.e. the simulator tells the "vision" system which pixels are the pod. That
validates the geometry but is not perception. Here each pod carries a small ArUco marker on its
aisle-facing wall instead. OpenCV's ArucoDetector finds it in the RGB image (which also gives the
pod's IDENTITY from the marker id), and solvePnP turns the four corners into a 6-DoF pose. The only
things used from the simulator are the camera's own pose (the head position, as on a real robot
with encoders) and the camera intrinsics.

The marker is built from real geometry (a white plate + black cell boxes), not a texture, so its
orientation in the render is exact and cannot be mirrored by PyBullet's UV mapping.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

DICTIONARY = cv2.aruco.DICT_4X4_100          # 100 ids >= the 72 pods
MARKER_CELLS = 6                             # 4x4 payload + 1-cell black border on each side
MARKER_SIDE_M = 0.066                        # black marker edge (cells are 11 mm)
PLATE_SIDE_M = 0.088                         # white quiet-zone plate (fits the 0.09 m trough wall)
MARKER_X_OFFSET_M = 0.0                      # centred on the wall: the three camera views (x = -0.11, 0,
                                             #   +0.11) then see it from -22, 0, +22 degrees, and the
                                             #   "serviced" tag (x = -0.19) stays clear of it
TILT_DEG = 40.0                              # the tag leans back like a small sign so its face points up
                                             #   toward the camera head (which looks down at ~45 degrees);
                                             #   a flat-on-the-wall tag is foreshortened to ~70 % at 320x240
TOP_EDGE_ABOVE_CENTER_M = 0.040              # the tag's top edge touches the wall this far above the wall centre
PLATE_THICKNESS_M = 0.002
CELL_RAISE_M = 0.0008                        # black cells stand this proud of the plate

# Marker frame (same convention used for solvePnP): origin = marker centre, x = image-right,
# y = image-down, z = into the surface.  Corners in the order the detector returns them.
_h = MARKER_SIDE_M / 2.0
MARKER_OBJECT_POINTS = np.array([[-_h, -_h, 0], [_h, -_h, 0], [_h, _h, 0], [-_h, _h, 0]], np.float32)


def aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(DICTIONARY)


def marker_id_for_pod(pod_index):
    return int(pod_index)


def _aisle_sign(pod):
    """+1 for a front-side pod (aisle toward +y), -1 for a back-side pod."""
    return 1 if pod["side"] == "front" else -1


def marker_rotation_world(pod):
    """Columns are the marker's x (image-right), y (image-down), z (into-surface) axes in world
    coordinates, for a viewer standing in the pod's aisle looking at the tag."""
    s = _aisle_sign(pod)
    t = math.radians(TILT_DEG)
    normal = np.array([0.0, s * math.cos(t), math.sin(t)])   # visible face: out into the aisle, tipped upward
    x_m = np.array([-s, 0.0, 0.0])           # viewer's right: -x from the +y aisle, +x from the -y aisle
    z_m = -normal                            # into the surface
    y_m = np.cross(z_m, x_m)                 # image-down (world -z when the tag is vertical)
    return np.column_stack((x_m, y_m, z_m))


def marker_origin_world(pod, half_y):
    """Body origin of the tag: the back-centre of the plate. Placed so the plate's top edge touches the
    aisle-facing wall and the plate leans out and downward from there."""
    pos = pod["position_m"]
    s = _aisle_sign(pod)
    R = marker_rotation_world(pod)
    top_edge = np.array([pos["x"] + MARKER_X_OFFSET_M,
                         pos["y"] + s * half_y,
                         pos["z"] + TOP_EDGE_ABOVE_CENTER_M])
    return top_edge + R[:, 1] * (PLATE_SIDE_M / 2.0)         # image-down by half the plate


def marker_center_world(pod, half_y):
    """Centre of the marker's visible (black-cell) face, in world coordinates."""
    R = marker_rotation_world(pod)
    return marker_origin_world(pod, half_y) - R[:, 2] * (PLATE_THICKNESS_M + CELL_RAISE_M)


def _matrix_to_quaternion(R):
    """3x3 rotation matrix -> (x, y, z, w) as PyBullet wants it."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return q


def _black_runs(bits):
    """Per row of a 6x6 bit grid (1 = white, 0 = black): list of (row, first_col, last_col) black runs."""
    runs = []
    for r in range(bits.shape[0]):
        c = 0
        while c < bits.shape[1]:
            if bits[r, c] == 0:
                c0 = c
                while c + 1 < bits.shape[1] and bits[r, c + 1] == 0:
                    c += 1
                runs.append((r, c0, c))
            c += 1
    return runs


_BOX_SHAPES = {}          # (client, width_in_cells) -> visual shape id, shared by all markers


def _black_box_shape(p, client, width_cells):
    key = (client, width_cells)
    if key not in _BOX_SHAPES:
        cell = MARKER_SIDE_M / MARKER_CELLS
        _BOX_SHAPES[key] = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[width_cells * cell / 2, cell / 2, CELL_RAISE_M / 2],
            rgbaColor=[0.0, 0.0, 0.0, 1.0], physicsClientId=client)
    return _BOX_SHAPES[key]


def add_pod_marker(p, client, pod, marker_id, half_y):
    """Create the marker body: a white plate (the base) with the marker's black cells as fixed links on
    it, tilted toward the camera head and leaning out from the pod's aisle-facing wall. Visual only (no
    collision), like the rest of the scene. Returns the body id."""
    bits = cv2.aruco.generateImageMarker(aruco_dictionary(), int(marker_id), MARKER_CELLS, borderBits=1)
    bits = (np.asarray(bits) > 127).astype(np.uint8)          # 1 = white cell, 0 = black cell
    cell = MARKER_SIDE_M / MARKER_CELLS
    x0 = y0 = -MARKER_SIDE_M / 2                               # marker-frame top-left of the black marker

    plate = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[PLATE_SIDE_M / 2, PLATE_SIDE_M / 2, PLATE_THICKNESS_M / 2],
        visualFramePosition=[0.0, 0.0, -PLATE_THICKNESS_M / 2],     # outward (toward -z of the marker frame)
        rgbaColor=[1.0, 1.0, 1.0, 1.0], physicsClientId=client)
    link_shapes, link_positions = [], []
    for r, c0, c1 in _black_runs(bits):
        w = c1 - c0 + 1
        link_shapes.append(_black_box_shape(p, client, w))
        link_positions.append([x0 + c0 * cell + w * cell / 2, y0 + r * cell + cell / 2,
                               -PLATE_THICKNESS_M - CELL_RAISE_M / 2])
    n = len(link_shapes)
    return p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=plate,
        basePosition=[float(v) for v in marker_origin_world(pod, half_y)],
        baseOrientation=_matrix_to_quaternion(marker_rotation_world(pod)),
        linkMasses=[0.0] * n, linkCollisionShapeIndices=[-1] * n, linkVisualShapeIndices=link_shapes,
        linkPositions=link_positions, linkOrientations=[[0, 0, 0, 1]] * n,
        linkInertialFramePositions=[[0, 0, 0]] * n, linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=[0] * n, linkJointTypes=[p.JOINT_FIXED] * n, linkJointAxis=[[0, 0, 1]] * n,
        physicsClientId=client)


# ----------------------------------------------------------------------------------------------
# Vision-based localization: detect the marker in each rendered view, solvePnP, fuse the views.
# ----------------------------------------------------------------------------------------------
_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners: much better PnP
        _DETECTOR = cv2.aruco.ArucoDetector(aruco_dictionary(), params)
    return _DETECTOR


def _world_from_cv_camera(view_matrix):
    """PyBullet's view matrix is OpenGL (column-major, camera looks down -z, y up). Return the
    4x4 world-from-camera transform in OpenCV's convention (looks down +z, y down)."""
    V = np.asarray(view_matrix, dtype=np.float64).reshape((4, 4), order="F")
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(flip @ V)


def _solve_marker_pose(corners, K, dist, cam_pos_world, world_T_cam):
    """Best marker-from-camera pose for one detection. IPPE returns two mirror-ambiguous solutions;
    keep the lowest-reprojection-error one whose visible face actually points at the camera."""
    n, rvecs, tvecs, errs = cv2.solvePnPGeneric(MARKER_OBJECT_POINTS, corners.reshape(4, 2), K, dist,
                                                flags=cv2.SOLVEPNP_IPPE)
    best = None
    for rvec, tvec, err in zip(rvecs, tvecs, np.asarray(errs).ravel()):
        R, _ = cv2.Rodrigues(rvec)
        cam_T_marker = np.eye(4)
        cam_T_marker[:3, :3], cam_T_marker[:3, 3] = R, tvec.ravel()
        world_T_marker = world_T_cam @ cam_T_marker
        z_into = world_T_marker[:3, 2]                          # marker z points INTO the wall
        to_cam = np.asarray(cam_pos_world) - world_T_marker[:3, 3]
        if np.dot(-z_into, to_cam) <= 0:                         # face pointing away from the camera
            continue
        if best is None or err < best[0]:
            best = (float(err), world_T_marker)
    return best


def _rig_matrices(rig):
    """Calibrated intrinsics and the pose of camera B in camera A's frame, from the rig calibration."""
    K_a, K_b = np.asarray(rig["K_a"], float), np.asarray(rig["K_b"], float)
    camB_T_camA = np.eye(4)
    camB_T_camA[:3, :3], camB_T_camA[:3, 3] = np.asarray(rig["R_ba"], float), np.asarray(rig["T_ba"], float)
    return K_a, K_b, camB_T_camA


def _stereo_pod_centre(corners_a, corners_b, K_a, K_b, camB_T_camA, world_T_camA, d_marker):
    """Triangulate the marker's four corners from the two cameras and turn them into a pod-centre
    estimate: an independent cross-check on the per-camera PnP result (it uses no marker size)."""
    P_a = K_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_b = K_b @ camB_T_camA[:3, :]
    X = cv2.triangulatePoints(P_a, P_b, corners_a.reshape(4, 2).T.astype(np.float64),
                              corners_b.reshape(4, 2).T.astype(np.float64))
    X_cam = (X[:3] / X[3]).T                                    # 4 x 3, in camera A's frame
    pts = (world_T_camA @ np.vstack([X_cam.T, np.ones(4)]))[:3].T
    x_axis = pts[1] - pts[0]
    y_axis = pts[3] - pts[0]
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    R = np.column_stack((x_axis, y_axis, z_axis))
    return pts.mean(axis=0) + R @ d_marker


def estimate_pod_pose_from_markers(views, pod, marker_id, half_extents, rig=None, K=None, dist=None):
    """Pod pose from ArUco detections in the rendered views. Uses ONLY the RGB images, the camera
    intrinsics and the camera's own pose -- not the simulator's segmentation or depth. Returns a dict with
    the same keys estimate_pod_pose() returns, so the service agent can use either.

    With `rig` (the calibrated two-camera head: intrinsics of A and B and B's pose relative to A) each view
    carries a second image `rgba_b`; both cameras then give a PnP pose, camera B is placed in the world from
    the CALIBRATED extrinsics, and the marker's corners are also triangulated across the pair as a
    cross-check. Without a rig it is the single-camera version."""
    from datetime import datetime, timezone
    from .multiview_localization import camera_intrinsics

    if rig is not None:
        K_a, K_b, camB_T_camA = _rig_matrices(rig)
        dist_a, dist_b = np.asarray(rig["dist_a"], float), np.asarray(rig["dist_b"], float)
        camA_T_camB = np.linalg.inv(camB_T_camA)
    else:
        K_a = np.asarray(camera_intrinsics() if K is None else K, dtype=np.float64)
        K_b = dist_b = camB_T_camA = camA_T_camB = None
        dist_a = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
    half_y = float(half_extents[1])
    R_nom = marker_rotation_world(pod)
    nominal_center = np.array([pod["position_m"]["x"], pod["position_m"]["y"], pod["position_m"]["z"]])
    d_marker = R_nom.T @ (nominal_center - marker_center_world(pod, half_y))   # pod centre in marker frame

    detector = _detector()
    per_view, centres, yaws, stereo_centres, stereo_gaps = [], [], [], [], []
    views_with_a_good_camera = 0
    for view in views:
        entry = {"view_index": int(view["view_index"]),
                 "camera_position_world": [float(v) for v in view["camera_position_world"]],
                 "target_pixels": 0, "image_bbox_xywh": None, "reconstructed_points": 0,
                 "marker_detected": False, "reprojection_px": None}
        per_view.append(entry)
        world_T_camA = _world_from_cv_camera(view["view_matrix"])
        cams = [("A", view.get("rgba"), K_a, dist_a, world_T_camA)]
        if rig is not None and view.get("rgba_b") is not None:
            cams.append(("B", view["rgba_b"], K_b, dist_b, world_T_camA @ camA_T_camB))
        found = {}
        good_here = 0
        for name, rgba, Kc, dc, world_T_cam in cams:
            if rgba is None:
                continue
            gray = cv2.cvtColor(np.ascontiguousarray(rgba[:, :, :3]), cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or int(marker_id) not in [int(i) for i in ids.ravel()]:
                continue
            c = corners[[int(i) for i in ids.ravel()].index(int(marker_id))]
            found[name] = c
            entry["corners" + ("" if name == "A" else "_b")] = [[float(x), float(y)] for x, y in c.reshape(4, 2)]
            sol = _solve_marker_pose(c, Kc, dc, world_T_cam[:3, 3], world_T_cam)
            if sol is None:
                continue
            err_px, world_T_marker = sol
            tag = "" if name == "A" else "_b"
            entry["marker_detected" + tag] = True
            entry["reprojection_px" + tag] = round(err_px, 3)
            if name == "A":
                x, y, w, h = cv2.boundingRect(c.reshape(4, 2).astype(np.float32))
                entry.update(image_bbox_xywh=[int(x), int(y), int(w), int(h)],
                             target_pixels=int(cv2.contourArea(c.reshape(4, 2).astype(np.float32))),
                             reconstructed_points=4)
            if err_px > 3.0:                                     # a bad fit: don't let it vote
                entry["rejected" + tag] = "reprojection error"
                continue
            R_est = world_T_marker[:3, :3]
            centres.append(world_T_marker[:3, 3] + R_est @ d_marker)
            entry["center" + tag] = [float(v) for v in centres[-1]]      # this camera's own pod-centre estimate
            R_pod = R_est @ R_nom.T                              # pod rotation relative to nominal
            yaws.append(math.atan2(R_pod[1, 0], R_pod[0, 0]))
            good_here += 1
        views_with_a_good_camera += 1 if good_here else 0
        if rig is not None and "A" in found and "B" in found:
            st = _stereo_pod_centre(found["A"], found["B"], K_a, K_b, camB_T_camA, world_T_camA, d_marker)
            stereo_centres.append(st)
            entry["stereo_center"] = [float(v) for v in st]
            if centres:
                entry["stereo_vs_pnp_mm"] = round(float(np.linalg.norm(st - centres[-1]) * 1000), 2)
                stereo_gaps.append(entry["stereo_vs_pnp_mm"])

    good = len(centres)                                          # camera measurements that passed the gates
    valid = good >= 1
    center = yaw = quaternion = spread = None
    if good:
        arr = np.vstack(centres)
        center_arr = np.median(arr, axis=0)
        spread = float(np.max(np.linalg.norm(arr - center_arr, axis=1)))
        yaw = float(np.median(yaws))
        center = [float(v) for v in center_arr]
        quaternion = [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]
        # measurements that disagree by more than 3 cm mean at least one PnP solution flipped: don't trust it
        if good >= 2 and spread > 0.03:
            valid = False
        if abs(yaw) > math.radians(8):
            valid = False
    view_coverage = views_with_a_good_camera / max(1, len(views))
    pose = {
        "pod_id": pod["pod_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_frame": "pybullet_world",
        "method": "multi_view_aruco_pnp" + ("_stereo_rig" if rig is not None else ""),
        "valid": bool(valid),
        "confidence_score": float(round(view_coverage, 3)),
        "confidence_score_type": "heuristic_not_probability",
        "center_world": center,
        "orientation_quaternion_xyzw": quaternion,
        "orientation_yaw_rad": yaw,
        "orientation_yaw_deg": None if yaw is None else float(math.degrees(yaw)),
        "estimated_visible_dimensions_m": None,
        "geometry_residual_m": spread,
        "point_count": 4 * good,
        "view_count": len(views),
        "contributing_view_count": views_with_a_good_camera,
        "camera_measurement_count": good,
        "camera_matrix": K_a.tolist(),
        "distortion_coefficients": [float(v) for v in dist_a],
        "marker_id": int(marker_id),
        "views": per_view,
    }
    if rig is not None:
        pose["rig"] = {"calibrated": True, "baseline_m": rig["baseline_m"], "rms_px": rig["rms_px"],
                       "frames_used": rig["frames_used"]}
        if stereo_centres:
            pose["stereo_center_world"] = [float(v) for v in np.median(np.vstack(stereo_centres), axis=0)]
            pose["stereo_vs_pnp_mm_mean"] = float(round(np.mean(stereo_gaps), 2)) if stereo_gaps else None
    return pose
