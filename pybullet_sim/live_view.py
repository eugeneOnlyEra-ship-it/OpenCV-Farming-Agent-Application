"""Live view of the head's two cameras, inside the PyBullet window.

PyBullet's GUI has no API for drawing image panels on screen (its RGB / depth / segmentation previews are
built in and fixed). The closest thing available is a textured quad IN the 3D scene. So this keeps two small
"screens" -- camera A on top, camera B underneath, a vertical stack on the right edge of the window -- and
re-pins them in front of the debug camera every few steps, so they stay put on screen however you rotate or
zoom the view. Their textures are replaced with changeTexture each time a head captures a view.

Safeguards:
  * the screens are moved far away while a robot camera renders (otherwise a robot camera could photograph
    them if the debug camera happens to be near it);
  * if anything in the PyBullet route fails, it falls back to a plain OpenCV window (A over B), and if that
    fails too the live view switches itself off. The simulation never depends on it.

Modes (gantry_robot.LIVE_CAMERA_VIEW):  "pybullet" | "window" | "off"
"""
from __future__ import annotations

import contextlib
import os
import tempfile

import cv2
import numpy as np

PANE_PX = (160, 120)             # texture size (w, h). Small on purpose: PyBullet streams texture data through a
                                 #   size-limited buffer, and 160 x 120 x 3 = 57 KB is well inside it
PANE_SCREEN_FRACTION = 0.20      # each screen's width as a fraction of the window width
SCREEN_MARGIN = 0.02             # gap to the window edge / between the two screens (fraction of window)
HUD_SIDE = "right"               # "right" | "left"  (the built-in previews live on the left)
HUD_DISTANCE_M = 0.30            # how far in front of the debug camera the screens float
UPDATE_EVERY_N_STEPS = 4         # how often the screens are re-pinned to the debug camera
# PyBullet's box texture mapping can come out mirrored or upside down depending on the build; if the caption
# text on the screens reads wrong, flip here.
FLIP_VERTICAL = False
FLIP_HORIZONTAL = False
_FAR_AWAY = (0.0, 0.0, -1000.0)


def _matrix_to_quaternion(R):
    from .pod_markers import _matrix_to_quaternion as f
    return f(R)


def hud_pose(camera_info, index, side=HUD_SIDE, distance=HUD_DISTANCE_M):
    """World position + rotation matrix for screen `index` (0 = top, 1 = below it) from
    p.getDebugVisualizerCamera()'s result. Also returns the screen's size in metres at that distance."""
    V = np.asarray(camera_info[2], dtype=float).reshape((4, 4), order="F")
    P = np.asarray(camera_info[3], dtype=float).reshape((4, 4), order="F")
    cam_to_world = np.linalg.inv(V)
    right, up, back = cam_to_world[:3, 0], cam_to_world[:3, 1], cam_to_world[:3, 2]
    forward = -back
    half_w, half_h = distance / P[0, 0], distance / P[1, 1]          # half extent of the view at that distance
    pane_w = PANE_SCREEN_FRACTION * 2 * half_w
    pane_h = pane_w * PANE_PX[1] / PANE_PX[0]
    margin_x, margin_y = SCREEN_MARGIN * 2 * half_w, SCREEN_MARGIN * 2 * half_h
    x = (half_w - margin_x - pane_w / 2) * (1 if side == "right" else -1)
    y = half_h - 2 * margin_y - pane_h / 2 - index * (pane_h + margin_y)
    position = cam_to_world[:3, 3] + forward * distance + right * x + up * y
    R = np.column_stack((right, up, back))                           # +z of the screen faces the viewer
    return position, R, (pane_w, pane_h)


class LiveCameraView:
    def __init__(self, p, client, gui, mode="pybullet"):
        self.p, self.client = p, client
        self.mode = mode if mode in ("pybullet", "window", "off") else "off"
        if self.mode == "pybullet" and not gui:
            self.mode = "off"                       # no window to draw into
        self.bodies, self.textures = [], []
        self._steps = 0
        self._hidden = False
        self._window_ok = True
        if self.mode == "pybullet":
            try:
                self._build_screens()
            except Exception as exc:                # noqa: BLE001
                print(f"[live_view] in-GUI screens unavailable ({type(exc).__name__}: {exc}); using an OpenCV window")
                self._drop_screens()
                self.mode = "window"

    # ---- PyBullet screens ------------------------------------------------------------------------------
    def _build_screens(self):
        p, c = self.p, self.client
        info = p.getDebugVisualizerCamera(physicsClientId=c)
        for index, label in enumerate(("A", "B")):
            path = os.path.join(tempfile.gettempdir(), f"pod_live_view_{label}.png")
            cv2.imwrite(path, self._test_pane(label))
            texture = p.loadTexture(path, physicsClientId=c)
            _, _, (w, h) = hud_pose(info, index)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[w / 2, h / 2, 0.0005], rgbaColor=[1, 1, 1, 1],
                                         specularColor=[0, 0, 0], physicsClientId=c)
            body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=visual,
                                     basePosition=list(_FAR_AWAY), physicsClientId=c)
            p.changeVisualShape(body, -1, textureUniqueId=texture, physicsClientId=c)
            self.bodies.append(body)
            self.textures.append(texture)
        self.tick(force=True)

    def _drop_screens(self):
        for body in self.bodies:
            with contextlib.suppress(Exception):
                self.p.removeBody(body, physicsClientId=self.client)
        self.bodies, self.textures = [], []

    @staticmethod
    def _test_pane(label):
        img = np.full((PANE_PX[1], PANE_PX[0], 3), 40, np.uint8)
        cv2.putText(img, f"camera {label}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "waiting...", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        return img

    def tick(self, force=False):
        """Called every simulation step; every few steps re-pin the screens to the debug camera."""
        if self.mode != "pybullet" or self._hidden:
            return
        self._steps += 1
        if not force and self._steps % UPDATE_EVERY_N_STEPS:
            return
        try:
            info = self.p.getDebugVisualizerCamera(physicsClientId=self.client)
            for index, body in enumerate(self.bodies):
                position, R, _ = hud_pose(info, index)
                self.p.resetBasePositionAndOrientation(body, [float(v) for v in position],
                                                       _matrix_to_quaternion(R), physicsClientId=self.client)
        except Exception as exc:                    # noqa: BLE001
            print(f"[live_view] screens stopped ({type(exc).__name__}: {exc}); using an OpenCV window")
            self._drop_screens()
            self.mode = "window"

    @contextlib.contextmanager
    def hidden(self):
        """Wrap a robot camera's render: the screens are parked far away so they cannot appear in the frame."""
        if self.mode != "pybullet" or not self.bodies:
            yield
            return
        self._hidden = True
        try:
            for body in self.bodies:
                self.p.resetBasePositionAndOrientation(body, list(_FAR_AWAY), [0, 0, 0, 1],
                                                       physicsClientId=self.client)
            yield
        finally:
            self._hidden = False
            self.tick(force=True)

    # ---- new frames ---------------------------------------------------------------------------------------
    @staticmethod
    def _pane(rgba, caption):
        bgr = cv2.cvtColor(np.ascontiguousarray(rgba[:, :, :3]), cv2.COLOR_RGB2BGR)
        pane = cv2.resize(bgr, PANE_PX, interpolation=cv2.INTER_AREA)
        cv2.putText(pane, caption, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pane, caption, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return pane

    def update(self, rgba_a, rgba_b, caption=""):
        """Show the newest pair of frames (A on top, B below). rgba_*: (H, W, 4) uint8 from getCameraImage."""
        if self.mode == "off" or rgba_a is None:
            return
        panes = [self._pane(rgba_a, f"A  {caption}")]
        if rgba_b is not None:
            panes.append(self._pane(rgba_b, f"B  {caption}"))
        if self.mode == "pybullet":
            try:
                for pane, texture in zip(panes, self.textures):
                    rgb = cv2.cvtColor(pane, cv2.COLOR_BGR2RGB)
                    if FLIP_VERTICAL:
                        rgb = rgb[::-1]
                    if FLIP_HORIZONTAL:
                        rgb = rgb[:, ::-1]
                    self.p.changeTexture(texture, np.ascontiguousarray(rgb).flatten().tolist(),
                                         PANE_PX[0], PANE_PX[1], physicsClientId=self.client)
                return
            except Exception as exc:                # noqa: BLE001
                print(f"[live_view] texture update failed ({type(exc).__name__}: {exc}); using an OpenCV window")
                self._drop_screens()
                self.mode = "window"
        if self.mode == "window" and self._window_ok:
            try:
                big = [cv2.resize(x, (320, 240), interpolation=cv2.INTER_LINEAR) for x in panes]
                cv2.imshow("head cameras: A (top) / B (bottom)", np.vstack(big))
                cv2.waitKey(1)
            except Exception as exc:                # noqa: BLE001
                print(f"[live_view] OpenCV window unavailable ({type(exc).__name__}); live view off")
                self._window_ok = False
                self.mode = "off"

    def close(self):
        self._drop_screens()
        if self.mode == "window":
            with contextlib.suppress(Exception):
                cv2.destroyWindow("head cameras: A (top) / B (bottom)")
