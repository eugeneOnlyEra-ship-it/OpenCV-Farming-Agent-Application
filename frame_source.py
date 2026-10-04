"""
frame_source.py  --  "where does the NEW photo come from after a robot finishes?"

Closing the loop needs a fresh frame after the camera agent has carried out a
RECAPTURE or VERIFY order. On the real robots that frame comes from the
physical camera. This build has no physical camera -- only a dataset of photos --
so SimulatedCamera DERIVES a new frame from the original photo by doing, in
OpenCV, what the order told the camera agent to do:

    RECAPTURE with suspected regions  -> crop + upscale the suspected region   ("zoom in")
    RECAPTURE, nothing readable       -> tighter centre crop + brighten        ("re-frame, more light")
    RECAPTURE, growth stage unclear   -> CLAHE contrast on the lightness       ("better lighting")
    VERIFY (after a treatment)        -> inpaint the regions the worker was told to remove
                                         ("post-treatment frame")

!!  EVERYTHING THIS MODULE PRODUCES IS SIMULATED.  !!
    The downstream analysis (OpenCV DNN + decision + comparison) is real, but the
    frame is synthetic. Never report results from these frames as measured
    accuracy, and say so in the technical report. The VERIFY frame in particular
    assumes the worker removed the targeted tissue; the untargeted disease stays,
    which is why a widespread infection correctly fails verification.

To use real frames, provide any object with the same method:

    capture(order, out_dir) -> {"path": <image file>, "note": <text>, "simulated": False}

and assign it to FarmEngine.frame_source.
"""

import os

import cv2
import numpy as np


def _union_box(targets, w, h, margin=0.35):
    xs, ys, xe, ye = [], [], [], []
    for t in targets:
        x, y, bw, bh = t["box_norm"]
        xs.append(x * w)
        ys.append(y * h)
        xe.append((x + bw) * w)
        ye.append((y + bh) * h)
    x0, y0, x1, y1 = min(xs), min(ys), max(xe), max(ye)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
    x1, y1 = min(w, int(x1 + mx)), min(h, int(y1 + my))
    if x1 - x0 < 32 or y1 - y0 < 32:                 # degenerate box: fall back to the whole frame
        return 0, 0, w, h
    return x0, y0, x1, y1


class SimulatedCamera:
    simulated = True

    def capture(self, order, out_dir):
        img = cv2.imread(order.evidence["image"])
        if img is None:
            raise FileNotFoundError(order.evidence["image"])
        h, w = img.shape[:2]
        if order.task == "VERIFY":
            out, note = self._post_treatment(img, order.targets, w, h)
        else:
            out, note = self._recapture(img, order, w, h)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{order.order_id}_after.jpg")
        cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {"path": path, "note": note, "simulated": True}

    # ----- strategies ---------------------------------------------------
    def _recapture(self, img, order, w, h):
        ev = order.evidence
        adjust = next((s for s in order.steps if s["action"] == "ADJUST_VIEW"), None)
        focus = adjust["params"].get("focus_regions", 0) if adjust else 0
        if focus and order.targets:
            x0, y0, x1, y1 = _union_box(order.targets[:focus], w, h)
            crop = img[y0:y1, x0:x1]
            scale = max(1.0, 640.0 / max(crop.shape[:2]))
            scale = min(scale, 3.0)
            out = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            return out, f"zoomed x{scale:.1f} on the suspected region"
        if ev.get("growth_stage") is None and ev.get("n_disease_detections", 0) == 0:
            mx, my = int(w * 0.10), int(h * 0.10)
            crop = img[my:h - my, mx:w - mx]
            out = cv2.convertScaleAbs(crop, alpha=1.0, beta=18)
            gamma = np.array([((i / 255.0) ** 0.75) * 255 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(out, gamma), "re-framed closer (80% centre) and brightened"
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), "improved contrast / lighting (CLAHE)"

    def _post_treatment(self, img, targets, w, h):
        mask = np.zeros((h, w), np.uint8)
        for t in targets:
            x, y, bw, bh = t["box_norm"]
            pad = 0.03
            x0, y0 = max(0, int((x - pad) * w)), max(0, int((y - pad) * h))
            x1, y1 = min(w, int((x + bw + pad) * w)), min(h, int((y + bh + pad) * h))
            mask[y0:y1, x0:x1] = 255
        out = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
        return out, f"SIMULATED post-treatment: {len(targets)} targeted region(s) inpainted (tissue removed)"
