"""pod_labels.py -- printed pod tags, and the OpenCV reader that identifies a pod from a camera frame.

WHY THIS EXISTS
    The application names pods like "R1-L-F1" (rail 1, left station, front, slot 1) and used to TELL the
    simulated camera agent which pod it was at. A real robot cannot be told that: it drives to where the
    pod should be, and then has to LOOK at the pod to find out which one it is really standing in front
    of. So every simulated trough now carries a printed tag with its pod number, and the camera agent reads
    that tag from its own rendered frames with OpenCV. The number it reads -- not the dispatch list -- is
    what its scan, scores and any follow-up work are attached to.

WHAT THE TAG LOOKS LIKE
    A white plate with a thick black border and the pod number in black, fixed format ``R<1-3>-<L|M|R>-<F|B><1-4>``
    (7 characters, 72 possible tags). It is mounted flat on the aisle-facing wall of the trough, like the
    tag clipped to a real growing tray. ``render_plate()`` draws it; the simulator wraps that image around
    a thin quad as a texture (see GantryRobot._add_pod_label).

HOW IT IS READ  (PodLabelReader, no OCR engine and no extra dependency -- only cv2 + numpy)
    1. FIND   threshold the very dark pixels; the plate's black border is a closed ring whose outer
              contour is a four-sided shape of plausible proportions.
    2. FLATTEN perspective-rectify that quadrilateral to the plate's canonical size (cv2.warpPerspective).
    3. SLICE  the format is fixed-width, so the interior is cut into 7 character cells.
    4. MATCH  each cell is compared (shift-tolerant normalised cross-correlation) against glyph templates
              drawn with the same font that printed the tag.
    5. VOTE   the camera takes several views of a pod; reads are combined, and a tag is only accepted if the
              views agree (or one view is very clean) and nothing contradicts it.

    This is a purpose-built reader for the farm's own tag format -- NOT a general-purpose OCR. That is what
    keeps it small, dependency-free and checkable. The production-grade version of the same idea is a
    fiducial (ArUco / QR) printed next to the human-readable number.

    A cell is decoded within the characters its position allows (position 1 is a digit 1-3, position 3 is
    L/M/R ...), but the best match over the WHOLE alphabet is kept too: a cell that looks more like some
    other character than anything allowed there marks the view unreliable rather than being forced into
    a valid-looking tag. The reader never "snaps" to the nearest real pod: an uncertain read is a failed
    read, and a failed read means the robot does not touch the pod.
"""
from __future__ import annotations

import re
from collections import defaultdict

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# ----------------------------------------------------------------------------------------- tag format
LABEL_RE = re.compile(r"^R[1-3]-[LMR]-[FB][1-4]$")
STATION_LETTERS = ("L", "M", "R")                 # station (column) 0, 1, 2 -> the letter on the tag
# characters allowed at each of the 7 positions of the tag
POSITION_ALPHABETS = ("R", "123", "-", "LMR", "-", "FB", "1234")
ALPHABET = "R123-LMFB4"                           # every character that can appear on a tag (4 is listed once)
N_CELLS = len(POSITION_ALPHABETS)

# physical plate (metres): sized to sit on the 0.09 m tall trough wall. The wall is shared with main's ArUco marker
# (pod_markers.MARKER_X_OFFSET_M = -0.10, plate 0.088 wide: x -0.144..-0.056) and the "serviced" disc at x = -0.19, so the tag
# is 0.255 wide and sits to the RIGHT of the marker (x -0.045..+0.21). Centred and 0.30 wide it hid behind the marker and
# no view could be read (see HANDOFF.md, "Wall layout"). Keep the same 576:150 aspect if you resize it.
PLATE_W_M = 0.255
PLATE_H_M = 0.0664
PLATE_X_OFFSET_M = 0.0825                         # plate centre, along the trough, from the trough centre
PLATE_STANDOFF_M = 0.002                          # plate floats this far proud of the wall (no z-fighting)

# printed plate image (pixels)
PLATE_PX_W, PLATE_PX_H = 576, 150                 # same aspect as the physical plate
BORDER_PX = 14
_INNER_X0, _INNER_Y0 = BORDER_PX, BORDER_PX
_INNER_W, _INNER_H = PLATE_PX_W - 2 * BORDER_PX, PLATE_PX_H - 2 * BORDER_PX
_CELL_W = _INNER_W / N_CELLS
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GLYPH_THICKNESS = 11
_GLYPH_HEIGHT_FRAC = 0.74                         # glyph cap height as a fraction of the inner height

# The reader works on a smaller flattened plate than the one that is printed: the camera frames are only ~320 px
# wide, so the tag is never seen at print resolution, and matching at print size is ~8x slower for no gain.
READ_SCALE = 0.5
READ_W, READ_H = int(PLATE_PX_W * READ_SCALE), int(PLATE_PX_H * READ_SCALE)
_R_BORDER = BORDER_PX * READ_SCALE
_R_INNER_X0 = _R_INNER_Y0 = int(round(_R_BORDER))
_R_INNER_W, _R_INNER_H = READ_W - 2 * _R_INNER_X0, READ_H - 2 * _R_INNER_Y0
_R_CELL_W = _R_INNER_W / N_CELLS

# reader thresholds
DARK_MAX_CHANNEL = 45                             # "ink-dark" = brightest channel below this. Printed black is ~0; the dark-GREY
                                                  # poles in the scene sit at 60-80, so they must not count (they once fused with the plate)
MIN_PLATE_AREA_PX = 300                           # smaller than this is not a readable plate
MIN_PLATE_WIDTH_PX = 55
MIN_CELL_SCORE = 0.50                             # a cell must correlate at least this well with its character
MIN_CELL_MARGIN = 0.04                            # ... and beat the runner-up (within the allowed set) by this
SINGLE_VIEW_CONFIDENCE = 0.78                     # a lone view must be at least this clean to be trusted by itself


def label_text(row, col, layer, side):
    """The text printed on the pod at (row, col, layer, side): row 0 -> "R1", column 0/1/2 -> L/M/R, layer 0 -> "1"."""
    return f"R{int(row) + 1}-{STATION_LETTERS[int(col)]}-{str(side)[0].upper()}{int(layer) + 1}"


def label_for_pod(pod):
    """The tag text of a simulator pod dict (as built by pod_registry)."""
    return label_text(pod["row"], pod["col"], pod["layer"], pod["side"])


def is_valid_label(text):
    return bool(text) and LABEL_RE.match(str(text)) is not None


# ----------------------------------------------------------------------------------------- printing
def _glyph_scale():
    """Hershey scale at which a capital letter is _GLYPH_HEIGHT_FRAC of the plate's inner height."""
    (_, h), _ = cv2.getTextSize("R", _FONT, 1.0, _GLYPH_THICKNESS)
    return (_INNER_H * _GLYPH_HEIGHT_FRAC) / max(1, h)


_SCALE = _glyph_scale()


def _draw_char(canvas, ch, cell_index, offset_x=0, offset_y=0):
    """Draw one character centred in cell `cell_index` (shared by printing and template generation, so the
    templates are exactly what the printer produces)."""
    (tw, th), _ = cv2.getTextSize(ch, _FONT, _SCALE, _GLYPH_THICKNESS)
    cx = offset_x + _INNER_X0 + (cell_index + 0.5) * _CELL_W
    cy = offset_y + _INNER_Y0 + _INNER_H / 2.0
    cv2.putText(canvas, ch, (int(round(cx - tw / 2.0)), int(round(cy + th / 2.0))), _FONT, _SCALE, 0,
                _GLYPH_THICKNESS, cv2.LINE_AA)


def render_plate(text):
    """The printed tag for `text`, as a BGR uint8 image (PLATE_PX_H x PLATE_PX_W): white plate, black border, black text."""
    if not is_valid_label(text):
        raise ValueError(f"not a valid pod tag: {text!r}")
    gray = np.full((PLATE_PX_H, PLATE_PX_W), 255, np.uint8)
    cv2.rectangle(gray, (0, 0), (PLATE_PX_W - 1, PLATE_PX_H - 1), 0, thickness=BORDER_PX * 2 - 1)
    for i, ch in enumerate(text):
        _draw_char(gray, ch, i)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------------------------- templates
_TPL_TRIM = 3              # templates are cropped this much tighter than a cell ...
_CELL_PAD = 1              # ... and the cell is read with this much context, so a template can slide +/-(trim+pad) = 4 px
_TEMPLATE_PAD = _CELL_PAD  # (name kept for the helpers below)
_CELL_PX_W, _CELL_PX_H = int(round(_R_CELL_W)), int(_R_INNER_H)


def _cell_ink(gray_plate, cell_index, pad=0):
    """One cell of a flattened (reading-size) plate, with `pad` pixels of context each side (white outside the plate)."""
    x0 = int(round(_R_INNER_X0 + cell_index * _R_CELL_W)) - pad
    y0 = _R_INNER_Y0 - pad
    x1, y1 = x0 + _CELL_PX_W + 2 * pad, y0 + _CELL_PX_H + 2 * pad
    h, w = gray_plate.shape
    patch = np.full((y1 - y0, x1 - x0), 255, np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = gray_plate[sy0:sy1, sx0:sx1]
    return patch


def _build_templates():
    """One soft-ink template per character: the printer's own output for that glyph (drawn at print size, then shrunk
    to reading size), cropped a little tighter than its cell so that, while matching, it can slide around inside the
    padded cell to absorb mis-registration."""
    templates = {}
    t = _TPL_TRIM
    for ch in sorted(set(ALPHABET)):
        canvas = np.full((PLATE_PX_H, PLATE_PX_W), 255, np.uint8)
        _draw_char(canvas, ch, 0)
        small = cv2.resize(canvas, (READ_W, READ_H), interpolation=cv2.INTER_AREA)
        cell = _cell_ink(small, 0)
        ink = (255.0 - cell.astype(np.float32)) / 255.0
        ink = cv2.GaussianBlur(ink, (0, 0), 1.5)             # a little blur: real frames are never razor sharp
        templates[ch] = np.ascontiguousarray(ink[t:-t, t:-t])
    return templates


_TEMPLATES = _build_templates()


def _normalise_ink(gray_cell):
    """Gray cell -> soft ink image in 0..1, stretched to the cell's own contrast so lighting/shading does not matter."""
    g = gray_cell.astype(np.float32)
    lo, hi = float(np.percentile(g, 3)), float(np.percentile(g, 97))
    if hi - lo < 25.0:                                    # no contrast at all: nothing printed here
        return np.zeros_like(g)
    ink = 1.0 - np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    return cv2.GaussianBlur(ink, (0, 0), 1.5)


def _ncc_grid(image, template):
    """Normalised cross-correlation of `template` at every position inside `image` (same maths as cv2.TM_CCOEFF_NORMED).
    Done directly in numpy because the shift grid here is tiny (about 9 x 9): cv2.matchTemplate's per-call overhead
    (it goes through an FFT) cost ~2 ms a call, 10 calls a cell, and made reading a tag take ~100 ms."""
    th, tw = template.shape
    windows = sliding_window_view(image, (th, tw))                      # (H', W', th, tw), no copy
    n = float(th * tw)
    t0 = template - template.mean()
    t_norm = float(np.sqrt((t0 * t0).sum()))
    cross = np.tensordot(windows, t0, axes=([2, 3], [0, 1]))
    s1 = windows.sum(axis=(2, 3))
    s2 = np.einsum("ijkl,ijkl->ij", windows, windows)
    var = np.maximum(s2 - s1 * s1 / n, 1e-6)
    return cross / (np.sqrt(var) * max(t_norm, 1e-6))


def _match_cell(gray_plate, cell_index, allowed):
    """Score the glyph templates against one cell. Returns {char: best normalised correlation}.

    The characters this position ALLOWS are searched over every small shift of the cell. Every other character is
    only checked in a small window around the best alignment found (that is all the 'does this cell look like
    something that cannot be here?' test needs)."""
    cell = _normalise_ink(_cell_ink(gray_plate, cell_index, pad=_CELL_PAD))
    scores, best_char, best_loc = {}, None, (_TPL_TRIM + _CELL_PAD, _TPL_TRIM + _CELL_PAD)
    for ch in set(allowed):
        res = _ncc_grid(cell, _TEMPLATES[ch])
        iy, ix = np.unravel_index(int(np.argmax(res)), res.shape)
        scores[ch] = float(res[iy, ix])
        if best_char is None or scores[ch] > scores[best_char]:
            best_char, best_loc = ch, (int(ix), int(iy))
    slack = 2
    x0, y0 = max(0, best_loc[0] - slack), max(0, best_loc[1] - slack)
    for ch, tpl in _TEMPLATES.items():
        if ch in scores:
            continue
        th, tw = tpl.shape
        window = cell[y0:best_loc[1] + th + slack, x0:best_loc[0] + tw + slack]
        if window.shape[0] < th or window.shape[1] < tw:
            window = cell
        scores[ch] = float(_ncc_grid(window, tpl).max())
    return scores


# ----------------------------------------------------------------------------------------- the reader
def _order_corners(pts):
    """Four points -> top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def _drop_weakest_vertex(poly):
    """Remove the polygon vertex whose triangle with its two neighbours is smallest (it contributes least to the shape)."""
    n = len(poly)
    areas = [abs(np.cross(poly[(i + 1) % n] - poly[i], poly[(i - 1) % n] - poly[i])) / 2.0 for i in range(n)]
    return np.delete(poly, int(np.argmin(areas)), axis=0)


def _quad_from_contour(contour):
    """A four-corner approximation of a contour, or None. The epsilon is grown until four corners remain (aliased edges
    at low resolution otherwise give five or six). If the approximation skips from five corners straight to three, the
    least significant corner is dropped instead, so one stray pixel cannot hide a perfectly good plate."""
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    last = None
    for eps in (0.02, 0.03, 0.045, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2).astype(np.float32)
        if len(approx) == 4:
            return approx
        if len(approx) < 4:
            break
        last = approx
    if last is not None and len(last) <= 6:
        while len(last) > 4:
            last = _drop_weakest_vertex(last)
        return last
    return None


class PodLabelReader:
    """Reads the printed pod tag out of camera frames. See the module docstring for the method."""

    def __init__(self, min_cell_score=MIN_CELL_SCORE, min_cell_margin=MIN_CELL_MARGIN,
                 single_view_confidence=SINGLE_VIEW_CONFIDENCE):
        self.min_cell_score = float(min_cell_score)
        self.min_cell_margin = float(min_cell_margin)
        self.single_view_confidence = float(single_view_confidence)

    # -- 1. find ------------------------------------------------------------------------
    def find_plates(self, bgr):
        """Candidate plate quadrilaterals in a frame, as ordered (4, 2) float32 corner arrays, biggest first."""
        if bgr is None or bgr.ndim != 3:
            return []
        dark = (bgr.max(axis=2) < DARK_MAX_CHANNEL).astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for c in contours:
            if cv2.contourArea(c) < MIN_PLATE_AREA_PX:
                continue
            quad = _quad_from_contour(c)
            if quad is None:
                continue
            quad = _order_corners(quad)
            top, bottom = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
            left, right = np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
            width, height = (top + bottom) / 2.0, (left + right) / 2.0
            if width < MIN_PLATE_WIDTH_PX or height < 8:
                continue
            if not (2.0 <= width / max(height, 1e-6) <= 10.0):
                continue
            if not cv2.isContourConvex(quad.astype(np.int32).reshape(-1, 1, 2)):
                continue
            found.append((cv2.contourArea(quad), quad))
        found.sort(key=lambda t: -t[0])
        return [q for _, q in found]

    # -- 2. flatten ---------------------------------------------------------------------
    @staticmethod
    def flatten(bgr, quad):
        dst = np.array([[0, 0], [READ_W - 1, 0], [READ_W - 1, READ_H - 1], [0, READ_H - 1]], np.float32)
        matrix = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
        flat = cv2.warpPerspective(bgr, matrix, (READ_W, READ_H), flags=cv2.INTER_CUBIC)
        return cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)

    # -- 3-4. slice + match ---------------------------------------------------------------
    def decode_plate(self, gray_plate):
        """Decode a flattened plate. Returns (text, confidence, details). `text` is None when it cannot be trusted."""
        # the interior of a real plate is mostly white; a dark blob that merely looks rectangular is not a tag
        inner = gray_plate[_R_INNER_Y0:_R_INNER_Y0 + _R_INNER_H, _R_INNER_X0:_R_INNER_X0 + _R_INNER_W]
        bright = float((inner > 0.5 * (float(inner.max()) + float(inner.min()))).mean())
        if inner.max() - inner.min() < 40 or not (0.55 <= bright <= 0.97):
            return None, 0.0, {"reason": "interior is not a white plate with dark print", "cells": []}
        chars, cells, confidence, reason = [], [], 1.0, None
        for i in range(N_CELLS):
            allowed = POSITION_ALPHABETS[i]
            scores = _match_cell(gray_plate, i, allowed)
            ranked = sorted(((scores[c], c) for c in set(allowed)), reverse=True)
            best_score, best = ranked[0]
            runner_up = ranked[1][0] if len(ranked) > 1 else max(
                (s for c, s in scores.items() if c != best), default=0.0)
            overall_best = max(scores, key=scores.get)
            margin = best_score - runner_up
            # best match over the WHOLE alphabet must not be a character this position cannot hold
            # (position-wise decoding is only trusted when the cell does not look like something else entirely)
            conflict = overall_best not in allowed and scores[overall_best] - best_score > self.min_cell_margin
            cells.append({"index": i, "char": best, "score": round(best_score, 3), "margin": round(margin, 3),
                          "overall_best": overall_best, "conflict": bool(conflict)})
            chars.append(best)
            confidence = min(confidence, best_score)
            if reason is None:
                if best_score < self.min_cell_score:
                    reason = f"cell {i} ('{best}') matches poorly ({best_score:.2f})"
                elif margin < self.min_cell_margin:
                    reason = f"cell {i} is ambiguous ('{best}' {best_score:.2f} vs {runner_up:.2f})"
                elif conflict:
                    reason = f"cell {i} looks like '{overall_best}', which cannot appear there"
        text = "".join(chars)
        if reason is None and not is_valid_label(text):
            reason = f"decoded text {text!r} is not a valid tag"
        return (None if reason else text), float(confidence), {"reason": reason, "raw_text": text, "cells": cells}

    def read_frame(self, bgr):
        """Read the best tag in one frame. Always returns a dict: {text (or None), valid, confidence, corners, reason, ...}."""
        quads = self.find_plates(bgr)
        if not quads:
            return {"text": None, "valid": False, "confidence": 0.0, "corners": None, "flat": None,
                    "reason": "no tag-shaped plate found in the frame", "raw_text": None, "cells": []}
        best = None
        for quad in quads[:4]:                               # a frame rarely holds more than a couple of candidates
            flat = self.flatten(bgr, quad)
            text, confidence, detail = self.decode_plate(flat)
            result = {"text": text, "valid": text is not None, "confidence": round(confidence, 3), "flat": flat,
                      "corners": quad.round(1).tolist(), "reason": detail.get("reason"),
                      "raw_text": detail.get("raw_text"), "cells": detail.get("cells", [])}
            if best is None or (result["valid"], result["confidence"]) > (best["valid"], best["confidence"]):
                best = result
        return best

    # -- 5. vote ------------------------------------------------------------------------
    def read_views(self, frames, keep_evidence=False):
        """Combine the reads from several views of ONE pod into a single identity decision.

        Returns {"label", "confirmed", "confidence", "views_read", "views_total", "agreeing_views", "conflict",
                 "reason", "per_view": [...]} where `label` is None unless the tag is confirmed. A tag is confirmed when
        at least two views agree, or one view is very clean, and no view clearly reads a DIFFERENT tag."""
        per_view = [self.read_frame(f) for f in frames]
        votes = defaultdict(list)
        for i, r in enumerate(per_view):
            if r["valid"]:
                votes[r["text"]].append(r["confidence"])
        summary = {"label": None, "confirmed": False, "confidence": 0.0, "views_total": len(per_view),
                   "views_read": sum(1 for r in per_view if r["valid"]), "agreeing_views": 0, "conflict": False,
                   "reason": None,
                   "per_view": [{k: r[k] for k in ("text", "valid", "confidence", "reason", "raw_text")} for r in per_view]}
        if keep_evidence:
            # what the reader SAW, so it can be shown (render_reading_sheet): every frame with the plate it found, and the
            # flattened plate + per-character scores of the most convincing view
            candidates = [i for i, r in enumerate(per_view) if r.get("flat") is not None]
            best = max(candidates, key=lambda i: (per_view[i]["valid"], per_view[i]["confidence"]), default=None)
            summary["evidence"] = {
                "views": [{"frame": f, "corners": r["corners"], "text": r["text"], "valid": r["valid"],
                           "confidence": r["confidence"], "reason": r["reason"]} for f, r in zip(frames, per_view)],
                "best": best, "flat": None if best is None else per_view[best]["flat"],
                "cells": [] if best is None else per_view[best]["cells"]}
        if not votes:
            reasons = [r["reason"] for r in per_view if r["reason"]]
            summary["reason"] = ("no view produced a readable tag" + (f" ({reasons[0]})" if reasons else ""))
            return summary
        ranked = sorted(votes.items(), key=lambda kv: (len(kv[1]), sum(kv[1])), reverse=True)
        top_text, top_conf = ranked[0]
        summary.update(label=top_text, agreeing_views=len(top_conf), confidence=round(float(np.mean(top_conf)), 3))
        if len(ranked) > 1:
            summary["conflict"] = True
            summary["reason"] = f"views disagree: {', '.join(f'{t} x{len(c)}' for t, c in ranked)}"
            summary["label"] = None
            return summary
        if len(top_conf) >= 2 or max(top_conf) >= self.single_view_confidence:
            summary["confirmed"] = True
        else:
            summary["label"] = None
            summary["reason"] = f"only one view read {top_text} and not cleanly (confidence {max(top_conf):.2f})"
        return summary



def plate_geometry():
    """Geometry of the flattened (reading-size) plate, so an interface can draw the reading: its size and the pixel box
    (x0, y0, x1, y1) of each of the 7 character cells."""
    boxes = [(_R_INNER_X0 + i * _R_CELL_W, _R_INNER_Y0, _R_INNER_X0 + (i + 1) * _R_CELL_W, _R_INNER_Y0 + _R_INNER_H)
             for i in range(N_CELLS)]
    return {"width": READ_W, "height": READ_H, "cells": boxes}


# ----------------------------------------------------------------------------------------- showing the reading
_SHEET_BG = (38, 28, 20)          # BGR, close to the application's dark panels
_SHEET_TEXT = (235, 225, 215)
_SHEET_DIM = (150, 140, 130)
_GOOD, _WARN, _BAD = (90, 200, 80), (60, 190, 235), (80, 90, 235)


def _put(img, text, org, scale=0.5, color=_SHEET_TEXT, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render_reading_sheet(evidence, min_cell_score=MIN_CELL_SCORE):
    """One picture of how the camera read a pod's tag, for the application to show (ASCII text only: OpenCV's fonts).

    Row 1: the camera frames, with the plate OpenCV located outlined (green = read, orange = found but unreadable).
    Row 2: the best view's plate after perspective correction, split into its 7 character cells, with the character
           chosen for each cell and how well it matched the printed glyph."""
    tw, th, pad = 256, 192, 8
    views = evidence["views"]
    width = max(3, len(views)) * tw + (max(3, len(views)) + 1) * pad
    plate_w = width - 2 * 50
    scale = plate_w / READ_W
    plate_h = int(READ_H * scale)
    height = 30 + th + 30 + 30 + plate_h + 66 + pad
    sheet = np.full((height, width, 3), _SHEET_BG, np.uint8)

    _put(sheet, "1  CAMERA FRAMES  -  the tag plate OpenCV found is outlined", (pad, 20), 0.55, _SHEET_TEXT, 1)
    for i, v in enumerate(views):
        x0, y0 = pad + i * (tw + pad), 30
        thumb = cv2.resize(v["frame"], (tw, th), interpolation=cv2.INTER_AREA)
        if v["corners"] is not None:
            pts = (np.array(v["corners"], np.float32) * [tw / v["frame"].shape[1], th / v["frame"].shape[0]]).astype(np.int32)
            cv2.polylines(thumb, [pts.reshape(-1, 1, 2)], True, _GOOD if v["valid"] else _WARN, 2, cv2.LINE_AA)
        sheet[y0:y0 + th, x0:x0 + tw] = thumb
        if v["valid"]:
            _put(sheet, f"view {i + 1}: read {v['text']}  ({v['confidence']:.2f})", (x0, y0 + th + 20), 0.5, _GOOD)
        elif v["corners"] is not None:
            _put(sheet, f"view {i + 1}: unreadable", (x0, y0 + th + 20), 0.5, _WARN)
        else:
            _put(sheet, f"view {i + 1}: no tag found", (x0, y0 + th + 20), 0.5, _BAD)

    y = 30 + th + 30
    _put(sheet, "2  PLATE FLATTENED, then read one character at a time", (pad, y + 18), 0.55, _SHEET_TEXT, 1)
    y += 30
    flat = evidence.get("flat")
    if flat is None:
        _put(sheet, "no plate could be flattened - nothing to read", (50, y + plate_h // 2), 0.6, _BAD)
        return sheet
    x0 = 50
    big = cv2.resize(flat, (int(plate_w), plate_h), interpolation=cv2.INTER_CUBIC)
    sheet[y:y + plate_h, x0:x0 + int(plate_w)] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    cells = {c["index"]: c for c in evidence.get("cells", [])}
    for i in range(N_CELLS):
        cx0 = int(x0 + (_R_INNER_X0 + i * _R_CELL_W) * scale)
        cx1 = int(x0 + (_R_INNER_X0 + (i + 1) * _R_CELL_W) * scale)
        cy0, cy1 = int(y + _R_INNER_Y0 * scale), int(y + (_R_INNER_Y0 + _R_INNER_H) * scale)
        c = cells.get(i)
        color = _SHEET_DIM if c is None else (_GOOD if c["score"] >= 0.75 else (_WARN if c["score"] >= min_cell_score else _BAD))
        cv2.rectangle(sheet, (cx0 + 1, cy0), (cx1 - 1, cy1), color, 2)
        if c is not None:
            (tw_, _), _b = cv2.getTextSize(c["char"], cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            _put(sheet, c["char"], ((cx0 + cx1) // 2 - tw_ // 2, y + plate_h + 30), 1.1, color, 2)
            _put(sheet, f"{c['score']:.2f}", ((cx0 + cx1) // 2 - 16, y + plate_h + 54), 0.45, _SHEET_DIM)
    return sheet

_DEFAULT_READER = None


def default_reader():
    global _DEFAULT_READER
    if _DEFAULT_READER is None:
        _DEFAULT_READER = PodLabelReader()
    return _DEFAULT_READER
