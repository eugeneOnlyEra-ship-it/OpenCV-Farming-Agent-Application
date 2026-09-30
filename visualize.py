"""
visualize.py

Draws what classify_pod() actually detected directly onto the pod's
photo -- bounding boxes, class labels, confidences -- so there's visual
proof the CV pipeline is really running detection, not just returning
numbers. classify_pod() already computes all of this (growth_detections
/ disease_detections, real pixel-coordinate boxes from the real YOLOv8
models); this module's only job is drawing it, not detecting anything
itself.

Growth-model boxes draw in cyan, disease-model boxes in orange/red so
the two models' output stays visually distinguishable on one image
(classify_pod runs both on every pod, remember -- a growth-verified
pod still gets a real disease reading drawn too, exactly as uncurated
as the number in the log).
"""

import cv2

GROWTH_COLOR = (255, 200, 0)     # cyan-ish (BGR)
DISEASE_COLOR = (0, 100, 255)    # orange-red (BGR)
BANNER_BG = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
MAX_BOXES_DRAWN = 6  # a real diseased leaf can have dozens of genuine small-lesion detections;
                      # drawing every one makes the image unreadable, so only the highest-confidence
                      # few get a label -- the rest still count in classification, just not drawn

ACTION_BANNER_COLOR = {
    "log_healthy": (60, 170, 60),
    "flag_for_harvest": (10, 150, 230),
    "schedule_frequent_monitoring": (10, 200, 230),
    "flag_for_treatment": (40, 40, 220),
}


def _draw_box(img, det, color, font_scale):
    img_h, img_w = img.shape[:2]
    x, y, w, h = det["box"]
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{det['class_name']} {det['confidence']:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    label_x1 = min(max(0, x1), img_w - tw - 6)  # clamp so the label box can't run off either edge
    ty1 = max(0, y1 - th - 8)
    cv2.rectangle(img, (label_x1, ty1), (label_x1 + tw + 6, ty1 + th + 6), color, -1)
    cv2.putText(img, label, (label_x1 + 3, ty1 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def _fit_text(text, max_width_px, font_scale):
    """Truncates `text` with a trailing ellipsis so it fits max_width_px
    at the given font scale, instead of silently running off the edge
    of the image."""
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] <= max_width_px:
        return text
    while text and cv2.getTextSize(text + "...", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] > max_width_px:
        text = text[:-1]
    return text + "..." if text else "..."


def annotate(image_path, pod, classification, decision, trend_note=""):
    """Returns an annotated BGR image (numpy array): the highest-confidence
    detection boxes the real models found (capped at MAX_BOXES_DRAWN per
    model so a genuinely multi-lesion leaf stays readable), plus a banner
    summarizing the pod, the decision, and the trend note."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read {image_path}")
    h, w = img.shape[:2]
    font_scale = max(0.35, min(0.6, w / 640 * 0.5))

    def top_boxes(dets):
        return sorted(dets, key=lambda d: -d["confidence"])[:MAX_BOXES_DRAWN]

    growth_shown = top_boxes(classification["growth_detections"])
    disease_shown = top_boxes(classification["disease_detections"])
    for det in growth_shown:
        _draw_box(img, det, GROWTH_COLOR, font_scale)
    for det in disease_shown:
        _draw_box(img, det, DISEASE_COLOR, font_scale)

    n_hidden = (len(classification["growth_detections"]) - len(growth_shown)) + \
               (len(classification["disease_detections"]) - len(disease_shown))

    banner_h = int(70 * max(1.0, font_scale / 0.5))
    canvas = cv2.copyMakeBorder(img, 0, banner_h, 0, 0, cv2.BORDER_CONSTANT, value=BANNER_BG)

    action = decision["action"]
    banner_color = ACTION_BANNER_COLOR.get(action, BANNER_BG)
    cv2.rectangle(canvas, (0, h), (w, h + banner_h), banner_color, -1)

    disease_bit = classification["disease_name"] if classification["disease_flag"] else "no disease"
    line1 = f"{pod['pod_id']}  {pod['crop_type']}  stage={classification['growth_stage']}  {disease_bit}"
    line2 = f"-> {action}"
    if trend_note:
        line2 += f"  ({trend_note})"
    if n_hidden > 0:
        line2 += f"  [+{n_hidden} more detections not shown]"

    max_w = w - 16
    line1 = _fit_text(line1, max_w, font_scale)
    line2 = _fit_text(line2, max_w, font_scale)

    ty1 = h + int(24 * font_scale / 0.5)
    ty2 = h + int(48 * font_scale / 0.5)
    cv2.putText(canvas, line1, (8, ty1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (8, ty2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)

    return canvas


def save_annotated(image_path, pod, classification, decision, out_path, trend_note=""):
    canvas = annotate(image_path, pod, classification, decision, trend_note)
    cv2.imwrite(str(out_path), canvas)
    return out_path
