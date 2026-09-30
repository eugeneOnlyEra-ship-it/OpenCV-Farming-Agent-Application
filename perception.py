"""
Perception layer for OpenCV-Farming-Agent.

Implements the classify_pod(pod_id, crop_type, image) interface: loads the
per-crop growth-stage and disease-detection ONNX models (YOLOv8, exported
per the training notebooks) and runs both through cv2.dnn to produce a
single structured reading for a pod.

Model files are expected as flat .onnx files in MODEL_DIR, named exactly as
they came out of the training notebooks:
    cabbage_growth_stage_v2.onnx   lettuce_growth_stage_v2.onnx   mushroom_growth_stage_v2.onnx
    cabbage_disease.onnx           lettuce_disease.onnx           mushroom_disease.onnx

Set MODEL_DIR (below) to wherever you keep them before calling classify_pod().
"""

from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_DIR = Path("./models")  # change this to wherever your .onnx files live

CONF_THRESHOLD = 0.25   # per-detection confidence floor before NMS
NMS_THRESHOLD = 0.45    # IoU threshold for suppressing overlapping boxes
INPUT_SIZE = 640        # must match the imgsz the models were exported at

# Class names and index order are read directly from each ONNX file's own
# embedded metadata (verified against the actual exported models), not
# retyped by hand — the one exception is cabbage's growth-stage labels
# (S2_3 / S4_5 / S6_7 / S8 / S9), which the dataset's own README already
# flags as an unverified best-effort leaf-count interpretation, not a
# confirmed definition from the dataset author. That caveat is preserved
# below rather than presented as fact.
MODEL_REGISTRY = {
    "cabbage": {
        "growth": {
            "path": "cabbage_growth_stage_v2.onnx",
            "classes": {0: "S2_3", 1: "S4_5", 2: "S6_7", 3: "S8", 4: "S9"},
            "harvest_stage": "S9",  # unverified interpretation: "9+ leaves / head formation-maturity"
        },
        "disease": {
            "path": "cabbage_disease.onnx",
            "classes": {
                0: "Cabbage-Alternaria_leaf_spot",
                1: "Cabbage-Bacterial_leaf_spot",
                2: "Cabbage-Black_rot",
                3: "Cabbage-Downy_mildew",
                4: "Cabbage-Fusarium_wilt",
                5: "Cabbage-Healthy",
                6: "Cabbage-Powdery_mildew",
                7: "Cabbage-Ringspot",
                8: "Cabbage-Sclerotinia_rot",
            },
            "healthy_class": "Cabbage-Healthy",
        },
    },
    "lettuce": {
        "growth": {
            "path": "lettuce_growth_stage_v2.onnx",
            "classes": {
                0: "Stage 01- Early Growth",
                1: "stage 02- Leafy growth",
                2: "stage 03- Head formation",
                3: "stage 04- Harvest stage",
            },
            "harvest_stage": "stage 04- Harvest stage",
        },
        "disease": {
            "path": "lettuce_disease.onnx",
            "classes": {
                0: "Bacterial",
                1: "Downy_mildew_on_lettuce",
                2: "Powdery_mildew_on_lettuce",
                3: "Septoria_Blight_on_lettuce",
                4: "Viral",
                5: "Wilt_and_leaf_blight_on_lettuce",
                6: "healthy",
            },
            "healthy_class": "healthy",
        },
    },
    "mushroom": {
        "growth": {
            "path": "mushroom_growth_stage_v2.onnx",
            "classes": {0: "Harvest", 1: "Intermediate", 2: "Juvenile"},
            "harvest_stage": "Harvest",
        },
        "disease": {
            "path": "mushroom_disease.onnx",
            "classes": {
                0: "Bacterial Blotch",
                1: "Dry Bubble",
                2: "Healthy",
                3: "Trichoderma",
                4: "Wilt",
            },
            "healthy_class": "Healthy",
        },
    },
}

_net_cache = {}
_net_cache_lock = __import__("threading").Lock()


def _get_net(crop_type, kind):
    """Lazily loads and caches a cv2.dnn.Net for (crop_type, kind).
    Locked because cloud_pipeline.LocalCloudClient calls classify_pod() from
    background threads — without this, two threads racing on a cold cache
    could both load the same model at once."""
    key = (crop_type, kind)
    with _net_cache_lock:
        if key not in _net_cache:
            model_path = MODEL_DIR / MODEL_REGISTRY[crop_type][kind]["path"]
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Expected model at {model_path} — set perception.MODEL_DIR to the "
                    f"folder holding your .onnx exports before calling classify_pod()."
                )
            _net_cache[key] = cv2.dnn.readNetFromONNX(str(model_path))
        return _net_cache[key]


def warm_up(crop_types=None):
    """Loads every model into the cache up front. Call this once before
    your PyBullet loop starts submitting frames, the same way a real
    deployment would use provisioned-concurrency Lambdas or a pre-warmed
    EC2/Fargate service — otherwise the first frame for each crop pays a
    one-time model-load cost that has nothing to do with actual inference
    speed, and will skew any latency numbers you measure."""
    for crop_type in (crop_types or MODEL_REGISTRY.keys()):
        _get_net(crop_type, "growth")
        _get_net(crop_type, "disease")


# ---------------------------------------------------------------------------
# Pre/post-processing (standard YOLOv8-ONNX decode via cv2.dnn)
# ---------------------------------------------------------------------------

def _letterbox(image, new_size=INPUT_SIZE, color=(114, 114, 114)):
    """Resizes with aspect ratio preserved and pads to a square, the same
    way Ultralytics does internally — needed so box coordinates decoded
    from the model's output map back to the original image correctly."""
    h, w = image.shape[:2]
    r = min(new_size / h, new_size / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def _preprocess(image):
    padded, scale, pad = _letterbox(image, INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, size=(INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    return blob, scale, pad


def _postprocess(output, scale, pad, class_names, conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD):
    """Decodes a raw YOLOv8-ONNX output tensor (shape [1, 4+nc, 8400]) into
    a list of detections in original-image pixel coordinates."""
    preds = output[0].T  # (8400, 4+nc)

    boxes, confidences, class_ids = [], [], []
    for row in preds:
        cx, cy, w, h = row[:4]
        scores = row[4:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        if conf < conf_threshold:
            continue
        x = (cx - w / 2 - pad[0]) / scale
        y = (cy - h / 2 - pad[1]) / scale
        boxes.append([x, y, w / scale, h / scale])
        confidences.append(conf)
        class_ids.append(class_id)

    if not boxes:
        return []

    keep = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections = []
    for i in np.array(keep).flatten():
        detections.append({
            "class_id": class_ids[i],
            "class_name": class_names[class_ids[i]],
            "confidence": confidences[i],
            "box": boxes[i],  # [x, y, w, h] in original image pixels, top-left origin
        })
    return detections


def _load_image(image):
    """Accepts either a file path/str or an already-loaded BGR ndarray
    (e.g. a frame captured in PyBullet), so classify_pod doesn't force a
    round-trip through disk when the caller already has pixels in memory."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image}")
        return img
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"image must be a file path or numpy ndarray, got {type(image)}")


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_pod(pod_id, crop_type, image, conf_threshold=CONF_THRESHOLD):
    """
    Runs both the growth-stage and disease-detection models for `crop_type`
    against `image` and returns a single structured reading.

    Returns a dict with (at minimum) the three fields from the original
    interface contract — growth_stage, disease_flag, confidence — plus
    additional fields (disease_name, the two split-out confidences, and the
    raw per-model detections) that weren't in the original three-field
    contract but are available now that real models are wired in. Trim the
    dict down before handing it to code that expects exactly the original
    three keys, if that's what it was written against.
    """
    crop_type = crop_type.lower()
    if crop_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown crop_type '{crop_type}'. Expected one of {list(MODEL_REGISTRY)}.")

    cfg = MODEL_REGISTRY[crop_type]
    img = _load_image(image)
    blob, scale, pad = _preprocess(img)

    growth_net = _get_net(crop_type, "growth")
    growth_net.setInput(blob)
    g_out = growth_net.forward()
    g_dets = _postprocess(g_out, scale, pad, cfg["growth"]["classes"], conf_threshold)

    disease_net = _get_net(crop_type, "disease")
    disease_net.setInput(blob)
    d_out = disease_net.forward()
    d_dets = _postprocess(d_out, scale, pad, cfg["disease"]["classes"], conf_threshold)

    growth_stage, growth_confidence = None, 0.0
    if g_dets:
        best = max(g_dets, key=lambda d: d["confidence"])
        growth_stage, growth_confidence = best["class_name"], best["confidence"]

    healthy_class = cfg["disease"]["healthy_class"]
    non_healthy = [d for d in d_dets if d["class_name"] != healthy_class]
    disease_flag, disease_name, disease_confidence = False, None, 0.0
    if non_healthy:
        best_d = max(non_healthy, key=lambda d: d["confidence"])
        disease_flag, disease_name, disease_confidence = True, best_d["class_name"], best_d["confidence"]
    elif d_dets:
        best_d = max(d_dets, key=lambda d: d["confidence"])
        disease_confidence = best_d["confidence"]

    return {
        "pod_id": pod_id,
        "crop_type": crop_type,
        "growth_stage": growth_stage,
        "growth_confidence": growth_confidence,
        "disease_flag": disease_flag,
        "disease_name": disease_name,
        "disease_confidence": disease_confidence,
        # single-value "confidence" for callers written against the original
        # 3-field contract — the more conservative (lower) of the two reads
        "confidence": min(c for c in (growth_confidence, disease_confidence) if c > 0) if (growth_confidence or disease_confidence) else 0.0,
        "growth_detections": g_dets,
        "disease_detections": d_dets,
    }
