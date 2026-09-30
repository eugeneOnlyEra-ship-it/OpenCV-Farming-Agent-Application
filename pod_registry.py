"""
pod_registry.py

Pod metadata + real image assignment for the pod farm's data-processing
pipeline. This is the PyBullet project's pod_registry.py with the
physical/geometric half removed (rail heights, station X positions,
trough half-extents, bay span) -- none of that means anything without a
simulated rig to place it in. What's kept is everything about DATA:
which pods exist, what crop and physical grouping (rail/station/side)
each belongs to, which real image represents it, and what ground truth
that image is actually verified for.

Layout (grouping only, no coordinates): 3 station rails x 3 stations x
2 sides = 18 physical troughs; each trough holds 4 pods (a 3-plant
cluster each) = 72 pods, 24 per crop. One crop per station, shared by
both its sides and all 8 pods there.

Every pod's image is a REAL photo from the fixed Roboflow datasets --
none are synthetic placeholders, and none repeat pixel-identical across
pods. Front pods are growth-stage-verified (ground truth is a real
confirmed stage); back pods are disease-verified (ground truth is a
real confirmed disease/healthy status). classify_pod() still runs both
models on every pod regardless of which aspect is verified -- the
*other* aspect's reading on a given pod is real model output with no
curated answer to check it against, not a wrong one.

Growth stages and disease classes are spread evenly across a crop's 12
same-side pods (a running cycle across all its pods, not reset per
station), so every stage/disease a crop can show actually appears
across its pods.

GAP CLOSED: mushroom's 12 front pods originally fell back to real
mushroom *disease*-dataset photos with growth_stage left unverified,
because no real mushroom growth-stage source images were available.
That dataset (mushroom-growth-stages-2vmfq, Roboflow) was supplied
later and all 12 now use real growth-stage-verified photos, same as
cabbage and lettuce -- see pods_manifest.json.

Every front pod also carries a full growth-stage TRAJECTORY -- real
photos for every stage from wherever it started through to harvest,
57 additional real images beyond the base 72, none repeated -- so
multi-visit runs (see main.py --visits) can swap in a genuinely later
photo each time rather than replaying the same snapshot forever.

Class names are validated against perception.py's MODEL_REGISTRY at
import time, so a label here the real model was never trained to
produce fails loudly instead of silently logging a ground truth
nothing backs up.
"""

import json
import os

from perception import MODEL_REGISTRY

STATION_RAILS = [1, 2, 3]        # physical grouping only -- no rig geometry here
STATIONS_PER_RAIL = 3            # left / middle / right
STATION_LABELS = ("L", "M", "R")
SIDES = ("front", "back")
PODS_PER_SIDE = 4                # pods sharing one physical trough

IMAGE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "pods_manifest.json")


def trough_id(rail, station, side):
    """The physical trough a pod belongs to -- 18 of these. Useful for
    grouping/reporting (e.g. "how did rail 3's pods do") even without a
    robot moving between them."""
    return f"R{rail}-{STATION_LABELS[station]}-{side[0].upper()}"


def _validate_manifest(entries):
    for e in entries:
        crop = e["crop_type"]
        gt = e["ground_truth"]
        if gt["verified_aspect"] == "growth" and gt["growth_stage"] is not None:
            valid = list(MODEL_REGISTRY[crop]["growth"]["classes"].values())
            assert gt["growth_stage"] in valid, (
                f"{e['pod_id']}: {gt['growth_stage']!r} not a real {crop} growth-stage class: {valid}"
            )
        if gt["disease_name"] is not None:
            valid = list(MODEL_REGISTRY[crop]["disease"]["classes"].values())
            assert gt["disease_name"] in valid, (
                f"{e['pod_id']}: {gt['disease_name']!r} not a real {crop} disease class: {valid}"
            )


def build_pod_registry():
    with open(MANIFEST_PATH) as f:
        entries = json.load(f)
    _validate_manifest(entries)

    pods = []
    for index, e in enumerate(entries):
        rail, station, side = e["rail"], e["station"], e["side"]
        trajectory = None
        if e.get("trajectory"):
            crop = e["crop_type"]
            valid_stages = list(MODEL_REGISTRY[crop]["growth"]["classes"].values())
            trajectory = []
            for step in e["trajectory"]:
                assert step["growth_stage"] in valid_stages, (
                    f"{e['pod_id']}: trajectory stage {step['growth_stage']!r} not a real "
                    f"{crop} growth-stage class: {valid_stages}"
                )
                trajectory.append({
                    "growth_stage": step["growth_stage"],
                    "image_path": os.path.join(IMAGE_DIR, step["image"]),
                })

        pods.append({
            "pod_id": e["pod_id"],
            "index": index,
            "rail": rail,
            "station": station,
            "side": side,
            "trough_id": trough_id(rail, station, side),
            "crop_type": e["crop_type"],
            "ground_truth": e["ground_truth"],
            "image_path": os.path.join(IMAGE_DIR, e["image"]),
            "trajectory": trajectory,  # None for back (disease-verified) pods
        })
    return pods


PODS = build_pod_registry()


def get_pod(pod_id):
    for p in PODS:
        if p["pod_id"] == pod_id:
            return p
    raise KeyError(f"Unknown pod_id: {pod_id}")


def pods_by_crop(crop_type):
    return [p for p in PODS if p["crop_type"] == crop_type]


if __name__ == "__main__":
    print(f"{len(STATION_RAILS)} station rails x {STATIONS_PER_RAIL} stations x 2 sides "
          f"x {PODS_PER_SIDE} pods/side = {len(PODS)} pods "
          f"({len(STATION_RAILS) * STATIONS_PER_RAIL * len(SIDES)} physical troughs)")
    by_crop = {}
    for p in PODS:
        by_crop[p["crop_type"]] = by_crop.get(p["crop_type"], 0) + 1
    print("pods per crop:", by_crop)
    for p in PODS[:6]:
        print(p["pod_id"], p["crop_type"], p["trough_id"], p["ground_truth"])
