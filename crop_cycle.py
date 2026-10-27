"""
crop_cycle.py  --  a realistic life cycle for every growing pod.

PROBLEM THIS SOLVES
    pod_registry.py gave every front pod a "staircase": one photo per stage from where it started up to harvest, so
    every visit jumped a whole stage and EVERY pod of a crop finished its run at harvest (all lettuce / all
    mushrooms harvested together, a third of the pods even started at harvest).

THE MODEL
    * AGE.  Each pod has a crop age counted in visits. Age 0 is a freshly planted crop.
    * ORDERED STAGES THAT LINGER.  A crop spends several visits in each stage before moving on (STAGE_VISITS).
      While it lingers, the pod's photo of that stage simply repeats -- natural progress, stalled.
    * HARVEST WAITS.  When the crop reaches its harvest stage its age stops advancing: a ripe crop stays ripe
      (the same harvest-stage photo keeps coming back) until the robots harvest it AND the REPLANT job finishes.
      Then the age resets to 0 and the next photo is stage 1 of a new crop.
    * STAGGERED.  Pods start at different ages (spread evenly across the cycle, neighbours deliberately
      different), so a monitoring run only brings SOME pods to harvest, never all of a crop at once.
    * PERSISTENT.  Ages are saved (dynamo_table_cycle.json), so the next run continues where this one stopped.

    Durations are RELATIVE and compressed (one visit = one monitoring pass): cabbage is the slowest crop,
    mushrooms the fastest. They are plain numbers in STAGE_VISITS: edit them freely.

    * VARIETY.  Each pod owns a curated set of real photos (pybullet_sim/pods/<POD_ID>/). Where it has several photos
      of one stage (lettuce / mushroom keep extra, different harvest-stage photos), successive visits in that stage
      walk through them instead of repeating one.

    Limitation (simulation only): after a replant the pod shows the same pod's stage photos again, i.e. the
    "new" plant looks like the previous one. There are only a handful of real photos per pod to work with.

No UI, no robot code: farm_engine.py asks this module which photo a pod shows next.
"""

import json
import os
import threading

from pod_registry import PODS

# Visits spent in each stage BEFORE the next one. The final (harvest) stage has no entry: the crop waits there.
STAGE_VISITS = {
    "cabbage": {"S2_3": 5, "S4_5": 5, "S6_7": 5, "S8": 3},                                       # harvest at age 18
    "lettuce": {"Stage 01- Early Growth": 4, "stage 02- Leafy growth": 4, "stage 03- Head formation": 4},   # age 12
    "mushroom": {"Juvenile": 3, "Intermediate": 5},                                              # age 8
}
# Stage names in growth order (planting -> harvest). Explicit rather than derived from the registry's photo labels,
# because of the mushroom labelling issue described below.
STAGE_ORDER = {
    "cabbage": ["S2_3", "S4_5", "S6_7", "S8", "S9"],
    "lettuce": ["Stage 01- Early Growth", "stage 02- Leafy growth", "stage 03- Head formation", "stage 04- Harvest stage"],
    "mushroom": ["Juvenile", "Intermediate", "Harvest"],
}

# MUSHROOM GROUND-TRUTH LABELS LOOK SWAPPED  (evidence, not certainty -- please confirm with your own eyes)
#   * The mushroom model declares class 0 = Harvest, 2 = Juvenile in its own ONNX metadata (perception.py matches it).
#   * On the project's labelled photos it says "Harvest" for 4/4 photos labelled Juvenile and "Juvenile" for 10/12 photos
#     labelled Harvest; cabbage and lettuce models agree with their labels ~100%.
#   * Looking at the photos: those labelled Juvenile show large, fully open caps (mature); most photos labelled Harvest
#     are long thin stems with small caps (young).
#   Model metadata + model behaviour + appearance all point the same way, so the pods_manifest.json ground-truth labels
#   for mushroom growth stage are most likely Juvenile<->Harvest swapped. perception.py is left untouched; the swap is
#   corrected HERE (labels only, nothing is renamed on disk). Set APPLY_LABEL_CORRECTIONS = False to use the labels as
#   shipped. Legacy scripts (main.py etc.) still read the uncorrected manifest.
LABEL_CORRECTIONS = {("mushroom", "Juvenile"): "Harvest", ("mushroom", "Harvest"): "Juvenile"}
APPLY_LABEL_CORRECTIONS = True


def corrected(crop, stage):
    """The manifest's stage label for a photo, corrected if that is switched on."""
    return LABEL_CORRECTIONS.get((crop, stage), stage) if APPLY_LABEL_CORRECTIONS else stage


STRIDE = 5          # spreads initial ages so pods in the same trough are in different stages (coprime with 12)


def has_cycle(pod):
    """Front pods (growth-verified, with a photo per stage) follow a life cycle; back pods are single disease photos."""
    return pod["side"] == "front" and bool(pod.get("trajectory"))


def _front_pods(crop):
    return [p for p in PODS if p["crop_type"] == crop and has_cycle(p)]


def stage_order(crop):
    """Stages from planting to harvest."""
    return STAGE_ORDER[crop]


def harvest_age(crop):
    return sum(STAGE_VISITS[crop].values())


def _check():
    for crop, durations in STAGE_VISITS.items():
        order = stage_order(crop)
        assert list(durations) == order[:-1], f"{crop}: STAGE_VISITS must list every stage before harvest, in order: {order}"


_check()


def stage_at(crop, age):
    """-> (stage, first_age_of_that_stage). Ages at or beyond the harvest age are all the harvest stage."""
    start = 0
    for stage, visits in STAGE_VISITS[crop].items():
        if age < start + visits:
            return stage, start
        start += visits
    return stage_order(crop)[-1], start


def _build_photos():
    """-> (photos, stage_photos).
    stage_photos[pod_id][stage] = ALL of that pod's photos of the stage, in the order of its growth timeline
    (a pod's own curated set: the data pipeline gives every front pod a real photo for every stage, and extra
    different photos for the stages a faster crop has to wait in, e.g. lettuce/mushroom at harvest).
    photos[pod_id][stage] = the first of those (one path per stage).
    A stage a pod has no photo for (only possible with a hand-edited pod.json) is filled from the crop's real photos
    of that stage, spread across pods."""
    pool, own = {}, {}
    for p in PODS:
        if not has_cycle(p):
            continue
        crop = p["crop_type"]
        own[p["pod_id"]] = {}
        for t in p["trajectory"]:
            own[p["pod_id"]].setdefault(corrected(crop, t["growth_stage"]), []).append(t["image_path"])
        own[p["pod_id"]].setdefault(corrected(crop, p["ground_truth"]["growth_stage"]), [p["image_path"]])
        for stage, paths in own[p["pod_id"]].items():
            pool.setdefault((crop, stage), set()).update(paths)
    for crop in STAGE_VISITS:
        for k, p in enumerate(_front_pods(crop)):
            for stage in stage_order(crop):
                if stage not in own[p["pod_id"]]:
                    photos = sorted(pool[(crop, stage)])
                    own[p["pod_id"]][stage] = [photos[k % len(photos)]]
    return {pid: {st: paths[0] for st, paths in d.items()} for pid, d in own.items()}, own


PHOTOS, STAGE_PHOTOS = _build_photos()


def _initial_ages():
    ages = {}
    for crop in STAGE_VISITS:
        pods, H = _front_pods(crop), harvest_age(crop)
        n = len(pods)
        spread = [round(i * H / (n - 1)) for i in range(n)] if n > 1 else [0]
        for k, p in enumerate(pods):
            ages[p["pod_id"]] = spread[(k * STRIDE) % n]
    return ages


INITIAL_AGE = _initial_ages()


class CycleStore:
    """Each pod's crop age (and how many crops it has grown). path=None keeps it in memory only."""

    def __init__(self, path=None):
        self.path, self._lock, self._data = path, threading.Lock(), {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self._data = json.load(f)
            except (OSError, ValueError):
                self._data = {}

    def _entry(self, pod_id):
        return self._data.setdefault(pod_id, {"age": INITIAL_AGE[pod_id], "cycles": 0})

    def _save(self):
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f, indent=1)
            os.replace(tmp, self.path)

    def info(self, pod):
        """Where this pod's crop is right now."""
        with self._lock:
            e = self._entry(pod["pod_id"])
            crop, age = pod["crop_type"], e["age"]
            stage, start = stage_at(crop, age)
            H = harvest_age(crop)
            # visits already spent in this stage; a ripe crop (age frozen at H) counts its waiting visits instead
            in_stage = e.get("waited", 0) if age >= H else age - start
            return {"age": age, "harvest_age": H, "stage": stage, "waiting_for_harvest": age >= H,
                    "cycle": e["cycles"] + 1, "stage_visit": in_stage}

    def image(self, pod):
        """-> (photo path, ground-truth stage of that photo, info) for this pod's NEXT visit.
        When the pod has several photos of the current stage, each visit in that stage shows the next one (never the
        same photo twice in a row), so a lingering or waiting crop still looks like a real, changing plant."""
        info = self.info(pod)
        photos = STAGE_PHOTOS[pod["pod_id"]][info["stage"]]
        return photos[info["stage_visit"] % len(photos)], info["stage"], info

    def advance(self, pod):
        """One visit has happened. The crop ages, except a ripe crop which keeps waiting for the harvest."""
        with self._lock:
            e = self._entry(pod["pod_id"])
            if e["age"] < harvest_age(pod["crop_type"]):
                e["age"] += 1
            else:
                e["waited"] = e.get("waited", 0) + 1       # ripe and still waiting: next visit shows a different photo
            self._save()

    def reset(self, pod_id):
        """Replanting finished: a new crop at age 0."""
        with self._lock:
            e = self._entry(pod_id)
            e["age"], e["cycles"], e["waited"] = 0, e["cycles"] + 1, 0
            self._save()

    def clear(self):
        with self._lock:
            self._data = {}
            if self.path and os.path.exists(self.path):
                os.remove(self.path)
