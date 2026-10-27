"""pod_data.py -- every pod's data set, stored in its own folder inside the simulation.

LAYOUT
    pybullet_sim/pods/<POD_ID>/          one folder per pod; the folder name IS the number printed on the pod's tag
        pod.json                         who/where/what the pod is (see below)
        <photo>.jpg ...                  the pod's real photos (front pods: one per visit of its growth timeline;
                                         back pods: the single disease-verified photo)

    e.g.  pybullet_sim/pods/R1-L-F1/pod.json
          pybullet_sim/pods/R1-L-F1/cabbage_R1-L-F1_v1_S2_3_img146_...jpg   (visit 1)
          ...

WHY IN THE SIMULATION
    The pod's printed tag, its 3-D location in PyBullet and its data set now share one key. The camera agent reads
    the tag (pod_labels.PodLabelReader), and the tag text is the folder name -- so "which pod am I looking at"
    leads straight to "this is its data" with no lookup table in between. The application's own registry
    (pod_registry.py) builds PODS from these same folders, so there is a single copy of the data.

pod.json
    pod_id, tag            "R1-L-F1" (identical; tag is what is physically printed)
    index                  0..71, the application's queue order (trough by trough)
    crop_type, rail, station (0/1/2 = L/M/R), side, slot (1-4), trough_id
    sim_location           {row, col, layer, side, sim_pod_id}: where the trough is in the PyBullet scene
    ground_truth           what the pod's photo is verified for (growth stage OR disease, never both)
    image                  the pod's first / only photo (file name inside this folder)
    trajectory             front pods only: [{visit, growth_stage, image}, ...] -- the 5-visit growth timeline

This module has no PyBullet dependency, so the application can read pod data even when the robot simulation is off.
"""
from __future__ import annotations

import json
import os

POD_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pods")
POD_FILE = "pod.json"
EXPECTED_PODS = 72


def pod_folder(pod_id):
    """The folder holding one pod's data set."""
    return os.path.join(POD_DATA_DIR, str(pod_id))


def load_pod(pod_id):
    """One pod's pod.json as a dict, with 'folder' added. KeyError if there is no such pod."""
    path = os.path.join(pod_folder(pod_id), POD_FILE)
    if not os.path.isfile(path):
        raise KeyError(f"No data folder for pod {pod_id!r} (expected {path})")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["folder"] = pod_folder(pod_id)
    return data


def pod_ids():
    """Every pod that has a data folder, in the application's queue order."""
    if not os.path.isdir(POD_DATA_DIR):
        return []
    found = []
    for name in os.listdir(POD_DATA_DIR):
        if os.path.isfile(os.path.join(POD_DATA_DIR, name, POD_FILE)):
            found.append(name)
    return sorted(found, key=lambda pid: load_pod(pid).get("index", 1 << 30))


def load_all():
    return [load_pod(pid) for pid in pod_ids()]


def image_path(pod, filename):
    """Absolute path of one of a pod's photos."""
    return os.path.join(pod["folder"] if "folder" in pod else pod_folder(pod["pod_id"]), filename)


def validate():
    """Consistency check of the whole data layer. Returns a list of problems (empty = healthy)."""
    try:
        from .pod_labels import label_text, is_valid_label
    except ImportError:                      # run as a script: python pybullet_sim/pod_data.py
        from pod_labels import label_text, is_valid_label
    problems = []
    pods = load_all()
    if len(pods) != EXPECTED_PODS:
        problems.append(f"expected {EXPECTED_PODS} pod folders, found {len(pods)}")
    seen_tags, seen_idx = set(), set()
    for pod in pods:
        pid, folder = pod["pod_id"], pod["folder"]
        if os.path.basename(folder) != pid:
            problems.append(f"{pid}: folder name does not match pod_id")
        if pod.get("tag") != pid or not is_valid_label(pid):
            problems.append(f"{pid}: tag {pod.get('tag')!r} is not this pod's valid tag")
        loc = pod.get("sim_location") or {}
        try:
            if label_text(loc["row"], loc["col"], loc["layer"], loc["side"]) != pid:
                problems.append(f"{pid}: sim_location {loc} would carry a different tag")
        except (KeyError, ValueError, IndexError):
            problems.append(f"{pid}: sim_location is missing or malformed")
        if pid in seen_tags:
            problems.append(f"{pid}: duplicate pod")
        seen_tags.add(pid)
        if pod.get("index") in seen_idx:
            problems.append(f"{pid}: duplicate index {pod.get('index')}")
        seen_idx.add(pod.get("index"))
        files = [pod.get("image")] + [t.get("image") for t in pod.get("trajectory") or []]
        for name in files:
            if not name or not os.path.isfile(os.path.join(folder, name)):
                problems.append(f"{pid}: missing photo {name!r}")
        if pod.get("side") == "front" and len(pod.get("trajectory") or []) != 5:
            problems.append(f"{pid}: a front pod needs the 5-visit growth timeline")
    return problems


if __name__ == "__main__":
    issues = validate()
    print(f"{len(pod_ids())} pod folders in {POD_DATA_DIR}")
    print("OK: data layer is consistent" if not issues else "PROBLEMS:\n  " + "\n  ".join(issues))
    raise SystemExit(1 if issues else 0)
