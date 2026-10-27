#!/usr/bin/env python3
"""Build pybullet_sim/pods/<POD_ID>/ from the OpenCV-Farming-Agent-DataPipeline package.

    python pybullet_sim/tools/migrate_pod_folders.py /path/to/OpenCV-Farming-Agent-DataPipeline

Reads the pipeline's pods_manifest.json and sample_images/, and for every pod creates one folder holding that pod's
photos and a pod.json (see pod_data.py for the layout). Photos are COPIED with their original file names (they carry
the dataset provenance); the pipeline folder is left untouched. Safe to re-run: existing pod folders are rebuilt.

Pod order (the `index` field) follows the application's queue order -- trough by trough, front slots then back slots --
not the pipeline manifest's order (which lists all front pods first), so queue behaviour does not change.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from pybullet_sim import pod_data                                  # noqa: E402
from pybullet_sim.pod_labels import label_text, STATION_LETTERS    # noqa: E402


def _order_key(entry):
    return entry["rail"], entry["station"], 0 if entry["side"] == "front" else 1, _slot(entry["pod_id"])


def _slot(pod_id):
    return int(pod_id.rsplit("-", 1)[-1][1:])


def migrate(pipeline_dir, out_dir=None, verbose=True):
    out_dir = out_dir or pod_data.POD_DATA_DIR
    with open(os.path.join(pipeline_dir, "pods_manifest.json"), encoding="utf-8") as f:
        entries = json.load(f)
    src_images = os.path.join(pipeline_dir, "sample_images")
    entries = sorted(entries, key=_order_key)
    copied = 0
    for index, e in enumerate(entries):
        pid = e["pod_id"]
        slot = _slot(pid)
        row, col, layer, side = e["rail"] - 1, e["station"], slot - 1, e["side"]
        assert label_text(row, col, layer, side) == pid, f"{pid}: naming scheme does not match the tag format"
        folder = os.path.join(out_dir, pid)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        files = [e["image"]] + [t["image"] for t in e.get("trajectory") or []]
        for name in dict.fromkeys(files):                          # each file once, order preserved
            shutil.copy2(os.path.join(src_images, name), os.path.join(folder, name))
            copied += 1
        record = {
            "pod_id": pid,
            "tag": pid,
            "index": index,
            "crop_type": e["crop_type"],
            "rail": e["rail"],
            "station": e["station"],
            "station_letter": STATION_LETTERS[e["station"]],
            "side": side,
            "slot": slot,
            "trough_id": f"R{e['rail']}-{STATION_LETTERS[e['station']]}-{side[0].upper()}",
            "sim_location": {"row": row, "col": col, "layer": layer, "side": side,
                             "sim_pod_id": f"R{row}C{col}-L{layer}-{side[0].upper()}"},
            "ground_truth": e["ground_truth"],
            "image": e["image"],
        }
        if e.get("trajectory"):
            record["trajectory"] = [{"visit": i + 1, "growth_stage": t["growth_stage"], "image": t["image"]}
                                    for i, t in enumerate(e["trajectory"])]
        with open(os.path.join(folder, pod_data.POD_FILE), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
            f.write("\n")
    if verbose:
        print(f"wrote {len(entries)} pod folders ({copied} photos) to {out_dir}")
    return len(entries), copied


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline_dir", help="the OpenCV-Farming-Agent-DataPipeline folder")
    ap.add_argument("--out", default=None, help="output folder (default: pybullet_sim/pods)")
    args = ap.parse_args()
    migrate(args.pipeline_dir, args.out)
    problems = pod_data.validate() if args.out is None else []
    print("OK: data layer is consistent" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(1 if problems else 0)
