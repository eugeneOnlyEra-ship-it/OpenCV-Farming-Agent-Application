"""
run_logger.py

Writes one row per pod visit: timestamp, pod id, crop, growth stage,
disease flag/name, confidences, and the decision that was made. This is
the artifact that (a) later feeds the DynamoDB per-pod health history
table and the report's evaluation section, and (b) is the direct
evidence for the Agentic Vision Award -- a reviewer can open this file
and see "saw X -> decided Y" for every stop in the run, without needing
to watch the GUI live.

Writes both CSV (for spreadsheet/report use) and JSON (for programmatic
use, e.g. later DynamoDB batch-loading) from the same in-memory rows.

Ground truth is only verified for ONE aspect per pod (see
pod_registry.py's module docstring -- a real photo is sourced from
either the growth-stage dataset or the disease dataset, not both), so
summary() scores growth-verified and disease-verified pods separately
rather than requiring an impossible simultaneous match on both.
"""

import csv
import json
import os
from datetime import datetime, timezone

FIELDNAMES = [
    "timestamp", "visit", "pod_id", "rail", "station", "side", "crop_type",
    "growth_stage", "growth_confidence",
    "disease_flag", "disease_name", "disease_confidence", "confidence",
    "action", "reason", "trend",
    "verified_aspect", "ground_truth_stage", "ground_truth_diseased", "ground_truth_disease_name",
]


class RunLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.csv_path = os.path.join(out_dir, f"run_{run_id}.csv")
        self.json_path = os.path.join(out_dir, f"run_{run_id}.json")
        self.rows = []

    def log(self, pod, classification, decision):
        gt = pod["ground_truth"]
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pod_id": pod["pod_id"],
            "rail": pod["rail"],
            "station": pod["station"],
            "side": pod["side"],
            "crop_type": pod["crop_type"],
            "growth_stage": classification["growth_stage"],
            "growth_confidence": classification["growth_confidence"],
            "disease_flag": classification["disease_flag"],
            "disease_name": classification["disease_name"],
            "disease_confidence": classification["disease_confidence"],
            "confidence": classification["confidence"],
            "action": decision["action"],
            "reason": decision["reason"],
            "verified_aspect": gt["verified_aspect"],
            "ground_truth_stage": gt["growth_stage"],
            "ground_truth_diseased": gt["diseased"],
            "ground_truth_disease_name": gt["disease_name"],
        }
        self.rows.append(row)
        return row

    def flush(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.rows)
        with open(self.json_path, "w") as f:
            json.dump(self.rows, f, indent=2)
        return self.csv_path, self.json_path

    def summary(self):
        total = len(self.rows)
        by_action = {}
        for r in self.rows:
            by_action[r["action"]] = by_action.get(r["action"], 0) + 1

        growth_rows = [r for r in self.rows if r["verified_aspect"] == "growth" and r["ground_truth_stage"] is not None]
        disease_rows = [r for r in self.rows if r["verified_aspect"] == "disease"]

        growth_correct = sum(1 for r in growth_rows if r["growth_stage"] == r["ground_truth_stage"])
        disease_correct = sum(1 for r in disease_rows if r["disease_flag"] == r["ground_truth_diseased"])

        return {
            "total_pods_visited": total,
            "actions": by_action,
            "growth_stage_accuracy": f"{growth_correct}/{len(growth_rows)}" if growth_rows else "n/a",
            "disease_flag_accuracy": f"{disease_correct}/{len(disease_rows)}" if disease_rows else "n/a",
        }
