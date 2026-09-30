"""
main.py

Data-processing pipeline for OpenCV-Farming-Agent, with no PyBullet /
robot simulation -- just the perception + decision pipeline running
against real pod images, now across multiple simulated visits so
growth progress and disease persistence can actually be tracked over
time instead of just single-snapshot.

For every pod, every visit:
  1. camera "captures" (looks up) the pod's image for THIS visit and
     submits it -- it does NOT classify anything itself (see camera.py).
     Front (growth-verified) pods use their trajectory: visit 1 shows
     their earliest curated stage, each later visit swaps in the real
     photo for the next stage along, capping at the last (harvest)
     stage rather than cycling past it. Back (disease-verified) pods
     use the same photo every visit -- see the note below for why
     that's a deliberate choice, not a shortcut.
  2. the cloud pipeline (real classify_pod() + real decide_action(),
     see perception.py / agent.py) processes it; this loop waits for
     that result as feedback (see cloud_pipeline.py)
  3. the result is written to the persistent per-pod history table
     (dynamo_client.py -- a local stand-in for the proposal's DynamoDB
     table, survives across separate runs, not just this one)
  4. a trend note (trend.py) compares this visit against the pod's
     prior history and gets logged alongside the decision
  5. log the full row (run_logger.py); optionally save an annotated
     image + build an HTML gallery (visualize.py / gallery.py)

Why back-side pods don't get a changing photo: this round's ask was
specifically about growth stages progressing (early -> late), and
disease status changing turn to turn would need its own curated
before/after image set the way growth stages got one. A disease-verified
pod re-inspected with no treatment applied SHOULD read the same each
visit -- that's not a flat demo, it's what "diseased, still diseased 3
visits later, still not treated" actually looks like, and trend.py
still surfaces that persistence explicitly rather than silently
repeating the same line with nothing to say about it.

Run modes:
  python3 main.py                       # 1 visit, all 72 pods (same as before)
  python3 main.py --visits 5            # 5 simulated visits, growth trajectories play out
  python3 main.py --crop mushroom --visits 3
  python3 main.py --pods 8 --visits 3   # quick smoke test
  python3 main.py --reset-history       # clear dynamo_table.json before this run
  python3 main.py --save-images         # also save annotated images + an HTML gallery (see visualize.py / gallery.py)
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import perception
from pod_registry import PODS, pods_by_crop, IMAGE_DIR
from camera import PodCamera
from cloud_pipeline import LocalCloudClient
from dynamo_client import LocalDynamoTable
from trend import summarize_trend
from run_logger import RunLogger
from visualize import save_annotated
from gallery import build_gallery

FEEDBACK_TIMEOUT_S = 15.0       # generous margin over the ~1-3.5s a real classify_pod() call takes on CPU
VISIT_INTERVAL = timedelta(days=1)  # simulated spacing between visits, for readable timestamps only


def image_for_visit(pod, cumulative_step):
    """cumulative_step is 0-based and comes from the pod's PERSISTED visit
    count (history.visit_count), not the local loop index -- that's what
    makes growth continue correctly across separate `python3 main.py`
    invocations, not just within one run's --visits N. Front pods
    progress through their curated trajectory, capped at the last
    (harvest) stage rather than cycling past it once reached. Back pods,
    and any front pod with no curated trajectory, use their single
    assigned image every visit."""
    traj = pod.get("trajectory")
    if not traj:
        return pod["image_path"], pod["ground_truth"]["growth_stage"]
    step = min(cumulative_step, len(traj) - 1)
    return traj[step]["image_path"], traj[step]["growth_stage"]


def run(crop=None, max_pods=None, visits=1, reset_history=False, save_images=False):
    pods = pods_by_crop(crop) if crop else PODS
    if max_pods:
        pods = pods[:max_pods]

    history_path = "dynamo_table.json"
    if reset_history and os.path.exists(history_path):
        os.remove(history_path)
    history = LocalDynamoTable(path=history_path)

    print("warming up models (loads all 6 .onnx files once, so per-pod timing "
          "below reflects real inference speed, not first-call model-load cost)...")
    perception.warm_up()

    logger = RunLogger(out_dir="logs")
    cloud = LocalCloudClient()
    camera = PodCamera(cloud)

    images_dir = Path("logs/annotated")
    gallery_entries = []
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== OpenCV-Farming-Agent | data-processing pipeline | "
          f"{len(pods)} pod(s) x {visits} visit(s) ===\n")

    base_time = datetime.now(timezone.utc)

    for visit_index in range(visits):
        visit_num = visit_index + 1
        visit_time = base_time + visit_index * VISIT_INTERVAL
        print(f"\n--- Visit {visit_num}/{visits} ({visit_time.date()}) ---\n")

        for pod in pods:
            cumulative_step = history.visit_count(pod["pod_id"])  # persisted, survives across runs
            image_path, this_visit_stage_gt = image_for_visit(pod, cumulative_step)

            request_id = camera.capture_and_submit(pod["pod_id"], pod["crop_type"], image_path)
            record = cloud.get_feedback(request_id, block=True, timeout=FEEDBACK_TIMEOUT_S)
            if record["status"] != "done":
                print(f"[{pod['pod_id']:>9}] WARNING: no feedback within {FEEDBACK_TIMEOUT_S}s, skipping")
                continue

            classification = record["classification"]
            decision = record["action"]

            prior_rows = history.query(pod["pod_id"])
            trend_note = summarize_trend(prior_rows, classification)
            cumulative_visit_num = cumulative_step + 1

            history.put_item({
                "pod_id": pod["pod_id"],
                "timestamp": visit_time.isoformat(),
                "visit": cumulative_visit_num,
                "crop_type": pod["crop_type"],
                "growth_stage": classification["growth_stage"],
                "growth_confidence": classification["growth_confidence"],
                "disease_flag": classification["disease_flag"],
                "disease_name": classification["disease_name"],
                "disease_confidence": classification["disease_confidence"],
                "action": decision["action"],
                "ground_truth_stage_this_visit": this_visit_stage_gt,
            })

            pod_for_log = dict(pod)  # ground_truth reflects THIS visit's image, not just visit 1
            pod_for_log["ground_truth"] = {
                **pod["ground_truth"],
                "growth_stage": this_visit_stage_gt if pod["side"] == "front" else pod["ground_truth"]["growth_stage"],
            }
            row = logger.log(pod_for_log, classification, decision)
            row["visit"] = cumulative_visit_num
            row["trend"] = trend_note

            if save_images:
                img_filename = f"{pod['pod_id']}_v{cumulative_visit_num}.jpg"
                save_annotated(
                    image_path, pod, classification, decision,
                    images_dir / img_filename, trend_note=trend_note,
                )
                gallery_entries.append({
                    "pod_id": pod["pod_id"], "crop_type": pod["crop_type"],
                    "growth_stage": classification["growth_stage"],
                    "disease_flag": classification["disease_flag"],
                    "disease_name": classification["disease_name"],
                    "action": decision["action"], "trend": trend_note,
                    "image_filename": img_filename,
                })

            disease_bit = f"disease={classification['disease_name']}" if classification["disease_flag"] else "disease=None"
            print(
                f"[v{cumulative_visit_num} {pod['pod_id']:>9}] {pod['crop_type']:<9} "
                f"saw stage='{classification['growth_stage']}' "
                f"{disease_bit} "
                f"(g_conf={classification['growth_confidence']:.2f} d_conf={classification['disease_confidence']:.2f})  "
                f"-> {decision['action']:<28} | {trend_note}"
            )

    csv_path, json_path = logger.flush()
    summary = logger.summary()

    print("\n--- run summary ---")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nper-visit log written to:\n  {csv_path}\n  {json_path}")
    print(f"persistent per-pod history ({len(history.all_pod_ids())} pods tracked): {history_path}")

    if save_images:
        gallery_path = build_gallery(
            gallery_entries, out_path="logs/gallery.html", images_dir_name="annotated",
            subtitle=f"{len(gallery_entries)} pod-visit(s) — growth-stage boxes in cyan, disease boxes in orange-red",
        )
        print(f"\nannotated images: {images_dir}/ ({len(gallery_entries)} images)")
        print(f"open in a browser: {gallery_path}")

    return logger, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenCV-Farming-Agent data-processing pipeline")
    parser.add_argument("--crop", choices=["cabbage", "lettuce", "mushroom"], default=None,
                         help="only process one crop's 24 pods")
    parser.add_argument("--pods", type=int, default=None, help="only process the first N pods")
    parser.add_argument("--visits", type=int, default=1,
                         help="simulate this many visits per pod, progressing front pods' growth trajectory each time")
    parser.add_argument("--reset-history", action="store_true",
                         help="clear dynamo_table.json before this run instead of appending to it")
    parser.add_argument("--save-images", action="store_true",
                         help="save annotated images (detection boxes + decision banner) per pod-visit, plus an HTML gallery")
    args = parser.parse_args()

    run(crop=args.crop, max_pods=args.pods, visits=args.visits, reset_history=args.reset_history,
        save_images=args.save_images)
