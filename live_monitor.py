"""
live_monitor.py

Real-time OpenCV desktop application for OpenCV-Farming-Agent: a live
window that shows the actual detection pipeline running, pod by pod,
as it happens -- not a report generated afterward. Same underlying
pipeline as main.py (camera -> cloud (classify_pod + decide_action) ->
history -> trend -> logger), reused as-is, not reimplemented; this
module's only job is putting a live window and a running dashboard on
top of it.

Run:
    python3 live_monitor.py
    python3 live_monitor.py --crop lettuce --visits 3
    python3 live_monitor.py --delay-ms 800   # slow down for a watchable demo

Controls (window must be focused):
    q / ESC   quit early (still flushes logs + history for everything
              processed so far -- an interrupted run is not a lost run)
    p         pause / resume
    n         step one pod forward while paused
    +/-       speed up / slow down the per-pod delay

Needs a real display (or Xvfb) -- this opens an actual OpenCV window
via cv2.imshow, unlike the rest of this project, which runs headless.
If you're on a server/SSH session with no display, run main.py
--save-images and open the HTML gallery instead.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import perception
from pod_registry import PODS, pods_by_crop
from camera import PodCamera
from cloud_pipeline import LocalCloudClient
from dynamo_client import LocalDynamoTable
from trend import summarize_trend
from run_logger import RunLogger
from visualize import annotate, ACTION_BANNER_COLOR

WINDOW_NAME = "OpenCV-Farming-Agent — Live Monitor"
FRAME_W = 720                # fixed display width; every frame gets scaled to this so the window never jumps around
DASHBOARD_H = 110
BG = (24, 24, 24)
FG = (235, 235, 235)
MUTED = (150, 150, 150)
FEEDBACK_TIMEOUT_S = 15.0


class LiveMonitor:
    def __init__(self, pods, visits=1, delay_ms=600, reset_history=False):
        self.pods = pods
        self.visits = visits
        self.delay_ms = delay_ms
        self.paused = False

        history_path = "dynamo_table.json"
        if reset_history and Path(history_path).exists():
            Path(history_path).unlink()
        self.history = LocalDynamoTable(path=history_path)

        self.logger = RunLogger(out_dir="logs")
        self.cloud = LocalCloudClient()
        self.camera = PodCamera(self.cloud)

        self.counts = {"log_healthy": 0, "flag_for_harvest": 0,
                        "schedule_frequent_monitoring": 0, "flag_for_treatment": 0}
        self.processed = 0
        self.total = len(pods) * visits
        self.start_time = time.time()

    # ---------- one pod-visit ----------

    def _image_for_visit(self, pod, cumulative_step):
        traj = pod.get("trajectory")
        if not traj:
            return pod["image_path"], pod["ground_truth"]["growth_stage"]
        step = min(cumulative_step, len(traj) - 1)
        return traj[step]["image_path"], traj[step]["growth_stage"]

    def process_one(self, pod, visit_time):
        cumulative_step = self.history.visit_count(pod["pod_id"])
        image_path, this_visit_stage_gt = self._image_for_visit(pod, cumulative_step)

        request_id = self.camera.capture_and_submit(pod["pod_id"], pod["crop_type"], image_path)
        record = self.cloud.get_feedback(request_id, block=True, timeout=FEEDBACK_TIMEOUT_S)
        if record["status"] != "done":
            return None

        classification = record["classification"]
        decision = record["action"]
        prior_rows = self.history.query(pod["pod_id"])
        trend_note = summarize_trend(prior_rows, classification)
        cumulative_visit_num = cumulative_step + 1

        self.history.put_item({
            "pod_id": pod["pod_id"], "timestamp": visit_time, "visit": cumulative_visit_num,
            "crop_type": pod["crop_type"], "growth_stage": classification["growth_stage"],
            "growth_confidence": classification["growth_confidence"],
            "disease_flag": classification["disease_flag"], "disease_name": classification["disease_name"],
            "disease_confidence": classification["disease_confidence"], "action": decision["action"],
            "ground_truth_stage_this_visit": this_visit_stage_gt,
        })

        pod_for_log = dict(pod)
        pod_for_log["ground_truth"] = {
            **pod["ground_truth"],
            "growth_stage": this_visit_stage_gt if pod["side"] == "front" else pod["ground_truth"]["growth_stage"],
        }
        row = self.logger.log(pod_for_log, classification, decision)
        row["visit"] = cumulative_visit_num
        row["trend"] = trend_note

        self.counts[decision["action"]] = self.counts.get(decision["action"], 0) + 1
        self.processed += 1

        frame = annotate(image_path, pod, classification, decision, trend_note=trend_note)
        return frame

    # ---------- rendering ----------

    def _scaled(self, frame):
        h, w = frame.shape[:2]
        scale = FRAME_W / w
        return cv2.resize(frame, (FRAME_W, int(h * scale)))

    def _dashboard(self, pod, elapsed_s):
        panel = np.full((DASHBOARD_H, FRAME_W, 3), BG, dtype=np.uint8)
        pct = self.processed / self.total if self.total else 0

        cv2.putText(panel, f"pod-visit {self.processed}/{self.total}  ({pct*100:.0f}%)",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FG, 1, cv2.LINE_AA)
        cv2.putText(panel, f"elapsed {elapsed_s:5.1f}s   avg {elapsed_s / max(1, self.processed):.2f}s/pod",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, MUTED, 1, cv2.LINE_AA)

        bar_w = FRAME_W - 20
        cv2.rectangle(panel, (10, 50), (10 + bar_w, 58), (60, 60, 60), -1)
        cv2.rectangle(panel, (10, 50), (10 + int(bar_w * pct), 58), (90, 200, 90), -1)

        x = 10
        for action, color in ACTION_BANNER_COLOR.items():
            label = f"{action}: {self.counts.get(action, 0)}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(panel, (x, 68), (x + tw + 12, 90), color, -1)
            cv2.putText(panel, label, (x + 6, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            x += tw + 20

        state = "PAUSED (p=resume, n=step)" if self.paused else "q=quit  p=pause  +/-=speed"
        cv2.putText(panel, state, (10, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.42, MUTED, 1, cv2.LINE_AA)
        return panel

    def run(self, show_window=True):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE) if show_window else None
        quit_requested = False

        try:
            for visit_index in range(self.visits):
                visit_time = time.strftime("%Y-%m-%dT%H:%M:%S")
                for pod in self.pods:
                    if quit_requested:
                        break

                    while self.paused:
                        key = cv2.waitKey(50) & 0xFF
                        if key == ord('p'):
                            self.paused = False
                        elif key == ord('n'):
                            break
                        elif key in (ord('q'), 27):
                            quit_requested = True
                            break
                    if quit_requested:
                        break

                    frame = self.process_one(pod, visit_time)
                    if frame is None:
                        continue

                    scaled = self._scaled(frame)
                    dash = self._dashboard(pod, time.time() - self.start_time)
                    canvas = np.vstack([scaled, dash])

                    if show_window:
                        cv2.imshow(WINDOW_NAME, canvas)
                        key = cv2.waitKey(max(1, self.delay_ms)) & 0xFF
                        if key in (ord('q'), 27):
                            quit_requested = True
                        elif key == ord('p'):
                            self.paused = True
                        elif key == ord('+'):
                            self.delay_ms = max(30, int(self.delay_ms * 0.7))
                        elif key == ord('-'):
                            self.delay_ms = int(self.delay_ms * 1.4)
                if quit_requested:
                    break
        finally:
            if show_window:
                cv2.destroyAllWindows()
            self.cloud.shutdown()
            csv_path, json_path = self.logger.flush()
            summary = self.logger.summary()
            print("\n--- run summary ---")
            for k, v in summary.items():
                print(f"{k}: {v}")
            print(f"\nlog written to:\n  {csv_path}\n  {json_path}")
            print(f"persistent per-pod history ({len(self.history.all_pod_ids())} pods tracked): dynamo_table.json")

        return self.logger, self.history


def main():
    parser = argparse.ArgumentParser(description="OpenCV-Farming-Agent live monitor")
    parser.add_argument("--crop", choices=["cabbage", "lettuce", "mushroom"], default=None)
    parser.add_argument("--pods", type=int, default=None)
    parser.add_argument("--visits", type=int, default=1)
    parser.add_argument("--delay-ms", type=int, default=600,
                         help="minimum ms each frame stays on screen (inference itself already takes ~500-800ms)")
    parser.add_argument("--reset-history", action="store_true")
    parser.add_argument("--no-window", action="store_true",
                         help="run the same loop without opening a window (for headless smoke-testing this script itself)")
    args = parser.parse_args()

    pods = pods_by_crop(args.crop) if args.crop else PODS
    if args.pods:
        pods = pods[:args.pods]

    print("warming up models...")
    perception.warm_up()

    monitor = LiveMonitor(pods, visits=args.visits, delay_ms=args.delay_ms, reset_history=args.reset_history)
    monitor.run(show_window=not args.no_window)


if __name__ == "__main__":
    main()
