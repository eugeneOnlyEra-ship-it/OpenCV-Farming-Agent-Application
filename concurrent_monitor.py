"""
concurrent_monitor.py

Real-time OpenCV desktop app that processes MULTIPLE pods truly
simultaneously — pod 1 advancing through its image 1 -> 2 -> 3 growth
sequence at the same time pod 2 advances through its own, shown as a
grid of live panels (this is what the original sketch asked for: pods
side by side, each with its own image + growth/disease scores).

Architecture, and why it's shaped this way:

  - A fixed number of SLOTS (--slots, default 4) are shown on screen.
    Each slot is a worker thread that pulls the next pod off a shared
    queue, runs that pod's full image sequence to completion (front/
    growth-verified pods: their whole trajectory, early stage through
    harvest; back/disease-verified pods: their one image), then pulls
    the next pod. This is a fixed-lane dashboard, not "launch 72
    threads at once" -- with 72 pods and a handful of crops, most pods
    share a model, and unbounded concurrency was measured to make
    things slower, not faster (see cloud_pipeline.py).

  - Within one pod's sequence, images are processed IN ORDER -- growth
    is chronological, image 2 has to follow image 1 -- but different
    SLOTS (different pods) run genuinely concurrently, each blocking
    only on its own current step, via the same bounded, thread-safe
    LocalCloudClient main.py and live_monitor.py use. Nothing here
    reimplements classification or decision-making; it's the same
    camera.py -> cloud_pipeline.py -> perception.py/agent.py chain,
    just with multiple pods in flight through it at once instead of one
    at a time.

  - Every step still writes to the same persistent history
    (dynamo_client.py) and gets the same trend note (trend.py) as
    main.py/live_monitor.py -- this is a third way to advance a pod's
    history, not a separate concept from it. Run this, then run
    main.py later, and pods pick up exactly where this left off, same
    as running main.py multiple times does.

Run:
    python3 concurrent_monitor.py                    # all 72 pods, 4 slots
    python3 concurrent_monitor.py --slots 6           # 6 pods in flight at once
    python3 concurrent_monitor.py --crop mushroom
    python3 concurrent_monitor.py --pods 12 --slots 3
    python3 concurrent_monitor.py --no-window         # headless smoke test

Controls: q/Esc quit (finishes in-flight steps, then flushes and exits
cleanly -- no partial/corrupt log rows), p pause new pod pickup (lets
in-flight pods finish their current step without starting new pods).

Needs a real display, same as live_monitor.py -- use --no-window on a
headless machine.
"""

import argparse
import queue
import threading
import time

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

PANEL_W = 340
FEEDBACK_TIMEOUT_S = 15.0
BG = (24, 24, 24)
FG = (235, 235, 235)
MUTED = (150, 150, 150)
IDLE_LABEL_COLOR = (90, 90, 90)


class Slot:
    """One visual lane: whatever pod it's currently working on, and the
    latest rendered frame for that pod, guarded by its own lock so the
    render loop (main thread) and the worker thread (this slot's own)
    never touch the same frame mid-write."""

    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.lock = threading.Lock()
        self.current_pod_id = None
        self.frame = None          # latest annotated (unscaled) frame, or None if idle
        self.status_text = "waiting for a pod..."
        self.pods_completed = 0

    def set_frame(self, pod_id, frame, status_text):
        with self.lock:
            self.current_pod_id = pod_id
            self.frame = frame
            self.status_text = status_text

    def snapshot(self):
        with self.lock:
            return self.current_pod_id, self.frame, self.status_text


class ConcurrentMonitor:
    def __init__(self, pods, n_slots=4, delay_ms_per_step=150, reset_history=False, max_concurrency=None):
        self.pods = pods
        self.n_slots = n_slots
        self.delay_ms_per_step = delay_ms_per_step
        self.paused = False
        self.quit_requested = False

        history_path = "dynamo_table.json"
        if reset_history:
            import os
            if os.path.exists(history_path):
                os.remove(history_path)
        self.history = LocalDynamoTable(path=history_path)

        self.logger = RunLogger(out_dir="logs")
        self.logger_lock = threading.Lock()  # RunLogger.rows isn't thread-safe on its own
        self.cloud = LocalCloudClient(max_concurrency=max_concurrency or n_slots)
        self.camera = PodCamera(self.cloud)

        self.pod_queue = queue.Queue()
        for pod in pods:
            self.pod_queue.put(pod)

        self.slots = [Slot(i) for i in range(n_slots)]
        self.action_counts = {a: 0 for a in ACTION_BANNER_COLOR}
        self.counts_lock = threading.Lock()
        self.total_pods = len(pods)
        self.pods_started = 0
        self.pods_done = 0
        self.start_time = time.time()

    # ---------- worker (one per slot) ----------

    def _image_for_step(self, pod, step):
        traj = pod.get("trajectory")
        if not traj:
            return pod["image_path"], pod["ground_truth"]["growth_stage"]
        step = min(step, len(traj) - 1)
        return traj[step]["image_path"], traj[step]["growth_stage"]

    def _n_steps(self, pod):
        traj = pod.get("trajectory")
        return len(traj) if traj else 1

    def _process_step(self, pod, step):
        image_path, gt_stage = self._image_for_step(pod, step)
        request_id = self.camera.capture_and_submit(pod["pod_id"], pod["crop_type"], image_path)
        record = self.cloud.get_feedback(request_id, block=True, timeout=FEEDBACK_TIMEOUT_S)
        if record["status"] != "done":
            return None

        classification = record["classification"]
        decision = record["action"]
        prior_rows = self.history.query(pod["pod_id"])
        trend_note = summarize_trend(prior_rows, classification)
        visit_num = step + 1

        self.history.put_item({
            "pod_id": pod["pod_id"], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "visit": visit_num, "crop_type": pod["crop_type"],
            "growth_stage": classification["growth_stage"],
            "growth_confidence": classification["growth_confidence"],
            "disease_flag": classification["disease_flag"], "disease_name": classification["disease_name"],
            "disease_confidence": classification["disease_confidence"], "action": decision["action"],
            "ground_truth_stage_this_visit": gt_stage,
        })

        pod_for_log = dict(pod)
        pod_for_log["ground_truth"] = {
            **pod["ground_truth"],
            "growth_stage": gt_stage if pod["side"] == "front" else pod["ground_truth"]["growth_stage"],
        }
        with self.logger_lock:
            row = self.logger.log(pod_for_log, classification, decision)
            row["visit"] = visit_num
            row["trend"] = trend_note

        with self.counts_lock:
            self.action_counts[decision["action"]] = self.action_counts.get(decision["action"], 0) + 1

        frame = annotate(image_path, pod, classification, decision, trend_note=trend_note)
        return frame, decision["action"], trend_note

    def _slot_worker(self, slot):
        while not self.quit_requested:
            if self.paused:
                time.sleep(0.1)
                continue
            try:
                pod = self.pod_queue.get_nowait()
            except queue.Empty:
                slot.set_frame(None, None, "all pods done" if self.pods_done >= self.total_pods else "waiting...")
                time.sleep(0.2)
                continue

            self.pods_started += 1
            n_steps = self._n_steps(pod)
            start_step = self.history.visit_count(pod["pod_id"])

            for step in range(start_step, n_steps):
                if self.quit_requested:
                    break
                result = self._process_step(pod, step)
                if result is None:
                    continue
                frame, action, trend_note = result
                status = f"{pod['pod_id']}  step {step + 1}/{n_steps}  -> {action}"
                slot.set_frame(pod["pod_id"], frame, status)
                time.sleep(self.delay_ms_per_step / 1000.0)

            slot.pods_completed += 1
            self.pods_done += 1

    # ---------- rendering ----------

    def _panel_for_slot(self, slot):
        pod_id, frame, status_text = slot.snapshot()
        if frame is None:
            panel = np.full((int(PANEL_W * 1.1), PANEL_W, 3), BG, dtype=np.uint8)
            cv2.putText(panel, f"slot {slot.slot_id}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, MUTED, 1, cv2.LINE_AA)
            cv2.putText(panel, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, IDLE_LABEL_COLOR, 1, cv2.LINE_AA)
            return panel

        h, w = frame.shape[:2]
        scale = PANEL_W / w
        scaled = cv2.resize(frame, (PANEL_W, int(h * scale)))
        strip = np.full((26, PANEL_W, 3), (50, 50, 50), dtype=np.uint8)
        cv2.putText(strip, f"slot {slot.slot_id}: {status_text}"[:60], (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, FG, 1, cv2.LINE_AA)
        return np.vstack([strip, scaled])

    def _dashboard_row(self, total_w):
        panel = np.full((70, total_w, 3), BG, dtype=np.uint8)
        elapsed = time.time() - self.start_time
        pct = self.pods_done / self.total_pods if self.total_pods else 0
        cv2.putText(panel, f"pods done {self.pods_done}/{self.total_pods} ({pct*100:.0f}%)   elapsed {elapsed:.1f}s",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FG, 1, cv2.LINE_AA)
        bar_w = total_w - 20
        cv2.rectangle(panel, (10, 30), (10 + bar_w, 38), (60, 60, 60), -1)
        cv2.rectangle(panel, (10, 30), (10 + int(bar_w * pct), 38), (90, 200, 90), -1)

        x = 10
        with self.counts_lock:
            counts = dict(self.action_counts)
        for action, color in ACTION_BANNER_COLOR.items():
            label = f"{action}: {counts.get(action, 0)}"
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(panel, (x, 46), (x + tw + 10, 66), color, -1)
            cv2.putText(panel, label, (x + 5, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            x += tw + 16
        return panel

    def _compose_grid(self):
        panels = [self._panel_for_slot(s) for s in self.slots]
        max_h = max(p.shape[0] for p in panels)
        panels = [cv2.copyMakeBorder(p, 0, max_h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=BG) for p in panels]
        row = np.hstack(panels)
        dash = self._dashboard_row(row.shape[1])
        return np.vstack([row, dash])

    # ---------- run ----------

    def run(self, show_window=True):
        threads = [threading.Thread(target=self._slot_worker, args=(s,), daemon=True) for s in self.slots]
        for t in threads:
            t.start()

        window = "OpenCV-Farming-Agent — Concurrent Monitor"
        if show_window:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

        try:
            while self.pods_done < self.total_pods and not self.quit_requested:
                if show_window:
                    canvas = self._compose_grid()
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(80) & 0xFF
                    if key in (ord('q'), 27):
                        self.quit_requested = True
                    elif key == ord('p'):
                        self.paused = not self.paused
                else:
                    time.sleep(0.1)

            if show_window and not self.quit_requested:
                # one last frame so the final state (e.g. "all pods done") is visible briefly
                cv2.imshow(window, self._compose_grid())
                cv2.waitKey(500)
        finally:
            self.quit_requested = True
            for t in threads:
                t.join(timeout=FEEDBACK_TIMEOUT_S + 1)
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
    parser = argparse.ArgumentParser(description="OpenCV-Farming-Agent concurrent multi-pod monitor")
    parser.add_argument("--crop", choices=["cabbage", "lettuce", "mushroom"], default=None)
    parser.add_argument("--pods", type=int, default=None, help="only process the first N pods")
    parser.add_argument("--slots", type=int, default=4, help="how many pods run simultaneously")
    parser.add_argument("--delay-ms", type=int, default=150,
                         help="extra pause after each image so a fast run is still watchable (inference itself already takes ~500-800ms)")
    parser.add_argument("--reset-history", action="store_true")
    parser.add_argument("--no-window", action="store_true", help="run headlessly (for servers / smoke tests)")
    args = parser.parse_args()

    pods = pods_by_crop(args.crop) if args.crop else PODS
    if args.pods:
        pods = pods[:args.pods]

    print(f"warming up models ({args.slots} slot(s) will each pre-warm their own worker on first use)...")
    perception.warm_up()

    monitor = ConcurrentMonitor(
        pods, n_slots=args.slots, delay_ms_per_step=args.delay_ms,
        reset_history=args.reset_history, max_concurrency=args.slots,
    )
    monitor.run(show_window=not args.no_window)


if __name__ == "__main__":
    main()
