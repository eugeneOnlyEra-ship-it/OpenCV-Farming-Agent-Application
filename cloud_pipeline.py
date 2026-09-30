"""
Cloud pipeline — the AWS side of the loop.

Simulates the round-trip described in the proposal's AWS architecture table
(S3 upload -> Lambda/EC2 inference -> DynamoDB log -> IoT Core feedback)
without needing real AWS resources wired up yet. The robot/camera side only
ever calls submit_frame() and get_feedback() — it never imports perception.py
or agent.py directly, because in the real architecture those run in the
cloud, not on the robot.

This is a LOCAL SIMULATION, not a real AWS client. When you're ready to
point this at actual AWS resources, swap LocalCloudClient for a class with
the same two methods (submit_frame, get_feedback) that does:
    submit_frame -> boto3 s3.upload_fileobj(...), the Lambda trigger does the
                     rest (classify_pod + decide_action run inside the Lambda,
                     same perception.py / agent.py code, unchanged)
    get_feedback -> read the result back from DynamoDB (or an IoT Core
                     shadow/message), keyed by the same request_id
The robot-side code in camera.py doesn't change either way — that's the
point of keeping this behind one small interface.
"""

import random
import threading
import time
import uuid

from perception import classify_pod
from agent import decide_action


class LocalCloudClient:
    def __init__(self, simulated_latency_range=(0.05, 0.3)):
        """simulated_latency_range: standing in for real-world S3 upload +
        Lambda cold-start + inference time, in seconds. Tune these once you
        have real numbers from an actual deployment; until then they just
        need to be nonzero so the async behavior (robot doesn't block
        waiting) actually gets exercised rather than resolving instantly."""
        self._results = {}
        self._lock = threading.Lock()
        self._latency_range = simulated_latency_range
        self._log = []  # completed records, standing in for a DynamoDB table

    def submit_frame(self, pod_id, crop_type, image):
        """Robot/camera side calls this after capturing a frame. Returns a
        request_id immediately without waiting for classification — this is
        the 'camera just collects images' half of the split. Processing
        happens on a background thread, standing in for the Lambda
        invocation actually happening off-robot, in the cloud."""
        request_id = str(uuid.uuid4())
        submitted_at = time.time()
        with self._lock:
            self._results[request_id] = {
                "status": "processing",
                "pod_id": pod_id,
                "crop_type": crop_type,
                "submitted_at": submitted_at,
            }

        thread = threading.Thread(
            target=self._process,
            args=(request_id, pod_id, crop_type, image, submitted_at),
            daemon=True,
        )
        thread.start()
        return request_id

    def _process(self, request_id, pod_id, crop_type, image, submitted_at):
        """Runs off the calling thread — this is the 'cloud' half: the
        actual classify_pod() + decide_action() call. In a real deployment
        this is the code that runs inside the Lambda function, not on the
        robot."""
        time.sleep(random.uniform(*self._latency_range))

        classification = classify_pod(pod_id, crop_type, image)
        action = decide_action(classification)
        completed_at = time.time()

        record = {
            "status": "done",
            "pod_id": pod_id,
            "crop_type": crop_type,
            "classification": classification,
            "action": action,
            "submitted_at": submitted_at,
            "completed_at": completed_at,
            "latency_seconds": completed_at - submitted_at,
        }
        with self._lock:
            self._results[request_id] = record
            self._log.append(record)

    def get_feedback(self, request_id, block=False, timeout=5.0):
        """Robot side calls this to check whether the cloud has a decision
        back yet.

        block=False (default): returns immediately, with status='processing'
        if the result isn't ready yet — lets the robot move on to its next
        pod and check back later rather than stalling in place.

        block=True: waits up to `timeout` seconds for a result, for the
        cases where the robot genuinely needs to hold position for a
        response (e.g. the demo script below).
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                record = self._results.get(request_id)
            if record is None:
                raise KeyError(f"Unknown request_id: {request_id}")
            if record["status"] == "done" or not block:
                return record
            if time.time() > deadline:
                return record
            time.sleep(0.02)

    def history(self):
        """All completed records so far — stands in for querying DynamoDB's
        per-pod health history table, useful for the evaluation-report
        latency numbers the proposal calls for."""
        with self._lock:
            return list(self._log)
