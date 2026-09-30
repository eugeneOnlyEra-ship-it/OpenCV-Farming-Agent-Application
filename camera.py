"""
Client-side capture interface (robot camera in the PyBullet build; a
dataset image lookup here, where there's no simulated robot at all).

Its only job is to capture a frame at a pod and hand it to the cloud
pipeline — it does not import perception.py or agent.py, and never runs
a classification itself. That split is deliberate: it's what makes the
"visual evidence changes the next action" loop genuinely live in the
cloud layer (matching the proposal's S3 -> Lambda -> DynamoDB -> IoT
Core architecture), not just something that happens to run on whatever
client captured the frame.
"""


class PodCamera:
    def __init__(self, cloud_client):
        self.cloud_client = cloud_client

    def capture_and_submit(self, pod_id, crop_type, image):
        """Call this once you have a frame for a pod -- a PyBullet camera
        capture (numpy array) in the simulated build, or a dataset image
        path here. This function does nothing with it except pass it
        along. Returns a request_id; the classification/decision comes
        back later via cloud_client.get_feedback(request_id).
        """
        return self.cloud_client.submit_frame(pod_id, crop_type, image)
