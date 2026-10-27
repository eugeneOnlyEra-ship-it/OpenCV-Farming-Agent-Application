"""Offline, dependency-light checks for the POST /observations Lambda and local outbox.
Run from the project root: python dev/check_lambda_observations.py
Uses a fake DynamoDB table and never contacts AWS.
"""
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

class FakeTable:
    def __init__(self): self.items, self.updates = {}, []
    def get_item(self, Key):
        item = self.items.get((Key["pk"], Key["sk"]))
        return {"Item": item} if item else {}
    def put_item(self, Item, **kwargs): self.items[(Item["pk"], Item["sk"])] = Item
    def update_item(self, Key, **kwargs): self.updates.append((Key, kwargs))

class FakeResource:
    def __init__(self): self.table = FakeTable()
    def Table(self, name): return self.table

resource = FakeResource()
boto3 = types.ModuleType("boto3")
boto3.resource = lambda *args, **kwargs: resource
sys.modules["boto3"] = boto3
os.environ["TABLE_NAME"] = "FarmPods"
os.environ["API_KEY"] = "test-secret"
spec = importlib.util.spec_from_file_location("observation_api_under_test", ROOT / "aws/lambda/observation_api.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)

BASE = {"observation_id": "event-12345678", "pod_id": "R1-L-F1", "crop_type": "lettuce",
        "timestamp": "2026-10-26T10:00:00+00:00", "visit": 1, "growth_stage": "stage 02- Leafy growth",
        "growth_confidence": 0.92, "disease_flag": True, "disease_name": "Downy_mildew_on_lettuce",
        "disease_confidence": 0.91, "local_action": "log_healthy"}

def call(payload, headers=None):
    response = api.handler({"routeKey": "POST /observations", "headers": headers if headers is not None else
                            {"x-api-key": "test-secret"}, "body": json.dumps(payload)}, None)
    return response["statusCode"], json.loads(response["body"])

checks = []
status, body = call(BASE)
checks.append(("high-confidence disease creates treatment work order", status == 200 and body["work_order"]["action"] == "flag_for_treatment"))
checks.append(("observation, inspection, and order are persisted", all(any(k[1].startswith(prefix) for k in resource.table.items)
              for prefix in ("OBSERVATION#", "INSPECTION#", "ORDER#"))))
checks.append(("pod profile is updated", bool(resource.table.updates)))
status, body = call({**BASE, "observation_id": "event-87654321", "disease_flag": False, "disease_name": None,
                     "disease_confidence": 0.1, "growth_stage": "stage 04- Harvest stage"})
checks.append(("harvest-stage observation creates harvest work order", status == 200 and body["work_order"]["action"] == "flag_for_harvest"))
status, _ = call({**BASE, "observation_id": "event-87654322", "image": {"pixels": [1, 2]}})
checks.append(("image-like payload is rejected", status == 400))
status, _ = call(BASE, headers={"x-api-key": "wrong"})
checks.append(("write endpoint enforces API key", status == 401))

from lambda_cloud_client import LambdaObservationClient
class Response:
    status = 200
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self): return json.dumps({"accepted": True, "work_order": {"order_id": "lambda-test", "pod_id": "R1-L-F1",
                                                                            "action": "flag_for_treatment", "task": "TREAT"}}).encode()
with tempfile.TemporaryDirectory() as tmp:
    client = LambdaObservationClient(api_url="https://example.invalid", api_key="test-secret",
        outbox=os.path.join(tmp, "outbox.jsonl"), results=os.path.join(tmp, "orders.json"), log=lambda *a: None)
    with patch("urllib.request.urlopen", return_value=Response()):
        order = client.submit_observation({"pod_id": "R1-L-F1", "crop_type": "lettuce"},
            {"growth_stage": "stage 02", "growth_confidence": .9, "disease_flag": True, "disease_name": "x", "disease_confidence": .8},
            {"action": "log_healthy", "reason": "local"}, "2026-10-26T10:00:00+00:00", 1)
    checks.append(("online client returns Lambda work order", bool(order and order["action"] == "flag_for_treatment")))
    client.close()
with tempfile.TemporaryDirectory() as tmp:
    client = LambdaObservationClient(api_url="https://example.invalid", api_key="test-secret",
        outbox=os.path.join(tmp, "outbox.jsonl"), results=os.path.join(tmp, "orders.json"), log=lambda *a: None)
    with patch("urllib.request.urlopen", side_effect=TimeoutError("offline")):
        order = client.submit_observation({"pod_id": "R1-L-F1", "crop_type": "lettuce"},
            {"growth_stage": "stage 02", "growth_confidence": .9, "disease_flag": False, "disease_confidence": .1},
            {"action": "log_healthy"}, "2026-10-26T10:00:00+00:00", 1)
    checks.append(("offline client persists observation to outbox", order is None and len(Path(tmp, "outbox.jsonl").read_text().splitlines()) == 1))
    client.close()

for label, passed in checks: print(("PASS" if passed else "FAIL") + "  " + label)
if not all(passed for _, passed in checks): sys.exit(1)
print(f"\nAll {len(checks)} Lambda/offline checks passed. No AWS calls were made.")
