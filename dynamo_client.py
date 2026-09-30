"""
dynamo_client.py

Local stand-in for the DynamoDB per-pod health history table from the
proposal's AWS architecture ("Logs per-pod health history over time
(growth stage, disease flags, timestamps)"). Same two operations a real
table needs -- put an item, query a pod's items back in order -- but
persisted to a local JSON file instead of a real table, so history
actually survives across separate `python3 main.py` invocations (unlike
cloud_pipeline.LocalCloudClient's history(), which only covers whatever
happened in the current run).

Table shape mirrors what a real DynamoDB table for this would use:
partition key = pod_id, sort key = timestamp. query(pod_id) returns
every item for that pod sorted by timestamp, same as a real DynamoDB
Query against that key schema would.

Swapping to real DynamoDB later: same two methods (put_item, query),
backed by boto3 (`table.put_item(Item=...)` /
`table.query(KeyConditionExpression=Key('pod_id').eq(pod_id))`) instead
of a local file. Nothing else in this project needs to change --
main.py only ever calls put_item/query, never touches the file
directly.
"""

import json
from pathlib import Path


class LocalDynamoTable:
    def __init__(self, path="dynamo_table.json"):
        self.path = Path(path)
        self._items = self._load()

    def _load(self):
        if self.path.exists():
            return json.loads(self.path.read_text())
        return []

    def _save(self):
        self.path.write_text(json.dumps(self._items, indent=2))

    def put_item(self, item):
        """item must include "pod_id" (partition key) and "timestamp"
        (sort key) -- same requirement a real DynamoDB PutItem against
        this key schema would have."""
        assert "pod_id" in item and "timestamp" in item, \
            "items need pod_id (partition key) and timestamp (sort key)"
        self._items.append(item)
        self._save()

    def query(self, pod_id):
        """Every item for pod_id, oldest first -- same shape as a real
        DynamoDB Query against a pod_id+timestamp key schema."""
        rows = [i for i in self._items if i["pod_id"] == pod_id]
        return sorted(rows, key=lambda r: r["timestamp"])

    def latest(self, pod_id):
        rows = self.query(pod_id)
        return rows[-1] if rows else None

    def all_pod_ids(self):
        return sorted(set(i["pod_id"] for i in self._items))

    def visit_count(self, pod_id):
        return len(self.query(pod_id))
