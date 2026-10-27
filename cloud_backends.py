"""
cloud_backends.py -- where the farm's records physically live.

    DynamoBackend   a real Amazon DynamoDB table (boto3).
    LocalBackend    a JSON file on this computer that behaves the same way, so the application runs, and can be developed
                    and tested, without an AWS account. (It is NOT the cloud, and the application says so.)

Both implement the same five operations; everything above this file (cloud_sync.py, cloud_cli.py, the UI) is identical
for either. Every write passes `assert_no_images` first: images can never reach either backend.
"""
from __future__ import annotations

import json
import os
import threading
import time

from cloud_model import (GSI_CROP, GSI_STATUS, PK, SK, TABLE_DEFINITION, assert_no_images, from_dynamo, to_dynamo)

_INDEX_KEYS = {GSI_CROP: ("gsi1pk", "gsi1sk"), GSI_STATUS: ("gsi2pk", "gsi2sk")}


class Backend:
    name = "backend"
    where = ""

    def put(self, item):                                  # create / replace one item
        raise NotImplementedError

    def update(self, pk, sk, values, remove=()):          # upsert: set `values`, delete `remove` (None values are removed too)
        raise NotImplementedError

    def get(self, pk, sk):
        raise NotImplementedError

    def query(self, pk, sk_prefix=None, limit=None, newest_first=False):
        raise NotImplementedError

    def query_index(self, index, value, limit=None):
        raise NotImplementedError

    def ping(self):                                       # -> (ok, message)
        raise NotImplementedError


# ================================================================================================ local file
class LocalBackend(Backend):
    """Same semantics as the table, kept in memory and saved to a JSON file. One lock; atomic file replace."""
    name = "local"

    def __init__(self, path):
        self.path = path
        self.where = os.path.abspath(path)
        self._lock = threading.RLock()
        self._items = {}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for it in json.load(f):
                        self._items[(it[PK], it[SK])] = it
            except (OSError, ValueError):
                self._items = {}

    def _save(self):
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(list(self._items.values()), f, indent=1, default=str)
        os.replace(tmp, self.path)

    @staticmethod
    def _clean(item):
        return {k: v for k, v in from_dynamo(to_dynamo(item)).items()}

    def put(self, item):
        assert_no_images(item)
        item = self._clean(item)
        with self._lock:
            self._items[(item[PK], item[SK])] = item
            self._save()

    def update(self, pk, sk, values, remove=()):
        assert_no_images(values)
        drop = list(remove) + [k for k, v in values.items() if v is None]       # BEFORE cleaning: cleaning drops Nones
        values = self._clean(values)
        with self._lock:
            item = dict(self._items.get((pk, sk), {PK: pk, SK: sk}))
            for k in drop:
                item.pop(k, None)
            item.update(values)
            self._items[(pk, sk)] = item
            self._save()

    def get(self, pk, sk):
        with self._lock:
            it = self._items.get((pk, sk))
            return dict(it) if it else None

    def query(self, pk, sk_prefix=None, limit=None, newest_first=False):
        with self._lock:
            rows = [dict(v) for (p, s), v in self._items.items() if p == pk and (sk_prefix is None or s.startswith(sk_prefix))]
        rows.sort(key=lambda r: r[SK], reverse=newest_first)
        return rows[:limit] if limit else rows

    def query_index(self, index, value, limit=None):
        hk, rk = _INDEX_KEYS[index]
        with self._lock:
            rows = [dict(v) for v in self._items.values() if v.get(hk) == value]
        rows.sort(key=lambda r: r.get(rk, ""))
        return rows[:limit] if limit else rows

    def ping(self):
        return True, f"local file {self.where}"

    def all_items(self):
        with self._lock:
            return [dict(v) for v in self._items.values()]


# ================================================================================================ DynamoDB
class DynamoBackend(Backend):
    """A DynamoDB table. Credentials come from the normal AWS chain (environment, ~/.aws, an instance/role...): they are
    never stored by this application. A boto3 resource is created per thread (resources are not thread-safe)."""
    name = "aws"

    def __init__(self, table, region=None, endpoint_url=None):
        self.table_name, self.region, self.endpoint_url = table, region, endpoint_url
        self.where = f"DynamoDB table '{table}'" + (f" ({region})" if region else "")
        self._tls = threading.local()

    def _table(self):
        t = getattr(self._tls, "table", None)
        if t is None:
            import boto3
            from botocore.config import Config
            res = boto3.session.Session().resource(
                "dynamodb", region_name=self.region, endpoint_url=self.endpoint_url,
                config=Config(retries={"max_attempts": 3, "mode": "standard"}, connect_timeout=5, read_timeout=10))
            t = self._tls.table = res.Table(self.table_name)
        return t

    def put(self, item):
        assert_no_images(item)
        self._table().put_item(Item=to_dynamo(item))

    def update(self, pk, sk, values, remove=()):
        assert_no_images(values)
        drop = list(dict.fromkeys(list(remove) + [k for k, v in values.items() if v is None]))
        values = to_dynamo({k: v for k, v in values.items() if v is not None})
        drop = [k for k in drop if k not in values]
        names, vals, sets = {}, {}, []
        for i, (k, v) in enumerate(values.items()):
            names[f"#s{i}"], vals[f":s{i}"] = k, v
            sets.append(f"#s{i} = :s{i}")
        rem = []
        for j, k in enumerate(drop):
            names[f"#r{j}"] = k
            rem.append(f"#r{j}")
        expr = ("SET " + ", ".join(sets) if sets else "") + (" REMOVE " + ", ".join(rem) if rem else "")
        if not expr:
            return
        kwargs = {"Key": {PK: pk, SK: sk}, "UpdateExpression": expr.strip(), "ExpressionAttributeNames": names}
        if vals:
            kwargs["ExpressionAttributeValues"] = vals
        self._table().update_item(**kwargs)

    def get(self, pk, sk):
        r = self._table().get_item(Key={PK: pk, SK: sk}, ConsistentRead=True)
        return from_dynamo(r["Item"]) if "Item" in r else None

    def _paged(self, **kw):
        out, limit = [], kw.pop("_limit", None)
        while True:
            r = self._table().query(**kw)
            out += r.get("Items", [])
            if limit and len(out) >= limit:
                return out[:limit]
            if "LastEvaluatedKey" not in r:
                return out
            kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]

    def query(self, pk, sk_prefix=None, limit=None, newest_first=False):
        from boto3.dynamodb.conditions import Key
        cond = Key(PK).eq(pk) & Key(SK).begins_with(sk_prefix) if sk_prefix else Key(PK).eq(pk)
        kw = {"KeyConditionExpression": cond, "ScanIndexForward": not newest_first, "ConsistentRead": True}
        if limit:
            kw["Limit"] = limit
        return [from_dynamo(i) for i in self._paged(_limit=limit, **kw)]

    def query_index(self, index, value, limit=None):
        from boto3.dynamodb.conditions import Key
        hk, _ = _INDEX_KEYS[index]
        kw = {"IndexName": index, "KeyConditionExpression": Key(hk).eq(value)}
        if limit:
            kw["Limit"] = limit
        return [from_dynamo(i) for i in self._paged(_limit=limit, **kw)]

    def ping(self):
        try:
            d = self._table().meta.client.describe_table(TableName=self.table_name)["Table"]
            return True, f"{self.where}: {d['TableStatus']}"
        except Exception as exc:                                   # noqa: BLE001 - any failure means "not reachable"
            return False, f"{self.where}: {type(exc).__name__}: {str(exc)[:160]}"

    def create_table(self, wait=True):
        """Create this table from TABLE_DEFINITION (idempotent). Normally done once by aws/farm_table.yaml instead."""
        client = self._table().meta.client
        try:
            client.create_table(TableName=self.table_name, **TABLE_DEFINITION)
        except client.exceptions.ResourceInUseException:
            return False
        if wait:
            client.get_waiter("table_exists").wait(TableName=self.table_name, WaiterConfig={"Delay": 2, "MaxAttempts": 30})
        return True
