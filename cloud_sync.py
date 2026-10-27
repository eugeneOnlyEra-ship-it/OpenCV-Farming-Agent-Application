"""
cloud_sync.py -- connecting the running farm to its cloud database.

    FarmCloud       what the rest of the application talks to: record_inspection / record_order_event / record_tag_read.
                    Each is NON-BLOCKING and NEVER RAISES -- the farm must keep working whatever the network does.
    CloudSync       the background worker behind it: a bounded queue, retries with back-off, and an on-disk outbox so
                    nothing is lost while AWS is unreachable (replayed when it comes back).
    FarmCloudStore  applies each event (history item + updated pod profile) and answers queries.
    CloudConfig     which backend: FARM_CLOUD = aws | local | off   (default: aws if FARM_DDB_TABLE is set, else local)

        FARM_DDB_TABLE      DynamoDB table name           (aws)
        FARM_AWS_REGION     region                        (aws; else AWS_REGION / AWS_DEFAULT_REGION / your AWS config)
        FARM_DDB_ENDPOINT   custom endpoint, e.g. DynamoDB Local   (optional)
    Credentials are NOT configured here: boto3 finds them the standard way (environment, ~/.aws, an instance role).

Images never get here: events carry only numbers and text, and every write is checked by `assert_no_images`.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from cloud_backends import DynamoBackend, LocalBackend
from cloud_model import (GSI_CROP, GSI_STATUS, PK, SK, STATUSES, ImageDataRejected, apply_event, assert_no_images,
                         event_item, event_pod_id, event_pod_meta, inspection_event, new_profile, order_event, plain,
                         pod_pk, tag_read_event)


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ======================================================================================================== config
@dataclass
class CloudConfig:
    backend: str = "local"                       # aws | local | off
    table: str = "FarmPods"
    region: str | None = None
    endpoint_url: str | None = None
    local_path: str = "farm_cloud_local.json"
    outbox_path: str = "farm_cloud_outbox.jsonl"


def config_from_env(env=None, base="farm_cloud"):
    """Build the config from environment variables. `base` is the path stem for the local file and the outbox."""
    env = os.environ if env is None else env
    backend = (env.get("FARM_CLOUD") or ("aws" if env.get("FARM_DDB_TABLE") else "local")).strip().lower()
    if backend not in ("aws", "local", "off"):
        raise ValueError(f"FARM_CLOUD must be aws, local or off (got {backend!r})")
    return CloudConfig(
        backend=backend, table=env.get("FARM_DDB_TABLE") or "FarmPods",
        region=env.get("FARM_AWS_REGION") or env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION"),
        endpoint_url=env.get("FARM_DDB_ENDPOINT") or None,
        local_path=f"{base}_cloud.json", outbox_path=f"{base}_cloud_outbox.jsonl")


def make_backend(cfg):
    if cfg.backend == "aws":
        return DynamoBackend(cfg.table, cfg.region, cfg.endpoint_url)
    if cfg.backend == "local":
        return LocalBackend(cfg.local_path)
    return None


# ======================================================================================================== the store
def _idem_key(ev):
    t = ev["type"]
    if t == "inspection":
        return (t, event_pod_id(ev), ev["ts"], ev.get("visit"), ev.get("kind"))
    if t == "order":
        o = ev["order"]
        return (t, o["order_id"], o["created"], o["status"], ev.get("event"))
    return (t, event_pod_id(ev), ev["ts"])


class FarmCloudStore:
    """Applies events and answers queries. `apply` must only be called from ONE thread (the sync worker); the query
    methods are safe from any thread."""

    def __init__(self, backend):
        self.backend = backend
        self._mirror, self._applied = {}, set()
        self._persisted = set()                  # pods whose PROFILE item already exists in the table

    # -------------------------------------------------------------------- writing
    def _profile_for_write(self, pid, meta):
        p = self._mirror.get(pid)
        if p is None:
            stored = self.backend.get(pod_pk(pid), "PROFILE")             # a restart carries on where the cloud left off
            p = ({k: v for k, v in stored.items() if k not in (PK, SK)} if stored else new_profile(meta))
            if stored:
                self._persisted.add(pid)
        return p

    def apply(self, ev):
        """Write the event's history item and update the pod's profile. Returns False if already applied. Raises on a
        backend failure WITHOUT having committed the in-memory profile, so a retry never double-counts."""
        key = _idem_key(ev)
        if key in self._applied:
            return False
        pid = event_pod_id(ev)
        if not pid:
            raise KeyError("event has no pod id")
        base = self._profile_for_write(pid, event_pod_meta(ev))
        new = apply_event(base, ev, _utc())
        item = event_item(ev)
        if item:
            self.backend.put(item)
        # a pod the table has never seen gets EVERY attribute; after that only what changed
        before = base if pid in self._persisted else {}
        values = {k: v for k, v in new.items() if before.get(k) != v}
        values.update({k: None for k in before if k not in new})
        if values:
            self.backend.update(pod_pk(pid), "PROFILE", values)
        self._mirror[pid] = new
        self._persisted.add(pid)
        self._applied.add(key)
        return True

    # -------------------------------------------------------------------- reading
    def profile(self, pod_id):
        return self.backend.get(pod_pk(pod_id), "PROFILE")

    def inspections(self, pod_id, limit=20):
        return self.backend.query(pod_pk(pod_id), "INSPECTION#", limit=limit, newest_first=True)

    def orders(self, pod_id, task=None, limit=50):
        rows = self.backend.query(pod_pk(pod_id), "ORDER#", newest_first=True)
        return [r for r in rows if task is None or r.get("task") == task][:limit]

    def treatments(self, pod_id):
        return self.orders(pod_id, task="TREAT")

    def harvests(self, pod_id):
        return self.orders(pod_id, task="HARVEST")

    def tag_reads(self, pod_id, limit=20):
        return self.backend.query(pod_pk(pod_id), "TAGREAD#", limit=limit, newest_first=True)

    def pods_of_crop(self, crop):
        return self.backend.query_index(GSI_CROP, f"CROP#{crop}")

    def pods_with_status(self, status):
        return self.backend.query_index(GSI_STATUS, f"STATUS#{status}")

    def status_counts(self):
        return {s: len(self.pods_with_status(s)) for s in STATUSES}


# ======================================================================================================== the worker
class CloudSync:
    """Background delivery of events to the store. `submit` returns immediately."""
    OFFLINE_RETRY_S = 15.0

    def __init__(self, store, outbox_path=None, log=None, max_queue=5000, backoff=(0.4, 1.2, 3.0)):
        self.store, self.outbox_path = store, outbox_path
        self.log = log or (lambda msg, level="info": None)
        self.backoff = tuple(backoff)
        self._q = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._outbox_lock = threading.Lock()
        self._offline_until = 0.0
        self._thread = None
        self.stats = {"sent": 0, "spooled": 0, "dropped": 0, "replayed": 0, "online": None, "last_error": None,
                      "last_ok_at": None}

    # -------------------------------------------------------------------- public
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cloud-sync", daemon=True)
            self._thread.start()

    def submit(self, event):
        """Queue an event for delivery. Never blocks and never raises; returns False if the event was refused."""
        try:
            assert_no_images(event)
        except ImageDataRejected as exc:
            self._count("dropped")
            self.log(f"cloud: event REFUSED (image data never goes to the cloud): {exc}", "error")
            return False
        try:
            self._q.put_nowait(event)
        except queue.Full:                      # the worker is far behind: keep the event safely on disk instead
            self._spool(event)
        return True

    def pending(self):
        return self._q.unfinished_tasks

    def join(self, timeout=10.0):
        """Wait until everything submitted so far has been delivered (or spooled). True if the queue drained."""
        end = time.time() + timeout
        while self._q.unfinished_tasks and time.time() < end:
            time.sleep(0.02)
        return not self._q.unfinished_tasks

    def close(self, timeout=10.0):
        self.join(timeout)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def status(self):
        with self._lock:
            s = dict(self.stats)
        s.update(queued=self.pending(), outbox=self.outbox_size())
        return s

    # -------------------------------------------------------------------- internals
    def _count(self, key, n=1):
        with self._lock:
            self.stats[key] += n

    def _set(self, **kw):
        with self._lock:
            self.stats.update(kw)

    def _run(self):
        ok, msg = self.store.backend.ping()
        self._set(online=ok, last_error=None if ok else msg)
        self.log(("cloud: connected - " if ok else "cloud: NOT reachable yet (events are kept on disk and sent when it is) - ") + msg,
                 "ok" if ok else "warn")
        while not self._stop.is_set():
            try:
                ev = self._q.get(timeout=0.5)
            except queue.Empty:
                if self.outbox_size() and time.time() >= self._offline_until:
                    self.replay_outbox()
                continue
            try:
                self._deliver(ev)
            except Exception as exc:                                    # noqa: BLE001 - the worker must never die
                self._count("dropped")
                self.log(f"cloud: dropped a malformed event ({type(exc).__name__}: {exc})", "error")
            finally:
                self._q.task_done()

    def _deliver(self, ev):
        if time.time() < self._offline_until:                           # known offline: don't make every event wait
            return self._spool(ev)
        attempts = 1 + len(self.backoff)
        for i in range(attempts):
            try:
                self.store.apply(ev)
                self._count("sent")
                self._set(online=True, last_ok_at=_utc(), last_error=None)
                if self.outbox_size():
                    self.replay_outbox()
                return
            except (ImageDataRejected, KeyError):
                raise
            except Exception as exc:                                    # noqa: BLE001 - network / AWS errors
                self._set(last_error=f"{type(exc).__name__}: {str(exc)[:200]}")
                if i < attempts - 1 and not self._stop.is_set():
                    time.sleep(self.backoff[i])
        self._set(online=False)
        self._offline_until = time.time() + self.OFFLINE_RETRY_S
        self.log(f"cloud: cannot reach the database ({self.stats['last_error']}); keeping events on disk", "warn")
        self._spool(ev)

    # -------------------------------------------------------------------- outbox
    def _spool(self, ev):
        if not self.outbox_path:
            self._count("dropped")
            return
        with self._outbox_lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.outbox_path)), exist_ok=True)
            with open(self.outbox_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"spooled_at": _utc(), "event": ev}) + "\n")
        self._count("spooled")

    def outbox_size(self):
        if not self.outbox_path or not os.path.exists(self.outbox_path):
            return 0
        with self._outbox_lock, open(self.outbox_path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def replay_outbox(self):
        """Send spooled events oldest-first; stop at the first failure and keep the rest. Returns how many were sent."""
        if not self.outbox_path or not os.path.exists(self.outbox_path):
            return 0
        with self._outbox_lock:
            with open(self.outbox_path, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
        sent = 0
        for ln in lines:
            try:
                self.store.apply(json.loads(ln)["event"])
            except (ImageDataRejected, KeyError, ValueError):
                sent += 1                                               # can never succeed: discard rather than loop forever
                self._count("dropped")
                continue
            except Exception as exc:                                    # noqa: BLE001 - still offline
                self._set(online=False, last_error=f"{type(exc).__name__}: {str(exc)[:200]}")
                self._offline_until = time.time() + self.OFFLINE_RETRY_S
                break
            sent += 1
            self._count("replayed")
        with self._outbox_lock:
            rest = lines[sent:]
            tmp = self.outbox_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(rest)
            os.replace(tmp, self.outbox_path)
        if sent:
            self.log(f"cloud: sent {sent} event(s) that were waiting on disk", "ok")
            self._set(online=True, last_ok_at=_utc(), last_error=None)
        return sent


# ======================================================================================================== the facade
class FarmCloud:
    """What the application uses. Build one with FarmCloud.from_env(); every record_* call is safe to make anywhere."""

    def __init__(self, config, backend, log=None):
        self.config, self.backend = config, backend
        self.log = log or (lambda msg, level="info": None)
        self.store = FarmCloudStore(backend) if backend else None
        self.sync = CloudSync(self.store, config.outbox_path, self.log) if backend else None
        if self.sync:
            self.sync.start()

    @classmethod
    def from_env(cls, env=None, base="farm_cloud", log=None):
        cfg = config_from_env(env, base)
        return cls(cfg, make_backend(cfg), log)

    @property
    def enabled(self):
        return self.sync is not None

    # -------------------------------------------------------------------- recording (non-blocking, never raise)
    def _submit(self, build):
        if not self.enabled:
            return False
        try:
            return self.sync.submit(build())
        except Exception as exc:                                        # noqa: BLE001
            self.log(f"cloud: could not record an event ({type(exc).__name__}: {exc})", "error")
            return False

    def record_inspection(self, pod, classification, decision, ts, visit, trend="", cycle=None,
                          ground_truth_stage=None, image_path=None, kind="scan"):
        return self._submit(lambda: inspection_event(pod, classification, decision, ts, visit, trend, cycle,
                                                     ground_truth_stage, image_path, kind))

    def record_order_event(self, event, order, extra=None):
        return self._submit(lambda: order_event(event, order, extra))

    def record_tag_read(self, read, ts=None):
        return self._submit(lambda: tag_read_event(read, ts or _utc()))

    # -------------------------------------------------------------------- status / lifecycle
    def status(self):
        base = {"backend": self.config.backend, "table": self.config.table, "region": self.config.region,
                "where": self.backend.where if self.backend else "disabled"}
        if self.sync:
            base.update(self.sync.status())
        return base

    def close(self, timeout=10.0):
        if self.sync:
            self.sync.close(timeout)
