"""Offline-first client for the Farm Agent's API Gateway -> Lambda -> DynamoDB path.

Set FARM_LAMBDA_API_URL to enable it. Images never leave the machine; only compact
OpenCV classification metadata is sent. If the network is unavailable, events are
appended to a local JSONL outbox and replayed by a daemon thread.
"""
from __future__ import annotations
import json, os, threading, time, urllib.error, urllib.request, uuid
from datetime import datetime, timezone

ACTIONS = {"log_healthy", "flag_for_harvest", "schedule_frequent_monitoring", "flag_for_treatment"}

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

class LambdaObservationClient:
    def __init__(self, api_url=None, api_key=None, outbox=None, results=None, timeout=1.5, log=None):
        self.api_url = (api_url if api_url is not None else os.getenv("FARM_LAMBDA_API_URL", "")).strip().rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("FARM_LAMBDA_API_KEY", "")
        self.outbox = outbox or os.getenv("FARM_LAMBDA_OUTBOX", "farm_lambda_outbox.jsonl")
        self.results = results or os.getenv("FARM_LAMBDA_RESULTS", "farm_lambda_work_orders.json")
        self.timeout, self.log = timeout, log or (lambda message, level="info": None)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self.last_status = "disabled" if not self.api_url else "waiting"
        self._offline_until = 0.0
        if self.api_url:
            self._thread = threading.Thread(target=self._replay_loop, name="lambda-outbox", daemon=True)
            self._thread.start()

    @property
    def enabled(self): return bool(self.api_url)

    def _request(self, event):
        data = json.dumps(event, separators=(",", ":")).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key: headers["x-api-key"] = self.api_key
        req = urllib.request.Request(self.api_url + "/observations", data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"Lambda API returned HTTP {response.status}")
            return body

    def _queue(self, event):
        os.makedirs(os.path.dirname(os.path.abspath(self.outbox)), exist_ok=True)
        with self._lock, open(self.outbox, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.last_status = "queued_offline"

    def _save_work_order(self, result):
        order = result.get("work_order") if isinstance(result, dict) else None
        if not isinstance(order, dict) or order.get("action") not in ACTIONS: return
        os.makedirs(os.path.dirname(os.path.abspath(self.results)), exist_ok=True)
        with self._lock:
            try:
                with open(self.results, encoding="utf-8") as f: data = json.load(f)
            except (OSError, ValueError): data = {}
            pending = data.setdefault("pending", [])
            if not any(x.get("order_id") == order.get("order_id") for x in pending):
                pending.append(order)
            tmp = self.results + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            os.replace(tmp, self.results)

    def take_work_order(self, pod_id):
        """Return one cloud decision waiting for this pod, removing it from the local response queue."""
        if not self.enabled: return None
        with self._lock:
            try:
                with open(self.results, encoding="utf-8") as f: data = json.load(f)
            except (OSError, ValueError): return None
            pending = data.get("pending", [])
            idx = next((i for i, o in enumerate(pending) if o.get("pod_id") == pod_id), None)
            if idx is None: return None
            order = pending.pop(idx)
            tmp = self.results + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
            os.replace(tmp, self.results)
            return order

    def submit_observation(self, pod, classification, decision, timestamp, visit, trend=""):
        """Try Lambda quickly; return its work order if online, otherwise persist and continue locally."""
        if not self.enabled: return None
        event = {"observation_id": str(uuid.uuid4()), "pod_id": str(pod.get("pod_id", "")),
                 "crop_type": str(pod.get("crop_type", "")), "timestamp": timestamp or _now(),
                 "visit": int(visit or 0), "growth_stage": classification.get("growth_stage"),
                 "growth_confidence": float(classification.get("growth_confidence") or 0),
                 "disease_flag": bool(classification.get("disease_flag")),
                 "disease_name": classification.get("disease_name") if classification.get("disease_flag") else None,
                 "disease_confidence": float(classification.get("disease_confidence") or 0),
                 "local_action": decision.get("action"), "local_reason": str(decision.get("reason", ""))[:500],
                 "trend": str(trend or "")[:500]}
        if time.time() < self._offline_until:
            self._queue(event)
            return self.take_work_order(pod.get("pod_id", ""))
        try:
            result = self._request(event)
            self._offline_until = 0.0
            self.last_status = "online"
            self._save_work_order(result)
            # This was returned synchronously; remove it from the deferred queue so it is not used twice.
            order = result.get("work_order")
            if order:
                with self._lock:
                    try:
                        with open(self.results, encoding="utf-8") as f: data = json.load(f)
                    except (OSError, ValueError): data = {"pending": []}
                    data["pending"] = [x for x in data.get("pending", []) if x.get("pod_id") != order.get("pod_id")]
                    tmp = self.results + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
                    os.replace(tmp, self.results)
            return order
        except Exception as exc:
            self._offline_until = time.time() + 15.0
            self._queue(event)
            self.last_status = "queued_offline"
            self.log(f"AWS Lambda unavailable; observation saved to local outbox ({type(exc).__name__}).", "warn")
            return None

    def _replay_loop(self):
        while not self._stop.wait(8.0):
            if not os.path.exists(self.outbox): continue
            try:
                with self._lock, open(self.outbox, encoding="utf-8") as f: lines = f.readlines()
                if not lines: continue
                sent_ids = set()
                for index, line in enumerate(lines):
                    try: event = json.loads(line)
                    except ValueError: continue
                    try:
                        result = self._request(event)
                        self._save_work_order(result)
                        sent_ids.add(event.get("observation_id"))
                        self._offline_until = 0.0
                        self.last_status = "online"
                    except Exception:
                        self._offline_until = time.time() + 15.0
                        self.last_status = "queued_offline"
                        break
                # Remove only IDs that were confirmed delivered. Re-read under lock so
                # events appended while HTTP calls were in flight cannot be overwritten.
                with self._lock:
                    with open(self.outbox, encoding="utf-8") as f: current = f.readlines()
                    remaining = []
                    for line in current:
                        try: oid = json.loads(line).get("observation_id")
                        except ValueError: remaining.append(line); continue
                        if oid not in sent_ids: remaining.append(line)
                    tmp = self.outbox + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as f: f.writelines(remaining)
                    os.replace(tmp, self.outbox)
            except OSError as exc:
                self.log(f"Could not replay Lambda outbox: {exc}", "warn")

    def close(self, timeout=1.0):
        self._stop.set()
        if self._thread: self._thread.join(timeout)
