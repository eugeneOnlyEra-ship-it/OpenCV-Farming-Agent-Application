"""
dev/check_cloud_api.py -- the read-only web API (aws/lambda/farm_api.py behind API Gateway, deployed by aws/farm_api.yaml).

Run from the project root:   python dev/check_cloud_api.py        (~10 s; needs boto3, moto, PyYAML, cfn-lint)

What is proven, and what is not:
  PROVEN   the handler returns the right records for realistic API Gateway (HTTP API v2) events, against a mock DynamoDB
           filled by the application's own cloud code; it is read-only; it rejects bad input before touching the table;
           it returns no images; it never leaks an internal error; the template's inline code IS the source file, fits the
           4096-character limit, passes cfn-lint, and grants the Lambda only GetItem + Query.
  NOT PROVEN  that API Gateway, IAM and your account are wired up (nothing here can reach AWS): deploy it and open the
           TryIt URL from the stack outputs. See aws/DEPLOY.md.
"""
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

for k, v in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_SESSION_TOKEN": "testing",
             "AWS_DEFAULT_REGION": "us-east-1"}.items():
    os.environ[k] = v
for k in ("FARM_CLOUD", "FARM_DDB_TABLE", "AWS_PROFILE", "API_KEY"):
    os.environ.pop(k, None)

from moto import mock_aws                                           # noqa: E402

import cloud_model as M                                             # noqa: E402
from cloud_backends import DynamoBackend                            # noqa: E402
from cloud_sync import FarmCloudStore                               # noqa: E402
from worker_planner import WorkOrder                                # noqa: E402

FAILS = []
TABLE = "FarmPods"
SRC = os.path.join(ROOT, "aws", "lambda", "farm_api.py")
TEMPLATE = os.path.join(ROOT, "aws", "farm_api.yaml")


def ok(cond, name, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


def load_handler(**env):
    """Import a FRESH copy of the Lambda module, the way Lambda would, with these environment variables."""
    for k in ("API_KEY",):
        os.environ.pop(k, None)
    os.environ["TABLE_NAME"] = TABLE
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("farm_api_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def event(route, tag=None, query=None, headers=None):
    """What API Gateway's HTTP API (payload format 2.0) hands the Lambda."""
    ev = {"version": "2.0", "routeKey": route, "rawPath": "/" + route.split(" ", 1)[1].replace("{tag}", tag or ""),
          "rawQueryString": "", "headers": headers or {"host": "x.execute-api.us-east-1.amazonaws.com"},
          "requestContext": {"http": {"method": route.split()[0], "path": "/x"}}, "isBase64Encoded": False}
    if query:
        ev["queryStringParameters"] = query
    if tag is not None:
        ev["pathParameters"] = {"tag": tag}
    return ev


def call(mod, *a, **kw):
    r = mod.handler(event(*a, **kw), None)
    return r["statusCode"], json.loads(r["body"]), r


# ------------------------------------------------------------------------------------------ data
def order(task, status, pod_id, oid, created, updated, outcome=None):
    return WorkOrder(order_id=oid, task=task, role="worker", agent="worker_front", pod_id=pod_id, crop_type="lettuce",
                     trough_id=pod_id[:-1], side="front", location="x", priority=1, title=f"{task} {pod_id}", rationale="r",
                     evidence={}, targets=[], steps=[1, 2, 3], requires_approval=False, status=status, created=created,
                     updated=updated, outcome=outcome or {})


def cls(stage, flag=False):
    return {"growth_stage": stage, "growth_confidence": 0.91, "disease_flag": flag,
            "disease_name": "Septoria" if flag else None, "disease_confidence": 0.6 if flag else 0.1, "disease_detections": [1] * flag}


def populate(store):
    """Three pods in different states, written by the application's own cloud code."""
    lettuce = {"pod_id": "R1-L-F1", "crop_type": "lettuce", "side": "front"}
    for i in range(12):                                                   # many inspections, so ?limit can be tested
        store.apply(M.inspection_event(lettuce, cls("stage 02- Leafy growth", flag=(i == 0)), {"action": "log_healthy", "reason": "ok"},
                                       f"2026-10-25T10:{i:02d}:00+00:00", i + 1, "trend",
                                       {"age": i, "harvest_age": 12, "stage": "s", "waiting_for_harvest": False, "cycle": 1}, None, None))
    for st in ("dispatched", "completed"):
        store.apply(M.order_event("status", order("TREAT", st, "R1-L-F1", "WO-0001", "2026-10-25T10:00:30+00:00",
                                                  "2026-10-25T10:01:00+00:00"), None))
    store.apply(M.order_event("loop_closed", order("VERIFY", "completed", "R1-L-F1", "WO-0002", "2026-10-25T10:02:00+00:00",
                                                   "2026-10-25T10:02:30+00:00"), {"outcome": {"result": "treatment_worked", "headline": "cleared"}}))
    store.apply(M.tag_read_event({"state": "confirmed", "label": "R1-L-F1", "expected": "R1-L-F1", "confidence": 0.96,
                                  "agreeing_views": 3, "views_total": 3, "crop_type": "lettuce", "location": "rail 1 station L layer 1 front"},
                                 "2026-10-25T10:00:10+00:00"))
    ripe = {"pod_id": "R2-M-F4", "crop_type": "lettuce", "side": "front"}
    store.apply(M.inspection_event(ripe, cls("stage 04- Harvest stage"), {"action": "flag_for_harvest", "reason": "ripe"},
                                   "2026-10-25T11:00:00+00:00", 1, "", {"age": 12, "harvest_age": 12, "stage": "s", "waiting_for_harvest": True, "cycle": 1}, None, None))
    sick = {"pod_id": "R1-R-F1", "crop_type": "mushroom", "side": "front"}
    store.apply(M.inspection_event(sick, cls("Juvenile", flag=True), {"action": "flag_for_treatment", "reason": "wilt"},
                                   "2026-10-25T11:05:00+00:00", 1, "", None, None, None))
    store.apply(M.tag_read_event({"state": "failed", "label": None, "expected": "R3-R-B4", "reason": "no tag found"}, "2026-10-25T11:10:00+00:00"))


def main():
    with mock_aws():
        backend = DynamoBackend(TABLE, "us-east-1")
        backend.create_table()
        store = FarmCloudStore(backend)
        populate(store)
        h = load_handler()

        print("\n[1] GET /pods/{tag}: one pod's record")
        code, body, raw = call(h, "GET /pods/{tag}", "R1-L-F1")
        pod = body.get("pod", {})
        ok(code == 200 and pod.get("pod_id") == "R1-L-F1" and pod.get("crop_type") == "lettuce", "200 with the pod's identity and crop type", str(body)[:200])
        want = store.profile("R1-L-F1")
        shown = {k: v for k, v in M.from_dynamo(want).items() if k not in ("pk", "sk") and not k.startswith(("gsi", "x_"))}
        ok(pod == shown, "the profile is exactly the record the application wrote (same records, nothing added or lost)")
        ok(all(k in pod for k in ("growth_stage", "disease_status", "harvest_readiness", "treatments_completed", "status", "tag_state")),
           "growth stage, disease status, treatment history, harvest readiness and tag state are all present")
        ok(pod["treatments_completed"] == 1 and pod["last_treatment_result"] == "worked" and pod["tag_state"] == "confirmed" and pod["tag_views"] == "3/3",
           "treatment and tag facts are right")
        ok(not any(k in pod for k in ("pk", "sk", "gsi1pk", "gsi2pk", "x_ripe")), "internal keys are not exposed")
        ok(raw["headers"]["content-type"] == "application/json" and raw["headers"]["access-control-allow-origin"] == "*" and
           raw["headers"]["cache-control"] == "no-store", "JSON, CORS and no-store headers")
        ok(isinstance(pod["visit_count"], int) and isinstance(pod["growth_confidence"], float), "numbers come back as numbers (not Decimals or strings)")
        ins = body["inspections"]
        ok(len(ins) == 10 and ins[0]["visit"] == 12 and ins[-1]["visit"] == 3, "inspections: the 10 newest, newest first")
        ok(len(body["treatments"]) == 1 and body["treatments"][0]["task"] == "TREAT" and body["treatments"][0]["status"] == "completed",
           "treatments: the treatment history")
        ok(body["harvests"] == [] and len(body["orders"]) == 2 and len(body["tag_reads"]) == 1 and body["tag_reads"][0]["tag_read"] == "R1-L-F1",
           "harvests, orders and tag reads are listed")
        ok(not any(k in ins[0] for k in ("pk", "sk")), "history items are cleaned too")
        _, b3, _ = call(h, "GET /pods/{tag}", "R1-L-F1", {"limit": "3"})
        ok(len(b3["inspections"]) == 3 and b3["inspections"][0]["visit"] == 12, "?limit=3 returns the 3 newest")
        _, b99, _ = call(h, "GET /pods/{tag}", "R1-L-F1", {"limit": "999"})
        ok(len(b99["inspections"]) == 12, "?limit=999 is clamped (all 12 available, never more than 50)")
        _, bx, _ = call(h, "GET /pods/{tag}", "R1-L-F1", {"limit": "abc"})
        ok(len(bx["inspections"]) == 10, "a nonsense limit falls back to the default")
        _, b1, _ = call(h, "GET /pods/{tag}", "R1-L-F1", {"limit": "0"})
        ok(len(b1["inspections"]) == 1, "?limit=0 is raised to 1")
        _, b0, _ = call(h, "GET /pods/{tag}", "R1-L-F1", {"history": "0"})
        ok(list(b0) == ["pod"], "?history=0 returns just the profile")

        print("\n[2] bad input is rejected before the table is touched")
        calls = []

        class Spy:
            def __getattr__(self, name):
                def f(*a, **k):
                    calls.append(name)
                    raise AssertionError("the table must not be touched")
                return f
        real, h.T = h.T, Spy()
        for bad in ("R9-X-F9", "r1-l-f1", "R1-L-F1; DROP", "../../etc/passwd", "R1-L-F1 ", "R1-L-F10", "", "A" * 5000, "R1-L-F1\n"):
            code, body, _ = call(h, "GET /pods/{tag}", bad)
            ok(code == 400 and "R1-L-F1" in body["error"], f"tag {bad[:22]!r} -> 400")
        for q in ({"status": "bogus"}, {"status": "harvest_ready' OR '1'='1"}, {"crop": "Lettuce"}, {"crop": "lettuce'"}, {"crop": "x" * 99}):
            code, _, _ = call(h, "GET /pods", None, q)
            ok(code == 400, f"query {q} -> 400")
        ok(calls == [], "none of those reached DynamoDB")
        h.T = real
        code, body, _ = call(h, "GET /pods/{tag}", "R3-M-F2")
        ok(code == 404 and "R3-M-F2" in body["error"], "a valid tag with no record -> 404")
        code, _, _ = call(h, "GET /nothing")
        ok(code == 404, "an unknown route -> 404")

        print("\n[3] GET /pods: find pods by status or crop")
        code, body, _ = call(h, "GET /pods")
        ids = sorted(p["pod_id"] for p in body["pods"])
        ok(code == 200 and ids == ["R1-L-F1", "R1-R-F1", "R2-M-F4", "R3-R-B4"] and body["count"] == 4, "every pod, once", str(ids))
        ok(sum(body["by_status"].values()) == 4 and body["by_status"].get("harvest_ready") == 1, "counts by status", str(body["by_status"]))
        ok(set(body["pods"][0]) == {"pod_id", "crop_type", "status", "growth_stage", "disease_status", "harvest_readiness", "tag_state", "updated_at"},
           "the list is compact (the full record is one call away)")
        _, body, _ = call(h, "GET /pods", None, {"status": "harvest_ready"})
        ok([p["pod_id"] for p in body["pods"]] == ["R2-M-F4"], "?status=harvest_ready finds the ripe pod")
        _, body, _ = call(h, "GET /pods", None, {"status": "identity_unconfirmed"})
        ok([p["pod_id"] for p in body["pods"]] == ["R3-R-B4"], "?status=identity_unconfirmed finds the pod whose tag could not be read")
        _, body, _ = call(h, "GET /pods", None, {"crop": "lettuce"})
        ok(sorted(p["pod_id"] for p in body["pods"]) == ["R1-L-F1", "R2-M-F4"], "?crop=lettuce finds the lettuce pods")
        _, body, _ = call(h, "GET /pods", None, {"crop": "cabbage"})
        ok(body["pods"] == [] and body["count"] == 0, "a crop with no pods -> empty list, not an error")

        print("\n[4] /health and the optional API key")
        code, body, _ = call(h, "GET /health")
        ok(code == 200 and body == {"ok": True}, "/health")
        hk = load_handler(API_KEY="s3cret-key")
        for name, headers in (("no key", {}), ("wrong key", {"x-api-key": "nope"}), ("almost the key", {"x-api-key": "s3cret-ke"}),
                              ("empty key", {"x-api-key": ""})):
            code, body, _ = call(hk, "GET /pods/{tag}", "R1-L-F1", headers=headers)
            ok(code == 401 and "s3cret" not in json.dumps(body), f"API key set, {name} -> 401 (and the key is not echoed)")
        code, body, _ = call(hk, "GET /pods/{tag}", "R1-L-F1", headers={"x-api-key": "s3cret-key"})
        ok(code == 200 and body["pod"]["pod_id"] == "R1-L-F1", "the right key -> 200")
        code, _, _ = call(hk, "GET /health", headers={})
        ok(code == 401, "even /health needs the key when one is set")

        print("\n[5] it fails safe")
        import botocore.exceptions as be

        class Boom:
            def get_item(self, **k):
                raise be.ClientError({"Error": {"Code": "AccessDeniedException", "Message": "SECRET internal arn:aws:dynamodb:..."}}, "GetItem")
            query = get_item
        h.T = Boom()
        code, body, raw = call(h, "GET /pods/{tag}", "R1-L-F1")
        ok(code == 500 and body == {"error": "internal error"} and "SECRET" not in raw["body"],
           "a database error -> 500 'internal error' (details go to the log, never to the caller)")
        h.T = real

        print("\n[6] no images, and read-only")
        everything = [call(h, "GET /pods/{tag}", t)[1] for t in ("R1-L-F1", "R2-M-F4", "R1-R-F1")] + [call(h, "GET /pods")[1]]
        try:
            for b in everything:
                M.assert_no_images(b)
            ok(True, "no response contains image data (checked with the same guard the writer uses)")
        except M.ImageDataRejected as exc:
            ok(False, "no response contains image data", str(exc))
        tree = ast.parse(open(SRC, encoding="utf-8").read())
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        ok(not (attrs & {"put_item", "update_item", "delete_item", "batch_write_item", "batch_writer", "scan", "create_table",
                         "delete_table", "transact_write_items"}), "the handler contains no write, delete or scan call", str(sorted(attrs)))
        ok({a for a in attrs if a in ("get_item", "query")} == {"get_item", "query"}, "it makes only GetItem and Query calls")
        before = len(backend.query(M.pod_pk("R1-L-F1")))
        snapshot = json.dumps(M.from_dynamo(store.profile("R1-L-F1")), sort_keys=True, default=str)
        for _ in range(3):
            call(h, "GET /pods/{tag}", "R1-L-F1")
            call(h, "GET /pods")
        ok(len(backend.query(M.pod_pk("R1-L-F1"))) == before and
           json.dumps(M.from_dynamo(store.profile("R1-L-F1")), sort_keys=True, default=str) == snapshot, "repeated requests change nothing in the table")

    print("\n[7] the CloudFormation template")
    import yaml

    class Loader(yaml.SafeLoader):
        pass
    Loader.add_multi_constructor("!", lambda l, s, n: l.construct_scalar(n) if isinstance(n, yaml.ScalarNode) else
                                 (l.construct_sequence(n) if isinstance(n, yaml.SequenceNode) else l.construct_mapping(n)))
    tpl = yaml.load(open(TEMPLATE, encoding="utf-8"), Loader=Loader)
    res = tpl["Resources"]
    inline = res["ApiFunction"]["Properties"]["Code"]["ZipFile"]
    source = open(SRC, encoding="utf-8").read()
    ok(inline.strip() == source.strip(), "the template's inline code IS aws/lambda/farm_api.py (run aws/sync_inline_code.py after editing)")
    ok(len(inline.strip()) <= 4096, f"it fits CloudFormation's 4096-character inline limit ({len(inline.strip())})")
    actions = {a for pol in res["ApiRole"]["Properties"]["Policies"] for st in pol["PolicyDocument"]["Statement"]
               for a in (st["Action"] if isinstance(st["Action"], list) else [st["Action"]])}
    ok(actions == {"dynamodb:GetItem", "dynamodb:Query"}, "the Lambda's role can only GetItem and Query", str(actions))
    ok(all(st["Effect"] == "Allow" and "*" not in "".join(r for r in st["Resource"] if "/index/*" not in r)
           for pol in res["ApiRole"]["Properties"]["Policies"] for st in pol["PolicyDocument"]["Statement"]),
       "scoped to the one table and its indexes (no wildcard resources)")
    routes = {r["Properties"]["RouteKey"] for r in res.values() if r["Type"] == "AWS::ApiGatewayV2::Route"}
    ok(routes == {"GET /health", "GET /pods/{tag}", "GET /pods"}, "exactly three GET routes, nothing that writes", str(routes))
    ok("ThrottlingRateLimit" in res["Stage"]["Properties"]["DefaultRouteSettings"], "the API is throttled")
    ok(res["ApiFunction"]["Properties"]["Runtime"].startswith("python3."), "a current Python runtime")
    ok(tpl["Parameters"]["ApiKey"]["NoEcho"] is True and tpl["Parameters"]["ApiKey"]["Default"] == "", "the API key is optional and never echoed")
    try:
        r = subprocess.run([shutil.which("cfn-lint") or "cfn-lint", TEMPLATE], capture_output=True, text=True, timeout=120)
        ok(r.returncode == 0, "cfn-lint finds no problems in the template", (r.stdout + r.stderr)[-400:])
    except Exception as exc:                                         # noqa: BLE001
        print(f"  skip  cfn-lint not available ({exc})")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("\n" + ("ALL API CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}") + f"   ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAILS else 0)
