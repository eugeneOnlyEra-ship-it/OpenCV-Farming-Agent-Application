# Farm Agent read-only API: Lambda behind an API Gateway HTTP API. Reads the farm's DynamoDB table, never writes.
# GET /health | GET /pods/{tag} | GET /pods?status=..|crop=..   See aws/DEPLOY.md. Facts only: the table holds no images.
import hmac, json, os, re
from decimal import Decimal
import boto3
from boto3.dynamodb.conditions import Key

T = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
API_KEY = os.environ.get("API_KEY", "")
TAG = re.compile(r"^R[1-3]-[LMR]-[FB][1-4]$")
CROP = re.compile(r"^[a-z]{3,20}$")
STATUSES = ("identity_unconfirmed", "treatment_in_progress", "disease_flagged", "verifying_treatment",
            "harvest_in_progress", "harvest_ready", "harvested", "growing")
BRIEF = ("pod_id", "crop_type", "status", "growth_stage", "disease_status", "harvest_readiness", "tag_state", "updated_at")


def clean(o):
    if isinstance(o, Decimal):
        return int(o) if o == o.to_integral_value() else float(o)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k not in ("pk", "sk") and not k.startswith(("gsi", "x_"))}
    return [clean(v) for v in o] if isinstance(o, list) else o


def out(code, body):
    return {"statusCode": code, "body": json.dumps(clean(body)), "headers": {
        "content-type": "application/json", "access-control-allow-origin": "*", "cache-control": "no-store"}}


def newest(tag, prefix, n):
    return T.query(KeyConditionExpression=Key("pk").eq("POD#" + tag) & Key("sk").begins_with(prefix),
                   ScanIndexForward=False, Limit=n)["Items"]


def handler(event, context):
    try:
        if API_KEY and not hmac.compare_digest((event.get("headers") or {}).get("x-api-key", ""), API_KEY):
            return out(401, {"error": "missing or wrong x-api-key header"})
        route, q = event.get("routeKey", ""), event.get("queryStringParameters") or {}
        if route == "GET /health":
            return out(200, {"ok": True})
        if route == "GET /pods/{tag}":
            tag = (event.get("pathParameters") or {}).get("tag", "")
            if not TAG.fullmatch(tag):
                return out(400, {"error": "tag must look like R1-L-F1"})
            pod = T.get_item(Key={"pk": "POD#" + tag, "sk": "PROFILE"}).get("Item")
            if not pod:
                return out(404, {"error": "no record for " + tag})
            body = {"pod": pod}
            if q.get("history") != "0":
                n = int(q["limit"]) if q.get("limit", "").isdigit() else 10
                n = max(1, min(n, 50))
                orders = newest(tag, "ORDER#", 50)
                body.update(inspections=newest(tag, "INSPECTION#", n), tag_reads=newest(tag, "TAGREAD#", 5),
                            treatments=[o for o in orders if o.get("task") == "TREAT"][:n],
                            harvests=[o for o in orders if o.get("task") == "HARVEST"][:n], orders=orders[:n])
            return out(200, body)
        if route == "GET /pods":
            status, crop = q.get("status"), q.get("crop")
            if status is not None and status not in STATUSES or crop is not None and not CROP.fullmatch(crop):
                return out(400, {"error": "status must be one of " + ", ".join(STATUSES) + "; crop like lettuce"})
            keys = ([("gsi_crop", "gsi1pk", "CROP#" + crop)] if crop else
                    [("gsi_status", "gsi2pk", "STATUS#" + s) for s in ([status] if status else STATUSES)])
            pods = [i for ix, k, v in keys for i in T.query(IndexName=ix, KeyConditionExpression=Key(k).eq(v))["Items"]]
            counts = {}
            for p in pods:
                counts[p["status"]] = counts.get(p["status"], 0) + 1
            return out(200, {"count": len(pods), "by_status": counts, "pods": [{k: p.get(k) for k in BRIEF} for p in pods]})
        return out(404, {"error": "not found"})
    except Exception as exc:
        print("error:", type(exc).__name__, exc)        # the details go to CloudWatch, never to the caller
        return out(500, {"error": "internal error"})
