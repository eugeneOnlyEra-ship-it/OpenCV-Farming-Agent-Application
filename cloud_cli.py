#!/usr/bin/env python3
"""
cloud_cli.py -- look at, and set up, the farm's cloud database from the command line.

    python cloud_cli.py ping                     is the database reachable? (the real check of your AWS setup)
    python cloud_cli.py create-table             create the DynamoDB table (idempotent; or use aws/farm_table.yaml)
    python cloud_cli.py summary                  how many pods are in each state
    python cloud_cli.py pods --status harvest_ready        which pods need what  (statuses: see --help)
    python cloud_cli.py pods --crop lettuce
    python cloud_cli.py show R1-L-F1             one pod: identity, crop, growth, disease, treatment, harvest, tag
    python cloud_cli.py history R1-L-F1 --kind treatments   inspections | orders | treatments | harvests | tags
    python cloud_cli.py sync                     send anything that was waiting on disk (after an outage)

Add --json to any read command for machine-readable output.
Which database: the same environment variables as the application (FARM_CLOUD / FARM_DDB_TABLE / FARM_AWS_REGION; see
cloud_sync.py). With none set it reads the LOCAL file the application writes (dynamo_table_cloud.json).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from cloud_model import STATUSES                                             # noqa: E402
from cloud_sync import CloudSync, FarmCloudStore, config_from_env, make_backend   # noqa: E402

FIELDS = [
    ("identity", ("pod_id", "tag", "crop_type", "rail", "station_letter", "side", "slot")),
    ("tag (robot camera)", ("tag_state", "tag_last_read_at", "tag_confidence", "tag_views", "tag_failure_reason")),
    ("growth", ("growth_stage", "growth_confidence", "crop_age", "harvest_age", "crop_cycle", "visit_count")),
    ("last inspection", ("last_inspection_at", "last_action", "last_reason", "last_trend_note")),
    ("disease", ("disease_status", "disease_flag", "disease_name", "disease_confidence", "disease_streak",
                 "disease_first_flagged_at")),
    ("treatment history", ("treatments_completed", "treatments_failed", "last_treatment_at", "last_treatment_order",
                           "last_treatment_result", "open_treatment")),
    ("harvest", ("harvest_readiness", "harvest_progress", "harvests_completed", "last_harvest_at", "open_harvest",
                 "replanted_at")),
    ("overall", ("status", "updated_at")),
]


def get_store(args):
    cfg = config_from_env(base=args.base)
    backend = make_backend(cfg)
    if backend is None:
        sys.exit("FARM_CLOUD=off: there is no database to read.")
    return cfg, backend, FarmCloudStore(backend)


def out(args, data, text):
    print(json.dumps(data, indent=2, default=str) if args.json else text)


def show(args):
    cfg, backend, store = get_store(args)
    p = store.profile(args.pod)
    if not p:
        sys.exit(f"{args.pod}: no record in {backend.where}")
    if args.json:
        return out(args, p, "")
    print(f"\n{args.pod}   ({backend.where})")
    for title, keys in FIELDS:
        rows = [(k, p[k]) for k in keys if p.get(k) is not None]
        if rows:
            print(f"\n  {title}")
            for k, v in rows:
                print(f"    {k:<26} {v}")
    n = {k: len(getattr(store, k)(args.pod)) for k in ("inspections", "orders", "tag_reads")}
    print(f"\n  history items: {n['inspections']} inspections, {n['orders']} orders, {n['tag_reads']} tag reads "
          f"(python cloud_cli.py history {args.pod})\n")


def history(args):
    cfg, backend, store = get_store(args)
    rows = {"inspections": store.inspections, "orders": store.orders, "treatments": store.treatments,
            "harvests": store.harvests, "tags": store.tag_reads}[args.kind](args.pod)
    if args.json:
        return out(args, rows, "")
    print(f"\n{args.pod} - {args.kind} (newest first)")
    for r in rows:
        if r["entity"] == "INSPECTION":
            d = f"{r['ts']}  visit {r.get('visit')}  {r.get('growth_stage')}  disease={r.get('disease_name') or 'no'}  -> {r.get('action')}"
        elif r["entity"] == "ORDER":
            d = f"{r['created']}  {r['order_id']}  {r['task']:<9} {r['status']:<10} {r.get('title')}" + (
                f"  [{r['outcome_result']}]" if r.get("outcome_result") else "")
        else:
            d = f"{r['ts']}  {r['state']:<9} tag read: {r.get('tag_read')}  crop(from tag): {r.get('crop_type_from_tag')}  conf {r.get('confidence')}"
        print("  " + d)
    if not rows:
        print("  (none)")
    print()


def pods(args):
    cfg, backend, store = get_store(args)
    if args.status:
        rows = store.pods_with_status(args.status)
    elif args.crop:
        rows = store.pods_of_crop(args.crop)
    else:
        sys.exit("give --status or --crop")
    if args.json:
        return out(args, rows, "")
    for r in rows:
        print(f"  {r['pod_id']:<9} {r.get('crop_type'):<9} {r['status']:<22} stage={r.get('growth_stage')}  disease={r['disease_status']}  harvest={r['harvest_readiness']}")
    print(f"  ({len(rows)} pods)")


def summary(args):
    cfg, backend, store = get_store(args)
    counts = store.status_counts()
    if args.json:
        return out(args, counts, "")
    print(f"\n{backend.where}")
    for s in STATUSES:
        print(f"  {s:<24} {counts[s]}")
    print(f"  {'total':<24} {sum(counts.values())}\n")


def ping(args):
    cfg, backend, store = get_store(args)
    good, msg = backend.ping()
    print(("OK   " if good else "FAIL ") + msg)
    if not good and cfg.backend == "aws":
        print("     Check: the table exists (python cloud_cli.py create-table / aws/farm_table.yaml), the region is right "
              "(FARM_AWS_REGION), and your credentials can call dynamodb:DescribeTable (aws/iam_policy.json).")
    sys.exit(0 if good else 1)


def create_table(args):
    cfg, backend, store = get_store(args)
    if cfg.backend != "aws":
        sys.exit("create-table is for the real DynamoDB table: set FARM_DDB_TABLE (and FARM_AWS_REGION).")
    print("created" if backend.create_table() else "already exists", "-", backend.where)


def sync(args):
    cfg, backend, store = get_store(args)
    worker = CloudSync(store, cfg.outbox_path, log=lambda m, lvl="info": print(m))
    n = worker.outbox_size()
    sent = worker.replay_outbox()
    print(f"{n} event(s) were waiting; {sent} sent; {worker.outbox_size()} still waiting")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="dynamo_table", help="path stem of the local cloud file / outbox (default: dynamo_table)")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ping").set_defaults(fn=ping)
    sub.add_parser("create-table").set_defaults(fn=create_table)
    sub.add_parser("summary").set_defaults(fn=summary)
    sub.add_parser("sync").set_defaults(fn=sync)
    s = sub.add_parser("show")
    s.add_argument("pod")
    s.set_defaults(fn=show)
    h = sub.add_parser("history")
    h.add_argument("pod")
    h.add_argument("--kind", default="inspections", choices=["inspections", "orders", "treatments", "harvests", "tags"])
    h.set_defaults(fn=history)
    p = sub.add_parser("pods")
    p.add_argument("--status", choices=STATUSES)
    p.add_argument("--crop")
    p.set_defaults(fn=pods)
    args = ap.parse_args()
    try:
        args.fn(args)
    except BrokenPipeError:                       # e.g. `... | head`: the reader closed early; not an error
        sys.stderr.close()


if __name__ == "__main__":
    main()
