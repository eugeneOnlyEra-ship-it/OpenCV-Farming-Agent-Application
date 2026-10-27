"""
dev/regression.py -- headless regression checks for the whole application (no display needed).

Run from the PROJECT ROOT (needs the models/ folder):

    python dev/regression.py                    # everything (~2-3 minutes)
    python dev/regression.py guardrails cycle   # only some checks
    checks: guardrails  cycle  engine  modes

Exit code 0 = all passed. Everything is written to a temp folder; dynamo_table.json and logs/ are never touched.
A check that FAILS prints why. The `engine` check also prints the story of each pod (visit by visit) so you can read it.
"""
import collections
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import crop_cycle as cc                      # noqa: E402
import perception                            # noqa: E402
from farm_engine import SESSION_VISITS, FarmEngine   # noqa: E402
from pod_registry import PODS                # noqa: E402

BLOCKING = ("TREAT", "HARVEST", "REPLANT", "RECAPTURE")
FAILS = []


def ok(cond, name, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


def make_engine(tmp, name, ids, slots=2, manual=False):
    pods = [p for p in PODS if p["pod_id"] in ids]
    e = FarmEngine(pods, n_slots=slots, log_dir=os.path.join(tmp, "logs"),
                   history_path=os.path.join(tmp, f"{name}.json"), orders_dir=os.path.join(tmp, f"orders_{name}"))
    e.settings.delay_ms = 100                       # fast
    e.orders.policy.sim_step_seconds = 0.12
    e.orders.policy.auto_verify_delay_s = 1.0
    e.orders.policy.auto_approve_all = not manual
    return e


def wait_finished(e, limit):
    t = time.time()
    while e.state != "finished" and time.time() - t < limit:
        time.sleep(0.25)
    return e.state == "finished"


# ---------------------------------------------------------------------------
def check_guardrails():
    print("\n[guardrails] LLM planner guardrails (hostile / broken replies)")
    import io
    import contextlib
    import try_llm_planner
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = try_llm_planner.selftest()
    ok(rc == 0, "all guardrail self-tests pass", buf.getvalue()[-300:])


def check_cycle():
    print("\n[cycle] crop life-cycle model (pure, no models needed)")
    for crop in cc.STAGE_VISITS:
        order, H = cc.stage_order(crop), cc.harvest_age(crop)
        story = [cc.stage_at(crop, a)[0] for a in range(H + 4)]
        idx = [order.index(s) for s in story]
        ok(idx == sorted(idx), f"{crop}: stages only ever move forward")
        ok(all(story.count(st) == v for st, v in cc.STAGE_VISITS[crop].items()), f"{crop}: each stage lingers exactly its configured visits")
        ok(set(story[H:]) == {order[-1]}, f"{crop}: after the harvest age the crop stays at the harvest stage")
        pods = cc._front_pods(crop)
        ages = [cc.INITIAL_AGE[p["pod_id"]] for p in pods]
        ok(len(set(ages)) >= len(pods) // 2, f"{crop}: pods start at many different ages", str(ages))
        ripe_in_run = sum(1 for a in ages if a + SESSION_VISITS - 1 >= H)
        ok(0 < ripe_in_run < len(pods), f"{crop}: some but not all pods reach harvest in one run ({ripe_in_run}/{len(pods)})")
    p = next(x for x in PODS if x["pod_id"] == "R2-M-F4")
    path = os.path.join(tempfile.mkdtemp(), "c.json")
    s = cc.CycleStore(path)
    for _ in range(40):
        s.advance(p)
    ok(s.info(p)["age"] == cc.harvest_age("lettuce"), "a ripe crop stops ageing (waits for the harvest)")
    s.reset(p["pod_id"])
    ok(s.info(p)["age"] == 0 and s.info(p)["cycle"] == 2, "replanting resets the age and counts the new crop")
    ok(cc.CycleStore(path).info(p)["age"] == 0, "cycle state survives a restart (saved to disk)")
    missing = [(pid, st) for pid, d in cc.PHOTOS.items() for st, f in d.items() if not os.path.exists(f)]
    ok(not missing, "every pod has an existing photo for every stage", str(missing[:3]))
    ok(all(set(d) == set(cc.stage_order(next(x for x in PODS if x['pod_id'] == pid)['crop_type']))
           for pid, d in cc.PHOTOS.items()), "photo sets cover exactly the crop's stages")


def check_engine():
    print("\n[engine] full run: lane gating, harvest->replant, closed loop, history, persistence")
    perception.warm_up()
    tmp = tempfile.mkdtemp()
    ids = ["R2-M-F4", "R2-R-F1", "R1-M-F1", "R1-L-B2", "R1-L-B1"]   # ripe lettuce, mushroom near ripe, fresh lettuce, strong TREAT, RECAPTURE
    e = make_engine(tmp, "main", ids)
    scans, done_t, story = collections.defaultdict(list), {}, collections.defaultdict(list)
    orig_scan, orig_complete = e._scan, e.orders.complete

    def scan(pod, step, slot, rescan, **kw):
        t0 = time.time()
        r = orig_scan(pod, step, slot, rescan, **kw)
        scans[pod["pod_id"]].append((step, t0))
        if r is not None and not rescan:
            story[pod["pod_id"]].append((r.cycle.get("age"), r.verified.get("expected"), r.classification["growth_stage"],
                                         r.decision["action"]))
        return r

    def complete(oid, note="completed"):
        res = orig_complete(oid, note)
        if res:
            done_t[oid] = time.time()
        return res
    e._scan, e.orders.complete = scan, complete
    e.start()
    finished = wait_finished(e, 150)
    time.sleep(2)
    ok(finished, "run finishes by itself in auto mode")
    orders = e.orders.list_orders()
    print("  stories (crop age -> true stage / model reading / decision):")
    for pid in ids:
        print("   ", pid, " | ".join(f"{a}:{(t or '-')[:9]}/{(m or '-')[:9]}/{d.replace('flag_for_', '').replace('schedule_frequent_monitoring', 'monitor').replace('log_healthy', 'ok')}"
                                     for a, t, m, d in story[pid]))
    bad = []
    for o in orders:
        if o.task in BLOCKING and o.status == "completed" and o.order_id in done_t:
            for st, t0 in scans[o.pod_id]:
                if st > o.evidence["step"] and t0 < done_t[o.order_id] - 0.01:
                    bad.append((o.order_id, o.pod_id, st))
    ok(not bad, "a pod's next image never starts while a robot job is open on it", str(bad))
    ok(e.stats()["steps"] == sum(len(e.history.query(p)) for p in ids), "history rows == logged scans (follow-up frames never pollute history)")
    waiting = {o.task for o in orders if o.status == "awaiting_approval"}
    ok(waiting <= {"REVIEW"}, "auto mode: only human-review escalations wait", str(waiting))
    harvests = [o for o in orders if o.task == "HARVEST" and o.status == "completed"]
    replants = {o.follow_up_of: o for o in orders if o.task == "REPLANT"}
    ok(len(harvests) >= 1, "at least one harvest happened", str(len(harvests)))
    ok(all(h.order_id in replants and replants[h.order_id].status == "completed" for h in harvests),
       "every completed harvest was followed by a completed replant")
    for h in harvests:
        seq = [a for a, *_ in story[h.pod_id]]
        # anchor on the scan where the HARVEST decision was made: a ripe crop may legitimately be treated first
        # (the disease model can flag one of its several real harvest-stage photos) and harvested a visit later
        i = next((k for k, (a, _t, _m, d) in enumerate(story[h.pod_id])
                  if a is not None and a >= cc.harvest_age(h.crop_type) and "harvest" in str(d)), None)
        if i is not None and i + 1 < len(seq):
            ok(seq[i + 1] == 0, f"{h.pod_id}: the image after harvest+replant is a new crop (age 0)", str(seq))
    loops = [o for o in orders if o.outcome]
    ok(len(loops) >= 1, "closed loop produced at least one before/after outcome", str(len(loops)))
    ok(all(o.outcome["simulated"] for o in loops), "loop frames are flagged as simulated")
    ev = collections.Counter(json.loads(l)["event"] for l in open(os.path.join(e.orders.out_dir, "trace.jsonl")))
    ok({"order_created", "status", "loop_closed"} <= set(ev), "trace.jsonl has order_created / status / loop_closed events", str(dict(ev)))
    ok(os.path.isdir(os.path.join(e.orders.out_dir, "outbox")), "dispatched orders were written to the outbox")
    cyc_file = os.path.join(tmp, "main_cycle.json")
    ok(os.path.exists(cyc_file), "crop ages were saved")
    fresh_age_after = e.cycles.info(next(p for p in PODS if p["pod_id"] == "R1-M-F1"))["age"]
    e.close()

    e2 = make_engine(tmp, "main", ["R1-M-F1"], slots=1)          # SAME history/cycle files: a second run
    first = {}
    o2 = e2._scan

    def scan2(pod, step, slot, rescan, **kw):
        r = o2(pod, step, slot, rescan, **kw)
        if r is not None and not rescan:
            first.setdefault(pod["pod_id"], r.cycle.get("age"))
        return r
    e2._scan = scan2
    e2.start()
    wait_finished(e2, 60)
    ok(first.get("R1-M-F1") == fresh_age_after, "a second run continues each crop from its saved age", f"{first} vs {fresh_age_after}")
    e2.close()
    shutil.rmtree(tmp, ignore_errors=True)


def check_modes():
    print("\n[modes] manual approval stops the lane; switching to auto releases it")
    perception.warm_up()
    tmp = tempfile.mkdtemp()
    ids = ["R1-L-B2", "R1-L-B4", "R2-M-F4"]
    e = make_engine(tmp, "manual", ids, slots=1, manual=True)
    e.start()
    time.sleep(8)
    waiting = [o for o in e.orders.list_orders() if o.status == "awaiting_approval" and o.task != "REVIEW"]
    ok(e.state == "running" and e.stats()["pods_done"] == 0, "manual mode: the lane is stopped waiting for approval",
       f"state={e.state} done={e.stats()['pods_done']}")
    ok(len(waiting) >= 1 and e.slot_view(0)["status"] == "robot", "manual mode: orders wait and the lane shows 'robot'", str(len(waiting)))
    e.orders.set_auto_approve_all(True)
    ok(wait_finished(e, 90), "switching to auto releases waiting orders and the run completes")
    ok({o.task for o in e.orders.list_orders() if o.status == "awaiting_approval"} <= {"REVIEW"}, "only REVIEW escalations still wait")
    e.orders.set_auto_approve_all(False)
    e.close()
    shutil.rmtree(tmp, ignore_errors=True)


CHECKS = {"guardrails": check_guardrails, "cycle": check_cycle, "engine": check_engine, "modes": check_modes}

if __name__ == "__main__":
    want = sys.argv[1:] or list(CHECKS)
    t0 = time.time()
    for name in want:
        CHECKS[name]()
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + '; '.join(FAILS)}   ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAILS else 0)
