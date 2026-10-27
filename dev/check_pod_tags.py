"""
dev/check_pod_tags.py -- the pod number printed on every PyBullet trough, the OpenCV reader that identifies a pod
from it, the per-pod data folders, and the bridge's use of all three. Needs `pybullet` (it renders real frames).

Run from the project root:   python dev/check_pod_tags.py        (~1 min)

What is proven, against the REAL renderer and the REAL bridge (nothing here is mocked except the ONNX scoring of the
rendered views, which is irrelevant to identity):
  [1] the data layer: 72 pod folders, tag == folder name == pod_id, the application registry is built from them
  [2] the camera agent reads the right tag for all 72 pods from its own rendered frames (and never a wrong one)
  [3] the reader fails SAFE: a partly hidden tag is rejected, never misread as another pod
  [4] bridge + gantry end to end: the pod the camera read is the pod its scores are attached to
  [5] sabotage: a removed tag, or a tag swapped for another pod's, makes the agent refuse to work the pod and
      fail the order (instead of treating the wrong pod)
Exit code 0 = all passed.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dev"))

import cv2                                                    # noqa: E402
import cv2 as _cv2                                            # noqa: E402
import numpy as np                                            # noqa: E402
import pybullet as p                                          # noqa: E402

from pybullet_sim import pod_data, pod_labels, pod_registry as sim_registry   # noqa: E402
from pybullet_sim.bridge import PyBulletBridge                # noqa: E402
from pybullet_sim.gantry_robot import GantryRobot             # noqa: E402
from worker_planner import OrderBook, OrderPolicy             # noqa: E402
from check_bridge import add_order, wait_for                  # noqa: E402  (reuses its helpers)

FAILS = []


def ok(cond, name, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)
    return cond


def _no_scoring(robot):
    """Skip the ONNX scoring of rendered frames: identity does not depend on it and the models are heavy."""
    robot._score_rendered_views = staticmethod(lambda classify_pod, pod, items: {pos: {} for pos, _, _ in items})
    return robot


def check_data_layer():
    print("\n[1] every pod has its own data folder, named by the number on its tag")
    problems = pod_data.validate()
    ok(not problems, "pod_data.validate() finds no problems", "; ".join(problems[:3]))
    ok(len(pod_data.pod_ids()) == 72, "72 pod folders")
    import pod_registry
    ok(len(pod_registry.PODS) == 72 and all(os.path.isfile(x["image_path"]) for x in pod_registry.PODS),
       "the application registry is built from them and every photo exists")
    ok(all(os.path.dirname(x["image_path"]) == pod_data.pod_folder(x["pod_id"]) for x in pod_registry.PODS),
       "each pod's photo lives in that pod's own folder")
    front = [x for x in pod_registry.PODS if x["side"] == "front"]
    ok(all(len(x["trajectory"]) == 5 for x in front), "every front pod has its full 5-visit growth timeline")
    sim_ids = {(s["row"], s["col"], s["layer"], s["side"]) for s in sim_registry.PODS}
    ok(all((x["sim_location"]["row"], x["sim_location"]["col"], x["sim_location"]["layer"], x["sim_location"]["side"])
           in sim_ids for x in pod_data.load_all()), "every pod folder points at a real trough in the 3-D scene")


def check_reader_on_real_frames():
    print("\n[2] the camera agent reads every pod's tag from its own rendered frames")
    robot = _no_scoring(GantryRobot(pods=sim_registry.service_visit_order(), gui=False, speed_scale=8.0))
    ok(len(robot.pod_label_ids) == 72, "a tag plate is mounted on all 72 troughs", str(len(robot.pod_label_ids)))
    reads = {}

    def assess(pod, identity=None):
        reads[pod["pod_id"]] = identity
        return {"service_action": None, "camera_dwell": 0.0}

    robot.park_at_dock(sim_registry.TOP_DOCK_Y)
    for st in sim_registry.stations_in_route_order():
        given = {"front": False, "back": False}

        def poll(idle, st=st, given=given):
            out = {}
            for side in idle:
                if not given[side]:
                    given[side] = True
                    out[side] = st[side]
            return out
        robot.service_station_live(st["row"], st["col"], assess, poll, max_wall_seconds=120)
    robot.disconnect()
    ok(len(reads) == 72, "the assess callback was handed a tag read for all 72 pods", str(len(reads)))
    right = sum(1 for pid, i in reads.items() if i and i["confirmed"] and i["label"] == pod_labels.label_for_pod(
        next(x for x in sim_registry.PODS if x["pod_id"] == pid)))
    wrong = [pid for pid, i in reads.items() if i and i["confirmed"] and i["label"] != i["location_label"]]
    ok(right == 72, "72/72 pods identified correctly by OpenCV from the rendered frames", f"{right}/72")
    ok(not wrong, "no pod was ever confirmed as a DIFFERENT pod", str(wrong))
    ok(all(i["agreeing_views"] >= 2 for i in reads.values()), "every pod's tag was agreed by at least 2 of its 3 views")


def check_fail_safe_reader():
    print("\n[3] the reader fails safe on a damaged tag")
    reader = pod_labels.PodLabelReader()
    frame = cv2.imread("logs/multiview/R0C0-L0-F_visit1_view2.png")
    if frame is None:
        return ok(False, "a real captured frame is available (run [2] first)")
    base = reader.read_frame(frame)
    ok(base["valid"] and base["text"] == "R1-L-F1", "the untouched frame reads R1-L-F1", str(base["text"]))
    x, y, w, h = cv2.boundingRect(np.array(base["corners"], np.float32))
    wrong_valid = 0
    for name, (x0, x1) in {"left third hidden": (x, x + w // 3), "middle hidden": (x + w // 3, x + 2 * w // 3),
                           "right third hidden": (x + 2 * w // 3, x + w)}.items():
        damaged = frame.copy()
        damaged[y:y + h, x0:x1] = (170, 170, 170)
        r = reader.read_frame(damaged)
        if r["valid"] and r["text"] != "R1-L-F1":
            wrong_valid += 1
        ok(not r["valid"] or r["text"] == "R1-L-F1", f"{name}: rejected, or still the right tag (never another pod)",
           str(r["text"]))
    ok(wrong_valid == 0, "no damaged view was misread as a different valid tag")
    blank = np.full_like(frame, 200)
    ok(reader.read_views([blank, blank, blank])["label"] is None, "frames with no tag at all give no identity")
    mixed = reader.read_views([frame, cv2.imread("logs/multiview/R0C0-L1-F_visit1_view2.png"), frame])
    ok(mixed["label"] is None and mixed["conflict"], "views that disagree about the tag give no identity",
       str(mixed["reason"]))


def _bridge_with_real_robot():
    book = OrderBook(__import__("tempfile").mkdtemp(), OrderPolicy(auto_approve_all=True, simulate_robots=False))
    robots, logs = [], []

    def factory(bridge):
        r = _no_scoring(GantryRobot(pods=bridge._sim_pods, gui=False, speed_scale=8.0))
        robots.append(r)
        return r
    bridge = PyBulletBridge(gui=False, robot_factory=factory, on_log=lambda m, lvl="info": logs.append((lvl, m)))
    bridge.attach_orders(book)
    return bridge, book, robots, logs


def check_bridge_end_to_end():
    print("\n[4] bridge + real gantry: scores are attached to the pod the camera READ")
    from pod_registry import PODS
    bridge, book, robots, logs = _bridge_with_real_robot()
    ok(bridge.start(timeout=120)[0], "bridge starts with the real gantry")
    robot = robots[0]
    marks = []
    original = robot.mark_pod_decision
    robot.mark_pod_decision = lambda pid, action: (marks.append((pid, action)), original(pid, action))[1]
    pod = next(x for x in PODS if x["pod_id"] == "R1-L-F1")
    bridge.enqueue_visit(pod, {"action": "log_healthy"}, {"growth_stage": "S4_5"}, 1)
    ok(wait_for(lambda: bridge.wait_until_idle(0.05), 90), "the visit completes")
    rec = robot.localization_records.get("R0C0-L0-F") or {}
    ident = rec.get("identity") or {}
    ok(ident.get("confirmed") and ident.get("label") == "R1-L-F1", "the pose record carries the tag the camera read",
       str(ident.get("label")))
    ok(rec.get("application_pod_id") == "R1-L-F1" and rec.get("model_scores") == {"growth_stage": "S4_5"},
       "the application's scores are attached to that pod", str(rec.get("application_pod_id")))
    partner = (robot.localization_records.get("R0C0-L0-B") or {}).get("identity") or {}
    ok(partner.get("label") == "R1-L-B1", "its back-side counterpart was identified as R1-L-B1 too", str(partner.get("label")))
    ok(any("Camera read tag R1-L-F1" in m for _, m in logs), "the application log says which tag the camera read")
    tr = bridge.tag_read("R1-L-F1") or {}
    ok(tr.get("state") == "confirmed" and tr.get("label") == "R1-L-F1" and tr.get("agreeing_views", 0) >= 2,
       "the UI can ask the bridge which tag the camera read for this pod", str(tr))
    ok(bridge.tag_read("R3-R-B4") is None, "a pod the camera has not visited has no tag read yet")
    import cv2 as _cv2
    sheet = _cv2.imdecode(np.frombuffer(tr.get("sheet_png") or b"", np.uint8), _cv2.IMREAD_COLOR)
    ok(sheet is not None and sheet.shape[1] == 800 and sheet.shape[0] > 400,
       "the bridge holds a picture of the reading (camera frames, located plate, per-character decode) to show in the UI")
    folder_record = pod_data.load_pod("R1-L-F1")
    ok(tr.get("crop_type") == folder_record["crop_type"] and (tr.get("record") or {}).get("folder") == folder_record["folder"],
       "the tag read leads to that pod's record and crop type", f"{tr.get('crop_type')} vs {folder_record['crop_type']}")
    ok(any("-> " + folder_record["crop_type"] in m for _, m in logs), "the application log says which crop the tag belongs to")
    ok(len(marks) >= 1 and marks[0][0] == "R0C0-L0-F", "the pod was colour-coded only after its tag was confirmed")
    # the live feed the Pod tags monitor polls
    entries, latest = bridge.tag_feed_since(0)
    views_in = [e for e in entries if e["kind"] == "view"]
    reads_in = [e for e in entries if e["kind"] == "read"]
    ok(len(views_in) >= 6 and all(_cv2.imdecode(np.frombuffer(e["jpg"], np.uint8), _cv2.IMREAD_COLOR) is not None for e in views_in),
       "the live feed carries every camera frame as it was captured (3 per pod, front and back)", str(len(views_in)))
    ok(sorted({e["view"] for e in views_in}) == [1, 2, 3] and {e["side"] for e in views_in} == {"front", "back"},
       "frames are labelled with their view number and camera side")
    ok(len(reads_in) >= 2 and all(len((e["evidence"] or {}).get("views", [])) == 3 and (e["evidence"] or {}).get("flat_png")
                                  for e in reads_in), "each reading carries its evidence: 3 frames, the flattened plate, per-character results")
    ok([e["seq"] for e in entries] == sorted(e["seq"] for e in entries) and bridge.tag_feed_since(latest)[0] == [],
       "the feed is ordered and a poll with the latest sequence number returns nothing new")
    ok(all("jpg" not in ev for ev in robot.events), "camera frames are not piled up in the robot's permanent event log")
    # the dispatch cross-check itself (a tag that is readable but is not the pod that was dispatched)
    job = {"kind": "visit", "app_pod_id": "R1-L-F2"}
    good = {"confirmed": True, "label": "R1-L-F1", "at_expected_location": True, "confidence": .9,
            "agreeing_views": 3, "views_total": 3, "location_label": "R1-L-F1"}
    v = bridge._check_identity(job, good)
    ok(not v["ok"] and "R1-L-F2 was dispatched" in v["reason"], "a read that differs from the dispatched pod is rejected", v["reason"])
    ok(bridge._check_identity({"kind": "visit", "app_pod_id": "R1-L-F1"}, good)["ok"], "a read that matches the dispatch is accepted")
    ok(bridge._check_identity(job, None) is None, "a robot that reads no tags keeps the old positional behaviour")
    wrong_crop = {"kind": "order", "app_pod_id": "R1-L-F1", "order": {"pod_id": "R1-L-F1", "crop_type": "lettuce"}}
    v = bridge._check_identity(wrong_crop, good)
    ok(not v["ok"] and "cabbage" in v["reason"], "an order for a different crop than the tag's pod record is rejected", v["reason"])
    right_crop = {"kind": "order", "app_pod_id": "R1-L-F1", "order": {"pod_id": "R1-L-F1", "crop_type": folder_record["crop_type"]}}
    ok(bridge._check_identity(right_crop, good)["ok"], "an order whose crop matches the tag's pod record is accepted")
    bridge.request_stop(); bridge.join(10)


def check_sabotaged_tags():
    print("\n[5] a missing or wrong tag: the agent refuses to work the pod and the order FAILS")
    bridge, book, robots, logs = _bridge_with_real_robot()
    bridge.start(timeout=120)
    robot = robots[0]
    # (a) R1-L-F1's tag is torn off; (b) R1-L-F3 wears R1-L-F4's tag
    p.removeBody(robot.pod_label_ids["R0C0-L0-F"], physicsClientId=robot.client)
    wrong_tex = p.loadTexture(robot._label_texture("R1-L-F4"))
    p.changeVisualShape(robot.pod_label_ids["R0C0-L2-F"], -1, textureUniqueId=wrong_tex, physicsClientId=robot.client)
    o_missing = add_order(bridge, book, "WO-MISSING", "TREAT", "R1-L-F1", "front")
    o_swapped = add_order(bridge, book, "WO-SWAPPED", "TREAT", "R1-L-F3", "front")
    o_good = add_order(bridge, book, "WO-GOOD", "TREAT", "R1-L-F2", "front")
    ok(wait_for(lambda: all(o.status in ("completed", "failed") for o in (o_missing, o_swapped, o_good)), 180),
       "all three orders reach an end state")
    ok(o_missing.status == "failed", "missing tag: the order FAILED instead of treating an unidentified pod", o_missing.status)
    ok(o_swapped.status == "failed", "wrong tag: the order FAILED instead of treating the wrong pod", o_swapped.status)
    ok(o_good.status == "completed", "the pod with an intact tag was still treated normally", o_good.status)
    ok(not book.pod_busy("R1-L-F1") and not book.pod_busy("R1-L-F3"), "the application's lanes were released")
    ids = robot.pod_identities
    ok(not ids["R0C0-L0-F"]["confirmed"], "the torn-off tag was not confirmed", str(ids["R0C0-L0-F"]["reason"]))
    ok(ids["R0C0-L2-F"]["confirmed"] and ids["R0C0-L2-F"]["label"] == "R1-L-F4" and not ids["R0C0-L2-F"]["at_expected_location"],
       "the swapped tag was read faithfully as R1-L-F4 and recognised as out of place")
    ok(any(lvl == "error" and "NOT confirmed" in m for lvl, m in logs), "the application log reports the failed identification")
    ok((bridge.tag_read("R1-L-F1") or {}).get("state") == "failed" and (bridge.tag_read("R1-L-F3") or {}).get("state") == "failed",
       "the UI is told the tag was NOT confirmed for the missing and the swapped tag")
    ok((bridge.tag_read("R1-L-F2") or {}).get("state") == "confirmed", "and confirmed for the intact one")
    fails = [e for e in bridge.tag_feed_since(0)[0] if e["kind"] == "read" and e["state"] == "failed"]
    ok(len(fails) >= 2 and all(e["reason"] for e in fails), "failed readings reach the live feed with their reason")
    missing = next((e for e in fails if e["pod_id"] == "R0C0-L0-F"), None)
    ok(missing is not None and all(v["corners"] is None for v in missing["evidence"]["views"]),
       "the torn-off tag shows as 'no tag found' in every view")
    failed_sheet = _cv2.imdecode(np.frombuffer((bridge.tag_read("R1-L-F1") or {}).get("sheet_png") or b"", np.uint8), _cv2.IMREAD_COLOR)
    ok(failed_sheet is not None, "a failed read still comes with its picture, so you can see why it failed")
    bridge.request_stop(); bridge.join(10)


if __name__ == "__main__":
    t0 = time.time()
    check_data_layer()
    check_reader_on_real_frames()
    check_fail_safe_reader()
    check_bridge_end_to_end()
    check_sabotaged_tags()
    print("\n" + ("ALL POD-TAG CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}") + f"   ({time.time() - t0:.0f}s)")
    sys.exit(1 if FAILS else 0)
