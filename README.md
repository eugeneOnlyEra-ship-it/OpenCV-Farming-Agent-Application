# OpenCV Farming Agent — Integrated Application + PyBullet Gantry

This folder combines the OpenCV/ONNX farming application with the PyBullet gantry robot simulation. **Run the integrated application with `python3 app_ui.py`.** The `PyBullet robots` checkbox in the toolbar enables/disables the physical simulation layer.

## Multi-view pod localization (single camera agent)

The PyBullet camera agent now captures three calibrated RGB-D views while it
sweeps across a pod's three plant positions. Each view records the camera head's
actual simulated world position, its view/projection matrices, depth buffer, and
pod segmentation mask. `pybullet_sim/multiview_localization.py` unprojects the
masked depth pixels into the world frame and estimates the pod center and
long-axis yaw. The result is stored in `logs/pod_localization_latest.json` and
emitted as a `pod_localized` simulation event.

Before a service agent engages, it checks that the pod pose passed the minimum
point-count, multi-view, and orientation-alignment checks. A valid center estimate
is applied to a private copy of the pod and its plant target coordinates; invalid
localization withholds service, and work orders are marked failed rather than
reported as completed. The camera's image capture remains part of the existing
pause/resume/step/stop-controlled PyBullet job.

**Simulation limitation:** this first localization pass uses PyBullet's object
segmentation mask to isolate the target trough, and known simulated camera
intrinsics/extrinsics. It validates multi-view depth reconstruction and worker
coordination, but it is not yet a camera-only OpenCV detector and is not a physical
checkerboard calibration workflow. A real deployment would replace the simulator
mask with an OpenCV detector or fiducial/feature detector and calibrate the physical
camera. The current gantry service axes also cannot compensate for a genuinely
rotated pod; large estimated yaw is rejected rather than acted on.
## What is integrated

- The application remains the source of truth for image classification, decisions, crop history, logs, work-order approval, and follow-up scans.
- Every normal pod scan queues a corresponding camera-agent visit in PyBullet. The application decision colours the matching simulated pod.
- Dispatched `TREAT`, `HARVEST`, `RECAPTURE`, `VERIFY`, and `REPLANT` work orders are sent to the PyBullet executor instead of the old time-based simulated fleet. Robot work completes the same order in the application's order book.
- **Pause / Resume / Step / Stop are shared controls.** Pause prevents new robot jobs from starting while the active physical job finishes safely; Step releases one queued robot job while the app advances its scan step; Stop clears pending robot jobs, lets the current job finish, parks the gantry, and disconnects.
- The UI does not report the run as finished until its scan work and queued robot work have drained.
- The previous JSON work-order outbox is preserved for audit.

## Install and run (Fedora/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
sudo dnf install python3-tkinter
python dev/integration_smoke.py
python3 app_ui.py
```

The PyBullet GUI needs a working desktop display. If PyBullet is not installed or cannot initialise, the application logs a warning and falls back to its original built-in simulated fleet; uncheck **PyBullet robots** to intentionally use that mode. `python dev/integration_smoke.py` verifies the 72-pod mapping and shared control-state logic without requiring PyBullet, but it is not a substitute for a live motion test.

## Important simulation boundary

The gantry movement, camera-head motion, treatment/harvest/replant contact animation, and service markers are simulated in PyBullet. The camera image analyzed by OpenCV still comes from the application's image/frame pipeline; this integration does not yet render the live PyBullet camera view into OpenCV. Crop treatment and harvesting are demonstrations, not validated real-world robotic/agronomic operations. Real AWS services and physical hardware are not included.

---

## Application pipeline reference

This section describes the OpenCV/ONNX data-processing layer that remains the source of truth in the integrated build. The PyBullet robot layer is in `pybullet_sim/`; it consumes scan visits and approved work orders from this application rather than replacing the perception or decision code.

> **Full project documentation for developers / the next assistant: [`HANDOFF.md`](HANDOFF.md).** Regression checks: `python dev/regression.py`; UI screenshots: `dev/ui_screenshots.py`.

## What this does

For each of 72 pods, every visit:
1. **Capture** — `camera.py` looks up the pod's real image for this
   visit and submits it. It does not classify anything itself.
2. **Classify + decide** — `cloud_pipeline.py`'s `LocalCloudClient`
   runs the real `classify_pod()` (two YOLOv8-ONNX models per crop via
   `cv2.dnn`, see `perception.py`) and `decide_action()` (see
   `agent.py`) on a background thread, simulating the async
   S3 → Lambda → DynamoDB round-trip from the AWS architecture, without
   needing real AWS.
3. **Track history** — the result is written to a persistent per-pod
   history table (`dynamo_client.py`) and compared against that pod's
   prior visits for a trend note (`trend.py`).
4. **Log** — `run_logger.py` writes one row per pod-visit to CSV + JSON.
5. **Visualize** (optional) — `visualize.py` draws the real detection
   boxes onto the photo; `gallery.py` builds a browsable HTML page of
   every result.

`perception.py`, `agent.py`, `camera.py`, `cloud_pipeline.py`,
`run_logger.py`, `dynamo_client.py`, and `trend.py` are the application
layer. The application registry (`pod_registry.py`) holds crop/photo data;
`pybullet_sim/pod_registry.py` separately holds the physical geometry so
the two registries can use different pod IDs while mapping the same 72 pods.

## Farm structure (application data model)

3 station rails × 3 stations × 2 sides = 18 physical troughs; each
trough holds 4 pods (a 3-plant cluster each) → **72 pods, 24 per
crop**. `rail`/`station`/`side` are kept on every pod as grouping
metadata (useful for "how did rail 3 do" style breakdowns) and are mapped
to the matching rail, station, layer, and side in `pybullet_sim/`.

## Real images, real ground truth

All images are real photos from the fixed Roboflow datasets — none are
synthetic placeholders, and none repeat pixel-identical across pods.
Front pods are growth-stage-verified; back pods are disease-verified —
a single photo isn't independently verified for both, since it's
sourced from either the growth-stage dataset (a confirmed stage) or the
disease dataset (a confirmed disease/healthy status), never both at
once. `classify_pod()` still runs both models on every pod regardless —
the *other* aspect's reading is real model output with no curated
answer to check it against, not a wrong one.

Data augmentation wasn't actually needed for the base 72 images — real
per-class supply was abundant (500+ images for most growth-stage
classes, dozens to hundreds per disease class), enough to reach 72
unique real photos with zero repeats without manufacturing synthetic
variants.

## Multi-visit growth tracking

Originally every pod was a single snapshot — same image, forever, no
matter how many times you ran the pipeline. It now tracks real growth
progress across repeated visits:

- **`dynamo_client.py`** — a local stand-in for the DynamoDB per-pod
  health history table from the proposal's AWS architecture. Same two
  operations a real table needs (`put_item`, `query` by pod_id sorted
  by timestamp), persisted to a local JSON file so history survives
  across *separate* `python3 main.py` invocations, not just one run.
  Swapping to real DynamoDB later only touches this file.
- **Growth trajectories** — every front (growth-stage-verified) pod has
  a full sequence of real photos, one per stage from wherever it
  started through to harvest (e.g. a cabbage pod that started at S2_3
  has real S2_3 → S4_5 → S6_7 → S8 → S9 photos, all different real
  images, 57 additional real images beyond the base 72, none reused).
  Each visit swaps in the next one; once a pod reaches harvest, it
  stays there rather than cycling back to a seedling. Back
  (disease-verified) pods use the same photo every visit — deliberate,
  not a shortcut: a disease-verified pod re-inspected with no treatment
  applied should read the same each visit; that's not a flat demo,
  it's what "diseased, still diseased 3 visits later, still not
  treated" actually looks like, and `trend.py` still surfaces that
  persistence explicitly.
- **`trend.py`** — compares each visit's reading against that pod's
  history and produces a short note: `"stage advanced: S4_5 -> S6_7"`,
  `"disease flagged 3x in a row"`, `"disease cleared since last
  visit"`. Logged alongside the decision, not folded into
  `decide_action()` itself — that function's four actions are still a
  single-frame reading; the trend is an additive layer on top.
- **Progress continues correctly across separate runs**, not just
  within one `--visits N` call — a pod's trajectory position is driven
  by its *persisted* visit count (`dynamo_client`), not a counter local
  to the current invocation. Run `main.py` three separate times and
  you get the same progression as one `--visits 3` run — confirmed
  directly by testing three genuinely separate invocations.

## Visual output

`classify_pod()` was always computing real bounding boxes internally —
`growth_detections` / `disease_detections`, real pixel coordinates from
the real YOLOv8 models — but nothing drew them anywhere; they just fed
the single growth_stage/disease_flag summary. `--save-images` makes
that actually visible:

- Every processed pod gets an annotated JPEG under `logs/annotated/` —
  growth-model boxes in cyan, disease-model boxes in orange-red, each
  labeled with class + confidence, plus a color-coded banner (green /
  yellow / orange / red, matching the four actions) showing the pod ID,
  crop, stage, disease reading, decision, and trend note.
- A real diseased leaf can trigger dozens of genuine small-lesion
  detections (Alternaria leaf spot especially) — drawing every single
  one makes the image unreadable, so only the top 6 highest-confidence
  boxes per model get drawn; the banner says `[+N more detections not
  shown]` rather than silently dropping them.
- `logs/gallery.html` — one static page with every annotated image as a
  card (thumbnail, pod ID, crop, stage, disease, color-coded action).
  Open it directly in a browser, no server needed.

## Live monitor (real-time OpenCV window)

`--save-images` produces a report you review afterward. `live_monitor.py`
is a real desktop application — an actual `cv2.imshow` window showing
each pod being processed as it happens, not a report generated after
the fact. Same pipeline underneath (reuses `camera.py` /
`cloud_pipeline.py` / `dynamo_client.py` / `trend.py` /
`run_logger.py` exactly, nothing duplicated), with a live window and a
running dashboard on top:

```bash
python3 live_monitor.py                       # all 72 pods, live window
python3 live_monitor.py --crop lettuce --visits 3
python3 live_monitor.py --delay-ms 1200        # slower, more watchable pace
```

**Controls** (window focused): `q`/`Esc` quit (still flushes logs +
history for everything processed so far), `p` pause/resume, `n` step
one pod while paused, `+`/`-` speed up/slow down.

**Dashboard**, redrawn every frame: pod-visit progress bar, elapsed
time and average seconds/pod, and a live running count per action
(`log_healthy` / `flag_for_harvest` / `schedule_frequent_monitoring` /
`flag_for_treatment`), color-matched to the same banner colors used in
the saved images.

Needs a real display. Every image + box + overlay it draws goes through
the exact same `visualize.annotate()` used by `--save-images` — nothing
about the detection or drawing logic is different or duplicated between
the two, only how the frames get shown (one window, live, vs. files on
disk). If you're on a headless server/SSH session with no display, use
`main.py --save-images` and open the HTML gallery instead — or run
`live_monitor.py --no-window` to exercise the same loop headlessly as a
smoke test.

## Concurrent multi-pod processing

`live_monitor.py` shows one pod at a time. `concurrent_monitor.py` is
the sketch that kicked this off, built for real: **multiple pods
processing truly simultaneously** — pod 1 advancing through its image
1 → 2 → 3 growth sequence at the same time pod 2 advances through its
own, shown as a grid of live panels side by side.

**This needed a real fix first, not just new code.** Before building
it, I tested whether running two pods of the same crop concurrently
was actually safe — it wasn't. `perception.py` cached one shared model
object per crop, and calling `.forward()` on it from two threads at
once corrupted results *silently*: 24 out of 24 concurrent calls
returned different output than running the exact same calls one at a
time, with no exception raised. That's the dangerous kind of bug —
wrong answers that look like normal output.

Fixed at the root, not worked around: `perception.py`'s model cache is
now thread-local (each thread gets its own model instance, so there's
nothing to race on), and `cloud_pipeline.py`'s `LocalCloudClient` runs
work through a bounded thread pool instead of spawning an unlimited
thread per request — bounded because unbounded concurrency was also
measured to make things *slower* (three pods contending for one CPU
core already showed real overhead), and because a real Lambda
deployment has a concurrency limit too, so this isn't a workaround,
it's the architecturally honest version. Re-tested the exact same
race through the real `LocalCloudClient` path afterward: 0/6 mismatches.

**Architecture:** a fixed number of visual **slots** (`--slots`,
default 4) each run a worker thread that pulls the next pod off a
shared queue, runs that pod's full image sequence to completion (its
whole growth trajectory for front pods, one image for back pods), then
picks up the next pod — a fixed-lane dashboard, not "launch 72 threads
at once." Within one pod, images stay strictly in order (growth is
chronological); across slots, different pods genuinely run at the same
time. Every step still writes to the same persistent history
(`dynamo_client.py`) and gets the same trend note (`trend.py`) as
`main.py`/`live_monitor.py` — this is a third way to advance a pod's
history, not a separate concept from it.

```bash
python3 concurrent_monitor.py                # all 72 pods, 4 slots
python3 concurrent_monitor.py --slots 6       # 6 pods in flight at once
python3 concurrent_monitor.py --crop mushroom
python3 concurrent_monitor.py --pods 12 --slots 3
```

Controls: `q`/`Esc` quit (finishes in-flight steps first, no partial
log rows), `p` pause new pod pickup.

## Interactive app (`app_ui.py`) — the UI, not an OpenCV window

`python3 app_ui.py` opens a real desktop application (Tkinter — ships with
Python, so the dependencies are still just `opencv-python` + `numpy`; on
Linux you may need `sudo apt install python3-tk`). The photos are shown
plainly and every reading is rendered as UI beside them, laid out like the
design sketch: each pod is a card with a **disease** row and a **growth**
row (photo + score panel with one bar per class the model can output), and
alternate cards are mirrored so the scores sit on the inside.

- **Run controls:** Start / Pause / Resume / Step / Stop (`Space` pauses, `N` steps while paused).
  Pause lets in-flight scans finish and then starts nothing new; Step advances every un-held pod by one image.
- **Per pod:** *Hold* a lane on its pod, *Skip* the pod, *Re-scan* the image on screen
  (not written to history/log, so it never advances growth), and a timeline to step back
  through that pod's earlier images (◀ ● ● ○ ▶ LIVE).
- **Queue:** pick any pod in the sidebar and *Run next*, or *Drop from run*.
- **Live tuning (no restart):** delay, model sensitivity (box confidence floor — applies to the next
  scan/re-scan), disease and growth action thresholds (shown decisions re-evaluate instantly; a `*`
  marks a decision that differs from what was logged).
- **Inspect:** click a photo for a large view with both models' boxes; click a class row to
  highlight only that class's boxes; hover any box for its label. Boxes are drawn by the UI over the
  photo (toggle in the sidebar), not baked into the pixels.
- **Setup:** crop filter, pod limit, 1–4 pods on screen, optional history reset, *Save log now*.

`farm_engine.py` holds all run logic with no UI code (worker threads, queue, every control above as a
plain method, events through a thread-safe queue), and reuses `camera.py` → `cloud_pipeline.py` →
`perception.py`/`agent.py` → `dynamo_client.py`/`trend.py`/`run_logger.py` unchanged apart from one
backwards-compatible change: `capture_and_submit` / `submit_frame` accept an optional `conf_threshold`.
Pods resume where `dynamo_table.json` left off; a pod whose images are all used is re-inspected on its last image.

## Robot orders: instructions for the camera + worker agents

The robot system has four agents, two per side of the troughs (**front**: `camera_front`, `worker_front`;
**back**: `camera_back`, `worker_back`). The camera agents collect the images; `worker_planner.py` turns each
analysed image into instructions for the agents. Click **🤖 Robot orders** in the app to see them.

| Decision (`agent.py`) | Order produced |
|---|---|
| `flag_for_treatment` | worker **TREAT** — steps chosen by the disease the model named (remove leaves / remove plant / isolate-and-review…); needs human approval **unless auto-approve is on (the default)**. Completing it spawns a camera **VERIFY** visit. |
| `flag_for_harvest` | worker **HARVEST** (approved automatically while auto-approve is on) |
| `schedule_frequent_monitoring` | camera **RECAPTURE** — tuned to what was unclear (re-frame / zoom on the suspected region / add light) |
| `log_healthy` | nothing |

**Robot progress lives in the score panels.** After a pod's scores have been shown for a moment, the robot job that belongs to
an aspect takes over that panel with its progress in detail (every step with its live state, the agent, a progress bar, and
Approve / Reject when it is waiting for you): treatments and treatment checks appear where the **disease** scores were;
harvesting and replanting appear where the **growth** scores were; camera re-captures go to whichever aspect was unclear.
`⇄ scores` flips back to the scores while a job runs, and the scores return a few seconds after it finishes.

**A pod's next image waits for its robots.** After each scan the lane waits until no robot job (treatment, harvest,
replanting, camera re-capture, and the follow-up frame that closes the loop) is still open for that pod; only then does the next
image start. In manual-approval mode this is where the lane waits for you. `⏭ Skip pod` releases a lane. Human reviews and
later verification visits do not block. After a **harvest** the worker gets a **REPLANT** job (clear the pod, plant, water/feed)
in the growth panel. The pod's growth history is *not* reset automatically; do that when the new crop is in.

**Watching it run.** The main window shows all four agents live (current order, step name and progress bar; queued and waiting-for-approval counts), and each pod card shows its own order's progress. Click a tile to jump to that order. You never need to open Robot orders to follow automatic mode.

**Auto-approve switch (default ON).** The `⚡ Auto-approve` button (main toolbar and Robot orders window) is a master
switch. ON: every robot order is approved and executed automatically, including treatments, and camera follow-ups run a few
seconds after a treatment (demo timing; set `verify_after_hours` for real use). Click it to switch to `✋ Manual approval`:
orders created from then on - every robot order, including camera re-captures - wait for you in 🤖 Robot orders. Switching
back to ON releases the orders that were waiting. Orders already approved are not recalled. Escalations to a human
(**REVIEW**) always wait for a person, in both modes. Every switch is recorded in `trace.jsonl`.

The vision output drives the order: detection boxes become the worker's **target regions** (normalised image
coordinates), confidence and spread (localised vs widespread) change the steps and warnings, a TREAT order
supersedes an open RECAPTURE for the same pod, and a later contradicting reading marks an unapproved order **stale**.
The playbook is deliberately generic (no products, no doses) — a starting point for an agronomist, not agronomic advice.

**Connecting your real robots.** Every dispatched order is written as JSON to
`work_orders/<run>/outbox/<agent>/<order_id>.json`; swap `FileOutbox` in `worker_planner.py` for MQTT / AWS IoT Core /
SQS / ROS 2 (any object with `.send(dict)`). Robots report back through `OrderBook.begin / advance / complete`.
`work_orders/<run>/trace.jsonl` records every perception → decision → order → status event (agent-trace evidence).
Until real robots are attached, tick **Simulate the 4 robots** and the built-in simulator executes dispatched orders.

## Crop life cycles (realistic growth, `crop_cycle.py`)

The old registry gave every front pod a "staircase" (one photo per stage up to harvest), so each visit jumped a whole stage
and every pod of a crop ended its run at harvest. The interactive app now gives every front pod a **life cycle**:

* **Ordered stages that linger.** A crop spends several visits in each stage (`STAGE_VISITS` in `crop_cycle.py`; relative,
  compressed durations: cabbage slowest, mushrooms fastest). While it lingers the pod's photo of that stage repeats.
* **Staggered.** Pods start at different crop ages (neighbours deliberately differ), so a run brings only *some* pods to
  harvest: with 4 visits per pod, about 3 of 12 cabbage, 4 of 12 lettuce and 5 of 12 mushroom pods per run.
* **Harvest waits for the robots.** A ripe crop stays at the harvest stage (same photo) until it is harvested *and* the
  REPLANT job finishes; then the next image is stage 1 of a new crop. A rejected harvest therefore leaves the crop ripe.
* **Persistent.** Crop ages are saved in `dynamo_table_cycle.json` next to the history file; "Reset history" clears both.

Each card shows `crop age 7/12` (and `RIPE, waiting for harvest`). Back (disease) pods are single photos and have no cycle.
`main.py`, `live_monitor.py` and `concurrent_monitor.py` still use the old staircase. Limitations: a replanted crop shows
the same pod's photos again (only ~12 real photos exist for late stages), and because photos repeat while a stage lingers, a
photo that looks diseased keeps looking diseased after treatment.

**Mushroom label finding (unresolved - please check).** On the project's own labelled photos the mushroom growth model says
"Harvest" for the photos labelled Juvenile (4/4) and "Juvenile" for most photos labelled Harvest (10/12). The model's own ONNX
metadata matches `perception.py`, and the photos labelled Juvenile show large open caps while most labelled Harvest show thin
young stems, so the *labels in `pods_manifest.json`* are the likely culprit, not `perception.py` (which is unchanged).
`crop_cycle.py` therefore swaps those two labels for the cycle (`APPLY_LABEL_CORRECTIONS = True`; set it to `False` to use the
manifest as shipped). Details and evidence: `HANDOFF.md` section 10.4.

## Closing the loop, and the optional AI planner

**The loop.** When a camera order (RECAPTURE after an uncertain reading, VERIFY after a treatment) finishes, the
system takes a NEW frame, runs the same OpenCV analysis on it, compares with what the order was based on, and acts:

| Order | New frame says | What happens next |
|---|---|---|
| RECAPTURE | treat / harvest | that order is created (linked to the re-capture) |
| RECAPTURE | healthy | uncertainty resolved, nothing more to do |
| RECAPTURE | still uncertain | escalated to a human (**REVIEW** order) |
| VERIFY | disease below threshold | treatment verified |
| VERIFY | still flagged | escalated to a human (**REVIEW** order) |

Every step is in `work_orders/<run>/trace.jsonl` (`loop_closed` events) and shown in the Robot orders window with
before/after images. REVIEW orders are never bulk-approved.

> **The new frames are SIMULATED.** This build has no physical camera, so `frame_source.py` derives the new frame from the
> original photo by doing what the order told the camera agent to do (zoom on the suspected region, re-frame and brighten,
> improve contrast, or — for VERIFY — inpaint the regions the worker was told to remove). The analysis and comparison
> are real; the frames are not. Do not report results from them as measured accuracy. They are never written to the
> pod history. To use real frames, give the engine any object with `capture(order, out_dir) -> {"path","note","simulated"}`.

**The AI planner (optional, off by default).** Pick it in the sidebar. For TREAT and HARVEST orders an AI planner receives
the numeric evidence plus the pod's recent history and proposes a task as strict JSON; plain-code guardrails
(`llm_planner.resolve`) decide whether to accept it. It can only agree, or ask for a RECAPTURE ("take a better look"); it can
never create or cancel work, cannot soften strong evidence (≥3 boxes at ≥0.85), and cannot ask twice in a row for the same
pod. Approval, dispatch and the robot steps stay in `worker_planner.py`. Backends: *Scripted demo* (a hand-written stand-in,
**not an LLM**, for testing the wiring) or *Local LLM via Ollama* (a real model running offline on your machine;
`ollama pull llama3.2:3b`). `python3 try_llm_planner.py --selftest` runs the guardrail tests without any model.

## Files

| File | Role |
|---|---|
| `pod_registry.py` | Builds the 72 pod dicts (metadata + growth trajectories, no rig geometry) from the per-pod folders in `pybullet_sim/pods/` |
| `pybullet_sim/pods/<POD_ID>/` | **Each pod's own data set**: `pod.json` + its real photos (front pods: a 5-visit growth timeline; back pods: one disease-verified photo). Replaces `pods_manifest.json` and `sample_images/` |
| `pybullet_sim/pod_data.py`, `tools/migrate_pod_folders.py` | Loader/validator for those folders; the tool that builds them from the DataPipeline package |
| `cloud_model.py`, `cloud_backends.py`, `cloud_sync.py`, `cloud_cli.py`, `aws/` | The cloud database layer: what is stored per pod and how events become records (no AWS), the DynamoDB + local-file backends, the non-blocking sync worker with its outbox, the command-line viewer, and the CloudFormation template / IAM policy / deployment guide |
| `pybullet_sim/pod_labels.py` | The number printed on every trough, and the OpenCV reader that identifies a pod from it |
| `perception.py` | `classify_pod()` — real ONNX models via `cv2.dnn`, thread-local model cache |
| `agent.py` | `decide_action()` — the 4-action decision logic |
| `camera.py` | Capture + submit only, no classification |
| `cloud_pipeline.py` | `LocalCloudClient` — simulated async cloud round-trip, bounded thread pool |
| `dynamo_client.py` | `LocalDynamoTable` — persistent per-pod history, stand-in for DynamoDB |
| `trend.py` | Turns a pod's history into a short human-readable trend note |
| `run_logger.py` | CSV + JSON logging, aspect-aware accuracy scoring, visit/trend-aware |
| `visualize.py` | Draws real detection boxes + a decision banner onto each pod's photo |
| `gallery.py` | Builds one static HTML page presenting every annotated image from a run |
| `main.py` | Batch driver — processes every pod, optionally saves images + gallery, no window |
| `live_monitor.py` | Real-time OpenCV desktop app — one pod at a time, live window with a dashboard |
| `concurrent_monitor.py` | Real-time OpenCV desktop app — multiple pods at once, grid of live slots |
| `app_ui.py` | Interactive desktop app (Tkinter) — pod cards with score panels, pause/hold/skip/re-scan, live tuning |
| `farm_engine.py` | UI-independent run engine behind `app_ui.py` |
| `llm_planner.py` / `try_llm_planner.py` | Optional AI planner with guardrails; command-line lab to try it offline |
| `frame_source.py` | Simulated camera that derives the new frame for the closed loop (replace with a real source) |
| `HANDOFF.md` | Complete project documentation (architecture, formats, decisions, issues, roadmap, starter prompt) |
| `dev/regression.py`, `dev/ui_screenshots.py` | Headless regression suite; virtual-display screenshot tool |
| `crop_cycle.py` | Crop life cycles: ordered lingering stages, staggered ages, harvest-wait, replant restart |
| `worker_planner.py` | Turns each analysis into orders for the camera/worker agents; order book, approvals, outbox, robot simulator |
| `run_app.sh` | Linux launcher (creates `.venv`, installs pinned OpenCV 5, starts the app) |
| `models/` | The 6 `.onnx` files `perception.py` loads |

## Running it

```bash
pip install -r requirements.txt   # just opencv-python and numpy

python3 main.py                          # all 72 pods, 1 visit
python3 main.py --visits 5               # 5 simulated visits, growth trajectories play out
python3 main.py --crop mushroom --visits 3
python3 main.py --pods 8 --visits 3      # quick smoke test
python3 main.py --reset-history          # clear dynamo_table.json before this run
python3 main.py --save-images            # also save annotated images + logs/gallery.html
python3 live_monitor.py                  # real-time OpenCV window, one pod at a time
python3 concurrent_monitor.py            # real-time OpenCV window, multiple pods at once
```

No `pybullet` in the dependency list at all — this build genuinely
doesn't need it.

## Tested this session — fully, for real


Every line of this was exercised for real — real models, real images,
nothing mocked:

```
total_pods_visited: 72
actions: log_healthy=19, schedule_frequent_monitoring=12, flag_for_harvest=11, flag_for_treatment=30
growth_stage_accuracy: 27/36
disease_flag_accuracy: 36/36
per-pod cloud latency: ~0.5-0.8s (real CPU inference, no GPU)
```

`flag_for_treatment` being the largest bucket is by design, not a
warning sign: most back-side pods in this curated set are deliberately
diseased (healthy is only 1 of 5-9 disease classes per crop) to show
disease variety, not a realistic healthy/diseased ratio. The nine
growth-stage misses out of 36 are genuine model behavior on real
images, left as-is rather than reshuffled to hide it.

**Multi-visit tracking, also tested for real:** ran the full 72-pod set
across 5 simulated visits (360 total pod-visits), and separately
confirmed history survives correctly across 3 genuinely separate script
invocations (not just one `--visits 3` call) — a pod picks up its
trajectory exactly where the last run left it.

```
total_pods_visited: 360 (72 pods x 5 visits)
growth_stage_accuracy: 127/180
disease_flag_accuracy: 180/180
```

Not directly comparable to the single-visit 27/36 — a pod capped at
harvest repeats the same image (and the same right-or-wrong prediction)
on every remaining visit, so one miss at the cap point gets counted
multiple times in this aggregate. The real finding underneath it:
mushroom's growth model mixes up "Juvenile" and "Harvest" on some of
the trajectory photos, confirmed directly against the individual files
(`perception.classify_pod()` run on each one in isolation, no visit
logic involved). Not a new problem — the very first confusion-matrix
check on this model, much earlier in this project, already flagged
Juvenile as its weakest class at 40% recall, most often confused with
Intermediate or Harvest.

**Visual output, also tested for real:** ran `--save-images` on the
full 72-pod set — all 72 annotated JPEGs and the HTML gallery generated
without error, spot-checked visually (not just "no exception thrown").
Caught and fixed two real rendering bugs this way: banner text running
off the edge of narrower images, and a genuinely multi-lesion diseased
leaf producing 85 overlapping boxes with illegible stacked labels
before the top-6-per-model cap was added.

**`live_monitor.py`, also tested for real:** ran it under Xvfb (a
virtual display) end to end — it opened a real Qt-backed `cv2` window,
processed pods, and shut down cleanly with a correct summary, no
crash. Screen-scraping that virtual display for a visual check turned
out to be unreliable, so verification instead compared the exact
composed frame the window shows (scaled detection image + dashboard,
saved to a file via the same code path `imshow` uses) against expected
output directly. That caught a real bug: a detection label near the
right edge of the frame ran off it, since only the banner text had
edge-fitting — not individual box labels. Fixed by clamping the label's
x-position in `visualize.py`, which also fixes the same edge case in
`--save-images`' output, since both share that code.

**`concurrent_monitor.py`, tested at full scale, headlessly and
visually:** a headless run across all 72 pods with 4 slots completed
cleanly (129 total image-steps, every pod picked up and finished,
`disease_flag_accuracy: 36/36`). Visual correctness was checked the
same way as `live_monitor.py` — composing the real grid frame mid-run
and saving it to a file — which showed four different pods' real
photos, detection boxes, and decisions rendering correctly in their
own slots at once, each on its own step count, exactly as intended.

One honest number: on this specific machine (1 CPU core), 4 slots
measured about 19% faster than 1 slot (14.3s vs 17.7s for the same 12
pods) — real, but modest, because there's only one core for threads to
share regardless of how many are runnable. The concurrency-safety fix
is what matters everywhere; the wall-clock speedup from it scales with
how many cores you actually run it on, and should be more pronounced
on typical multi-core hardware than what this sandbox could show.

## Pod tags: the camera agent reads which pod it is working on

Every trough in the PyBullet scene carries a printed tag with its pod number (`R1-L-F1` = rail 1, left station, front, slot 1) on the aisle-facing wall, exactly the number the application and the data folders use. The gantry still drives to **coordinates** (a real gantry must), but it is **not told which pod it reached**: its camera captures three views and `pod_labels.PodLabelReader` reads the tag from those frames with OpenCV:

1. find the plate's black border (a closed, four-sided ring), 2. perspective-rectify it, 3. cut the fixed 7-character format into cells, 4. match each cell against glyph templates, 5. vote across the three views (≥2 must agree, or one very clean view, and none may contradict).

What the number is used for:

* the application's scores/decision are attached to the pod the camera **read**, and only if that equals the pod the job was **dispatched** for (`PyBulletBridge._check_identity`);
* a tag that is unreadable, hidden, wrong, or not the one expected at that location means **no work on that pod**: the service agent stays stowed, no scores are attached, and a work order **fails** (lane released) instead of treating the wrong pod;
* the pod is colour-coded by its decision only once its tag is confirmed; the UI fleet panel shows the tag the camera read, and every pod card shows a **POD TAG** line at the top of its disease panel ("waiting…" until the robot camera has visited, then the number it read and how many of the 3 views agreed, or "NOT CONFIRMED" in red; it stays visible while a robot job takes over that panel). **Click that line** to open the *Camera tag reading* window, which shows the process: the camera's frames with the tag plate OpenCV found outlined, the plate flattened and read character by character (with each character's match score), the vote across the views, then tag → `pybullet_sim/pods/<tag>/pod.json` → **crop type**, checked against what the application dispatched (for work orders a crop mismatch also withholds work); the pose record carries an `identity` block (label, confidence, views agreeing, per-view reads).

The reader is a purpose-built reader for the farm's own tag format, not a general OCR; it never "snaps" to the nearest valid pod, so an uncertain read is a failed read. Measured on the real renderer: 72/72 pods identified, 216/216 single views correct, 0 wrong reads, ~100 ms per pod; hidden or damaged tags are rejected, never misread (`python dev/check_pod_tags.py`). The production-grade equivalent is a fiducial (ArUco/QR) next to the human-readable number.

### Pod tags · live (watch it happen)

The **Pod tags** tab (next to Operations and Calibration) is a live monitor of the tag reading, for as long as a run with *PyBullet robots* is going:

* **LIVE CAMERA** – the front and back camera agents' frames, updated the instant each one is captured, with where the camera is (rail / station / layer) and what it read.
* **OPENCV READING** – the latest reading played back in five stages: the 3 frames with the plate OpenCV found outlined → the plate flattened → each character decoded with its match score → the vote across views → the pod record the number leads to (crop type) and the check against what was dispatched. A failed read shows exactly where it failed (e.g. "no tag found" in every view).
* **HISTORY** – every reading with time, camera, tag read, dispatched pod, crop, result and confidence; click a row to look at that reading again (then **▶ Follow live** to resume).

The frames and results are live. The reading itself takes ~0.1 s, too fast to watch, so the analysis panel replays each finished reading at a human pace and speeds up on its own if readings pile up (the header shows how many are waiting).

## Cloud database: every pod's identity connected to its data (Amazon DynamoDB)

The farm keeps one record per pod in the cloud, under **the number on its tag** (`R1-L-F1`): its **crop type, growth stage, every inspection, disease status, treatment history and harvest readiness**, plus whether the robot camera has confirmed the tag. **No images are stored in the cloud**: photos stay on this computer (a file name and SHA-256 of each inspected photo is kept, so a cloud record can be matched to its photo).

```
robot camera reads the tag ──┐                       ┌─ PROFILE  the pod right now (crop, stage, disease, treatments, harvest, status)
models + decision rules ─────┼─► background queue ──►│  INSPECTION#…  every scan                       (one DynamoDB table,
work orders (treat/harvest…) ┘   (retries, disk outbox)│  ORDER#…       every treatment / harvest / replant   keyed by pod)
                                                      └─ TAGREAD#…     every tag the camera read
```

* **Switching it on:** `export FARM_DDB_TABLE=FarmPods FARM_AWS_REGION=us-east-1` (see `aws/DEPLOY.md`: create the table with `aws/farm_table.yaml`, attach the least-privilege policy). With nothing set the same pipeline writes a **local file** (`dynamo_table_cloud.json`) and the interface says "☁ local", not AWS.
* **Seeing it:** the **☁** button (top bar) shows the sync state (`☁ AWS ✓ 143`, or `☁ OFFLINE · 12 waiting`) and jumps to the **Cloud** tab, which reads the records back from the table (only while the tab is showing). From a terminal: `python cloud_cli.py show R1-L-F1`, `pods --status harvest_ready`, `summary`, `history R1-L-F1 --kind treatments`, `ping`.
* **Never blocks the farm:** writes happen in a background thread with retries; if AWS is unreachable events wait in an on-disk outbox and are sent, in order and without double counting, when it returns.
* **Images can't leak:** bytes, arrays, image-like fields, base64 and oversized text are refused in code before any write.
* **Status per pod** (for the `gsi_status` index): `identity_unconfirmed`, `treatment_in_progress`, `disease_flagged`, `verifying_treatment`, `harvest_in_progress`, `harvest_ready`, `harvested`, `growing`.
* **Web endpoint:** `GET /pods/{tag}`, `GET /pods`, and `/health` use a read-only Lambda; `POST /observations` uses a separate write-scoped Lambda to persist OpenCV metadata and generate a work order. The write route requires `ObservationApiKey`; the local app uses `FARM_LAMBDA_API_URL` and `FARM_LAMBDA_API_KEY`. Offline observations are retained in `farm_lambda_outbox.jsonl` and retried. See `aws/DEPLOY.md`, Part 2.
* **Tested** (`python dev/check_cloud.py`) against a mock of DynamoDB (`moto`), never a real account: model rules, privacy, schema/CloudFormation/IAM agreement, local-vs-DynamoDB parity, outage and recovery, a real engine run, and the robot camera's tag reads. Your first run against real AWS is the real test: `python cloud_cli.py ping`.

## Per-pod data folders

`pybullet_sim/pods/<POD_ID>/` holds each pod's data set: `pod.json` (who/where/what, ground truth, its 3-D location in the scene) and its real photos. The folder name is the number on the tag, so "the tag the camera just read" leads straight to "this pod's data". `pod_registry.py` builds `PODS` from these folders (one copy of the data; `pods_manifest.json` and `sample_images/` are no longer read and can be deleted). Source: the DataPipeline package — every front pod has its own curated photo for each of 5 visits (lettuce and mushroom keep extra, different harvest-stage photos for the visits they must wait in). `crop_cycle.py` keeps its age/lingering/harvest-wait model but now shows a pod's *different* photo on successive visits within a stage instead of repeating one. Rebuild the folders with `python pybullet_sim/tools/migrate_pod_folders.py <DataPipeline folder>`; check them with `python pybullet_sim/pod_data.py`.

## PyBullet integration boundary

The application registry intentionally remains geometry-free (pod folders record each pod's scene location, but the registry dicts carry none). The gantry model, URDF, and physical pod registry are in `pybullet_sim/`. The OpenCV pipeline still analyses its configured image/frame source (the pod's photos); the PyBullet cameras' own rendered frames are used for localization, for the per-frame diagnostic scores, and now to **read the pod tag**.
