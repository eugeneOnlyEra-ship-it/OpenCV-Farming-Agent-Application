# Farm Agent — Project Handoff

> **Read this first.** It is written for a *new assistant (or developer)* who has never seen this project and must
> continue it without the earlier conversation. Everything here was checked against the code at the time of writing
> (October 2026). Where something is **simulated, untested or uncertain, it says so** — please keep that honesty in
> anything you add.

---

## 0. TL;DR

**What it is.** A vertical-farm monitoring system built for the *OpenCV AI Competition 2026 (powered by AWS)*. Photos of
growing pods (cabbage, lettuce, mushrooms) are analysed with OpenCV (ONNX YOLO models through `cv2.dnn`); the analysis
decides what the **four robot agents** (a camera agent and a worker agent on each of two sides) should do; a human
approves or lets it run automatically; the robots act; a new photo is taken and analysed; the result decides what
happens next (a perception → decision → action → re-perception loop). It ships as a desktop app (Tkinter).

**State right now.** The full loop works end to end **in simulation**:

```
photo → OpenCV 5 + ONNX models → rule table picks a task → (optional) AI planner may ask for a better look
      → orders for camera/worker agents → human approval (or auto) → robots execute (simulated)
      → NEW photo (simulated) → re-analysed → result decides the next step → pod's next image
```

**Integration update (Oct 11, 2026).** This folder now combines the app and the former `OpenCV-Farming-Agent` PyBullet gantry project. `app_ui.py` is the entry point; `pybullet_sim/bridge.py` maps all 72 application pods to the physical geometry registry, queues a robot camera visit for each logged scan, and routes dispatched work orders to the gantry. When PyBullet is enabled, the old time-based fleet simulator is disabled so work orders are not executed twice. Pause / Resume / Step / Stop are shared through the engine; pause takes effect at a robot-job boundary (the active job is allowed to finish safely), and the run waits for queued robot work to drain before reporting completion.

**Verification caveat for this integration.** Python compilation and all 72 pod-ID/geometry mappings passed in the build environment. A live PyBullet movement test could not run there because `pybullet` is not installed and package-network access was unavailable. The full regression command also did not finish within the sandbox's 5-minute limit, so do not describe that suite as passed. Install `requirements.txt` in the project's Python environment and run `python dev/regression.py` plus a short GUI run before relying on the physical simulation path.

**The five things you most need to know**

1. **Simulated vs real.** Real: OpenCV analysis, decision logic, order/approval/loop logic, and UI. **Simulated:**
   robot motion (the PyBullet gantry when enabled, otherwise the built-in time-based fleet), follow-up frames derived from the
   configured image/crop-cycle pipeline, and cloud components (local stand-ins for S3/Lambda/DynamoDB). The PyBullet camera
   does not yet supply the pixels analyzed by OpenCV. **Nothing runs on AWS yet** — that remains the competition's biggest gap.
2. **Mushroom ground-truth labels look swapped (UNRESOLVED, needs the user's eyes).** See §10.4. `perception.py` is
   untouched and matches the model's own metadata; the correction lives in `crop_cycle.py` behind one switch.
3. **A real LLM has never been run** against this code (no model could be downloaded in the build sandbox). The "Scripted
   demo" planner is a hand-written stand-in, **not an LLM**. The Ollama backend follows Ollama's documented API but is untested.
4. **Run the regression suite after every change:** `python dev/regression.py` (≈1 minute, no display needed, exit code 0
   = pass). UI changes are verified with screenshots on a virtual display: `dev/ui_screenshots.py` (§14).
5. **Auto-approve is ON by default** (the user asked for this): every robot order — including treatments — is approved and
   executed automatically; the `⚡ Auto-approve` button switches to manual approval. Human-review escalations (`REVIEW`) always wait for a person.

**Quick start**

```bash
source <your-venv>/bin/activate
pip install -r requirements.txt            # opencv-python==5.0.0.93, numpy==2.5.3 (Tkinter comes from the system Python)
python app_ui.py                           # the application
python dev/regression.py                   # all headless checks
python try_llm_planner.py --selftest       # LLM-guardrail tests only
```
Fedora: `sudo dnf install python3-tkinter` (and for screenshots `xorg-x11-server-Xvfb ImageMagick`).

---

## 1. The user and how to work with them

* **Environment:** Fedora Linux, uses **their own existing virtual environment** (not `run_app.sh`'s `.venv`). Python on their
  machine is probably 3.14 (their committed `__pycache__` files are `cpython-314`); the code was built and tested on 3.12.
  Don't use syntax newer than 3.8 (an f-string quoting slip once broke older versions; see §17).
* **Experience:** new to LLMs/Bedrock/AWS. Explain concepts plainly (analogies worked: "rule table = vending machine, an
  LLM = a colleague who looks at the numbers and history", with a supervisor/guardrail in code). Keep explanations short and lead with the outcome.
* **Delivery habit that worked:** after each change, deliver **only the changed files** as a small zip (`…_update.zip`) plus a
  preview screenshot, with "copy these over your project and run `python app_ui.py`". The full project is ~160 MB
  (models + photos), so avoid re-zipping it unless asked.
* **They value:** honesty about what is simulated/untested; seeing the UI (they react to screenshots); being told when the
  assistant made a mistake; a UI that reads clearly at laptop size (1366×768).
* **Decisions the user made (do not silently reverse):** auto-approve **on** by default with a toggle; progress of robot jobs
  shown **inside the score panels** (disease panel for disease jobs, growth panel for growth jobs); a pod's **next image waits
  until its robots are done**; realistic crop growth (ordered stages, lingering, staggered pods, harvest waits); worker/camera
  instructions generated from the analysis; they dropped the half-built AWS backend (see §17) and asked for the LLM idea to be tried offline first.
* **Their original request (the very first message):** replace the OpenCV-window display with an interactive app with a pod
  layout (Pod 1 left, Pod 2 right, each with image + *Disease detection scores* and *Growth detection scores*), where the
  user can halt and interact. Their mock-up is reflected exactly in `PodCard` (§11).

---

## 2. Competition context and status

**Event:** OpenCV AI Competition 2026, powered by AWS (Devpost). Deadline **Oct 26, 2026, 11:59 pm Pacific** (the page's
countdown says Oct 27 12:45 pm GMT+6, slightly earlier — treat the earlier one as real). Prizes: $5,000 / $3,000 / $2,000,
plus $1,000 each for *Best Use of COOL* (Cloud-Optimized OpenCV Library on AWS Graviton) and the *Agentic Vision Award*.
Teams with a compute grant must hold a 30-minute Zoom check-in Oct 7–14.

**Hard requirements (from the page):** (a) OpenCV 5 for substantive image/video analysis; (b) a **meaningful component on
AWS**. **Submission:** technical report (problem, users, architecture, OpenCV 5 implementation, AWS deployment, evaluation,
limitations, responsible use); judge-accessible code + **pinned dependencies** + build/deploy/test instructions; architecture
diagram; a working web endpoint **or** arranged live demo; a public/unlisted video ≤ 5 minutes; evaluation evidence **including
failure cases**.

**Agentic Vision Award test (paraphrased):** an agent uses OpenCV 5 tools in a multi-step perception → decision → action loop, and
image results must change a *later* plan, tool call, action, or request for human approval. A chatbot merely explaining a fixed
result is not enough. Evidence wanted: agent-workflow diagram; a trace showing OpenCV output changing a later decision; evaluation
of task success, failure handling, observability and **human control**. (A coding assistant writing the entry does not make it agentic.)

**Judging weights:** technical execution 30 · innovation 20 · real-world impact 20 · UX 10 · documentation/presentation 10 ·
cloud delivery/reproducibility/responsible operation 10.

**Where the project stands**

| Requirement | Status |
|---|---|
| OpenCV 5 analysis | ✅ ONNX models via `cv2.dnn`; `requirements.txt` pins `opencv-python==5.0.0.93`. Verified: identical detections on 4.13 and 5.0; whole app + loop run on 5.0.0. |
| Agentic loop (image result changes a later action / approval request) | ✅ in simulation (but only ONE real OpenCV analysis tool and the agent does not call OpenCV tools itself — see §2.1): orders from detections, camera re-capture, follow-up frame, before/after comparison, escalation to a human, `trace.jsonl`. ⚠ the "new frame" is simulated. |
| Human control | ✅ approval switch, manual mode, REVIEW escalations, skip/hold/pause. |
| **AWS component** | ❌ **none.** `cloud_pipeline.py` / `dynamo_client.py` are local stand-ins. Biggest risk (and 10 % of the score). |
| Pinned dependencies / run instructions | ✅ `requirements.txt`, `README.md`, this file. |
| Technical report, architecture diagram, ≤5-min video, live endpoint/demo | ❌ not started. |
| Evaluation with failure cases | ⚠ ground-truth accuracy is logged; no evaluation write-up/ablation yet (§18). |
| COOL award (Graviton) | ❌ optional, not started. |
| Generative-AI element | ⚠ planner interface + guardrails built and tested with a stand-in; no real LLM run (§8). |

### 2.1 OpenCV tool audit (does "an agent calls OpenCV 5 tools" hold?) — measured Oct 2026

Counted from the code (AST scan of every `cv2.*` call):
* **13 distinct `cv2` functions in the interactive app, 20 in the whole repo** (the other 7 are the legacy OpenCV-window apps: `imshow, namedWindow, waitKey, destroyAllWindows, putText, getTextSize, rectangle`).
* **Real analysis = ONE tool**, the DNN detector (`perception.py`: `dnn.readNetFromONNX`, `dnn.blobFromImage`, `net.setInput/forward`, `dnn.NMSBoxes`, `copyMakeBorder` letterbox, `resize`, `imread`), run with two models per pod (growth + disease).
* **Four *simulated* camera operations** (`frame_source.py`: crop+`resize` zoom, `convertScaleAbs`+`LUT` brighten, `createCLAHE`+`cvtColor` contrast, `inpaint`). They stand in for a physical camera; they are not analysis.
* Support/IO only: `imread, imwrite, imencode, cvtColor, resize` in the UI/planner.
* **The agent never *calls* an OpenCV tool.** The rule table / LLM planner receives the detector's output and chooses a **task**; it does not choose which OpenCV operation to run or with what parameters. The only upstream decision that selects an OpenCV operation is a hard-coded `if` in `frame_source.py` (simulation). So the qualifying evidence today rests on the *"action / request for human approval"* clause of the award, not on tool use.
* **Nothing in our code is OpenCV-5-specific:** every function exists in 4.x (identical results on 4.13 and 5.0.0). What *is* 5-specific, implicitly: OpenCV 5.0's **new DNN engine** (the 5.0 notes: it coexists with the classic engine; `readNetFromONNX` takes an `engine` argument defaulting to `ENGINE_AUTO` = try the new engine first, fall back to classic). **Verified:** with `OPENCV_FORCE_DNN_ENGINE=2` (new engine, fail if it cannot load) all six models load and run, and detections on 12 sample pods are identical to the classic engine (`=1`) and to auto — so auto very likely uses the new engine (inferred, not directly observed). Do not claim speed differences: timings were within noise.
* **Consequence for judging:** "correctness and depth of the OpenCV 5 implementation" is 30 %; a single detector plus simulated image ops is thin, and the "agent uses OpenCV tools" wording is only weakly met.
* **Recommended fix (§18.1 item 2):** an explicit OpenCV **tool layer** the agent calls in a loop and whose next call depends on the previous result. Candidate real analysis tools: image-quality gate (Laplacian-variance blur, exposure histogram, glare) *before* trusting a detection; `detect(model, conf, roi)` the agent can re-run on a region/at other scales/tiles; enhance (CLAHE/gamma/denoise); plant segmentation → leaf-area / lesion-area **fraction** (quantitative severity instead of box counts); frame comparison (alignment + `absdiff`) for a *real* before/after check; and OpenCV-5-specific ones to investigate (neural feature matching — ALIKED/DISK/LightGlue in the new Features module — for alignment between visits; engine selection/benchmarking, relevant to the COOL/Graviton award; the new DNN engine's built-in LLM/VLM support per the 5.0 notes — Python API and models not verified). Expose them behind a small registry with schemas (and optionally an MCP server, which the competition page names as an allowed route).

---

## 3. System overview

### 3.1 The physical idea
Rows of growing troughs. **72 pods** = 3 rails (`R1..R3`) × 3 stations (`L`=cabbage, `M`=lettuce, `R`=mushroom) × 2 sides
(`F` front, `B` back) × 4 slots. Pod ids look like `R1-L-F1` (trough id `R1-L-F`). Two robots per side:

```
FRONT side:  camera_front   worker_front          BACK side:  camera_back   worker_back
```
* **Front** pods carry the *growth* ground truth (the photo is a growth-stage shot). **Back** pods carry the *disease*
  ground truth (a disease close-up). This is why back pods have a single photo and no crop cycle (§10).

### 3.2 Software architecture

```
                       ┌────────────────────────── app_ui.py (Tkinter, main thread) ──────────────────────────┐
                       │  PodCard ×1-4 (ImageView + ScorePanel/WorkerPanel)   FleetStrip   OrdersWindow  sidebar │
                       └───────▲──────────────────────────────────────────────────────────────┬──────────────────┘
        events (thread-safe)   │ engine.events: slot / queue / orders / state / log / finished  │ method calls
                               │                                                               ▼ (pause, approve, skip…)
 ┌─────────────────────────────┴───────────────────────────────  farm_engine.py  ─────────────────────────────────┐
 │ FarmEngine: one worker thread per slot ("lane"), pod queue, pause/step/hold/skip/rescan, settings, stats       │
 │   _scan(): image ─► camera.py ─► cloud_pipeline (local) ─► perception.py (OpenCV DNN) ─► decide_action        │
 │             ─► history (dynamo_client) + trend + run_logger + crop_cycle ─► OrderBook.plan_from_result()        │
 │   _await_orders(): the lane waits until the pod's robot jobs are finished                                      │
 │   _run_followup(): after a camera order finishes: frame_source ─► same analysis ─► OrderBook.close_loop()      │
 └───────────────┬───────────────────────────────────────────────────────────────┬───────────────────────────────┘
                 ▼                                                               ▼
   worker_planner.py  (orders)                                       crop_cycle.py (which photo next)
   OrderBook · WorkOrder · playbook · AI consult · loop logic        ages, stages that linger, staggering, harvest-wait
   SimulatedFleet (4 fake robots) · FileOutbox · trace.jsonl         frame_source.py (simulated new frame)
                 ▲
   llm_planner.py: evidence → LLM (Scripted | Ollama) → strict JSON → guardrails (plain code)
```

### 3.3 The core loop in one paragraph
A **lane** (slot) takes a pod and visits it `SESSION_VISITS` (4) times. Each **visit**: pick the pod's photo (from its crop age),
analyse it (growth model + disease model), decide an action with the rule table, write history/log, advance the crop's age, and
create **orders** from the result. Orders go to the robots (automatically, or after a human approves). **The lane waits** until
every robot job on that pod is finished (including the follow-up frame of a re-capture), then shows the finished job briefly,
then takes the next visit. A *harvest* triggers a *replant* job; when that finishes the crop's age resets to 0 so the next photo
is stage 1 of a new crop.

### 3.4 Vocabulary
* **Pod** one growing position; **lane/slot** one on-screen card + worker thread (1–4); **visit** one monitoring pass of a pod;
  **session/run** one press of Start; **order** an instruction for an agent (`WorkOrder`); **aspect** `disease` or `growth`
  (which score panel an order's progress is shown in); **ripe** crop at its harvest stage; **follow-up/loop** the re-analysis after a camera job.
* **Actions** (from `agent.py`): `log_healthy`, `flag_for_harvest`, `flag_for_treatment`, `schedule_frequent_monitoring`.
  Mapping to tasks: treatment→`TREAT`, harvest→`HARVEST`, monitoring→`RECAPTURE`, healthy→nothing.

---

## 4. Repository map

Legend: **O** = original (the user's pipeline, unchanged unless noted) · **N** = new in this work · **M** = modified original.

| File | Lines | | Purpose |
|---|---:|---|---|
| `perception.py` | 291 | O | ONNX/YOLOv8 inference through `cv2.dnn`: `MODEL_REGISTRY`, `classify_pod()`. Inference unchanged; §19 adds the pure display helper `human_stage_name()` (from the DataPipeline copy). |
| `agent.py` | 100 | O | `decide_action()` rule table (§5.2). |
| `pod_registry.py` | 150 | O/M | The 72 pods. *Change (§19):* built from `pybullet_sim/pods/<POD_ID>/pod.json` instead of `pods_manifest.json`. |
| `cloud_pipeline.py` | 164 | M | `LocalCloudClient` (thread pool standing in for Lambda). *Change:* optional `conf_threshold`. |
| `camera.py` | ~30 | M | `PodCamera.capture_and_submit()`. *Change:* optional `conf_threshold`. |
| `dynamo_client.py` | 66 | O | `LocalDynamoTable` (JSON file standing in for DynamoDB): per-pod history. |
| `trend.py` | 52 | O | `summarize_trend()` text ("stage advanced…", "disease flagged 2× in a row"). |
| `run_logger.py` | 96 | O | CSV/JSON run log with ground-truth accuracy summary. |
| `main.py`, `live_monitor.py`, `concurrent_monitor.py`, `visualize.py`, `gallery.py` | — | O | The original CLI/OpenCV-window apps. Still work; use the legacy staircase, not the crop cycle. |
| **`app_ui.py`** | 1920 | N | The Tkinter application (§11). |
| **`farm_engine.py`** | 781 | N | UI-independent run engine (§6). |
| **`worker_planner.py`** | 1066 | N | Orders for the robots: model, playbook, approvals, AI consult, closed loop, simulator (§7). |
| **`llm_planner.py`** | 209 | N | LLM-in-the-loop planner with hard guardrails (§8). |
| **`frame_source.py`** | 101 | N | Simulated camera: derives the "new frame" (§9). |
| **`crop_cycle.py`** | 202 | N | Crop life cycles (§10). |
| `try_llm_planner.py` | 128 | N | CLI lab for the planner; `--selftest` runs the guardrail tests. |
| `run_app.sh` | — | N | Launcher that makes a `.venv` (the user uses their own venv instead). |
| `dev/regression.py`, `dev/ui_screenshots.py` | 209 / 93 | N | Regression suite and UI screenshot tool (§14). |
| `requirements.txt` | — | M | Pinned: `opencv-python==5.0.0.93`, `numpy==2.5.3`. |
| `README.md` | — | M | Run instructions + feature sections (kept in sync with this file). |
| `models/` | 6 ONNX | O | `{cabbage,lettuce,mushroom}_growth_stage_v2.onnx` and `{cabbage,lettuce,mushroom}_disease.onnx`. |
| `sample_images/`, `pods_manifest.json` | — | — | **Superseded (§19):** photos now live in `pybullet_sim/pods/<POD_ID>/`. Safe to delete. |
| `dynamo_table.json` | — | O | Saved history (the user's real file — never overwrite/delete in tests). |
| `dynamo_table_cycle.json` | — | N (generated) | Saved crop ages (created on first run; do not ship it). |
| `logs/`, `work_orders/` | — | generated | Run logs (CSV/JSON) / per-run orders, outbox and `trace.jsonl`. **`logs/` contains the user's original run logs — never `rm` globs there.** |

---

## 5. The original pipeline (the user's code — understand it before touching anything)

### 5.1 Pods and ground truth (`pod_registry.py` + `pybullet_sim/pods/*/pod.json`, see §19)
`PODS` is a list of 72 dicts: `pod_id, index, rail, station (0/1/2 → `L`/`M`/`R` via `STATION_LABELS`; = cabbage/lettuce/mushroom), side ("front"|"back"),
trough_id, crop_type, image_path (absolute, resolved from the project folder at import), ground_truth, trajectory`.
* `ground_truth = {verified_aspect: "growth"|"disease", growth_stage, diseased, disease_name}`. **Only one aspect per pod is verified**:
  front pods → growth, back pods → disease (`engine.check_against_truth` scores exactly that one; `run_logger.summary()` does the same).
* `trajectory` (front pods only): the legacy "staircase" — one photo per stage from the pod's start stage to harvest. Used by `main.py`,
  `live_monitor.py`, `concurrent_monitor.py`. **The new app ignores it** except as a source of real photos per stage (`crop_cycle.py`).
* Each front pod owns exactly **one photo per stage**. Photos per crop/stage (distinct): cabbage 3/6/8/10/12 (S2_3…S9), lettuce 3/6/9/12, mushroom (as labelled) Juvenile 4 / Intermediate 8 / Harvest 12.

### 5.2 Perception (`perception.py`) — `classify_pod(pod_id, crop_type, image, conf_threshold=0.25)`
* Six ONNX YOLOv8 files in `models/`, run with `cv2.dnn`, 640×640 letterbox; per-thread network cache (`warm_up()` preloads all six).
  `CONF_THRESHOLD=0.25`, `NMS_THRESHOLD=0.45`. `image` may be a path or a BGR ndarray.
* `MODEL_REGISTRY[crop]` → `growth{path, classes{id:name}, harvest_stage}`, `disease{path, classes, healthy_class}`.
  Growth classes: cabbage `S2_3,S4_5,S6_7,S8,S9` (harvest `S9`; the labels are the dataset's own and flagged as an *unverified* leaf-count interpretation) ·
  lettuce `Stage 01- Early Growth, stage 02- Leafy growth, stage 03- Head formation, stage 04- Harvest stage` · mushroom `Harvest, Intermediate, Juvenile` (**class order exactly as the model's ONNX metadata declares it**; harvest `Harvest`).
  Disease classes: cabbage 9 (`Cabbage-Alternaria_leaf_spot, …Bacterial_leaf_spot, …Black_rot, …Downy_mildew, …Fusarium_wilt, …Healthy, …Powdery_mildew, …Ringspot, …Sclerotinia_rot`) ·
  lettuce `Bacterial, Downy_mildew_on_lettuce, Powdery_mildew_on_lettuce, Septoria_Blight_on_lettuce, Viral, Wilt_and_leaf_blight_on_lettuce, healthy` · mushroom `Bacterial Blotch, Dry Bubble, Healthy, Trichoderma, Wilt`.
  (The healthy-class name differs per crop; always use `registry[crop]["disease"]["healthy_class"]`.)
* **Returns** a dict: `pod_id, crop_type, growth_stage, growth_confidence, disease_flag, disease_name, disease_confidence, confidence, growth_detections, disease_detections`.
  `growth_stage`/`disease_*` come from the **top** detection (disease: top *non-healthy* detection). A detection is
  `{class_id, class_name, confidence, box:[x, y, w, h]}` with the box in **pixels of the original image** (numpy float32 — convert before JSON; `worker_planner._json_default` does).

### 5.3 Decision (`agent.py`) — `decide_action(classification, disease_threshold=0.50, growth_threshold=0.40)`
Returns `{action, reason, based_on}`. **Precedence matters:**
1. `disease_flag and disease_conf ≥ 0.50` → `flag_for_treatment`
2. growth stage missing, or `growth_conf < 0.40`, or (disease flagged but `< 0.50`) → `schedule_frequent_monitoring`
3. `growth_stage == harvest_stage` → `flag_for_harvest`
4. otherwise → `log_healthy`

`ACTION_DWELL_SECONDS` exists only for the legacy OpenCV apps (a sleep per action) — it is not a real physical response; don't cite it as evidence.
⚠ **Known quirk:** back (disease close-up) pods have no readable growth stage, so rule 2 fires for any *healthy-looking* back pod → a `RECAPTURE` order every time. Treating growth as "not applicable" for back pods would fix it (§17).

### 5.4 Cloud/camera/storage stand-ins
* `PodCamera.capture_and_submit(pod_id, crop, image, conf_threshold=None)` → `LocalCloudClient.submit_frame(...)` returns a request id; a thread pool runs
  `classify_pod` + `decide_action` after a simulated latency; `get_feedback(id, block, timeout)` returns a record `{status, classification, latency_seconds, worker_thread, …}`.
  **These are the interfaces an AWS backend must reproduce** (§18).
* `LocalDynamoTable(path)` — JSON list; `put_item(item)` (needs `pod_id`, `timestamp`), `query(pod_id)`, `latest`, `visit_count`, `all_pod_ids`. **Not thread-safe** → the engine wraps it in `_data_lock`.
  History row written per logged scan: `pod_id, timestamp (ISO-8601 UTC), visit, crop_type, growth_stage, growth_confidence, disease_flag, disease_name, disease_confidence, action, ground_truth_stage_this_visit`.
* `summarize_trend(prior_rows, current)` → text such as "first visit", "stage unchanged (S8)", "stage advanced: A -> B", "disease flagged 2x in a row".
* `RunLogger` rows (CSV+JSON in `logs/`): `timestamp, visit, pod_id, rail, station, side, crop_type, growth_stage, growth_confidence, disease_flag, disease_name, disease_confidence, confidence, action, reason, trend, verified_aspect, ground_truth_stage, ground_truth_diseased, ground_truth_disease_name`; `summary()` gives per-aspect accuracy.

---

## 6. The engine (`farm_engine.py`)

No Tk code. One object, `FarmEngine`, owns the run; everything the UI does is a method call, everything the UI learns arrives through `engine.events`.

```python
FarmEngine(pods, n_slots=2, settings=None, reset_history=False,
           log_dir="logs", history_path="dynamo_table.json", orders_dir=None)
```
Creates: `history` (`LocalDynamoTable`), `cycles` (`CycleStore` at `<history>_cycle.json`), `logger`, `cloud`+`camera`, `orders` (`OrderBook`, dir `work_orders/run_<stamp>` unless `orders_dir`),
`frame_source` (`SimulatedCamera`), a one-thread pool for follow-ups, `slots` (list of `Slot`). `reset_history=True` deletes the history file **and** the crop-age file.

**State** `engine.state`: `idle → running ⇄ paused → finished | stopping → stopped`. `finished` = every pod done (workers stay alive so you can still review/rescan; logs are flushed). `stop()` is non-blocking (a finalizer thread joins workers, shuts the pool, flushes logs, emits `stopped`); `close()` blocks until stopped.

**Settings** (live-editable dataclass, read fresh each scan): `delay_ms=700`, `conf_threshold=0.25` (model sensitivity, applies to the next scan/re-scan), `disease_threshold=0.50`, `growth_threshold=0.40`.
**Note:** decisions are computed in the engine with these thresholds (`decide_action(cls, st.disease_threshold, st.growth_threshold)`), *not* taken from the cloud record. `engine.redecide(classification)` re-evaluates a stored reading under current thresholds (the UI uses it so shown decisions update instantly; a `*` marks a decision that differs from what was logged).

**Control methods:** `start, pause, resume, step_once, stop, close, toggle_hold(slot), skip_pod(slot), rescan(slot), set_view(slot, idx|None), run_next(pod_id), skip_queued(pod_id), queue_position(pod_id), set_planner(kind, model, use_image), stats(), slot_view(i), elapsed()`.
* **Pause** blocks anything *new* from starting (in-flight scans finish); **`step_once`** (only while paused) gives each un-held slot a one-shot ticket; **Hold** freezes one lane; **Skip** ends the lane's current pod (overrides hold; when paused it ends the pod but does not pull a new one); **Re-scan** re-runs the models on the photo on screen — *not logged, no orders*, with the current sensitivity.
* `_await_orders` (below) is independent of pause: robots keep finishing while paused.

**Events** put on `engine.events` (a `queue.Queue`; the UI drains it): `("slot", i)`, `("queue", None)`, `("orders", None)`, `("state", s)`, `("log", (time, level, text))`, `("finished", info)`, `("stopped", info)`.
`slot_view(i)` returns a dict copy: `slot_id, pod, results, view_index, first_step, n_steps, next_step, done, status, hold, pods_completed`. **Slot statuses:** `idle, inspecting, showing, robot, done, skipped`.

### 6.1 The lane loop (`_worker` → `_advance`)
```
loop until stop:
  handle rescan request                      (works even while paused)
  handle skip (if not mid-scan)              (ends the current pod / clears the dwell)
  wait while  now < slot.not_before          (the delay between images / the dwell after a finished pod)
  wait unless _may_proceed(slot)             (hold blocks; paused needs a one-shot ticket)
  _advance(slot):
      if no pod or pod done: pull next pod from the queue → _begin_pod
      _run_step(slot)                        → _scan(...) → result appended, status "showing"
      waited = _await_orders(slot)           → lane blocks while orders.pod_busy(pod)  (status "robot")
      finished? → _finish_pod(extra_dwell = ROBOT_DONE_DWELL_S=3.0 if waited)    else not_before = now + max(delay, dwell)
```
`_begin_pod`: `n = SESSION_VISITS (4)` for pods with a crop cycle (`crop_cycle.has_cycle`), else the legacy trajectory length (back pods → 1). Cycle pods always start at visit 0 (the crop's *age*, not the visit, carries progress between runs); legacy pods resume at `history.visit_count`.

### 6.2 One scan (`_scan(pod, step, slot, rescan, override=None, cycle=None)`)
1. **Pick the photo:** re-scan → the photo on screen (`override`); cycle pod → `cycles.image(pod)` (also returns the photo's ground-truth stage and a `cycle` info dict); else legacy `image_for_step`.
2. `camera.capture_and_submit(...)` → `cloud.get_feedback(..., timeout=15s)`; a non-"done" record → the step is skipped with a log line.
3. `decide_action` with the live thresholds; `check_against_truth` (the one verified aspect).
4. **If not a re-scan** (inside `_data_lock`): `history.query` → `summarize_trend` → `history.put_item` → `logger.log` → counters → **`cycles.advance(pod)`** (a ripe crop does not age).
5. Build an immutable **`StepResult`** (`pod_id, crop_type, step_index, n_steps, visit_num, image_path, classification, decision, trend_note, latency_s, worker, timestamp, kind ("scan"|"rescan"|"followup"), logged, conf_threshold, verified, cycle`).
6. **If not a re-scan:** `orders.plan_from_result(pod, result, history_rows=…)` inside try/except (a planner problem must never break a lane) and log the new orders.

### 6.3 After a camera order finishes (the closed loop)
`OrderBook.complete()` → `on_complete` → `_run_followup(order_id)` on a one-thread pool: `frame_source.capture(order)` → same analysis on the new frame →
`decide_action` → `StepResult(kind="followup", logged=False)` → `orders.close_loop(...)` (outcome table in §7.5); `finally: orders.loop_done(order_id)`.
Follow-up frames **never** touch history, logs, or crop ages.

### 6.4 Planner switching
`set_planner("rules"|"scripted"|"ollama", model, use_image)` builds an `LlmPlanner` and hands it to `orders.set_advisor(...)` — live, any time. `planner_name` is for display.

### 6.5 Engine constants
`FEEDBACK_TIMEOUT_S=15`, `SESSION_VISITS=4`, `ROBOT_DONE_DWELL_S=3.0`, `MIN_DWELL_S=1.2`.

---

## 7. Orders for the robots (`worker_planner.py`)

An **order** (`WorkOrder`, mutable dataclass, lock-protected inside `OrderBook`) is one instruction for one agent. All reads from outside return **deep copies**.

### 7.1 Order types

| Task | Agent | Created when | Aspect (panel) | Needs approval* | Priority | Blocks the pod's next image |
|---|---|---|---|---|---|---|
| `TREAT` | `worker_<side>` | decision `flag_for_treatment` | disease | yes | 1 HIGH | yes |
| `HARVEST` | `worker_<side>` | decision `flag_for_harvest` | growth | yes | 2 NORMAL | yes |
| `REPLANT` | `worker_<side>` | a `HARVEST` completes | growth | no | 2 NORMAL | yes |
| `RECAPTURE` | `camera_<side>` | decision `schedule_frequent_monitoring` (or an AI "take a better look") | disease if a disease is suspected with boxes, else growth | no | 3 LOW | yes (incl. its follow-up frame) |
| `VERIFY` | `camera_<side>` | a `TREAT` completes | disease | no | 3 LOW | **no** (it is a later visit) |
| `REVIEW` | `human` | the loop could not settle a case | — | always (acknowledge) | 1 HIGH | **no** |

\*"Needs approval" is the order's own flag. **Gating** (`_open_for_work`): `REVIEW` always waits. With `policy.auto_approve_all` (default **True**) every other order is approved and dispatched immediately.
With it **off** (manual mode) **every** robot order waits for a human — including `RECAPTURE`. Switching it back **on** releases the waiting robot orders. `approve_all_pending()` never touches `REVIEW`.

### 7.2 Statuses
`planning → awaiting_approval | approved → dispatched → in_progress → completed`; also `rejected`, `cancelled` (superseded), `scheduled` (VERIFY waiting to be released).
Open = `planning, awaiting_approval, approved, dispatched, in_progress, scheduled`; **blocking** = `planning, awaiting_approval, approved, dispatched, in_progress` on tasks `TREAT, HARVEST, REPLANT, RECAPTURE` (plus a RECAPTURE whose follow-up frame has not been analysed yet — the `_loop_pending` marker).
`REVIEW` approve = acknowledge (→ `completed`, never dispatched).

### 7.3 Order contents
`order_id (WO-0001…), task, role, agent, pod_id, crop_type, trough_id, side, location, priority, title, rationale, evidence{…}, targets[…], steps[…], requires_approval, approval_note, human_notes[], status, progress, occurrences, stale, follow_up_of, not_before, created, updated, history[[ts,status,note]], ai{}, outcome{}, aspect`.
* **evidence**: `visit, step, n_steps, image, image_size, growth_stage/confidence, disease_flag/name/confidence, n_disease_detections, n_growth_detections, action, reason, trend, cycle{}, model_sensitivity, scan_time, rail` (+ `frame_source` for follow-up-created orders).
* **targets**: up to 8 detections as `{class, confidence, box_norm:[x,y,w,h] (0..1), center_norm:[cx,cy]}` — normalised so any arm/camera calibration can map them. TREAT targets = boxes of the diagnosed disease; HARVEST = boxes of the harvest stage; RECAPTURE = ≤3 suspected-disease boxes (the zoom region).
* **steps**: `[{seq, action, text, params}]`, actions drawn from `NAVIGATE, LOCATE, REMOVE_TISSUE, BAG_AND_SEAL, APPLY_TREATMENT, SANITIZE_TOOL, MARK_POD, REPORT, HARVEST, PLACE_IN_CRATE, CLEAR_POD, PLANT, WATER_FEED, ADJUST_VIEW, CAPTURE, UPLOAD, REQUEUE, COMPARE, HUMAN_INSPECT, HUMAN_DECIDE`.
* **progress** = number of *completed* steps; the current step is `progress + 1` (UI shows "step k/n" this way everywhere).

### 7.4 The treatment playbook (generic on purpose — for an agronomist to review; no products, no doses)
`DISEASE_CATEGORY[(crop, disease)] → category`; `CATEGORIES[category]` gives what to remove, whether a chemical class is named, and notes for the human:
`foliar_fungal` (remove affected leaves + fungicide class), `bacterial` (leaves/tissue + bactericide class), `soilborne` (whole plant + root zone, no chemical), `viral` (whole plant), `mushroom_bacterial` / `mushroom_fungal` (affected mushrooms / contaminated patch), `review` (unclear class → only mark the pod for human inspection).
Examples: Alternaria/mildews/Septoria/Ringspot → foliar_fungal; Black rot, bacterial spot → bacterial; Fusarium wilt, Sclerotinia → soilborne; lettuce Viral → viral; `Wilt_and_leaf_blight_on_lettuce` and mushroom `Wilt` → review.
Chemical steps use `product="HUMAN_SELECTED"`. ≥10 boxes of the disease = "widespread" → a note asks the human to decide between treating and removing the plant. Harvest verbs per crop: `HARVEST_MODE`.

### 7.5 Closing the loop (`close_loop`, called by the engine with the follow-up frame)

| Order | New frame says | Result | Next |
|---|---|---|---|
| RECAPTURE | treat / harvest | `confirmed_disease` / `confirmed_harvest` | that order is created (`follow_up_of` = the recapture) |
| RECAPTURE | healthy | `resolved_healthy` | nothing |
| RECAPTURE | still uncertain | `still_uncertain` | `REVIEW` order |
| VERIFY | disease no longer actionable | `treatment_worked` | nothing |
| VERIFY | still flagged | `treatment_not_resolved` | `REVIEW` order |

`order.outcome` stores `result, headline, before{…}, after{…}, after_boxes[…], after_size, frame, frame_note, simulated, next_order`; the UI shows before/after images. Other order-book rules: a **TREAT supersedes** an open RECAPTURE for the same pod; repeat sightings of the same task on a pod **update** the open order (`occurrences`) instead of duplicating; a later contradicting reading marks an unapproved TREAT/HARVEST **stale**.

### 7.6 AI consult (optional; see §8)
Only for `TREAT` and `HARVEST` (`CONSULT_TASKS`). With an advisor set, the new order enters `planning`; a single background thread asks the planner (the lane never waits for the LLM); then:
`downgraded` → the order is `cancelled` and a `RECAPTURE` is created (`ai` attached); anything else → continue to gating. `urgency=="high"` raises priority by one; `low` is never allowed to delay anything; `notes_for_human` are prepended as "AI: …". If the advisor is switched off or the run ends, planning orders fall back to the rules.

### 7.7 Robots, outbox, trace
* **SimulatedFleet** — four threads (one per agent) with a priority queue each; every step sleeps `policy.sim_step_seconds` then calls `book.advance`; at the end `book.complete`. Switch off (`policy.simulate_robots=False` / UI "Simulate robots") to let *external* agents drive orders.
* **Outbox** — on dispatch the order is written as JSON to `<orders_dir>/outbox/<agent>/<order_id>.json` via `FileOutbox.send(dict)`. **To connect real robots:** replace the transport (any object with `.send(dict)`: MQTT / AWS IoT Core / SQS / ROS 2) and have robots call `OrderBook.begin(order_id, agent)`, `advance(order_id)` (per finished step), `complete(order_id, note)` (alias `report`). The UI's agent strip and worker panels follow whatever status the book holds. A "Mark done" button appears when simulation is off.
* **`<orders_dir>/trace.jsonl`** — one JSON object per line: `order_created` (with the decision and evidence), `status` (with note), `ai_consulted`, `loop_closed` (with the outcome summary), `policy_changed`. This is the "agent trace" evidence for the competition.
* Also in the order dir: `orders/<id>.json` (current state of each order) and `frames/<order>_after.jpg` (simulated follow-up frames).
* `fleet_status()` (for the main-window strip) and `pod_panel_orders(pod_id, recent_s)` (which order each panel shows) are the two read APIs the UI polls.

---

## 8. The optional AI planner (`llm_planner.py`, `try_llm_planner.py`)

**Idea:** perception gives numbers → an LLM *proposes* a task as strict JSON → **plain code decides whether to accept it**. The LLM never touches robots and never writes robot steps.
* **Evidence** sent to the model (`build_evidence`): pod, growth (stage/conf/boxes/harvest stage), disease (flagged/name/conf/boxes), the rule table's task and reason, the last 3 visits, `allowed_tasks`.
* **Backends** (a backend is just `(system_prompt, evidence_dict, image_path) -> text`): `ScriptedBackend` — hand-written stand-in, **not an LLM** (downgrades weak TREAT evidence: conf < 0.60 with ≤ 2 boxes); `OllamaBackend(model, use_image, timeout)` — a real local LLM via `http://localhost:11434/api/chat` with `format: json`, `temperature: 0` (**untested against a real model**; text-only models work, vision models may also get the photo). A Bedrock backend would be one more such function.
* **Guardrails** (`resolve`, pure and tested): reply must parse into the schema (`task ∈ {TREAT,HARVEST,RECAPTURE,NONE}`, `urgency`, `reason ≤200`, `notes ≤3×160`; unknown keys like `"approve":true` are ignored and recorded) · the LLM may only **agree** with the rules or ask for a **RECAPTURE** (it can never create or cancel a TREAT/HARVEST) · it may not soften strong evidence (**≥3 boxes at ≥0.85**) · it may not ask twice in a row for the same pod (`prior_downgrade`) · on any error or invalid output the **rules stand** (`verdict="fallback"`). Verdicts: `agreed | downgraded | rejected | fallback`.
* **Tests:** `python try_llm_planner.py --selftest` (10 cases incl. hostile output). `python try_llm_planner.py --show-prompt` prints the exact prompt+evidence; `--backend ollama --model llama3.2:3b [--image]` for a real local model; `--pod`, `--visits N` select pods/visit counts (it walks pods through their crop cycle like the app).
* Ollama setup (user's machine): install from ollama.com, `ollama pull llama3.2:3b` (any model from ollama.com/library), keep it running.

---

## 9. Simulated camera (`frame_source.py`)

After a camera order finishes, the app needs a **new** photo. With no physical camera, `SimulatedCamera.capture(order, out_dir)` derives one *from the order's own instructions* (all OpenCV):
`RECAPTURE` with suspected regions → crop+upscale the region (≤3×) · nothing readable → centre 80 % crop + brighten (gamma) · growth unclear → CLAHE contrast on L · `VERIFY` → `cv2.inpaint` over the regions the worker was told to remove (only the ≤8 targeted regions, so a **widespread** infection correctly fails verification).
It returns `{"path", "note", "simulated": True}`. **Everything it produces is synthetic — the downstream analysis is real, the frames are not.** Never report results from these frames as measured accuracy; say so in the technical report. To use real frames, give the engine any object with the same `capture(order, out_dir) -> {...}` (and decide whether real follow-ups should also be written to history — currently they are not).

---

## 10. Crop life cycles (`crop_cycle.py`)

### 10.1 Why
The legacy registry gave every front pod a staircase (one photo per stage, each visit a whole new stage, every pod ending at harvest — a third even started there), so all pods of a crop harvested together and stages never lingered.

### 10.2 The model
* **Age** = visits since planting. `STAGE_VISITS` (visits per stage before the next; relative, compressed): cabbage `S2_3:5, S4_5:5, S6_7:5, S8:3` → harvest at **age 18**; lettuce `4,4,4` → **12**; mushroom `Juvenile:3, Intermediate:5` → **8**. `STAGE_ORDER` lists stages planting→harvest.
* **Lingering:** while a crop stays in a stage the pod's photo of that stage repeats (the user explicitly accepted repeated images to stall progress).
* **Harvest waits:** at the harvest age the age stops advancing (the same ripe photo returns) until a `HARVEST` *and* its `REPLANT` complete; `OrderBook.on_replant` → `CycleStore.reset(pod)` sets age 0 (and `cycles += 1`) **before** the REPLANT is marked complete, so the next photo is stage 1. A rejected harvest therefore leaves the crop ripe.
* **Staggered:** initial ages are spread evenly over `0..harvest_age` and permuted with `STRIDE=5` (coprime with 12) so neighbouring pods differ. With 4 visits per run roughly **3/12 cabbage, 4/12 lettuce, 5/12 mushroom** pods reach harvest per run.
* **Persistent:** `CycleStore(path)` keeps `{pod_id: {"age", "cycles"}}` in `dynamo_table_cycle.json` (atomic write). `path=None` = in memory (used by the CLI/tests).
* **Photos:** `PHOTOS[pod][stage]` = the pod's own photo for the stage, else a real photo of that stage from the crop's pool (rotated by pod index). Pods only have `has_cycle()` if `side=="front"` and a trajectory exists; back pods are single disease photos.
* API: `has_cycle(pod)`, `stage_order(crop)`, `harvest_age(crop)`, `stage_at(crop, age)`, `CycleStore.info/image/advance/reset/clear`.
* UI shows `crop age 7/12 · RIPE, waiting for harvest · crop #2` on each card; `VISIT k/4` badge on images.
* **Limitations:** a replanted crop shows the same pod's photos again (only ~12 real photos per late stage); because photos repeat, a diseased-looking photo stays diseased after treatment (repeated treatments are possible); durations are my choice (plain numbers — edit freely).

### 10.3 Ground-truth honesty
`check_against_truth` scores the model's reading against the **photo's** stage label (`gt_stage` from `cycles.image`), so accuracy is meaningful per visit.

### 10.4 ⚠ Mushroom ground-truth labels look swapped — UNRESOLVED, ask the user
**Observation (measured):** on the project's own labelled photos the mushroom growth model says *Harvest* for 4/4 photos labelled **Juvenile** and *Juvenile* for 10/12 photos labelled **Harvest** (Intermediate 7/8). Cabbage and lettuce models agree with their labels ~100 %.
**What the ONNX metadata says:** class `0=Harvest, 1=Intermediate, 2=Juvenile` — exactly what `perception.py`'s registry has. So the registry matches the model; the *labels in `pods_manifest.json`* are the suspect.
**What the photos show:** photos labelled Juvenile are large, fully open caps (look mature); most photos labelled Harvest are long thin stems with small caps (look young). Model behaviour + metadata + appearance all agree; the labels disagree with all three. (I am not a mycologist: the user should confirm visually.)
**What was done:** `perception.py` is untouched. `crop_cycle.py` has `LABEL_CORRECTIONS = {("mushroom","Juvenile"):"Harvest", ("mushroom","Harvest"):"Juvenile"}` and `APPLY_LABEL_CORRECTIONS = True`; photo labels are corrected there only. With it, mushroom photos per stage are Juvenile 12 / Intermediate 8 / Harvest 4 and the model agrees 10/12, 7/8, 4/4 (3 of the 4 Harvest photos trigger a harvest action), matching the other crops. Set the flag to `False` to use the manifest as shipped (then mushrooms show harvest-looking photos in their "young" stage, the model calls them Harvest, and harvest orders fire at the start of a mushroom's life).
**Not corrected:** the legacy scripts (`main.py` etc.) and `pod_registry`'s own ground truth still use the uncorrected labels. **An earlier attempt "fixed" this in `perception.py` — that was wrong and was reverted (§17).**

---

## 11. The application (`app_ui.py`, Tkinter, ~1900 lines)

### 11.1 Screen map
```
┌ toolbar: 🌱 title · [state chip] · ▶Start/New run · ⏸Pause/▶Resume · ⏭Step · ■Stop · 💾Save log · 🤖Robot orders[· N to approve] · ⚡Auto-approve / ✋Manual ┐
├ summary: "4/6 pods · 9 scans · 00:41 · 1.2 s avg" + progress bar │ chips HEALTHY n · HARVEST n · MONITOR n · TREAT n │ growth a/b  disease c/d ┤
├ fleet strip: FRONT·CAMERA │ FRONT·WORKER │ BACK·CAMERA │ BACK·WORKER   (current order, step name, progress bar; click = open that order) ┤
├ sidebar (300 px)           │ cards grid (1 column for 1 pod, else 2; odd columns MIRRORED so scores sit on the inside, like the user's sketch)
│  RUN SETUP  crop · pods(0=all) · on screen(1-4) · reset history
│  POD QUEUE  (double-click = run next) [Run next][Drop]      ┌ POD 1  R1-L-B2   [🤖 ROBOT WORKING][NEEDS TREATMENT] ┐
│  AI PLANNER rules / scripted demo / Ollama (+model, photo)  │ cabbage · R1-L-B · crop age 7/18 · RIPE…            │
│  LIVE TUNING delay · sensitivity · disease thr · growth thr │ [photo + boxes] [DISEASE scores │ WORKER · TREATMENT]   │
│               ☑Boxes ☑Box labels [Defaults]                 │ [photo + boxes] [GROWTH  scores │ WORKER · HARVEST …]   │
│                                                             │ → reason · trend · 🤖 order line · [Hold][Skip][Re-scan] timeline ┘
└─────────────────────────────────────────────────────────────┴ activity log (5 lines, coloured) ──────────────────────────────────────────────┘
```
Window 1560×940, minimum 1180×720; verified readable at **1366×768** (bottom sidebar rows scroll off at ≤768 px tall). Keys: **Space** pause/resume, **N** step (ignored while typing in a spinbox/combobox/entry).

### 11.2 Classes (all in `app_ui.py`)
| Class | Role |
|---|---|
| `Btn` | A `Label` acting as a button (native buttons ignore colours on macOS). `set_enabled`, `set_text`, `set_kind("ok"/"warn"/"danger"/"primary"/"ghost")`. |
| `Chip` | Coloured status label. |
| `ImageView` | `Canvas` showing the **plain photo**; detection boxes/labels are drawn by the UI on top (`layers=[{dets,color,healthy}]`), toggled from the sidebar; hover = tooltip; click = `ZoomWindow`; busy overlay "ANALYSING…"; badge bottom-right (`VISIT k/n` or `IMAGE k/n`). ≤25 boxes drawn, ≤6 labelled (more when a class is highlighted). |
| `ScorePanel` | "Disease/Growth detection scores": headline, one bar per class the model can output, a tick at the action threshold, click a row = highlight that class's boxes; compact mode (<215 px high) drops the secondary lines. |
| `WorkerPanel` | Same grid cell as a `ScorePanel`; shows the robot job for that aspect (title, status line, every step with ✓/▶/○, progress bar, Approve/Reject/Mark done); `⇄ scores` flips back for that job. |
| `Timeline` | `◀ ● ● ○ ▶ LIVE` per pod: one dot per visit (colour = decision, diamond = re-scan, hollow = pending); click = review that earlier reading. |
| `PodCard` | One lane. `render(slot_view)` redraws everything; `refresh_orders()` updates only the robot line (cheap, on every order event); `refresh_worker_panels()` decides scores vs worker panel per aspect. |
| `ZoomWindow` | Large view with both models' boxes and both score panels. |
| `FleetStrip` | The four agents in the main window (reads `OrderBook.fleet_status()`). |
| `OrdersWindow` | Robot orders: four agent tiles, Approve / Run now / Mark reviewed / Reject / Approve all robot jobs / Mark done / Copy JSON, toggles *Close the loop* and *Simulate robots*, the auto-approve button, the order list, and a detail pane (why, approval gate, AI verdict, target image and **before/after** images, steps, notes for the human). Polls the order book by its `version` counter. |
| `App` | Main window: builds everything, owns the engine, the `_pump`/`_tick` loops, run controls, planner and tuning handlers. |

### 11.3 Threading model — **the rule that must not be broken**
Engine threads (lanes, fleet simulator, planner thread, follow-up thread, timers) **never touch Tk**. They put tuples on `engine.events`.
`App._pump` (every **80 ms**) drains them: `slot` → `PodCard.render`; `queue` → refresh the queue list; `orders` → every card's `refresh_orders()` + `refresh_worker_panels()`; `log`, `state`, `finished`, `stopped`.
`App._tick` (every **250 ms**) updates the summary, the fleet strip, and time-based worker-panel flips (the "scores first" delay and the "done" display expire without any event).

### 11.4 What a pod card shows when (rules the user asked for)
1. A new reading appears → **scores are shown first** for `SCORE_VIEW_SECONDS = 1.8 s`.
2. Then the robot job for each aspect takes over *that* panel (`pod_panel_orders`: in-progress > sent > waiting for approval > planning; a just-finished job stays `WORKER_DONE_SECONDS = 3 s`). Disease jobs (TREAT/VERIFY, disease-type RECAPTURE) → disease panel; growth jobs (HARVEST/REPLANT, growth-type RECAPTURE) → growth panel. The other panel keeps its scores.
3. While reviewing an earlier reading (timeline) or analysing, scores are shown. `⇄ scores` pins the scores for that job.
4. The lane shows `🤖 ROBOT WORKING` while it waits; in manual mode the worker panel says "Waiting for your approval" with the steps previewed and Approve/Reject inside the panel.

### 11.5 Styling / Tk gotchas
Dark palette dict `C` (bg `#0f131a`, card `#171d27`, panel `#1e2633`, growth cyan `#38c8ea`, disease orange `#ff7a45`, good `#43b05c`, warn `#e6b422`, bad `#e0463c`, accent `#4c8dff`); action colours match the old OpenCV banners (healthy `#3caa3c`, harvest `#e6960a`, monitor `#c9a90a`, treat `#dc2828`). Font = first available of Segoe UI, SF Pro Text, Helvetica Neue, Ubuntu, Noto Sans, DejaVu Sans, Arial; emoji in labels depend on the system font.
* `app_ui` does `os.chdir(<its folder>)` at import (the pipeline uses `./models`, `./logs`, `./dynamo_table.json`) — anything importing it changes the working directory.
* Photos are shown via `PhotoImage` built from PPM bytes (PNG fallback); cached by `id(bgr)` + size; `load_bgr` is an `lru_cache(48)` returning **shared arrays — never mutate them**. Keep references to `PhotoImage`s.
* A `WorkerPanel` and a `ScorePanel` share a grid cell: swap with `grid_remove()` / `grid()` (the latter restores the old options).
* Pack order = priority when space is short (that's why the decision chip is re-packed before the pod id, and the footer is packed before the body).
* Overlays use canvas `stipple`; long texts update `wraplength` on `<Configure>`; resizes are debounced with `after`.
* The setup controls live in the **sidebar** (not the toolbar) because the toolbar overflowed at laptop width.

### 11.6 Toolbar/state behaviour
`_set_state`: Start enabled in `idle/finished/stopped` (label "New run" after one); Pause enabled while running/paused; Step only while paused; Stop while running/paused/finished; setup controls disabled while running. Closing the window asks to stop, saves the log, and `engine.close()`s. `Reset history` asks for confirmation (it deletes `dynamo_table.json` and the crop-age file).

---

## 12. Data files and formats

| File | Written by | Content |
|---|---|---|
| `dynamo_table.json` | `LocalDynamoTable` | list of history rows (§5.4). **The user's real data.** |
| `dynamo_table_cycle.json` | `CycleStore` | `{"R2-M-F4": {"age": 3, "cycles": 1}, …}` |
| `logs/run_<UTC>.csv/.json` | `RunLogger` (on finish/stop/"Save log now") | per-scan rows (§5.4) |
| `work_orders/run_<UTC>/trace.jsonl` | `OrderBook` | one JSON per line, e.g. `{"ts":…,"event":"status","order_id":"WO-0001","task":"TREAT","agent":"worker_back","pod_id":"R1-L-B2","status":"approved","note":"auto-approved (auto-approve is ON)"}`; events: `order_created` (+decision, evidence), `status`, `ai_consulted`, `loop_closed`, `policy_changed` |
| `…/orders/WO-0001.json` | `OrderBook` | current state of an order |
| `…/outbox/<agent>/WO-0001.json` | `FileOutbox` | the dispatched order — the integration surface for real robots |
| `…/frames/WO-0007_after.jpg` | `SimulatedCamera` | simulated follow-up frames |

Order JSON (real output, trimmed):
```json
{"order_id": "WO-0001", "task": "TREAT", "role": "worker", "agent": "worker_back", "pod_id": "R1-L-B2", "crop_type": "cabbage",
 "trough_id": "R1-L-B", "side": "back", "location": "rail 1 · trough R1-L-B · back", "priority": 1,
 "title": "Treat Alternaria leaf spot on R1-L-B2",
 "rationale": "Alternaria leaf spot detected at 0.92 confidence (85 detections) -> foliar fungal / mildew disease playbook.",
 "evidence": {"visit": 1, "image": ".../sample_images/….jpg", "image_size": [640, 640], "disease_name": "Cabbage-Alternaria_leaf_spot",
              "disease_confidence": 0.92, "action": "flag_for_treatment", "trend": "first visit", "cycle": {}, "rail": 1, "...": "..."},
 "targets": [{"class": "Cabbage-Alternaria_leaf_spot", "confidence": 0.918, "box_norm": [0.2569, 0.6016, 0.0562, 0.0592], "center_norm": [0.285, 0.6312]}],
 "steps": [{"seq": 1, "action": "NAVIGATE", "text": "Move the worker agent to R1-L-B and align with pod R1-L-B2.",
            "params": {"trough": "R1-L-B", "pod": "R1-L-B2", "side": "back", "rail": 1}}, "..."],
 "requires_approval": true, "approval_note": "Removes plant material and applies a fungicide - a human must approve.",
 "human_notes": ["Widespread symptoms (85 detections): decide whether to treat or remove the whole plant.", "..."],
 "status": "completed", "progress": 7, "occurrences": 1, "stale": false, "follow_up_of": "", "aspect": "disease",
 "history": [["2026-10-08T08:17:06+00:00", "approved", "auto-approved (auto-approve is ON)"], "..."], "ai": {}, "outcome": {}}
```

---

## 13. Configuration reference (all tunables)

| Where | Name | Default | Meaning |
|---|---|---|---|
| `perception.py` | `CONF_THRESHOLD` / `NMS_THRESHOLD` / `INPUT_SIZE` | 0.25 / 0.45 / 640 | detection floor / overlap suppression / export size |
| `agent.py` | `DISEASE_ACTION_THRESHOLD`, `GROWTH_ACTION_THRESHOLD` | 0.50, 0.40 | act on a disease / trust a growth stage |
| `farm_engine.py` | `SESSION_VISITS` | 4 | visits per pod per run (cycle pods) |
| | `ROBOT_DONE_DWELL_S`, `MIN_DWELL_S`, `FEEDBACK_TIMEOUT_S` | 3.0, 1.2, 15 | finished-job display, finished-pod dwell, analysis timeout |
| `Settings` (live, UI sliders) | `delay_ms`, `conf_threshold`, `disease_threshold`, `growth_threshold` | 700, 0.25, 0.50, 0.40 | |
| `OrderPolicy` (live) | `auto_approve_all` | **True** | master approval switch |
| | `auto_verify_delay_s` / `verify_after_hours` | 8.0 / 24 | camera check after a treatment: demo delay in auto mode / real delay in manual |
| | `recapture_revisit_hours` | 4 | text written into RECAPTURE steps |
| | `simulate_robots`, `sim_step_seconds` | True, 0.8 | built-in fleet and its step time |
| | `close_the_loop` | True | re-analyse a new frame when a camera order finishes |
| `worker_planner.py` | `MAX_TARGETS`, `WIDESPREAD_DETECTIONS` | 8, 10 | |
| | `PRIORITY`, `CONSULT_TASKS`, `BLOCKING_STATUSES/TASKS` | see §7 | |
| `llm_planner.py` | `STRONG_CONF`, `STRONG_BOXES` | 0.85, 3 | evidence the AI may not soften |
| `crop_cycle.py` | `STAGE_VISITS`, `STAGE_ORDER`, `STRIDE` | see §10 | crop timing / stage order / staggering |
| | `LABEL_CORRECTIONS`, `APPLY_LABEL_CORRECTIONS` | mushroom swap, **True** | §10.4 |
| `app_ui.py` | `SCORE_VIEW_SECONDS`, `WORKER_DONE_SECONDS`, `MAX_BOXES`, `MAX_LABELS` | 1.8, 3.0, 25, 6 | display rules; `_pump` 80 ms, `_tick` 250 ms |
| `cloud_pipeline.py` | simulated latency, `DEFAULT_MAX_CONCURRENCY` | 0.05–0.3 s | stand-in for Lambda |

---

## 14. Running, testing and verifying

**Environment.** Built/tested on Python 3.12; OpenCV 4.13 and **5.0.0** (pin `opencv-python==5.0.0.93`; the wheel is `abi3`, so it installs on any Python 3); numpy 2.5.3; Tkinter from the system Python. Use `python app_ui.py` from the project folder (or `./run_app.sh`).

**Regression suite — `python dev/regression.py [guardrails] [cycle] [engine] [modes]`** (≈1 min, headless, temp dirs, exit 0 = pass). It checks:
* `guardrails` — the 10 LLM-guardrail cases (hostile/broken output, invented/cancelled treatments, strong evidence, repeat requests, offline backend).
* `cycle` — stages only move forward; each lingers its configured visits; the crop waits at harvest; pods start at many ages; some-but-not-all pods ripen per run; persistence; every pod has an existing photo for every stage.
* `engine` — a full auto run (ripe lettuce, a mushroom near ripe, a fresh lettuce, a strong TREAT, a RECAPTURE): finishes by itself; **no pod's next image starts while a robot job is open**; history rows == logged scans; only REVIEW waits; every harvest is followed by a completed replant and the next image is age 0; closed loop produced outcomes flagged simulated; `trace.jsonl` event types; outbox exists; crop ages saved; **a second run continues from saved ages**. It prints each pod's story.
* `modes` — manual mode stops the lane and shows `robot`; switching to auto releases it and the run completes; only REVIEW still waits.

**UI verification — `dev/ui_screenshots.py`** drives the real app on a virtual display and saves screenshots:
`xvfb-run -a -s "-screen 0 1500x960x24" python dev/ui_screenshots.py [--manual] [--pods R2-M-F4 R2-R-F1] [--size 1366x768] [--seconds 14]` (needs `xorg-x11-server-Xvfb ImageMagick`). **Look at the pictures** — most UI bugs found so far (clipped chips, overflowing toolbar, stale footer text) were only visible this way. It uses the project's real data files (back them up, or restore `dynamo_table.json` and delete `dynamo_table_cycle.json` afterwards).

**What was verified, and how**

| Claim | Evidence |
|---|---|
| Pipeline unchanged on OpenCV 5.0.0 | identical detections to 4.13 on 12 pods; whole app + loop run on 5.0.0 |
| Lane gating, loop, replant, persistence, modes | `dev/regression.py` |
| UI layout | screenshots at 1500×960, 1600×960, 1366×768 |
| LLM guardrails | `--selftest` |
| **Not verified** | any real LLM; any AWS service; real robots/cameras; Python 3.14 / Fedora runtime; behaviour with an empty `dynamo_table.json` + many runs beyond the tests |

Sandbox notes for an assistant that builds in a similar sandbox: no AWS credentials, no model downloads, network limited to package mirrors; `apt install python3-tk` works as root; an OpenCV-5 venv (`python3 -m venv v5 && v5/bin/pip install "opencv-python>=5,<6" numpy`) was used to prove 5.x compatibility. Commands over ~5 min time out — run long tests in the background (`( cmd > log 2>&1 & )`, `python -u`) and poll the log.

---

## 15. Design decisions and why

* **Tkinter, not Qt/web:** ships with Python, only OpenCV+numpy as dependencies, matches a desktop demo. Custom `Btn` labels for cross-platform colours.
* **Engine separate from UI** (events + method calls): lets everything be tested headless and swapped (real robots, other front-ends).
* **Boxes drawn by the UI, not baked into pixels:** the user's explicit wish (scores and feedback in the UI, not the OpenCV image).
* **Rules first, LLM second, guardrails in code:** the LLM can only ask for a better look, never create/cancel work; failures fall back to rules. Directly supports "human control".
* **Orders are plain JSON + a trace file:** easy to hand to real robots/AWS, and the trace is competition evidence.
* **Manual mode holds *every* robot order** (including camera re-captures) and `REVIEW` escalations are never auto-approved or bulk-approved: the user asked for "any following orders require manual approval".
* **Lane waits for robots:** the user's rule that the next image only comes after the worker is done; implemented as `pod_busy` + a loop-pending marker so the lane can't slip through the gap between a re-capture finishing and its follow-up frame creating a treatment.
* **Replant reset happens before the REPLANT is marked complete:** otherwise a lane could race ahead and show the old crop.
* **Follow-up frames are never logged to history:** history drives crop/visit logic and accuracy; simulated frames must not pollute it.
* **Cycle state is its own file**, not rows in the history table (history rows are visit-numbered and used for trend text).
* **Ground-truth correction lives in `crop_cycle.py`, behind a flag,** not in `perception.py` (which matches the model's own metadata).
* **Small update zips:** the project folder is 160 MB.

---

## 16. Known issues and limitations (be honest about these in the report)

1. **Simulated:** robots, camera follow-up frames, and all "cloud" pieces. **No AWS.** No real LLM run.
2. **Back-pod quirk:** healthy-looking back pods always yield `RECAPTURE` (growth unreadable). Fix idea: treat growth as not applicable when `verified_aspect == "disease"`, or skip rule 2's growth check for back pods.
3. **Mushroom labels** (§10.4) — unresolved; legacy scripts uncorrected.
4. **Repeated photos:** a diseased-looking photo stays diseased after treatment → repeated treatment orders are possible; the simulated VERIFY frame inpaints only ≤8 targeted regions, so heavily infected pods correctly escalate to a human.
5. **Ripe crops that the model does not read as harvest-stage** (e.g. a harvest photo flagged diseased, or low confidence) are never harvested and keep returning ripe photos — realistic, but can look stuck.
6. **Manual mode stalls lanes by design** until you approve (or Skip pod).
7. **Order book is per run** (not reloaded next run); history/cycle files persist, orders don't.
8. **Decisions use the thresholds current at scan time;** the UI re-evaluates displayed cards live, but logs/orders keep the original.
9. `LocalDynamoTable` is not thread-safe (guarded by `_data_lock`); no authentication/security anywhere (local files).
10. Small screens: below ~800 px height the bottom of the sidebar scrolls out of view; minimum window 1180×720.
11. Crop durations are invented relative numbers; only ~12 real photos exist per late stage.
12. The playbook is generic and not agronomic advice.

---

## 17. Mistakes made in this project (so they are not repeated)

* **Mushroom "bug fix" in `perception.py` (WRONG, reverted).** I saw 0/12 accuracy, assumed the registry's labels were swapped and edited `perception.py`. The ONNX metadata and the photos later showed the *manifest* labels are the likely culprit. Lesson: check the model's own metadata and look at the images before editing a model registry; keep corrections where the evidence points (labels), behind a flag. If the user applied the old `crop_cycle_update.zip`, they must restore the original `perception.py` (`git checkout -- perception.py`).
* **Deleted the user's original run logs** with a cleanup glob (`logs/run_2026100*`) — restored from git. Never glob-delete in `logs/`; use `git clean` on untracked files only.
* **An f-string with nested same-type quotes** only parses on Python ≥3.12 — the user may run older versions.
* **A test that looked like a hang:** it waited for a manual-mode run to finish without approving anything; manual mode now (correctly) stops lanes. Tests must approve or use auto mode.
* **Toolbar overflow** introduced by adding buttons — fixed by moving setup controls to the sidebar; always screenshot at 1366×768.
* **Earlier policy reversed on request:** treatments used to *always* need approval; the user chose auto-approve on by default (kept: REVIEW always manual).
* **Abandoned AWS backend:** a boto3 backend was drafted and then deleted at the user's request (see §18 for the design so it can be rebuilt).

---

## 18. Roadmap and how to extend

### 18.1 Priorities for the competition (in order)
1. **Put a real component on AWS** (hard requirement; 10 % of the score). Suggested minimum, reusing the existing interfaces so the engine does not change:
   * `AwsCloudClient` (same methods as `LocalCloudClient`): `put_object` the frame to **S3**, invoke a **Lambda container image** that runs the *same* `perception.classify_pod` + `agent.decide_action` and returns the dict (convert numpy → plain Python); build with `public.ecr.aws/lambda/python:3.12`, `opencv-python-headless==5.0.0.93` (the wheel `opencv_python_headless-5.0.0.93-cp37-abi3-manylinux_2_28_x86_64.whl` exists; check arm64 for Graviton), bundle `models/`, load models at import (`perception.warm_up()` in INIT), `arch: arm64` optional.
   * `AwsHistoryTable` (same methods as `LocalDynamoTable`): table key `pod_id` (HASH) + `sk` = `"<timestamp>#<uuid8>"` (RANGE; avoids same-second overwrites); floats must be stored as `Decimal` and converted back.
   * Order dispatch: swap `FileOutbox` for an **AWS IoT Core** publisher on `farm/agents/<agent>/orders`; robots report via `farm/agents/<agent>/status` → IoT rule → **SQS** → a poller that calls `OrderBook.begin/advance/complete`.
   * Infra as code (SAM/CloudFormation): bucket (private, lifecycle), table (on-demand), Lambda, queue, IoT rule. Test client code offline with `moto` (S3/DynamoDB/SQS) and a fake invoker; Lambda/Bedrock/IoT publish can't be proven without an account — say so.
   * Keep the local backend selectable (a "Backend" switch) so the app still runs offline.
2. **OpenCV tool layer (agent-callable tools)** — see §2.1: registry of real analysis tools with schemas; the planner/LLM chooses and sequences calls, each depending on the last result; log every tool call in `trace.jsonl`. This is the most direct fix for the Agentic Vision wording and for "depth of the OpenCV 5 implementation".
3. **Generative AI step:** add a **Bedrock** backend to `llm_planner` (Converse API; model id from config/env, never hard-coded) — same guardrails. Optionally let it see the photo. Keep it advisory/downgrade-only.
4. **Evaluation harness** (competition asks for it): per-aspect accuracy from `run_logger`; ablations (rules only vs AI planner; loop on/off) over the 72 pods; failure cases (the mushroom labels; quirk §16.2; widespread-infection verification; low-confidence cases); loop outcomes distribution from `trace.jsonl`; human-control metrics (approvals/rejections/escalations). Clearly separate real-frame results from simulated-frame results.
5. **Submission artefacts:** technical report, architecture + agent-workflow diagram, ≤5-minute video, live endpoint or arranged screen-share, public/unlisted repo with pinned deps. Disclose simulated components plainly.
6. Optional: **COOL / Graviton** award (benchmark OpenCV on arm64 vs x86 with COOL; document reproducibly).

### 18.2 Recipes
* **Add a robot task type:** write `build_<task>(…)` in `worker_planner.py` returning a spec (`task, role, aspect, title, rationale, targets, steps, requires_approval, human_notes, approval_note`); add it to `PRIORITY`, `BLOCKING_TASKS` if the lane should wait, a trigger (`BUILDERS` / a `complete()` hook), UI maps (`TASK_COLOR`, `TASK_LABEL`), and a check in `dev/regression.py`.
* **Add a crop:** `perception.MODEL_REGISTRY` + ONNX files, a pod folder under `pybullet_sim/pods/`, `STAGE_ORDER`/`STAGE_VISITS` in `crop_cycle.py` (+`HARVEST_MODE`, `DISEASE_CATEGORY` entries in `worker_planner.py`), station letter/colour in the UI.
* **Change crop timing:** edit `STAGE_VISITS` (keys must be every stage before harvest, in order — `_check()` asserts), `SESSION_VISITS`.
* **Connect real robots:** transport with `.send(dict)` in `OrderBook(transport=…)`; call `begin/advance/complete`; turn off *Simulate robots*.
* **Real camera frames:** an object with `capture(order, out_dir)`; set `engine.frame_source`; decide whether to log real follow-ups to history/cycle.
* **New planner backend:** a callable `(system, evidence, image_path) -> text`; wrap with `LlmPlanner`; add a choice in `App._planner_changed` / `FarmEngine.set_planner`.
* **New live setting:** add a field to `Settings`/`OrderPolicy`, read it where used (workers read fresh), add a slider/checkbox in `_build_body`, mirror it in `README`.
* **New UI element:** build it in `_build_*`; update it from `_pump` (event-driven) or `_tick` (time-driven); never from an engine thread.

### 18.3 Idea backlog (not requested yet)
Treat growth as N/A for back pods (§16.2) · reload orders across runs · real-frame follow-ups written to history · a "Backend: local/AWS" switch · evaluation dashboard in the UI · export a one-click "demo report" (trace + screenshots).

---

## 19. Change history (what each delivery added)

1. **Interactive UI** (`app_ui.py` + `farm_engine.py`): pod cards per the user's sketch, score panels, pause/step/hold/skip/re-scan, queue, live tuning, zoom; `camera.py`/`cloud_pipeline.py` got an optional `conf_threshold`.
2. **Robot orders** (`worker_planner.py`): instructions for worker/camera agents from each analysis; approvals; Robot orders window; simulator; outbox; trace. **OpenCV 5 pin**, `run_app.sh`, Fedora instructions.
3. *(An AWS backend was drafted and dropped.)*
4. **AI planner** (`llm_planner.py`, `try_llm_planner.py`): guardrails, Scripted/Ollama backends, offline lab.
5. **Closed loop + planner integration** (`frame_source.py`, `close_loop`, REVIEW, before/after UI, AI verdict in orders).
6. **Auto-approve default ON** with a toggle; manual mode holds every robot order.
7. **Live fleet strip** in the main window; setup controls moved to the sidebar; short summary chips.
8. **Worker progress inside the score panels**; lanes wait for robots; **REPLANT** after harvest.
9. **Crop life cycles** (`crop_cycle.py`): ordered lingering stages, staggering, harvest-wait, persistence; header row for location/crop age. *(Its first version also edited `perception.py` — reverted, §17.)*
10. **Freeze fix + live front/back scheduling for the PyBullet bridge** (§21).
11. **This handoff, `dev/` tools, `orders_dir` option, mushroom finding corrected (label correction instead of a perception edit).**

---

## 20. Starter prompt for the next conversation

Paste this together with this file (and the project, or at least the code-only snapshot):

> I'm continuing a project called **Farm Agent** (OpenCV AI Competition 2026, powered by AWS). Please read `HANDOFF.md` fully before answering. Work in the existing code; don't rewrite what works. After any change run `python dev/regression.py` and, for UI changes, take screenshots with `dev/ui_screenshots.py` and look at them. Be explicit about what is simulated or untested. Deliver changes as a small zip of only the changed files plus a short note. I'm on Fedora Linux with my own venv and I'm new to LLMs/AWS, so explain plainly. The next thing I want to do is: **<your request>**.

**Working agreements to keep:** never reverse a user decision silently (§1); never delete from `logs/` or overwrite `dynamo_table.json`; never touch Tk from non-UI threads; label simulated things as simulated everywhere (UI, README, report); keep `requirements.txt` pinned; keep README and this file in sync with the code.

---

## 21. PyBullet integration and the scheduling / freeze fix (added later)

**What exists.** `pybullet_sim/` is a PyBullet model of the farm's gantry robot (`gantry_robot.py`: physics + the camera/service agents as generator tasks run in lock-step by a cooperative scheduler; `multiview_localization.py`; `pod_registry.py` geometry; URDFs; `docs/`). `pybullet_sim/bridge.py` connects it to the application: the engine calls `bridge.enqueue_visit(...)` for every logged scan and the order book's transport is wrapped (`wrap_transport`) so every dispatched order also goes to `bridge.enqueue_order(...)`. The bridge owns ONE physics thread; **every PyBullet call must stay on that thread**. The app builds it in `App._attach_pybullet` (background start, fallback to the built-in fleet).

**Scheduling model (v2, replaces the "wait for a partner order" design).** Work = *side jobs* (one pod, one side; kinds `visit` and `order`). The gantry serves one station (row, col) at a time via `GantryRobot.service_station_live(row, col, assess, poll)`: every physics step, for each *idle* side the gantry calls `poll(idle_sides)` (bridge `_poll_new_work`) which returns `{side: [pods]}` to start now, so a side picks up new work while the other side is still busy; the session ends when both sides are idle and nothing is queued for the station. One job per pod at a time. Order progress comes from the gantry's events (`_on_robot_event`: `plant_scan` / `assess` / `camera_left` / `service_start` / `service_contact` → step counter; `service_done` or `service_skip` → `complete`). Failures call `OrderBook.fail` (lane released). `start_async` / `abandon` / `request_stop` / `_shutdown_cleanup` guarantee no order is left dispatched/in-progress when the simulator cannot start or dies. The `GantryRobot.service_station` method is the original fixed-list version and is kept (used by `main.py`/tools).

**Why it froze (root causes found):** 60 s partner wait for lone orders (measured 60.1 s); `bridge.start()` and `engine.close()` on the Tk thread; failed jobs leaving orders in progress forever; `_pump`/`_tick` ending permanently after one exception (no `try/finally`); six ONNX inferences inside one physics-loop turn. Details and evidence: `FREEZE_FIX_NOTES.md`.

**Rules for future changes.** (1) The order book calls `bridge.enqueue_order` *while holding its own lock*, so the bridge must **never call the order book while holding its `_condition` lock** (it currently never does — keep it that way). (2) Periodic UI loops must be exception-proof (`try/except` + `finally: after(...)`). (3) No blocking calls (`bridge.start`, `engine.close`, `join`) on the Tk thread. (4) `poll()` runs on the physics thread, inside the step loop: keep it cheap.

**Tests.** `python dev/check_bridge.py` (27 checks, fake gantry, no PyBullet), `python dev/check_gantry_session.py` (15 checks; the real `service_station_live` with PyBullet stubbed), `xvfb-run … python dev/ui_live_check.py [--fallback] [--profile] [--shots dir]` (real app on a virtual display; measures window stalls and front/back overlap), plus `dev/regression.py`. **Not verified:** anything with real PyBullet (physics, rendering, GUI, collisions under the live scheduler; run `dev/integration_smoke.py` and `pybullet_sim/tools/verify_motion.py`), and the hypothesis that PyBullet rendering holds the GIL.

**Known limits.** One gantry = one station at a time; per side, jobs run one after another; a front pod gets 4 visits per run and a back pod 1, so the front side is busier; `App.PAIR_SIDES` interleaves front/back pods in the queue.


## 19. Pod tags, vision-based identity, and per-pod data folders (added Oct 2026)

**The user's request.** The simulated camera agent used to be *told* which pod it was at (the bridge mapped the app's pod id to a trough and handed
the id back through `assess`). They wanted the pod number **printed on the pod in PyBullet** so the agent identifies the pod with **OpenCV**, plus
each pod's data set stored **in its own folder inside the simulation**, using the DataPipeline package's pod names and data.

**What exists now.**
* `pybullet_sim/pod_labels.py` — `label_for_pod()`, `render_plate()` (what is printed), `PodLabelReader` (what is read: find border ring → rectify → 7 fixed
  cells → template match → vote over views). Reads at half scale with a direct numpy NCC (cv2.matchTemplate's per-call FFT overhead made it 10x slower).
* `GantryRobot._add_pod_label` mounts a textured quad on each trough's aisle wall; `_identify_pod` runs in `_g_camera` after the 3 views, **before** `assess`;
  `assess(pod, identity=...)` (signature-sniffed, old one-arg callbacks still work); `_g_service` refuses a pod whose tag is not confirmed AT the expected location.
* `PyBulletBridge._assess/_check_identity`: tag read vs dispatched pod; mismatch/unreadable → no scores, no service, order **FAILED**, error in the app log. The
  decision colour is applied only after confirmation (`robot.reads_pod_tags`); robots without that attribute (the FakeRobot in `dev/check_bridge.py`) keep the
  old positional behaviour. Late scores (`_apply_pending_updates`) attach only if the recorded tag matches.
* `pybullet_sim/pods/<POD_ID>/` (+ `pod_data.py`, `tools/migrate_pod_folders.py`): 72 folders, 216 photos. `pod_registry.PODS` is built from them; pod order
  (`index`) is the application's trough-by-trough queue order, NOT the DataPipeline manifest's front-then-back order.
* `crop_cycle.STAGE_PHOTOS`: all of a pod's photos per stage; successive visits in a stage rotate through them (`stage_visit`, `waited` in the saved cycle state).
  `PHOTOS` keeps its old shape (one path per stage) for the regression suite.

* UI: `PyBulletBridge.tag_read(app_pod_id)` (cleared when new work for the pod is queued) feeds `PodCard.refresh_tag()` -> a "POD TAG" line on the disease `ScorePanel` and the disease `WorkerPanel` (refreshed from `_tick_once`, 250 ms). Clicking it opens `TagReadWindow` (app_ui.py), fed by `bridge.tag_read()['sheet_png']`: `pod_labels.render_reading_sheet()` draws frames+outlined plate+flattened plate+per-cell decode (ASCII only, OpenCV fonts); the gantry encodes it once per pod (`_identify_pod`, `keep_evidence=True`) so raw frames are not retained for all 72 pods. `_check_identity` also looks the tag up in `pod_data` (crop type, record) and, for orders, rejects a crop that differs from the dispatched one. The line is short on purpose: the panel is ~250 px wide and longer text clipped. Verified with the real UI under Xvfb (`dev/ui_screenshots.py` opens the 3-D GUI, so patch `app_ui.PyBulletBridge` to `gui=False` on a headless box).

* Live monitor (`TagMonitorPanel`, the **Pod tags** tab, inside a `ScrollFrame`; polls slowly and skips frame decoding while hidden): gantry emits `camera_frame` events right after each captured view through `_emit_transient` (NOT stored in `robot.events`, so frames do not accumulate), the bridge turns them into `'view'` feed entries and every `_assess` with an identity into a `'read'` entry carrying structured `evidence` (3 JPEG frames + corners, flattened plate PNG, per-cell results). `bridge.tag_feed_since(seq)` is what the window polls (150 ms; deque of 300). The analysis canvas REPLAYS each finished read in stages (reading itself is ~0.1 s); it speeds up with backlog and jumps to the final state at >= 6 waiting. Tk gotcha: a `Label`'s width/height are pixels only if it holds an image, so camera tiles start with a blank `PhotoImage`.

**Gotchas learned the hard way.**
* PyBullet's texture `v` axis runs bottom-up (first attempt rendered the text upside down); the viewer's right is −X for front pods and +X for back pods.
* Dark-GREY scene poles (value 60–80) fused with the plate's black border under a 3×3 morphological close; the reader now thresholds < 45 and does no closing.
* The previous end-of-service handling (`service_skip` ⇒ job "completed") is unchanged for non-identity skips. Per the README an invalid localization should
  *fail* the order, but the bridge currently completes it. **Not changed here; flag to the user.**

**Behaviour change to expect.** Some pods' curated photos trip the disease model (e.g. R2-M-F4's first lettuce harvest photo, `disease_flag=True`), so that pod gets
a TREAT order and is harvested a visit later. `dev/regression.py` now anchors its harvest→replant check on the harvest *decision*, not the first ripe scan.
Existing `dynamo_table.json` history and `dynamo_table_cycle.json` ages belong to the old photos; delete them for a clean run (the app has a Reset-history button).

**Tests.** `python dev/check_pod_tags.py` (needs pybullet, ~40 s): data layer, 72/72 tag reads on real renders, damaged-tag fail-safety, bridge end-to-end,
removed/swapped-tag sabotage. `python dev/check_bridge.py`, `dev/check_gantry_session.py`, `dev/integration_smoke.py`, `dev/regression.py` all pass.


## 20. Cloud database: pod identity -> crop, growth, inspections, disease, treatments, harvest (added Oct 2026)

**The user's request.** Implement the cloud pipeline (for the OpenCV AI Competition 2026 with AWS: OpenCV 5 + a meaningful AWS component) so a pod's identity is
connected to its crop type, growth stage, inspection results, disease status, treatment history and harvest readiness in a cloud database; **no images in the
cloud, they stay on the computer.** (The existing `dynamo_client.LocalDynamoTable` / `cloud_pipeline.LocalCloudClient` were the intended seam; they still do the
local trend history and the simulated frame->score loop and are unchanged. The images-to-S3/Lambda design was NOT built, by the user's choice.)

**Files.** `cloud_model.py` (pure: table definition, key helpers, `assert_no_images`, event builders, `apply_event` = the profile rules), `cloud_backends.py`
(`DynamoBackend` boto3, `LocalBackend` JSON file with the same semantics), `cloud_sync.py` (`FarmCloudStore`, `CloudSync` queue+retry+outbox, `FarmCloud` facade,
`config_from_env`), `cloud_cli.py`, `aws/farm_table.yaml` (+ `iam_policy.json`, `DEPLOY.md`), `dev/check_cloud.py`, `dev/requirements-dev.txt`.

**Wiring (all best-effort, never raise, never block).** `FarmEngine.farm_cloud` (param `cloud=` for tests); `farm_engine._scan` -> `record_inspection(kind="scan")`;
the follow-up scan -> `kind="followup"`; `OrderBook.on_event` (new hook in `_trace`, fires on every order step) -> `record_order_event`; `PyBulletBridge.on_tag_read`
(new hook in `_assess`) -> `record_tag_read`, wired by the `FarmEngine.simulation_bridge` property setter. `_finalize` calls `farm_cloud.close()` (drains).
UI: `☁` button (state text refreshed in `_tick_once`) + `CloudPanel`, the **Cloud** tab (reads back from the store, in a thread: AWS reads are network calls; it reads only while the tab is visible, and once when opened).

### Merge note (Oct 26 2026): tabs and wall layout
* Notebook is now Operations(0) | Calibration(1) | Pod tags(2) | Cloud(3) | Log(4) | Settings(5); the `_*_tab_index` attributes in `App._build_body` hold the indices. The toolbar ☁ button only shows sync state and selects the Cloud tab.
* **Wall layout.** The ArUco marker (`pod_markers`) and the printed pod tag (`pod_labels`) share the aisle-facing trough wall (0.44 m long). Centred, the marker hid ~30 % of the tag and no view could be read. Now: "serviced" disc x -0.21..-0.17, marker plate x -0.144..-0.056 (`MARKER_X_OFFSET_M = -0.10`), tag x -0.045..+0.21 (`PLATE_W_M = 0.255`, `PLATE_X_OFFSET_M = 0.0825`). Checked with a synthetic render through the real reader/detector (60/60 tag reads; marker seen in >= 2 of 3 views; localization only needs one). **Not yet run against PyBullet**: run `python dev/check_pod_tags.py` and the localization checks after pulling this.

**Design facts worth knowing.**
* One table, `pk=POD#<tag>`, `sk` = `PROFILE` | `INSPECTION#<ts>#<visit>` | `ORDER#<created>#<id>` | `TAGREAD#<ts>`; sparse GSIs `gsi_crop`, `gsi_status`.
  **Order ids restart at WO-0001 every run, so `created` is in the order's sort key** (else runs overwrite each other). It must be STABLE: undated hand-built orders
  fall back to the literal "undated" (a moving fallback once split one order into two items).
* The profile is maintained from a per-pod in-memory mirror by ONE writer (`FarmCloudStore.apply`); the mirror is committed only after the writes succeed, so a retry
  never double-counts; first touch hydrates from the table (restarts continue). A pod the table has never seen writes EVERY attribute (an early bug diffed against
  defaults and skipped default-valued ones). `None` means REMOVE the attribute: both backends must do this (the local one once cleaned `None` away first).
* Absent attributes are normal (e.g. a back pod with no growth stage): read with `.get()`.
* Images: `image_name` + `image_sha256` only. The reset button never touches the cloud. Local-file mode lives next to the history file (`<history>_cloud.json`).
* Only 5 DynamoDB calls are made (Put/Update/Get/Query/DescribeTable): the IAM policy and a test pin this.

**Not done / honest limits.** Never run against real AWS (no access here): the first real run is the real test (`python cloud_cli.py ping`). The competition also asks for a
working web endpoint: that is now `aws/lambda/farm_api.py` + `aws/farm_api.yaml` (below), also never deployed to real AWS.
Single-writer only. The model/photo analysis still runs locally (by design: images stay here), so the AWS part of the architecture is persistence, not inference.

**Tests.** `python dev/check_cloud.py [section numbers]` (needs boto3, moto, models/, pybullet; ~3 min). Fake credentials are set so a real account can't be hit.

### 20.1 The read-only web API (added after 20)
`aws/lambda/farm_api.py` (Lambda, python3.12, boto3 from the runtime; no dependencies) behind an API Gateway **HTTP API** (payload v2; routes `GET /health`,
`GET /pods/{tag}`, `GET /pods`), deployed by `aws/farm_api.yaml`. The code is INLINED in the template (`ZipFile`, limit 4096 characters, now ~4020): no zip/S3/SAM.
**After editing `farm_api.py` run `python aws/sync_inline_code.py`**; `dev/check_cloud_api.py` fails if the two differ. The Lambda role has only `dynamodb:GetItem` + `Query`
on the table and `/index/*` (a test pins this, and that the source has no write/scan call). Public by default (that is the point for a judge); optional `ApiKey` parameter
(`x-api-key`, constant-time compare). Throttled (`RateLimit`/`BurstLimit`). 500s never leak details. Output strips `pk/sk/gsi*/x_*` and returns numbers as numbers.
Gotchas: validate with `re.fullmatch` (a `$` anchor accepts a trailing newline; found by the tests); the tag-read events create a profile for the pod they are about even when
no inspection has happened (so `GET /pods/<tag>` can be 200 for a pod with only a tag read). Not covered: a real deployment; list endpoints read at most one page (1 MB) per
index query, fine for 72 pods. `python dev/check_cloud_api.py` (~3 s).
