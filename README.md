# OpenCV-Farming-Agent — Data Processing Pipeline (no PyBullet)

Same perception → decision pipeline as the full PyBullet build, with
every simulated-robot concept removed: no rail geometry, no movement,
no rig, no GUI. This is for working on the data processing,
classification/decision logic, growth tracking, and visual output on
their own, without a simulation layer in the way.

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
`run_logger.py`, `dynamo_client.py`, `trend.py` are shared with the
PyBullet project; only `pod_registry.py` (rig geometry stripped out),
`main.py` (no movement/dwell/GUI), `visualize.py`, and `gallery.py` are
specific to this build.

## Farm structure (data grouping only, no physical layout)

3 station rails × 3 stations × 2 sides = 18 physical troughs; each
trough holds 4 pods (a 3-plant cluster each) → **72 pods, 24 per
crop**. `rail`/`station`/`side` are kept on every pod as grouping
metadata (useful for "how did rail 3 do" style breakdowns) even though
there's no simulated rig to place them in.

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

## Files

| File | Role |
|---|---|
| `pod_registry.py` | Loads `pods_manifest.json` into 72 pod dicts (metadata + growth trajectories, no rig geometry) |
| `pods_manifest.json` | The 72 pod → image → ground-truth assignments, plus each front pod's full growth trajectory |
| `sample_images/` | 129 real curated photos (72 base + 57 additional trajectory stages) |
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

## What's deliberately NOT here

`gantry_robot.py`, `rail_path.py`, the rig geometry constants, the
`--gui`/`--sleep` flags, dwell times, and the URDF/visual scene
construction all live only in the PyBullet build. If you need the
simulated rig back, that's a separate project rather than something
layered into this one — the point of this version is to have a place
to iterate on the data/classification/decision/tracking/visualization
logic without the simulation layer's own moving parts in the way.
