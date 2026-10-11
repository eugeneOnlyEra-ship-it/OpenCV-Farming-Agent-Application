# Design log

The history of the machine's design: what was asked for, what was built, **what went wrong and how it was
found**, and what was tried and removed. Read it to avoid re-litigating decisions and re-making mistakes.
(Newest work last. For how the current design works, see [ARCHITECTURE.md](ARCHITECTURE.md).)

---

## Before this log: the three earlier iterations

The project began as a PyBullet demo of a perception → decision → logging loop over a farm of pole/trough
"pod" stations, serviced by a robot. Three robot designs were tried before the current one; their code is kept
(marked LEGACY) because they still run and later code imports from them.

| Iteration | Files | What it was | Why it was set aside |
|---|---|---|---|
| Bridge-crane gantry | (the original `gantry_robot.py`) | carriage (X) + row (Y) + a mast with a telescoping hoist and a camera at the pod centre | the starting point of this log; rebuilt below |
| Rail-only trolley | `agent_robot.py`, `agent_rail_demo.py`, `README_rail_iteration.md` | two-axis trolley on the rail lattice with dock rails; no hoist/arm — just the route | scoped down deliberately to get the station structure and route right first |
| 6-DOF arm | `agent_arm_robot.py`, `agent_arm_demo.py`, `README_arm_iteration.md` | trolley + IK-driven 6-joint arm from a DH diagram | worked, but more sophisticated than the project needed |

---

## Iteration 1 — make it follow the sketch, and be physically credible

**Ask:** move 1 → 2 → 3 → 4 along the numbered horizontal poles from the user's sketch; the agent should be *attached
to the pole but keep its distance so it never seems to move through the pole or the pods*.

**Diagnosis of the old design:** the trolley's body sat at the rail's own height on its centreline, and the hoist rod
dropped straight down the **pole's own centreline**; the camera went to the trough's exact centre, i.e. *inside* it.

**Built:**
* The chassis now **rides on top of the rail on wheels** (plate ~4.5 cm above it); poles end flush with the rail top.
* A cross-slide swung the hoist 0.40 m to the side so the rod dropped through the empty aisle; the camera stood off the
  trough by ~11 cm.
* **Rail-only routing** (`_plan_rail_route`): X on a horizontal rail or Y on an outer vertical rail, one axis at a
  time, never a diagonal — 1 → 2 → 3 → 4 → end.
* Route and layout decisions taken from the sketch: start already parked on bar 1; only the two **outer** vertical
  rails exist (the middle vertical beams were removed); dock spacing 0.5 → 0.75 m so nothing brushes the dock bars.
* **`verify_motion.py` was created here**, checking every physics step. It became the backbone of the project.

**Lesson:** because the scene has no collision shapes, *a numeric per-step audit is the only safety net.* Everything
after this was verified against it.

## Iteration 2 — speed

**Ask:** make the animation faster. **Built:** `--speed N` (multiplies joint velocities) and verified it collision-free
from 1× to 4×. `--sleep 0.004` ≈ real time; `--speed` shortens the run rather than making it jumpy.

## Iteration 3 — the flickering rod (a lesson in PyBullet rendering)

**Symptom:** the lengthening rod flickered in the GUI. **Cause (measured):** the rod was one body deleted and
recreated on almost every physics step — ~350 rebuilds for a single one-layer move.

**Fix attempts, in order:**
1. *Nine fixed pieces of binary lengths, PyBullet's built-in cylinders.* No more flicker — but **bands**: PyBullet's
   cylinder mesh shades its ends differently from its middle, so stacked pieces looked like beads on a string.
2. *Many identical 5 cm pieces.* Made the beads **worse** (a highlight on every piece).
3. **Custom tube mesh with smooth radial normals** (`_tube_mesh`). A stack of pieces now renders indistinguishably from
   one cylinder (checked pixel-by-pixel against a single cylinder). Back to nine binary pieces.

**Lessons:** never delete/create bodies per step; move existing ones. And *compare renders against a reference* before
trusting that something "looks fine".

## Iteration 4 — a second agent, coordination, and four agents in all

**Ask:** add a *service* agent that waters / treats according to the camera's analysis; camera goes first, the service
agent follows after the camera has assessed the pod and moved on; put a camera and a service agent on **each side** of
the machine (4 total) so nothing has to swing across the rail.

**Interpretations made (worth revisiting if wrong):**
* "left / right" = the two **sides of the beam** (front +Y, back −Y) → one pair per side; the old cross-slide was dropped.
* **Service policy** (an assumption, one table in `decision_agent.py`): healthy → water; disease → chemical treatment;
  uncertain or near-harvest → do nothing (don't treat on a shaky reading, don't water a crop about to be harvested).

**Built:** four agents; a tiny **generator-based cooperative scheduler** (no threads) so front/back and camera/service
genuinely run in parallel and "the service agent waits for the camera" is a `while` loop that yields; front and back
faces worked simultaneously; pods top → bottom; per-station pipeline; event log; log columns for the service job.

**Caught by the verifier:**
* A too-strict "camera not near the pod" check false-alarmed when the camera was merely *passing through* that height on its
  way up. Replaced by "not *parked* there". Also introduced the idea of **settling tails** (a joint counts as arrived
  within a few mm at < 0.03 m/s, so "is it moving?" thresholds must sit above that).
* The camera-first rule couldn't actually fail with short camera holds, so the verifier now gives every 5th pod a long hold
  to *stress* it — and every rule was **mutation-tested** (break the behaviour, confirm the verifier fails).

## Iteration 5 — spray removed; the contact applicator; drive boxes

**Problem:** the spray effect (debug lines redrawn every few steps) slowed the GUI. **Measured:** ~12,100 debug-line calls
in three stations vs 5 (the route trail) after the fix.

**Replaced by** a *contact applicator*: a slim rod with a gold pad that slides out of the service head until it touches the
target (~2 mm short), holds, and slides back — one joint, zero per-frame cost. A tag on the pod records the job.
Also added **drive boxes** where each rod originates (motor, vents, LEDs) so it is credible where the "elongating poles"
come from and what drives them.

## Iteration 6 — one merged drive box per side, sliding in and out

**Ask:** camera and service boxes merge into one larger box per side (cheaper in the real world); the boxes move toward
each other ("disengage") when the machine changes horizontal pole and back out ("engage") at the next station.

**Built:** one two-tone box per side (teal camera bay, purple service bay) sliding on the outrigger rails; boxes in while
changing pole and on the docks, out at stations and while driving along the same pole.

**Caught by the verifier:** with the boxes slid in to 0.11 m, the service agent's *retracted applicator* (which points at
the beam) touched the **portal post at 0 mm** — invisible to the eye. Retract distance raised (0.14, later 0.26 when the
boxes got deeper). New rules: boxes only slide with everything stowed; only change pole with boxes in; hoists only deploy
with boxes out.

**Lesson:** the verifier finds contacts a human eye cannot. Every "looks fine" claim was backed by it.

## Iteration 7 — plants across each pod, and a telescoping traverse

**Ask:** more realistic behaviour — several plants across each pod, and camera/service heads that visibly address them one by
one via some mechanism at the head.

**Decisions:**
* 3 plants per pod, set in planting ports along the **aisle-facing wall** (where the agents already work), crop-specific
  looks (cabbage rosette / lettuce tuft / mushroom cap).
* Heads hang on a **horizontal traverse** that slides them along the pod.
* **The two agents' bays are staggered in depth** (camera 0.06 m further out, service 0.06 m further in). Reason: with both
  rods at the same y, a head sweeping across would hit the *other* agent's rod (the camera's rod stands at the height the
  service head sweeps, and vice versa).

**Caught by the verifier:** my first traverse was a single 0.5 m fixed rail across the rod. At the two outer poles the
machine sits over the outer vertical rail, and the rail **sliced through it** as the hoist descended past it. Fix: a
**two-stage telescoping** traverse whose sleeve sits *behind* the rod (away from the rail's line) and only extends once the
head is down at the pod. Plants were also moved to −0.11 / 0 / +0.11 so each agent's nearest plant is under its own rod.

**Performance regression found and fixed:** plants built as multi-link bodies (~18 static links per pod) made every physics
step ~4× slower (0.8 ms vs 0.2 ms; `main.py` 6 s → 42 s) because PyBullet steps every link of every body. Rebuilt as one
link-less body per pod with a compound visual → back to ~13 s for the full run.
**Crash discovered:** `createVisualShapeArray` **segfaults above 16 shapes** in one compound (a lettuce pod needed 18). Lettuce
simplified to 15; an assertion now guards it.

## Iteration 8 — labels and documentation

The colour-key text lines were removed from the scene (only the rail numbers remain). The project was then documented:
every source file commented (proven code-identical by comparing syntax trees with docstrings stripped), the URDF generator
moved into the project (`tools/generate_gantry_urdf.py`, importing shared dimensions from `gantry_robot.py`), `run_checks.py`
added, and these `docs/`. The doc pass also found and fixed stale statements (e.g. `main.py` mentioning a non-existent
`PERCEPTION_FN`; a docstring saying the drive boxes slide back out to 0.40 m rather than 0.445 m) and one real gap: the
instant checks did not build the scene, so a 4-plant lettuce layout passed them and only failed at run time — the
*robot-builds* check now catches it.

---

## Decisions that look odd but are deliberate

| Decision | Why |
|---|---|
| The scene has **no collision shapes** | Early poles with collision geometry made the carriage physically jam. Clearance is proven numerically instead. |
| The **URDF is generated** and shares constants with `gantry_robot.py` | Hand-editing 400 lines of repeated XML, and keeping it in sync with the Python numbers, was the main risk. |
| Rods are **not in the URDF** | A fixed-length link can't span 0.4–2.2 m without a gap. |
| Rods are stacks of **fixed pieces that only move** | Deleting/creating a body per step flickers. |
| **Custom tube mesh** for the rod | PyBullet's built-in cylinder shades stacked pieces like beads. |
| **Generators, not threads** | Deterministic; lets one agent wait for another with a plain loop; verifiable step by step. |
| **Machine stops centred over each pole** | At outer poles that is the crossing with the outer rail; the two agents straddle it. |
| **Bays staggered in depth** | So heads can slide past each other's rods. |
| **Telescoping traverse, sleeve behind the rod** | A long fixed rail crosses the outer rail's line during descent. |
| The applicator **stops ~2 mm short** | Joints settle ~1 mm short; the verifier requires −4…+6 mm ("a touch, not a push"). |
| Thresholds in the verifier are **generous after moves** | Settling tails — see EXTENDING.md recipe 9. |
| Only **two outer vertical rails** | From the user's sketch. |
| **Legacy files kept** | They still run and import `CROP_MARKER_COLOR` from `gantry_robot`. |
| `service_visit_order` puts the **front face then back face** per station | Flat order for `--pods N` and scene building; the agents themselves work both faces in parallel. |

## Assumptions to revisit with the user

1. **Service policy** — healthy → water; disease → treat; uncertain/near-harvest → nothing. (`decision_agent.SERVICE_ACTION_FOR_DECISION`)
2. **"Left and right" = the two sides of the beam.** (Alternative reading: fore/aft along the beam.)
3. **Plants sit in ports in the aisle-facing wall**, 3 per pod, all serviced together.
4. **One classification per pod**, although the camera images every plant (per-plant classification needs a new interface).
5. **Boxes slide in only for pole changes and docking**, not between stations on the same pole.

## Ideas discussed but not built

* **Re-check queue** — revisit `schedule_monitoring` pods later in the run for a second camera pass and then service.
* **Tank / refill** — a finite supply that makes the service agent return to a dock.
* **Per-plant decisions** — treat only the diseased plant (needs per-plant perception).
* **Real perception** — drop in the OpenCV/ONNX pipeline (see `perception_stub.py`).
* **A harvest tool** — a third tool per side (needs a bigger drive box; EXTENDING.md recipe 7).

## Test evidence recorded during development

* Full 72-pod verification passes at 1× and 3× speed (≈ 52,000 and ≈ 30,000 physics steps).
* Every verifier rule has been **mutation-tested** (deliberately broken → the verifier fails with a clear message).
* `main.py` full run: 72 rows (14 treated, 25 watered, 33 held), all service statuses `done`.
* Farm variants tried: **2 plants per pod** (runs, verifies); **4 plants per pod** (fails cleanly at scene build, 16-shape
  limit); **2 rows × 4 columns** (64 pods, 192 plants: the full verification passes, 46,121 steps, route 1 → 2 → 3 → end).
