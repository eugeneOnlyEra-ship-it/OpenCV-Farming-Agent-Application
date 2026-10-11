# Extending the project — recipes

Concrete, tested steps for the changes people are most likely to ask for. Read
[ARCHITECTURE.md](ARCHITECTURE.md) first if you have not; the "coupled numbers" table there
(§10) explains *why* several of these recipes have extra steps.

---

## 0. The workflow for EVERY change

```
edit  →  (regenerate the URDF if you touched a machine dimension)  →  python3 run_checks.py
      →  (if you touched motion/geometry/scheduler)  python3 verify_motion.py  [--pods 16 while iterating]
      →  before you finish: python3 run_checks.py --full
```

| Command | Takes | Use it |
|---|---|---|
| `python3 run_checks.py --quick` | ~5 s | after every small edit (no simulation) |
| `python3 run_checks.py` | ~1.5 min | before you consider something done |
| `python3 run_checks.py --full` | ~10 min | before handing the project over |
| `python3 verify_motion.py --pods 16` | ~1 min | tight loop while changing motion |

**Golden rule: a change is not done until `run_checks.py` passes.** The scene has no collision
shapes, so the numeric audit in `verify_motion.py` is the *only* thing that notices if the machine
now passes through something.

**Tip — proving a new check works ("mutation testing"):** temporarily break the behaviour on purpose
(delete the line that makes it true) and confirm the checker FAILS with a sensible message; then restore it.
A rule that has never failed has never been shown to work. Every rule in `verify_motion.py` was tested this way.

---

## 1. Change the decision policy

Thresholds and rules live in `decision_agent.py`.

* **Tune a threshold** — edit `CONFIDENCE_FLOOR`, `LOW_CONFIDENCE_RESCHEDULE`, or `NEAR_HARVEST_STAGES`. Nothing else changes.
* **Change what the service agent does for a decision** — edit `SERVICE_ACTION_FOR_DECISION`, e.g. to also water
  uncertain pods: `"schedule_monitoring": "water"`. Nothing else changes.
* **Change how long the camera lingers** — `ACTION_DWELL_SECONDS`.

`decide()` evaluates its rules **in order, first match wins** (documented in its docstring) — inserting a rule
above another changes what the lower one ever sees.

## 2. Add a new decision (e.g. `"quarantine"`)

1. Return it from `decide()` (`decision_agent.py`).
2. Add it as a **key** in `ACTION_DWELL_SECONDS` and `SERVICE_ACTION_FOR_DECISION` (value `None` = no service).
3. Add a colour in `gantry_robot.DECISION_COLOR` (otherwise the trough turns grey).
4. `python3 run_checks.py --quick` — the *decision-tables* check fails until all three tables agree.

## 3. Add a new service action (e.g. `"fertilize"`)

1. `gantry_robot.SERVICE_ACTIONS["fertilize"] = {"seconds": 0.3, "marker": (0.9, 0.6, 0.1, 1.0)}`
   — contact time per plant, and the RGBA of the tag left on the pod.
2. `decision_agent.SERVICE_ACTION_FOR_DECISION["<some decision>"] = "fertilize"`.
3. **`main.py` → `on_event()` has a verb dictionary** for the console line:
   `{"water": "watering", "apply_treatment": "applying treatment to"}[e["action"]]` — a dictionary lookup, so a new
   action raises `KeyError` until you add `"fertilize": "fertilizing"`. (This is the one place that is not table-driven.)
4. `python3 run_checks.py` — *decision-tables* verifies the action is defined; *main-smoke* exercises the loop.

The motion (slide in front of each plant, pad out, hold, pad in) is the same for every action; only the hold time
and tag colour differ. A genuinely different *motion* is recipe 8.

## 4. Change the number or spacing of plants per pod

Edit `pod_registry.PLANT_X_OFFSETS_M` (offsets in metres along the trough, from its centre). Everything else
(`PLANTS_PER_POD`, plant bodies, obstacle boxes, the camera/service loops, the verifier) follows automatically.

Constraints (all checked by `run_checks.py`):

* **Reach.** The camera rod hangs at x − 0.11 and its traverse can only extend toward **+x**; the service rod
  hangs at x + 0.11 and extends only toward **−x**; each by at most 0.23 m (two stages of 0.115 m). Both agents
  must reach every plant, which leaves plant offsets **between −0.11 and +0.11 m** from the trough's centre.
  The current `(−0.11, 0, +0.11)` puts each agent's nearest plant exactly under its own rod. To place plants
  outside that range you must change the traverse (recipe 5) — not just the plants: the retracted sleeve has to
  stay *behind* its rod (so it never crosses the outer rail's line), so the extension direction cannot simply be flipped.
* **Spacing.** Neighbouring plants must be at least one plant width apart (`2 × PLANT_HALF_SIZE_M = 0.07`).
* **Fit.** A plant must lie inside the trough (half-length 0.22).
* **The 16-shape limit.** Each pod's plants are one compound visual, and **PyBullet segfaults above 16 shapes**.
  Cabbage and mushroom plants use 3 shapes each, lettuce 5, so **3 plants per pod is the maximum with lettuce**
  (4 → 20 shapes; `_add_plants` asserts "20 shapes in one compound visual (max 16)" and `run_checks` *robot-builds* fails).
  To get more plants, simplify the lettuce tuft in `_add_plants`.

Tested: 2 plants per pod `(−0.11, 0.11)` runs and passes `verify_motion.py`.

## 5. Change a machine dimension (drive box, traverse, tool head, …)

Dimensions shared with the URDF live in **`gantry_robot.py` groups 3–4** (marked "SINGLE SOURCE OF TRUTH").
The URDF generator imports them.

1. Edit the constant in `gantry_robot.py`.
2. `python3 tools/generate_gantry_urdf.py` (regenerates `urdf/gantry.urdf`).
3. `python3 run_checks.py` — *design-constraints* tells you if you broke a relationship (e.g. plants now out of
   reach); *urdf-generator* tells you if you forgot step 2.
4. `python3 verify_motion.py` — the clearances. Read the "tightest clearance" table to see how much margin you spent.

Numbers that are tight and easy to break (all documented next to the constants): `BOX_RETRACTED_REACH_M`
(portal post / chassis plate), `BAY_OFFSET_M` (heads passing each other), `AGENT_X_OFFSET_M` (straddling the
outer rail), `BAR_CENTRE_BEHIND_M` (traverse must stay behind its rod).

Purely decorative changes (colours, vents, LEDs, motor sizes) are only in `tools/generate_gantry_urdf.py`
(`MATERIALS`, `drive_box()`); they don't need a constants change but still need step 2.

## 6. Add a new physical part to the machine

A part has to exist in **three** places:

1. **The URDF generator** (`tools/generate_gantry_urdf.py`): add a `visual(...)` (fixed to an existing link) or a
   new `link(...)` + `prismatic(...)` joint. Regenerate.
2. **`GantryRobot.robot_volumes()`**: add a `box(name, group, owner, x, y, z, hx, hy, hz)` with the same dimensions.
   *If you skip this the clearance check silently stops protecting the part.* Use group `"hoist"` for things that move
   with an agent, `"chassis"` for fixed ones. Name it `"<agent key> <part>"` for agent parts (the report groups by that).
3. **The joint-name lookups** in `GantryRobot.__init__` if it has its own joint (look joints up by name).

Then `run_checks.py` and `verify_motion.py`. Prove the new box protects it: shrink a gap in the design until the
verifier complains about that part by name.

## 7. Add an agent (e.g. a harvesting tool) or a fifth head

This is the most invasive change; do it in this order:

1. **Generator**: add the agent's links/joints (copy `agent()` in `tools/generate_gantry_urdf.py`); regenerate.
2. **`AGENT_ORDER`** and the joint lookups in `GantryRobot.__init__` (`Agent`, `a.stage_joints`, …).
3. **⚠ `_state_joints` is an order contract.** `_sync_rods()` reads all joints in one call and slices the result
   (`states[2]`, `states[3]` = boxes, `states[4:8]` = the four hoists, …). Adding agents changes those slices — update
   `_state_joints` *and* `_sync_rods` together.
4. **Collision volumes** for its parts (`robot_volumes`, recipe 6).
5. **Behaviour**: a new generator like `_g_camera`/`_g_service` (recipe 8) and start it in `service_station()`.
6. **Verifier**: `part_kind()` expects names of the form `"<side>_<role> <part>"`; the coordination checks assume
   the camera-then-service pipeline — extend them for your new role.
7. Give it a bay: the drive box is 0.315 m wide and houses exactly two bays; a third agent needs a bigger box.

## 8. Add a new behaviour (a new motion) to the scheduler

Express it as a **generator** from the three primitives, then run it alongside the others:

```python
def _g_my_job(self, agent, pod):
    yield from self._g_settle({agent.joint: hoist_extension_for(pod["position_m"]["z"])})  # move, wait to arrive
    self._emit("my_event", pod, agent=agent)          # tell the world (main.py / verifier can react)
    yield from self._g_wait(0.5)                      # hold
    yield from self._g_wait_until(lambda: something)  # wait for another task
```

* Give it to `_run([...])` together with the other tasks (see `service_station`).
* Keep the safety invariants (ARCHITECTURE §7). The usual trap: moving a head with the applicator out, or the chassis
  with something deployed. Stow with `_stow_all()` before driving.
* Emit an event for anything the verifier or main should know about, and document it in the module docstring's
  EVENT LOG list.
* Add a verifier rule for it (recipe 9). If you don't, nothing will notice when it breaks.

## 9. Add a rule to `verify_motion.py`

1. Decide *when* it can be checked: every step → `Checker.__call__`; at an event → `on_event()` inside `main()`;
   over the whole run → the coordination block after the run.
2. Write it as "condition must hold, else `self._fail("message")`" (or append to `at_rest` / `coord`).
3. Label it with the next `Rn` in the docstring's rule catalogue and in a comment at the code.
4. **Mutation-test it**: break the behaviour (e.g. comment out the line in `gantry_robot.py` that makes it true) and
   confirm the verifier now FAILS with your message. Restore.
5. Beware **settling tails**: a joint counts as "arrived" within a few mm and below 0.03 m/s, so for a moment it still
   creeps. Rules asking "is X moving?" should use thresholds well above that tail and well below real speeds
   (see `STAGE_STILL_SPEED`). If a new rule false-alarms right after a move, that is why.

## 10. Swap in the real perception pipeline

See the "HOW TO SWAP IN THE REAL PIPELINE" section of `perception_stub.py`. In short: same first three arguments
`(pod_id, crop_type, image)`, return `{growth_stage, disease_flag, confidence}`, change one import in `main.py`, and
either accept an ignored `rng=None` argument or delete `rng=rng` from the call in `main.assess()`.
The camera really stops at all 3 plants per pod, but the classifier is called once per pod — per-plant
classification would need a new interface (and a decision per plant).

## 11. Add a column to the log

`run_logger.py`: add the name to `FIELDNAMES`, fill it in `log()` (or in a method that updates the row later, like
`mark_service`), optionally aggregate it in `summary()`. `run_checks` *main-smoke* checks the CSV header equals
`FIELDNAMES`, so it will tell you if they drift.

## 12. Resize the farm (rows / columns / layers)

Edit `ROWS`, `COLUMNS`, `LAYERS_PER_STATION` in `pod_registry.py`. Rails, docks, poles, troughs, plants, the route and
the verifier's expectations all derive from them. **Tested: 2 rows × 4 columns** (see the result recorded in
DESIGN_LOG.md). Things to watch:

* More **layers** → higher top trough → higher rails → the lowest pod needs more hoist extension. The hoist joint
  limit is 2.6 m (`HOIST_UPPER_LIMIT` in the generator); *design-constraints* fails if it is exceeded.
* The two **outer** vertical rails are the only vertical connections (the middle has none, per the sketch). With many
  columns the machine still changes pole by driving to the nearest outer rail — longer trips, but correct.
* `main.py` labels rails `"rail <row+2>"` (rail 1 = top dock) — still right for any row count.
* Placeholder images: unchanged crops/stages need no new images; changing the ground-truth pattern does
  (`python3 generate_placeholder_images.py`).
* Re-run `verify_motion.py` (full) — the scene has no collision shapes, so this is the only proof.

## 13. Add something to the GUI only

* Text/lines: `p.addUserDebugText` / `addUserDebugLine` in `_add_rail_labels()` or `_drive_to()` (guarded by `self.gui`).
  **Do not draw per step** — an earlier spray effect made ~12,000 debug-line calls in three stations and slowed the GUI.
* Bodies: create them **once**, then only move them (`resetBasePositionAndOrientation`). Deleting/creating a body per step
  makes it flicker (that is what the rod pieces in `_Rod` avoid).
* Many static shapes: link-less bodies with a compound visual (**max 16 shapes**), not multi-link bodies (4× slower).
* Anything that is a physical obstacle must also be appended to `robot.obstacles` (recipe 6 for machine parts).

## 14. Speed and pacing knobs

| Knob | Where | Effect |
|---|---|---|
| `--speed N` | `main.py` CLI | multiplies every joint's max velocity (verified 1×–4×) |
| `--sleep S` | `main.py` CLI | seconds to sleep after each physics step (0.004 ≈ real time) |
| `CAMERA_PLANT_SECONDS` | `gantry_robot` | camera pause per plant |
| `SERVICE_ACTIONS[..]["seconds"]` | `gantry_robot` | applicator contact time per plant |
| `ACTION_DWELL_SECONDS` | `decision_agent` | camera hold after a decision |
| `JOINT_FORCE_VEL` | `gantry_robot` | per-joint-type max velocity (× speed_scale) |

Times are simulation seconds (240 steps/s) and are not changed by `--speed`.

---

## Pitfalls checklist (things that have already bitten this project)

* **Editing `urdf/gantry.urdf` by hand.** It is generated; your edit is lost on the next regeneration and
  `run_checks` (*urdf-generator*) will flag it. Edit `tools/generate_gantry_urdf.py`.
* **Forgetting `robot_volumes()`** when adding a part → the clearance check silently ignores it.
* **Plants: >16 shapes per compound visual** → PyBullet segfault (now an assertion).
* **Multi-link static bodies** → whole simulation ~4× slower.
* **Deleting/creating bodies per step** → flicker.
* **A long rigid rail across a rod** → slices through the outer vertical rail at outer poles (why the traverse telescopes).
* **`BOX_RETRACTED_REACH_M` too small** → service applicator touches the portal post (found by the verifier at 0.11).
* **Thresholds that ignore settling** → false alarms right after a move (see recipe 9).
* **Python bytecode cache** can mask a same-size edit made within the same second when you script edits + re-runs;
  delete `__pycache__` (or set `PYTHONDONTWRITEBYTECODE=1`) in such harnesses.
* **`main.py` verb dictionary** for new service actions (recipe 3).
* **Real perception signature** vs the `rng=` argument (recipe 10).
* **Legacy files import `CROP_MARKER_COLOR` from `gantry_robot`** — renaming it breaks them (*legacy-imports* check).
