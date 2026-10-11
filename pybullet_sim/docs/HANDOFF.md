# Handoff — paste this into a new chat

> **How to use:** start a new chat, attach the project (the zip), and paste everything below the line.
> It tells the assistant what the project is, where it stands, how to run and verify it, and the rules to
> follow so it doesn't break anything.

---

## Project: OpenCV-Farming-Agent (PyBullet simulation)

I'm continuing work on a PyBullet simulation of a vertical farm and a robot that services it. The project
folder is attached (`OpenCV-Farming-Agent`). **Please read these first, in this order:**

1. `README.md` — overview, how to run, file table.
2. `docs/ARCHITECTURE.md` — how the pieces fit, the kinematic tree, the scheduler, the coupled numbers.
3. `docs/EXTENDING.md` — step-by-step recipes for the changes I'm most likely to ask for.
4. `docs/DESIGN_LOG.md` — why it is the way it is, the mistakes already made and fixed.

Then run `python3 run_checks.py --quick` to confirm your environment works.

### What it is (30-second version)

* A 3 × 3 grid of vertical **poles**; each carries 4 layers of **trough pods** on each face (72 pods), each pod with
  **3 plants** (216). 
* **One machine** rides an overhead rail network (route: dock bar 1 → rails 2, 3, 4 → bottom dock) and carries **four
  agents**: on each side of the beam a **camera agent** and a **service agent**. Each agent is a lengthening rod ending
  in a telescoping traverse that slides its tool head along the pod plant by plant.
* At each station both faces are worked **at the same time**. On each face the **camera** images every plant, assesses the
  pod (stub classifier → decision agent → log), and moves on; **only then** the service agent waters / chemically treats the
  pod's plants (a gold pad on a slim rod touches each plant). The two drive boxes (one per side, each holding both agents'
  drives) **slide toward each other** while changing horizontal pole and back out at stations.
* Perception is a **stub**; everything else is real code.

### Current state

Complete and verified. `python3 run_checks.py --full` passes; the full 72-pod route passes `verify_motion.py` at 1× and 3× speed.
A full `python3 main.py --oracle --seed 42` run takes ~13 s headless and produces 72 log rows (14 treated, 25 watered, 33 held).

### How to run

```bash
pip install pybullet pillow numpy           # pybullet may compile from source (10-15 min, see below)
python3 main.py --gui --oracle --speed 3 --sleep 0.004      # watch it (needs a display)
python3 main.py --oracle --seed 42                          # headless; writes logs/run_*.csv/.json
python3 run_checks.py                                       # health check, ~1.5 min
python3 verify_motion.py --pods 16                          # the per-step safety audit, ~40 s
python3 tools/generate_gantry_urdf.py                       # regenerate urdf/gantry.urdf after editing shared dimensions
```

### Rules for working on this project (please follow these)

1. **Run `python3 run_checks.py` after every change** (`--quick` while iterating, full before finishing).
   Never call something done until it passes.
2. **The scene has no collision shapes.** `verify_motion.py` is the *only* thing that notices if the machine passes
   through something. Any change to geometry, motion or scheduling needs it re-run.
3. **`urdf/gantry.urdf` is GENERATED.** Never hand-edit it; edit `tools/generate_gantry_urdf.py` (decoration) or the shared
   constants in `gantry_robot.py` groups 3–4 (dimensions), then regenerate.
4. **If you add a physical part, add it to `GantryRobot.robot_volumes()` too** or the clearance check won't protect it.
5. **Prove a new safety rule works** by breaking the behaviour on purpose and confirming the verifier fails
   (see EXTENDING.md recipe 9). Real bugs in this project were found this way and by no other.
6. **Performance/GUI rules:** never create/delete bodies per step (flicker); never draw debug lines per step (slow); use
   link-less bodies with compound visuals, **max 16 shapes** (more segfaults), not multi-link bodies (4× slower).
7. **Everything that takes time is a generator** using `_g_settle` / `_g_wait` / `_g_wait_until`, run by `_run()`. No threads.
8. **Keep the safety invariants** (ARCHITECTURE §7): one axis at a time on rails; only drive/slide boxes with everything
   stowed; hoists only deploy with boxes out; camera assesses and moves on *before* the service agent starts.
9. When you change a number, check the "coupled numbers" table (ARCHITECTURE §10) — several dimensions are sized against each other.
10. Keep comments/docstrings accurate. If you change behaviour, update the docstring, `README.md`, and add a line to
    `docs/DESIGN_LOG.md`.

### Preferences I've shown (so you don't have to ask)

* I want visible, **physically credible** behaviour — machinery that looks like it could exist (drive boxes with motors and
  cables, telescoping arms, contact applicators) — and I notice when something passes through something else.
* The animation must be **stable (no flicker) and reasonably fast**; per-frame effects that slow it get removed.
* I don't want extra text drawn in the scene (only the rail numbers 1–4 and "start dock" / "end dock").
* I like it when claims are **backed by the verifier** and when you tell me honestly what you could and couldn't test
  (I run the GUI myself; you can only run headless).
* When you make an interpretation of an ambiguous request, say so, and make the assumption easy to change.

### Assumptions currently baked in (tell me if any is wrong)

1. Service policy: healthy → **water**; disease → **chemical treatment**; uncertain or near-harvest → **nothing**
   (`decision_agent.SERVICE_ACTION_FOR_DECISION`).
2. "Left and right" meant the two **sides of the beam** → one camera + one service agent per side.
3. Plants sit in ports in the aisle-facing wall, 3 per pod, all serviced together; one classification per pod.
4. Boxes slide in only for pole changes and docking.

### Ideas I may ask for next

Re-check queue for `schedule_monitoring` pods (second camera pass, then service) · a finite tank that sends the service agent
to a dock to refill · per-plant decisions (needs per-plant perception) · the real OpenCV/ONNX classifier (`perception_stub.py`
explains the 3-line swap) · a harvest tool · a different farm size (2 rows × 4 columns was tested and passes; see EXTENDING.md recipe 12).

---

## Working-environment tips (for an assistant with a sandboxed shell)

These cost time the first time; they may save you some.

* **PyBullet has no prebuilt wheel for Python 3.12 in that sandbox**; `pip install pybullet` compiles it (≈ 10–15 minutes on one
  core, longer than a single command's time limit). Start it detached and poll:
  `setsid nohup pip install pybullet --break-system-packages > /tmp/pb.log 2>&1 < /dev/null &`.
* **Long jobs** (`run_checks.py --full`, `verify_motion.py`) exceed a single command's limit: run them the same detached way
  (`setsid nohup sh -c 'python3 verify_motion.py > /tmp/v.log 2>&1' < /dev/null > /dev/null 2>&1 &`) and poll the log.
* **Don't `pkill -f <script name>` from inside a command that itself contains that name** — it kills your own shell.
* **Headless pictures:** `p.getCameraImage(..., renderer=p.ER_TINY_RENDERER)` works in `p.DIRECT` mode; that is how the images in
  this project were made. Debug text/lines are **not** rendered by it, and it can look different from the real GUI's OpenGL renderer
  (say so when reporting how something "looks").
* **Scripted edit-and-rerun loops:** Python's bytecode cache can serve a stale module when you edit a file to a same-size text
  within the same second; delete `__pycache__` or set `PYTHONDONTWRITEBYTECODE=1` in such harnesses.
* To document without risking behaviour, add comments by anchored insertion and **compare `ast.dump()` of before/after with
  docstrings stripped** — identical dumps prove the code is unchanged.
