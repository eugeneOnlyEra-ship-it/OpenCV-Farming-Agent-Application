# Freeze fix + simultaneous front/back agents (PyBullet integration)

## What was making the application "freeze sometimes"

I found five causes. The first two are the big ones; I measured them rather than guessing.

| # | Cause | Evidence |
|---|---|---|
| 1 | **The bridge held a lone order back waiting for a "partner" order on the other side** (20 s once the counterpart pod had been scanned, otherwise 60 s). The application's lane waits for its order, and orders at the two sides rarely coincide, so almost every order waited. The robot sat idle, the fleet strip said "idle", and the pod's lane was blocked. | Ran your original `bridge.py` against a fake gantry: a lone order stayed `dispatched` for **60.1 s** before the robot started it. New bridge: starts in ~0.05 s (`dev/check_bridge.py`). |
| 2 | **`bridge.start()` ran on the Tk (window) thread** with a 90 s timeout, so the whole window was unresponsive while PyBullet built its 3-D scene. "New run" and closing the window also blocked on the previous run's shutdown (up to 20 s). | Code reading; new `app.start()` returns in ~50 ms (`dev/ui_live_check.py`). |
| 3 | **A failed physical job left its order "in progress" forever**, so that pod's lane never continued. (`OrderBook.fail()` existed but the bridge never called it, and the UI had no "failed" status entry.) | Code reading; covered by `check_bridge.py` section 4. |
| 4 | **Both periodic UI loops (`_pump`, `_tick`) rescheduled themselves only on success.** Any exception inside them ended all screen updates permanently: the window still answered clicks but never changed again. | Code reading. Both are now exception-proof and isolate each widget's update. |
| 5 | **The camera task ran six ONNX inferences inside one physics-loop turn**, freezing the physics (both sides' agents stopped) for seconds per pod. | Code reading. Inference now runs on its own thread while the physics keeps stepping. |

Also: order progress was reported as a burst at the very end (0 → done), so the UI showed no movement during a physical job.

## How front and back now work simultaneously

* The gantry still serves **one station at a time** (it is one machine), but at a station the front pair and back pair of agents work **independently and at the same time**.
* Work is split into **side jobs** (one pod, one side). The instant a side's agents are idle, any queued job for that side at this station starts. If the other side's job arrives while this side is busy, it starts the moment *its own* agents are free: nothing waits for a "partner" any more (`GantryRobot.service_station_live` + the new scheduler in `bridge.py`).
* **Progress is live**: the gantry's events (`plant_scan`, `assess`, `camera_left`, `service_start`, `service_contact`, `service_done`) advance the order's steps while the machine moves, so the worker panels, the fleet strip and the pod cards show front and back progressing together.
* Camera visits are still issued for a front pod **and its back counterpart together**, so both camera agents are busy at the same time on every visit.
* The app queue now pairs sides (`F1, B1, F2, B2, …` instead of all front pods first) so the two lanes work on the same station together. Set `App.PAIR_SIDES = False` in `app_ui.py` to go back to registry order.
* A front pod gets 4 visits per run and a back pod 1, so the front side naturally has more work; the sides overlap whenever both have work at the station.

## Other behaviour changes

* PyBullet starts **in the background** (`start_async`). The run begins immediately; work queued meanwhile is held until the gantry is ready. If PyBullet is missing or fails to start, queued orders are handed to the built-in fleet simulator automatically (log line explains why).
* A physical job that fails is marked **failed** (lane released, red "robot job failed" status) and the gantry is stowed and carries on with the next job. A 20-minute wall-clock guard per station session raises instead of hanging.
* "New run" and closing the window no longer block: the old run shuts down in the background.
* Stop cancels jobs that have not started; jobs already moving finish first. Pause/Step still act at job boundaries.
* Worker panel text for a queued order now reads "(queued for the robot)".

## Files changed

`pybullet_sim/bridge.py` (scheduler rewritten), `pybullet_sim/gantry_robot.py` (adds `service_station_live`, `recover_after_failure`, off-thread scoring, same-layer contact sheets only; the original `service_station` is untouched), `app_ui.py`, and the new test tools `dev/check_bridge.py`, `dev/check_gantry_session.py`, `dev/ui_live_check.py`.

## How it was tested (and what was NOT)

| Check | Result |
|---|---|
| `python dev/check_bridge.py`: the real bridge + real order book with a scripted fake gantry (no PyBullet needed): lone order starts at once, front/back overlap, live step-by-step progress, other stations wait without deadlock, failure releases the lane, non-blocking start, fallback, stop, late score updates without CPU spin | **27 checks pass** |
| `python dev/check_gantry_session.py`: the **real** `service_station_live` loop with PyBullet stubbed: dynamic join, per-side counter reset (mutation-tested: removing it makes the test fail), guards | **15 checks pass** |
| `dev/regression.py` (engine, crop cycles, lanes, modes) | **all pass** |
| `dev/ui_live_check.py`: the real app on a virtual display against a fake gantry: `start()` returns in ~50 ms; typical window latency 20 ms; no internal UI errors; both camera agents busy at the same time; fallback to the built-in fleet works | pass |
| **Real PyBullet** (physics, rendering, GUI, collisions with the new live scheduler) | **NOT tested.** PyBullet has to be compiled from source and that was not possible in my environment. Please run `python dev/integration_smoke.py` and watch a run in the PyBullet window. |
| Fedora / your machine | not tested |

If the window still hitches while the camera renders images in the real PyBullet GUI, the next step would be running PyBullet in a separate process (PyBullet calls may hold Python's GIL during rendering). I could not measure that here, so treat it as a hypothesis.

One residual hitch remains: right at start-up the window can stall for up to about a second while the analysis models load and the first pod card is drawn (I measured 0.56-0.98 s on my single-CPU test machine, once per run; a normal desktop CPU should be several times faster). After that the typical window latency is 20 ms.
