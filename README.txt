frame_capture_patch  v7_2   (built on v7_1; nothing from v8-v11 except the things listed below)
==========================================================================================

Copy into OpenCV-Farming-Agent-Application/ (same layout, replace what exists):
  cpp/                                  C++ pod_capture module (unchanged from v7_1)
  capture_io.py, perception.py, frame_source.py     (unchanged from v7_1)
  app_ui.py                             NEW: tabbed window  Operations | Calibration | Log | Settings
  pybullet_sim/gantry_robot.py          camera agent: loads the saved calibration, publishes live data
  pybullet_sim/pod_markers.py           tag localization (+ per-camera details for the live tab)
  pybullet_sim/stereo_rig.py            rig maths: detect / solve / load_saved  (never runs anything itself)
  pybullet_sim/rig_calibration.py       the calibration process (streams JSON events)
  pybullet_sim/rig_feed.py              NEW: tiny thread-safe hand-over robot -> window
  pybullet_sim/sim_options.py           NEW: tiny shared options (Settings tab <-> robot)
  pybullet_sim/tools/generate_gantry_urdf.py, pybullet_sim/urdf/gantry.urdf
                                        BIGGER, BRIGHTER camera modules (A blue, B green-teal) on each camera head
  dev/                                  tests (see below)

Build the C++ module exactly as before (project root):
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
  cmake --build cpp/build --target pod_capture -j

What changed compared with v7_1
-------------------------------
1. The second camera is easy to see: each camera head carries two bright 30 mm camera modules on a small bracket bar
   (A = blue/left, B = green-teal/right), with dark lenses. This is URDF-only: no runtime code, so it cannot affect
   speed. (If you flip STEREO_RIG_ENABLED in gantry_robot.py, regenerate the .urdf from the project root:
   python -m pybullet_sim.tools.generate_gantry_urdf   -- add --check to verify. The generator's import of
   gantry_robot, which could not run as shipped, is fixed.)

2. Tabbed window (the layout we designed):
   Operations   only the live operation: pod queue, pod cards (headline result per aspect, bigger images), and a
                one-line LATEST ticker that keeps warning/error colours.
   Calibration  NEW, see below.
   Log          the full event log, Save log now, Clear; a "Log  ⚠ n" badge counts new warnings/errors.
   Settings     run setup, display (Scores: headline only / full per-class bars, boxes, labels), simulation
                (PyBullet robots, pod localization, use the two-camera rig), live tuning, AI planner.
   The toolbar keeps Start / Pause / Step / Stop, Robot orders and Auto-approve.

3. The Calibration tab -- the multi-camera calibration, live
   Header: status chip, [Calibrate camera pair], [Reload saved], progress bar.
   Live frames: camera A over camera B.
   Calculations: a running text of what is being computed, as it happens.
   Parameters + charts: the calibrated rig (baseline, reprojection error, rotation/translation B vs A, both
   cameras' focal lengths and centres, check against the true rig) and two charts.
   It shows two things:
   a) THE CALIBRATION: press [Calibrate camera pair] (or just press Start the first time -- see below). A separate
      process holds a chessboard in front of the two lenses at 20 poses in a scratch PyBullet world. For every pose
      you see both cameras' frames with the detected chessboard corners, whether both cameras found the board,
      and, from the 6th usable pair on, the available OpenCV stereo solver is re-solved after EVERY new pair:
      OpenCV 5's calibrateMultiviewExtended when available, otherwise OpenCV 4's calibrateCamera + stereoCalibrate.
      Reprojection error, baseline and rotation are shown; the baseline chart converges towards the designed 40 mm.
      Invalid, stale or poor-quality saved calibrations are rejected so the robot falls back to one camera. The result is saved to
      logs/stereo_rig_calibration.json, the frames to logs/calibration_frames/.
   b) THE PODS, during a run: for every pod a camera agent localizes, the tab shows both cameras' frames (cycling
      through the three views, tag outlined), whether each camera found the tag, each camera's own pod-position
      estimate (PnP) and its reprojection error, the stereo triangulation of the tag corners and how far it
      differs from PnP, the fused estimate, the true position, and the error; the error chart grows per pod.
   First Start with the PyBullet robots on and no saved calibration: Start first runs the calibration (watch it on
   the Calibration tab), then begins the run by itself. The robot only ever LOADS the saved calibration; it never
   calibrates inside its own step loop. No saved calibration (or stereo rig switched off): one camera per head.

4. Designed against the freeze you saw in v11 (I could not reproduce it without real PyBullet, so this is design,
   not proof):
   - the window never imports gantry_robot / PyBullet (the Settings tab talks to the robot through sim_options);
   - nothing in the robot's per-step loop talks to the window or to PyBullet more than v7 did; the robot hands the
     window ONE small sample per pod (a deque append + a few array copies, three times per pod visit);
   - the calibration is its own process, read through a thread + queue; the window polls every 150 ms;
   - the window only draws the frames while the Calibration tab is open.
   Measured here (no real PyBullet, a slow single-core sandbox): the longest window stall while the tab was receiving
   a full streamed calibration plus pod samples was 137-315 ms over repeated runs; the project's own
   dev/ui_live_check.py shows the same range as v7 itself (v7: 117-744 ms, new window: 104-250 ms, run to run noise).

Deliberately NOT included (they came after v7 and are what I suspect made v8-v11 freeze): live A/B screens inside the
PyBullet window, in-world visible chessboard calibration, 3-D debug drawings.

Tests (all run without PyBullet; the ones marked D need a display: on a headless box use  xvfb-run python dev/...):
  dev/check_pod_markers.py          tag localization of all 72 pods
  dev/check_stereo_rig.py           rig calibration + rig localization vs single camera
  dev/check_rig_stream.py           the streaming calibration events, convergence, saving, load_saved
  dev/check_camera_agent_run.py     a camera agent's WHOLE visit (with / without a saved calibration, live sample)
  dev/check_calibration_tab.py  D   the Calibration tab filling in live + no window stall + Start-calibrates-first
  dev/check_ui_tabs.py          D   the tab layout, ticker, Log badge, score-detail switch
  dev/check_local_imports.py        finds the trap that froze every agent in v9
  (plus your own dev/ui_live_check.py, integration_smoke.py, regression.py)

Not tested inside real PyBullet / on your machine (no PyBullet in the build sandbox): PyBullet's own rendering of the
tags and of the calibration board, and the first real calibration run. If the Calibration tab shows an error line,
or no frames arrive, send me that text.
