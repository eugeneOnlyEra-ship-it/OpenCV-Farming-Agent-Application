"""Run-time simulation options shared between the application window and the robot thread.

Kept deliberately tiny and free of heavy imports: the window sets these from the Settings tab, and
gantry_robot reads them when it needs them. (The window must never import gantry_robot itself -- that pulls in
PyBullet and OpenCV and can stall the UI.)
"""
OPTIONS = {
    "localization": "vision",     # "vision" = ArUco tags + solvePnP; "simulator_mask" = PyBullet's own mask (reference)
    "stereo_rig": True,           # use the calibrated two-camera head rig (needs a saved calibration)
}
