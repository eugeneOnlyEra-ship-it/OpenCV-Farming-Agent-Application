# Multi-view pod localization patch

This patch overlays the following files onto the existing `Integrated-OpenCV-Farming-Agent` directory:

- `pybullet_sim/multiview_localization.py` (new)
- `pybullet_sim/gantry_robot.py`
- `pybullet_sim/bridge.py`
- `worker_planner.py`
- `dev/integration_smoke.py`
- `README.md`

From the directory containing the patch ZIP, extract it over your existing project folder and allow the files to be replaced. For example, if the project folder is `~/Projects/Integrated-OpenCV-Farming-Agent`:

```bash
unzip -o MultiView-Pod-Localization-Patch.zip -d ~/Projects/Integrated-OpenCV-Farming-Agent
cd ~/Projects/Integrated-OpenCV-Farming-Agent
python3 dev/integration_smoke.py
```

The existing `requirements.txt` already includes OpenCV, NumPy, and PyBullet. If this project uses a virtual environment, activate it before running the smoke test or application.

The smoke test does not require PyBullet to be importable. A full localization/movement test does require the PyBullet dependency and a live application run.
