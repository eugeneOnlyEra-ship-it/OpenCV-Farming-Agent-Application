# Vertical plant geometry and top-view localization patch

This patch updates the integrated project so plants grow upward from the top surface of each trough and the camera agent captures elevated views aimed at the trough top.

## Changed files
- `pybullet_sim/pod_registry.py`: plant target heights and lateral contact surfaces now match plants growing from the trough top.
- `pybullet_sim/gantry_robot.py`: vertical crop meshes, raised camera capture position, worker engagement height, and initialization of localization result storage/callback.
- `pybullet_sim/multiview_localization.py`: estimates the trough centre using the visible top surface instead of assuming only the aisle-facing wall is visible.

## Apply
Extract this ZIP over the root of your existing `Integrated-OpenCV-Farming-Agent` directory, preserving the `pybullet_sim/` paths.

Then run:

```bash
python3 -m py_compile pybullet_sim/gantry_robot.py pybullet_sim/pod_registry.py pybullet_sim/multiview_localization.py
python3 app_ui.py
```

PyBullet is required for a live visual/motion test. In this authoring environment PyBullet was not installed, so the 3D scene and live movement could not be executed here. Python compilation and registry geometry checks passed for all 72 pods and 216 plants.

The current localization mask still uses PyBullet segmentation as a simulation-only target mask. This patch improves the geometry and camera view; it does not yet replace that mask with an RGB-only OpenCV pod detector.
