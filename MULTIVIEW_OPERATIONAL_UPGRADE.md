# Multi-view scores and visible plant operations

This patch changes the PyBullet executor so that the application’s already-computed OpenCV classification scores are passed into the camera visit. The gantry captures and saves three RGB-D views per pod, estimates a pose after the app assessment callback, and records the model scores and decision alongside the localization result. Captures are written to `logs/multiview/`.

Plants are now individual visual bodies. Approved whole-plant disease-removal steps (`REMOVE_TISSUE` with `mode: whole_plant`) map to a visible gripper close, upward pull, and removal of the selected plant body. Selection uses normalized detection x-centres to choose among the three simulated plants, with the middle plant as a fallback when no target box is available. HARVEST removes the simulated plant body; REPLANT restores missing bodies. Treatment that is not a whole-plant removal remains a treatment contact rather than silently uprooting the crop.

The two visible gripper jaws are visual-only bodies synchronized to the service tool tip; this is a kinematic animation, not a physics-based grasp with articulated finger joints. Plants are visual-only bodies as in the previous scene. The segmentation mask used by the localization estimator is still simulator ground truth, so this is not yet proof of RGB-only localization.
