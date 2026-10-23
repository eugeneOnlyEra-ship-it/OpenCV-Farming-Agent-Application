"""A small thread-safe hand-over of live camera-pair data from the robot thread to the application window.

The robot (a background thread in the same process) pushes one SAMPLE per pod it localizes: the two cameras' frames
(downsampled copies), the marker corners each camera found, each camera's own pod-position estimate, the stereo
triangulation, the fused estimate and the true position. The Calibration tab drains the queue a few times a second.
Nothing here blocks: push() is a deque append, and the queue is bounded, so a window that is not looking simply
loses the oldest samples.
"""
from __future__ import annotations

import time
from collections import deque

_SAMPLES = deque(maxlen=8)


def push(sample):
    sample["t"] = time.time()
    _SAMPLES.append(sample)


def drain():
    """All samples pushed since the last call, oldest first."""
    out = []
    while True:
        try:
            out.append(_SAMPLES.popleft())
        except IndexError:
            return out


def clear():
    _SAMPLES.clear()
