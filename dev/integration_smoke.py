#!/usr/bin/env python3
"""Fast, PyBullet-independent checks for the app ↔ gantry integration."""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pod_registry import PODS
from pybullet_sim.bridge import PyBulletBridge


def main():
    bridge = PyBulletBridge(gui=False)
    mapped = [bridge._map_pod(pod) for pod in PODS]
    assert len(PODS) == 72, f"expected 72 app pods, got {len(PODS)}"
    assert all(mapped), "at least one application pod has no PyBullet geometry mapping"
    mapped_ids = [pod["pod_id"] for pod in mapped]
    assert len(set(mapped_ids)) == 72, "multiple application pods map to the same geometry pod"
    for app_pod, sim_pod in zip(PODS, mapped):
        assert sim_pod["row"] == app_pod["rail"] - 1
        assert sim_pod["col"] == app_pod["station"]
        assert sim_pod["side"] == app_pod["side"]
    bridge.pause()
    assert bridge._paused
    bridge.step_once()
    assert bridge._step_tickets == 1
    bridge.resume()
    assert not bridge._paused and bridge._step_tickets == 0
    print("PASS: 72/72 unique pod mappings")
    print("PASS: shared pause / step / resume controller state")
    ok, reason = bridge.dependency_available()
    print("PyBullet runtime:", "available" if ok else f"not available ({reason})")
    if not ok:
        print("A real motion test requires installing requirements.txt on the target machine.")


if __name__ == "__main__":
    main()
