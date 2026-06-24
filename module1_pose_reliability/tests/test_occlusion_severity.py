from __future__ import annotations

import numpy as np

from m1.occlusion.severity import BoxOccluder, severity_label, visible_joint_ratio


def test_visible_joint_ratio_uses_box_occluders() -> None:
    pose = np.zeros((3, 18), dtype=float)
    pose[0] = np.arange(18)
    pose[1] = 5
    pose[2] = 1

    ratio = visible_joint_ratio(pose, [BoxOccluder(0, 0, 8, 10)])
    assert ratio == 0.5


def test_severity_label_bins() -> None:
    assert severity_label(0.95) == "none"
    assert severity_label(0.75) == "mild"
    assert severity_label(0.60) == "moderate"
    assert severity_label(0.25) == "severe"
