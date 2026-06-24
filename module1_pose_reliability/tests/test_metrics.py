from __future__ import annotations

import numpy as np

from m1.evaluation.metrics import mpjpe, pelvis_aligned_mpjpe, procrustes_aligned_mpjpe


def test_mpjpe_zero_for_identical_pose() -> None:
    pose = np.arange(24 * 3, dtype=float).reshape(24, 3)
    assert mpjpe(pose, pose) == 0.0


def test_pelvis_aligned_mpjpe_ignores_global_translation() -> None:
    pose = np.arange(24 * 3, dtype=float).reshape(24, 3)
    translated = pose + np.array([10.0, -2.0, 5.0])
    assert pelvis_aligned_mpjpe(translated, pose) == 0.0


def test_procrustes_aligned_mpjpe_ignores_similarity_transform() -> None:
    pose = np.arange(24 * 3, dtype=float).reshape(24, 3)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = 2.5 * pose @ rotation + np.array([3.0, 4.0, -5.0])
    assert procrustes_aligned_mpjpe(transformed, pose) < 1e-10
