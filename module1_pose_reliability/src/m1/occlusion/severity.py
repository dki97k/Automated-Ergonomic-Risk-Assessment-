"""Occlusion severity definitions used in Module 1 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BoxOccluder:
    """Axis-aligned occluder box in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, points_xy: np.ndarray) -> np.ndarray:
        points_xy = np.asarray(points_xy, dtype=np.float64)
        x_coord = points_xy[:, 0]
        y_coord = points_xy[:, 1]
        return (
            (x_coord >= self.x1)
            & (x_coord <= self.x2)
            & (y_coord >= self.y1)
            & (y_coord <= self.y2)
        )


def coco18_xy_conf(pose_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xy, confidence)`` from 3DPW's COCO-18 2D pose format.

    3DPW stores 2D keypoints as ``(3, 18)`` arrays, but this function also
    accepts ``(18, 3)`` to keep downstream code tolerant of converted records.
    """

    pose_2d = np.asarray(pose_2d, dtype=np.float64)
    if pose_2d.shape == (3, 18):
        return pose_2d[:2].T, pose_2d[2]
    if pose_2d.shape == (18, 3):
        return pose_2d[:, :2], pose_2d[:, 2]
    raise ValueError(f"expected COCO-18 2D pose with shape (3, 18) or (18, 3), got {pose_2d.shape}")


def visible_joint_ratio(
    pose_2d: np.ndarray,
    occluders: Iterable[BoxOccluder],
    min_confidence: float = 0.0,
) -> float:
    """Fraction of confident 2D joints not covered by synthetic occluders."""

    xy, confidence = coco18_xy_conf(pose_2d)
    valid = confidence > min_confidence
    if not np.any(valid):
        return 0.0

    occluded = np.zeros(len(xy), dtype=bool)
    for occluder in occluders:
        occluded |= occluder.contains(xy)

    visible = valid & ~occluded
    return float(visible.sum() / valid.sum())


def severity_label(ratio: float) -> str:
    """Map visible joint ratio to the manuscript severity bins."""

    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"visible joint ratio must be in [0, 1], got {ratio}")
    if ratio >= 0.90:
        return "none"
    if ratio >= 0.70:
        return "mild"
    if ratio >= 0.50:
        return "moderate"
    return "severe"
