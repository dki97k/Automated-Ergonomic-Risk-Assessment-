"""Joint mapping for 3DPW correctness evaluation.

3DPW provides SMPL-24 ground-truth joints. SAM 3D Body returns MHR-70
keypoints. The public correctness experiment evaluates only a conservative
body-joint subset shared by both definitions.
"""

from __future__ import annotations

import numpy as np


COMMON_BODY_JOINTS: tuple[str, ...] = (
    "pelvis",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


SMPL24_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}


MHR70_INDEX = {
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_hip": 9,
    "right_hip": 10,
    "left_knee": 11,
    "right_knee": 12,
    "left_ankle": 13,
    "right_ankle": 14,
    "right_wrist": 41,
    "left_wrist": 62,
    "neck": 69,
}


def smpl24_to_common_body(joints_smpl24: np.ndarray) -> np.ndarray:
    """Map 3DPW SMPL-24 joints to the common body subset."""

    joints_smpl24 = np.asarray(joints_smpl24, dtype=np.float64)
    if joints_smpl24.shape[-2:] != (24, 3):
        raise ValueError(f"expected SMPL-24 pose with shape (..., 24, 3), got {joints_smpl24.shape}")
    indices = [SMPL24_INDEX[name] for name in COMMON_BODY_JOINTS]
    return joints_smpl24[..., indices, :]


def mhr70_to_common_body(joints_mhr70: np.ndarray) -> np.ndarray:
    """Map SAM 3D Body MHR-70 keypoints to the common body subset."""

    joints_mhr70 = np.asarray(joints_mhr70, dtype=np.float64)
    if joints_mhr70.shape[-2:] != (70, 3):
        raise ValueError(f"expected MHR-70 pose with shape (..., 70, 3), got {joints_mhr70.shape}")

    mapped = []
    for name in COMMON_BODY_JOINTS:
        if name == "pelvis":
            left_hip = joints_mhr70[..., MHR70_INDEX["left_hip"], :]
            right_hip = joints_mhr70[..., MHR70_INDEX["right_hip"], :]
            mapped.append((left_hip + right_hip) / 2.0)
        else:
            mapped.append(joints_mhr70[..., MHR70_INDEX[name], :])
    return np.stack(mapped, axis=-2)
