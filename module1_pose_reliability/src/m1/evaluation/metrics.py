"""3D pose evaluation metrics for Module 1."""

from __future__ import annotations

import numpy as np


def _validate_pose_pair(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(f"pose shapes differ: predicted={predicted.shape}, target={target.shape}")
    if predicted.ndim < 2 or predicted.shape[-1] != 3:
        raise ValueError("poses must have shape (..., n_joints, 3)")
    return predicted, target


def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean per-joint position error in the input unit."""

    predicted, target = _validate_pose_pair(predicted, target)
    return float(np.linalg.norm(predicted - target, axis=-1).mean())


def pelvis_aligned_mpjpe(
    predicted: np.ndarray,
    target: np.ndarray,
    pelvis_index: int = 0,
) -> float:
    """MPJPE after translating both poses to the pelvis/root joint."""

    predicted, target = _validate_pose_pair(predicted, target)
    pred_root = np.take(predicted, [pelvis_index], axis=-2)
    target_root = np.take(target, [pelvis_index], axis=-2)
    return mpjpe(predicted - pred_root, target - target_root)


def scale_aligned_mpjpe(
    predicted: np.ndarray,
    target: np.ndarray,
    pelvis_index: int = 0,
) -> float:
    """MPJPE after pelvis translation and per-sample scale alignment."""

    predicted, target = _validate_pose_pair(predicted, target)
    flat_pred = predicted.reshape(-1, predicted.shape[-2], 3)
    flat_target = target.reshape(-1, target.shape[-2], 3)
    aligned = np.empty_like(flat_pred)

    for idx, (pred_pose, target_pose) in enumerate(zip(flat_pred, flat_target)):
        pred_centered = pred_pose - pred_pose[[pelvis_index]]
        target_centered = target_pose - target_pose[[pelvis_index]]
        denom = np.sum(pred_centered**2)
        scale = 1.0 if denom == 0 else float(np.sum(pred_centered * target_centered) / denom)
        aligned[idx] = pred_centered * scale

    target_centered = flat_target - flat_target[:, [pelvis_index], :]
    return mpjpe(aligned.reshape(predicted.shape), target_centered.reshape(target.shape))


def procrustes_aligned_mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """PA-MPJPE using similarity Procrustes alignment.

    This implementation follows the common evaluation protocol: center both
    poses, solve the optimal rotation by SVD, estimate a single global scale,
    and then compute MPJPE.
    """

    predicted, target = _validate_pose_pair(predicted, target)
    flat_pred = predicted.reshape(-1, predicted.shape[-2], 3)
    flat_target = target.reshape(-1, target.shape[-2], 3)
    aligned = np.empty_like(flat_pred)

    for idx, (pred_pose, target_pose) in enumerate(zip(flat_pred, flat_target)):
        pred_mean = pred_pose.mean(axis=0, keepdims=True)
        target_mean = target_pose.mean(axis=0, keepdims=True)
        pred_centered = pred_pose - pred_mean
        target_centered = target_pose - target_mean

        covariance = pred_centered.T @ target_centered
        u_mat, singular_values, v_t = np.linalg.svd(covariance)
        rotation = u_mat @ v_t
        if np.linalg.det(rotation) < 0:
            v_t[-1, :] *= -1
            singular_values[-1] *= -1
            rotation = u_mat @ v_t

        denom = np.sum(pred_centered**2)
        scale = 1.0 if denom == 0 else float(singular_values.sum() / denom)
        aligned[idx] = scale * pred_centered @ rotation + target_mean

    return mpjpe(aligned.reshape(predicted.shape), target)
