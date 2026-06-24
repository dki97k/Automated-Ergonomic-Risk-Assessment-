#!/usr/bin/env python3
"""Visualize worst-case correctness examples for SAM-3DB and AlphaPose-MotionBERT.

The figure contains two cases:
- the sample with the largest AlphaPose-MotionBERT PA-MPJPE;
- the sample with the largest SAM-3DB PA-MPJPE.

For each case, the original frame and root-aligned 3D skeleton comparisons are
shown for both methods against 3DPW ground truth.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MOTIONBERT_ROOT = PROJECT_ROOT / "vendor" / "MotionBERT"
for path in (SRC_ROOT, MOTIONBERT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.utils_data import crop_scale, flip_data  # noqa: E402
from m1.data.three_dpw import ThreeDPWFrame, iter_frames  # noqa: E402
from m1.evaluation.joint_mapping import mhr70_to_common_body, smpl24_to_common_body  # noqa: E402
from m1.evaluation.metrics import pelvis_aligned_mpjpe, procrustes_aligned_mpjpe  # noqa: E402


MOTIONBERT_TO_COMMON = [0, 8, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
EDGES = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (4, 6),
    (3, 5),
    (5, 7),
    (0, 8),
    (0, 9),
    (8, 10),
    (10, 12),
    (9, 11),
    (11, 13),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("<private_workspace>/data/3dpw/extracted"),
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument(
        "--sam-metrics",
        type=Path,
        default=PROJECT_ROOT / "results" / "metrics" / "e1_3dpw_correctness_stride50.csv",
    )
    parser.add_argument(
        "--apmb-metrics",
        type=Path,
        default=PROJECT_ROOT / "results" / "metrics" / "e1_alphapose_motionbert_3dpw_stride50.csv",
    )
    parser.add_argument(
        "--sam-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "sam3db_3dpw_test_stride50",
    )
    parser.add_argument(
        "--alphapose-2d",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "alphapose_coco17_3dpw_stride50.npz",
    )
    parser.add_argument(
        "--motionbert-config",
        type=Path,
        default=MOTIONBERT_ROOT / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml",
    )
    parser.add_argument(
        "--motionbert-checkpoint",
        type=Path,
        default=MOTIONBERT_ROOT
        / "checkpoint"
        / "pose3d"
        / "FT_MB_lite_MB_ft_h36m_global_lite"
        / "best_epoch.bin",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "figures" / "e1_correctness_worst_cases.png",
    )
    parser.add_argument(
        "--projection",
        choices=("3d", "bodyplane"),
        default="bodyplane",
        help="Use bodyplane for publication-friendly left-right/up skeletons.",
    )
    return parser.parse_args()


def worst_sample(metrics_path: Path, metric: str = "pa_mpjpe_mm") -> tuple[str, float]:
    rows = list(csv.DictReader(metrics_path.open()))
    row = max(rows, key=lambda item: float(item[metric]))
    return str(row["sample_key"]), float(row[metric])


def safe_sample_filename(sample_key: str) -> str:
    return sample_key.replace("/", "__") + ".npz"


def load_sam_prediction(sample_key: str, prediction_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(prediction_dir / safe_sample_filename(sample_key), allow_pickle=True)
    pred = np.asarray(data["pred_mhr70_m"][0], dtype=np.float64)
    cam_t = np.asarray(data["pred_cam_t_m"][0], dtype=np.float64)
    bbox = np.asarray(data["bbox"][0], dtype=np.float64)
    return mhr70_to_common_body(pred + cam_t[None, :]), bbox


def load_alphapose_2d(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keys = [str(key) for key in data["keys"]]
    poses = np.asarray(data["poses"], dtype=np.float64)
    return dict(zip(keys, poses, strict=True))


def predict_motionbert_for_needed(
    opts: argparse.Namespace,
    frames_by_key: dict[str, ThreeDPWFrame],
    poses_2d: dict[str, np.ndarray],
    needed_keys: set[str],
) -> dict[str, np.ndarray]:
    cfg = get_config(str(opts.motionbert_config))
    model = load_backbone(cfg)
    checkpoint = torch.load(str(opts.motionbert_checkpoint), map_location="cpu")
    state_dict = checkpoint["model_pos"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    grouped: dict[tuple[str, str, int], list[tuple[int, str, np.ndarray]]] = {}
    needed_groups = {
        (frames_by_key[key].split, frames_by_key[key].sequence, frames_by_key[key].person_id)
        for key in needed_keys
    }
    for key, pose in poses_2d.items():
        if key not in frames_by_key:
            continue
        frame = frames_by_key[key]
        group_key = (frame.split, frame.sequence, frame.person_id)
        if group_key in needed_groups:
            grouped.setdefault(group_key, []).append((frame.frame_index, key, pose))

    predictions = {}
    with torch.no_grad():
        for items in grouped.values():
            items.sort(key=lambda item: item[0])
            motion = crop_scale(np.stack([item[2] for item in items]), [1, 1]).astype(np.float32)
            chunks = []
            for start in range(0, len(motion), int(cfg.clip_len)):
                clip = motion[start : start + int(cfg.clip_len)]
                batch_input = torch.from_numpy(clip[None])
                if cfg.no_conf:
                    batch_input = batch_input[:, :, :, :2]
                if cfg.flip:
                    pred_1 = model(batch_input)
                    pred_flip = model(flip_data(batch_input))
                    pred = (pred_1 + flip_data(pred_flip)) / 2.0
                else:
                    pred = model(batch_input)
                if cfg.rootrel:
                    pred[:, :, 0, :] = 0
                else:
                    pred[:, 0, 0, 2] = 0
                if cfg.gt_2d:
                    pred[..., :2] = batch_input[..., :2]
                chunks.append(pred.cpu().numpy()[0])
            pred_h36m = np.concatenate(chunks, axis=0)
            for pred, (_, key, _) in zip(pred_h36m, items, strict=True):
                if key in needed_keys:
                    predictions[key] = pred[MOTIONBERT_TO_COMMON, :]
    return predictions


def root_align(pose: np.ndarray) -> np.ndarray:
    return pose - pose[[0]]


def canonical_body_view(target: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate root-aligned poses into an anatomy-friendly display frame.

    The target skeleton defines the display basis: X is left-to-right shoulder
    direction, Y is pelvis-to-neck, and Z is their cross product. This changes
    only the visualization, not the reported metrics.
    """

    target_ra = root_align(target)
    predicted_ra = root_align(predicted)

    shoulder_axis = target_ra[2] - target_ra[3]
    vertical_axis = target_ra[1] - target_ra[0]
    if np.linalg.norm(shoulder_axis) < 1e-8 or np.linalg.norm(vertical_axis) < 1e-8:
        return target_ra, predicted_ra

    x_axis = shoulder_axis / np.linalg.norm(shoulder_axis)
    y_axis = vertical_axis - np.dot(vertical_axis, x_axis) * x_axis
    if np.linalg.norm(y_axis) < 1e-8:
        return target_ra, predicted_ra
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    # Matplotlib displays the third coordinate as the vertical axis most
    # naturally, so store the anatomy frame as left-right, depth, up.
    basis = np.stack([x_axis, z_axis, y_axis], axis=1)

    return target_ra @ basis, predicted_ra @ basis


def equalize_3d_axes(ax, poses: list[np.ndarray]) -> None:
    points = np.concatenate(poses, axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    radius = max(radius, 0.3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_skeleton(ax, pose: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
    for start, end in EDGES:
        ax.plot(
            [pose[start, 0], pose[end, 0]],
            [pose[start, 1], pose[end, 1]],
            [pose[start, 2], pose[end, 2]],
            color=color,
            linewidth=2.0,
            linestyle=linestyle,
        )
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], color=color, s=16, label=label)


def plot_pose_comparison(ax, target: np.ndarray, predicted: np.ndarray, title: str, metric_text: str) -> None:
    target_display, predicted_display = canonical_body_view(target, predicted)
    plot_skeleton(ax, target_display, "#111111", "3DPW GT")
    plot_skeleton(ax, predicted_display, "#d62728", "Prediction", "--")
    equalize_3d_axes(ax, [target_display, predicted_display])
    ax.view_init(elev=5, azim=-90, roll=0)
    ax.set_title(f"{title}\n{metric_text}", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_box_aspect((1.0, 0.55, 1.35))
    ax.legend(loc="upper left", fontsize=8)


def equalize_2d_axes(ax, poses: list[np.ndarray]) -> None:
    points = np.concatenate([pose[:, :2] for pose in poses], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    radius = max(radius, 0.3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def plot_skeleton_2d(ax, pose: np.ndarray, color: str, label: str, linestyle: str = "-") -> None:
    for start, end in EDGES:
        ax.plot(
            [pose[start, 0], pose[end, 0]],
            [pose[start, 1], pose[end, 1]],
            color=color,
            linewidth=2.2,
            linestyle=linestyle,
        )
    ax.scatter(pose[:, 0], pose[:, 1], color=color, s=18, label=label, zorder=3)


def plot_pose_comparison_2d(ax, target: np.ndarray, predicted: np.ndarray, title: str, metric_text: str) -> None:
    target_display, predicted_display = canonical_body_view(target, predicted)
    plot_skeleton_2d(ax, target_display, "#111111", "3DPW GT")
    plot_skeleton_2d(ax, predicted_display, "#d62728", "Prediction", "--")
    equalize_2d_axes(ax, [target_display, predicted_display])
    ax.set_title(f"{title}\n{metric_text}", fontsize=10)
    ax.set_xlabel("Left-right")
    ax.set_ylabel("Up")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(loc="upper left", fontsize=8)


def main() -> None:
    opts = parse_args()
    sam_worst_key, sam_worst_pa = worst_sample(opts.sam_metrics)
    apmb_worst_key, apmb_worst_pa = worst_sample(opts.apmb_metrics)
    case_keys = [apmb_worst_key, sam_worst_key]

    annotation_root = opts.data_root / "sequenceFiles"
    image_root = opts.data_root / "imageFiles"
    frames_by_key = {
        f"{frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}": frame
        for frame in iter_frames(annotation_root, image_root, opts.split, opts.frame_stride)
        if frame.camera_pose_valid
    }
    poses_2d = load_alphapose_2d(opts.alphapose_2d)
    apmb_predictions = predict_motionbert_for_needed(opts, frames_by_key, poses_2d, set(case_keys))

    fig = plt.figure(figsize=(18, 10))
    case_titles = [
        f"AlphaPose-MotionBERT worst PA-MPJPE\n{apmb_worst_key}",
        f"SAM-3DB worst PA-MPJPE\n{sam_worst_key}",
    ]

    for row, (case_key, case_title) in enumerate(zip(case_keys, case_titles, strict=True)):
        frame = frames_by_key[case_key]
        target = smpl24_to_common_body(frame.joints_smpl24_camera_m)
        sam_pred, bbox = load_sam_prediction(case_key, opts.sam_predictions)
        apmb_pred = apmb_predictions[case_key]

        image = cv2.cvtColor(cv2.imread(str(frame.image_path)), cv2.COLOR_BGR2RGB)
        ax_img = fig.add_subplot(2, 3, row * 3 + 1)
        ax_img.imshow(image)
        x1, y1, x2, y2 = bbox
        ax_img.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="#ffcc00",
                linewidth=2.5,
            )
        )
        ax_img.text(
            x1,
            max(0, y1 - 8),
            "evaluated person",
            color="#ffcc00",
            fontsize=9,
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
        )
        ax_img.set_title(case_title, fontsize=10)
        ax_img.axis("off")

        sam_root = pelvis_aligned_mpjpe(sam_pred, target) * 1000.0
        sam_pa = procrustes_aligned_mpjpe(sam_pred, target) * 1000.0
        if opts.projection == "3d":
            ax_sam = fig.add_subplot(2, 3, row * 3 + 2, projection="3d")
            plot_pose_comparison(
                ax_sam,
                target,
                sam_pred,
                "SAM-3DB vs GT",
                f"Root {sam_root:.1f} mm | PA {sam_pa:.1f} mm",
            )
        else:
            ax_sam = fig.add_subplot(2, 3, row * 3 + 2)
            plot_pose_comparison_2d(
                ax_sam,
                target,
                sam_pred,
                "SAM-3DB vs GT",
                f"Root {sam_root:.1f} mm | PA {sam_pa:.1f} mm",
            )

        apmb_root = pelvis_aligned_mpjpe(apmb_pred, target) * 1000.0
        apmb_pa = procrustes_aligned_mpjpe(apmb_pred, target) * 1000.0
        if opts.projection == "3d":
            ax_apmb = fig.add_subplot(2, 3, row * 3 + 3, projection="3d")
            plot_pose_comparison(
                ax_apmb,
                target,
                apmb_pred,
                "AlphaPose-COCO17 -> MotionBERT vs GT",
                f"Root {apmb_root:.1f} mm | PA {apmb_pa:.1f} mm",
            )
        else:
            ax_apmb = fig.add_subplot(2, 3, row * 3 + 3)
            plot_pose_comparison_2d(
                ax_apmb,
                target,
                apmb_pred,
                "AlphaPose-COCO17 -> MotionBERT vs GT",
                f"Root {apmb_root:.1f} mm | PA {apmb_pa:.1f} mm",
            )

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.06, wspace=0.22, hspace=0.34)
    fig.savefig(opts.output, dpi=220)
    print(f"output={opts.output}")
    print(f"alphapose_motionbert_worst={apmb_worst_key} pa_mpjpe_mm={apmb_worst_pa:.3f}")
    print(f"sam3db_worst={sam_worst_key} pa_mpjpe_mm={sam_worst_pa:.3f}")


if __name__ == "__main__":
    main()
