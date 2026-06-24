#!/usr/bin/env python3
"""Run a MotionBERT 2D-to-3D baseline on the 3DPW stride subset.

This uses 3DPW's provided COCO/OpenPose-style 2D detections, not freshly run
AlphaPose detections. The result is therefore named MotionBERT-on-3DPW-2D and
should be described as a standard 2D-to-3D baseline rather than a reliable
reference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MOTIONBERT_ROOT = PROJECT_ROOT / "vendor" / "MotionBERT"
for path in (SRC_ROOT, MOTIONBERT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lib.utils.tools import get_config  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.utils_data import crop_scale, flip_data  # noqa: E402
from m1.data.three_dpw import iter_frames, load_sequence, sequence_files  # noqa: E402
from m1.evaluation.joint_mapping import smpl24_to_common_body  # noqa: E402
from m1.evaluation.metrics import (  # noqa: E402
    pelvis_aligned_mpjpe,
    procrustes_aligned_mpjpe,
    scale_aligned_mpjpe,
)


MOTIONBERT_TO_COMMON = [0, 8, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]


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
        "--config",
        type=Path,
        default=MOTIONBERT_ROOT / "configs" / "pose3d" / "MB_ft_h36m_global_lite.yaml",
    )
    parser.add_argument(
        "--checkpoint",
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
        default=PROJECT_ROOT / "results" / "metrics" / "e1_motionbert_3dpw_stride50.csv",
    )
    return parser.parse_args()


def coco18_to_motionbert_h36m17(pose_2d: np.ndarray) -> np.ndarray:
    """Map 3DPW COCO/OpenPose-18 2D detections to MotionBERT H36M-17 order."""

    pose = np.asarray(pose_2d, dtype=np.float64)
    if pose.shape == (3, 18):
        pose = pose.T
    if pose.shape != (18, 3):
        raise ValueError(f"expected 2D pose shape (3,18) or (18,3), got {pose.shape}")

    out = np.zeros((17, 3), dtype=np.float64)
    out[0] = (pose[8] + pose[11]) / 2.0  # pelvis
    out[1] = pose[8]  # right hip
    out[2] = pose[9]
    out[3] = pose[10]
    out[4] = pose[11]  # left hip
    out[5] = pose[12]
    out[6] = pose[13]
    out[8] = pose[1]  # neck
    out[7] = (out[0] + out[8]) / 2.0  # spine
    out[9] = pose[0]  # nose
    out[10] = pose[0]  # head proxy
    out[11] = pose[5]  # left shoulder
    out[12] = pose[6]
    out[13] = pose[7]
    out[14] = pose[2]  # right shoulder
    out[15] = pose[3]
    out[16] = pose[4]
    return out


def predict_motion(model: torch.nn.Module, args, motion_2d: np.ndarray, clip_len: int) -> np.ndarray:
    """Run MotionBERT over one full person track."""

    motion = crop_scale(motion_2d, [1, 1]).astype(np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(motion), clip_len):
            clip = motion[start : start + clip_len]
            batch_input = torch.from_numpy(clip[None])
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:
                pred_1 = model(batch_input)
                pred_flip = model(flip_data(batch_input))
                pred = (pred_1 + flip_data(pred_flip)) / 2.0
            else:
                pred = model(batch_input)
            if args.rootrel:
                pred[:, :, 0, :] = 0
            else:
                pred[:, 0, 0, 2] = 0
            if args.gt_2d:
                pred[..., :2] = batch_input[..., :2]
            preds.append(pred.cpu().numpy()[0])
    return np.concatenate(preds, axis=0)


def main() -> None:
    opts = parse_args()
    annotation_root = opts.data_root / "sequenceFiles"
    image_root = opts.data_root / "imageFiles"
    cfg = get_config(str(opts.config))

    print("loading MotionBERT...", flush=True)
    model = load_backbone(cfg)
    checkpoint = torch.load(str(opts.checkpoint), map_location="cpu")
    state_dict = checkpoint["model_pos"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    target_by_key = {
        f"{frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}": smpl24_to_common_body(
            frame.joints_smpl24_camera_m
        )
        for frame in iter_frames(annotation_root, image_root, opts.split, opts.frame_stride)
        if frame.camera_pose_valid
    }

    rows = []
    sequence_count = 0
    for annotation_file in sequence_files(annotation_root, opts.split):
        sequence = load_sequence(annotation_file)
        sequence_name = str(sequence["sequence"])
        poses2d = sequence.get("poses2d", [])
        for person_id, person_poses2d in enumerate(poses2d):
            motion_2d = np.stack([coco18_to_motionbert_h36m17(frame) for frame in person_poses2d])
            pred_h36m = predict_motion(model, cfg, motion_2d, clip_len=int(cfg.clip_len))
            sequence_count += 1

            for frame_index in range(0, len(pred_h36m), opts.frame_stride):
                sample_key = f"{opts.split}/{sequence_name}/p{person_id:02d}/f{frame_index:05d}"
                if sample_key not in target_by_key:
                    continue
                pred_common = pred_h36m[frame_index, MOTIONBERT_TO_COMMON, :]
                target_common = target_by_key[sample_key]
                rows.append(
                    {
                        "sample_key": sample_key,
                        "root_aligned_mpjpe_mm": f"{pelvis_aligned_mpjpe(pred_common, target_common) * 1000.0:.6f}",
                        "scale_aligned_mpjpe_mm": f"{scale_aligned_mpjpe(pred_common, target_common) * 1000.0:.6f}",
                        "pa_mpjpe_mm": f"{procrustes_aligned_mpjpe(pred_common, target_common) * 1000.0:.6f}",
                    }
                )
        print(f"processed_sequence={sequence_name} cumulative_rows={len(rows)}", flush=True)

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    with opts.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "root_aligned_mpjpe_mm",
                "scale_aligned_mpjpe_mm",
                "pa_mpjpe_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"output={opts.output}", flush=True)
    print(f"evaluated_samples={len(rows)}", flush=True)
    print(f"processed_person_tracks={sequence_count}", flush=True)
    for key in ["root_aligned_mpjpe_mm", "scale_aligned_mpjpe_mm", "pa_mpjpe_mm"]:
        values = np.array([float(row[key]) for row in rows])
        print(f"mean_{key}={values.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
