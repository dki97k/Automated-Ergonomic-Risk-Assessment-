#!/usr/bin/env python3
"""Run an AlphaPose-2D to MotionBERT-3D baseline on 3DPW.

The baseline controls person detection by using the same 3DPW-derived person
boxes used for crop-based evaluation. It does not use 3DPW-provided 2D
keypoints as MotionBERT input; AlphaPose predicts the 2D keypoints from pixels.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
ALPHAPOSE_ROOT = PROJECT_ROOT / "vendor" / "AlphaPose"
MOTIONBERT_ROOT = PROJECT_ROOT / "vendor" / "MotionBERT"
for path in (SRC_ROOT, ALPHAPOSE_ROOT, MOTIONBERT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alphapose.models import builder  # noqa: E402
from alphapose.utils.config import update_config  # noqa: E402
from alphapose.utils.presets import SimpleTransform  # noqa: E402
from alphapose.utils.transforms import get_func_heatmap_to_coord  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.utils_data import crop_scale, flip_data  # noqa: E402
from m1.data.three_dpw import ThreeDPWFrame, iter_frames  # noqa: E402
from m1.evaluation.joint_mapping import smpl24_to_common_body  # noqa: E402
from m1.evaluation.metrics import (  # noqa: E402
    pelvis_aligned_mpjpe,
    procrustes_aligned_mpjpe,
    scale_aligned_mpjpe,
)


MOTIONBERT_TO_COMMON = [0, 8, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]


@dataclass(frozen=True)
class AlphaPoseSample:
    key: str
    frame: ThreeDPWFrame
    bbox_xyxy: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("<private_workspace>/data/3dpw/extracted"),
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--alphapose-config",
        type=Path,
        default=ALPHAPOSE_ROOT / "configs" / "coco" / "resnet" / "256x192_res50_lr1e-3_1x.yaml",
    )
    parser.add_argument(
        "--alphapose-checkpoint",
        type=Path,
        default=ALPHAPOSE_ROOT / "pretrained_models" / "fast_res50_256x192.pth",
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
        default=PROJECT_ROOT / "results" / "metrics" / "e1_alphapose_motionbert_3dpw_stride50.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "alphapose_coco17_3dpw_stride50.npz",
    )
    return parser.parse_args()


def sample_key(frame: ThreeDPWFrame) -> str:
    return f"{frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}"


def bbox_from_pose2d(pose2d: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    pose = np.asarray(pose2d, dtype=np.float64)
    if pose.shape == (3, 18):
        pose = pose.T
    visible = pose[:, 2] > 0
    if visible.sum() < 4:
        raise ValueError("not enough visible 2D joints to derive a person box")
    xy = pose[visible, :2]
    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)
    width = x2 - x1
    height = y2 - y1
    pad = 0.15 * max(width, height)
    h, w = image_shape[:2]
    return np.array(
        [
            max(0.0, x1 - pad),
            max(0.0, y1 - pad),
            min(float(w - 1), x2 + pad),
            min(float(h - 1), y2 + pad),
        ],
        dtype=np.float32,
    )


def coco17_to_motionbert_h36m17(pose: np.ndarray) -> np.ndarray:
    """Map AlphaPose COCO-17 2D joints to MotionBERT H36M-17 order."""

    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (17, 3):
        raise ValueError(f"expected COCO-17 pose with shape (17, 3), got {pose.shape}")

    out = np.zeros((17, 3), dtype=np.float64)
    out[0] = (pose[11] + pose[12]) / 2.0
    out[1] = pose[12]
    out[2] = pose[14]
    out[3] = pose[16]
    out[4] = pose[11]
    out[5] = pose[13]
    out[6] = pose[15]
    out[8] = (pose[5] + pose[6]) / 2.0
    out[7] = (out[0] + out[8]) / 2.0
    out[9] = pose[0]
    out[10] = pose[0]
    out[11] = pose[5]
    out[12] = pose[7]
    out[13] = pose[9]
    out[14] = pose[6]
    out[15] = pose[8]
    out[16] = pose[10]
    return out


def load_alphapose(opts: argparse.Namespace):
    cfg = update_config(str(opts.alphapose_config))
    device = torch.device("cpu")
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    transform = SimpleTransform(
        pose_dataset,
        scale_factor=0,
        input_size=cfg.DATA_PRESET.IMAGE_SIZE,
        output_size=cfg.DATA_PRESET.HEATMAP_SIZE,
        rot=0,
        sigma=cfg.DATA_PRESET.SIGMA,
        train=False,
        add_dpg=False,
        gpu_device=device,
    )
    model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    state = torch.load(str(opts.alphapose_checkpoint), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return cfg, transform, model


def run_alphapose(opts: argparse.Namespace) -> tuple[dict[str, np.ndarray], dict[str, ThreeDPWFrame]]:
    cfg, transform, model = load_alphapose(opts)
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    annotation_root = opts.data_root / "sequenceFiles"
    image_root = opts.data_root / "imageFiles"

    samples = []
    frames_by_key = {}
    for frame in iter_frames(annotation_root, image_root, opts.split, opts.frame_stride):
        if not frame.camera_pose_valid or frame.joints_2d_coco18 is None or not frame.image_path.exists():
            continue
        image_bgr = cv2.imread(str(frame.image_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            bbox = bbox_from_pose2d(frame.joints_2d_coco18, image_rgb.shape)
        except ValueError:
            continue
        key = sample_key(frame)
        samples.append(AlphaPoseSample(key=key, frame=frame, bbox_xyxy=bbox))
        frames_by_key[key] = frame
        if opts.max_samples is not None and len(samples) >= opts.max_samples:
            break

    predictions = {}
    batch_inputs = []
    batch_meta = []
    with torch.no_grad():
        for index, sample in enumerate(samples, start=1):
            image_bgr = cv2.imread(str(sample.frame.image_path))
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            inp, cropped_box = transform.test_transform(image_rgb, torch.from_numpy(sample.bbox_xyxy))
            batch_inputs.append(inp)
            batch_meta.append((sample, cropped_box))

            if len(batch_inputs) == opts.batch_size or index == len(samples):
                heatmaps = model(torch.stack(batch_inputs, dim=0)).cpu()
                for heatmap, (meta_sample, meta_cropped_box) in zip(heatmaps, batch_meta, strict=True):
                    coords, scores = heatmap_to_coord(
                        heatmap,
                        meta_cropped_box,
                        hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE,
                        norm_type=cfg.LOSS.get("NORM_TYPE", None),
                    )
                    predictions[meta_sample.key] = coco17_to_motionbert_h36m17(
                        np.concatenate([coords, scores], axis=1)
                    )
                batch_inputs.clear()
                batch_meta.clear()
                print(f"alphapose_processed={index}/{len(samples)}", flush=True)

    opts.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        opts.predictions_output,
        keys=np.array(list(predictions.keys()), dtype=object),
        poses=np.stack([predictions[key] for key in predictions.keys()]),
    )
    print(f"alphapose_predictions={opts.predictions_output}", flush=True)
    return predictions, frames_by_key


def predict_motion(model: torch.nn.Module, cfg, motion_2d: np.ndarray, clip_len: int) -> np.ndarray:
    motion = crop_scale(motion_2d, [1, 1]).astype(np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(motion), clip_len):
            clip = motion[start : start + clip_len]
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
            preds.append(pred.cpu().numpy()[0])
    return np.concatenate(preds, axis=0)


def run_motionbert(opts: argparse.Namespace, poses_2d: dict[str, np.ndarray], frames_by_key: dict[str, ThreeDPWFrame]) -> list[dict[str, str]]:
    cfg = get_config(str(opts.motionbert_config))
    model = load_backbone(cfg)
    checkpoint = torch.load(str(opts.motionbert_checkpoint), map_location="cpu")
    state_dict = checkpoint["model_pos"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    grouped = {}
    for key, pose in poses_2d.items():
        frame = frames_by_key[key]
        group_key = (frame.split, frame.sequence, frame.person_id)
        grouped.setdefault(group_key, []).append((frame.frame_index, key, pose))

    rows = []
    for group_key, items in sorted(grouped.items()):
        items.sort(key=lambda item: item[0])
        motion_2d = np.stack([item[2] for item in items])
        pred_h36m = predict_motion(model, cfg, motion_2d, clip_len=int(cfg.clip_len))
        for pred, (_, key, _) in zip(pred_h36m, items, strict=True):
            frame = frames_by_key[key]
            pred_common = pred[MOTIONBERT_TO_COMMON, :]
            target_common = smpl24_to_common_body(frame.joints_smpl24_camera_m)
            rows.append(
                {
                    "sample_key": key,
                    "root_aligned_mpjpe_mm": f"{pelvis_aligned_mpjpe(pred_common, target_common) * 1000.0:.6f}",
                    "scale_aligned_mpjpe_mm": f"{scale_aligned_mpjpe(pred_common, target_common) * 1000.0:.6f}",
                    "pa_mpjpe_mm": f"{procrustes_aligned_mpjpe(pred_common, target_common) * 1000.0:.6f}",
                }
            )
        print(f"motionbert_processed={group_key} cumulative_rows={len(rows)}", flush=True)
    return rows


def main() -> None:
    opts = parse_args()
    poses_2d, frames_by_key = run_alphapose(opts)
    rows = run_motionbert(opts, poses_2d, frames_by_key)

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
    for key in ["root_aligned_mpjpe_mm", "scale_aligned_mpjpe_mm", "pa_mpjpe_mm"]:
        values = np.array([float(row[key]) for row in rows])
        print(f"mean_{key}={values.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
