#!/usr/bin/env python3
"""Run AlphaPose -> MotionBERT on field frames for plausibility analysis."""

from __future__ import annotations

import argparse
import csv
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


MOTIONBERT_TO_COMMON = [0, 8, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manifests" / "field_plausibility_by_severity_balanced_manifest.csv",
    )
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
        default=PROJECT_ROOT / "outputs" / "predictions" / "field_plausibility_alphapose_motionbert.npz",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "plausibility" / "field_alphapose_motionbert_summary.csv",
    )
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def bbox_from_row(row: dict[str, str]) -> np.ndarray:
    return np.array(
        [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])],
        dtype=np.float32,
    )


def has_bbox(row: dict[str, str]) -> bool:
    try:
        bbox_from_row(row)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def coco17_to_motionbert_h36m17(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
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


def run_alphapose(opts: argparse.Namespace, rows: list[dict[str, str]]) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    cfg, transform, model = load_alphapose(opts)
    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    poses_2d: dict[str, np.ndarray] = {}
    summary = []
    with torch.no_grad():
        for idx, row in enumerate(rows, start=1):
            if not has_bbox(row):
                summary.append({**row, "pipeline_status": "failed_no_bbox_available", "error": "no bbox available"})
                continue
            try:
                image_bgr = cv2.imread(row["frame_path"])
                if image_bgr is None:
                    raise FileNotFoundError(row["frame_path"])
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                inp, cropped_box = transform.test_transform(image_rgb, torch.from_numpy(bbox_from_row(row)))
                heatmap = model(inp.unsqueeze(0)).cpu()[0]
                coords, scores = heatmap_to_coord(
                    heatmap,
                    cropped_box,
                    hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE,
                    norm_type=cfg.LOSS.get("NORM_TYPE", None),
                )
                poses_2d[row["sample_id"]] = coco17_to_motionbert_h36m17(np.concatenate([coords, scores], axis=1))
                summary.append({**row, "pipeline_status": "alphapose_ok", "error": ""})
            except Exception as exc:
                summary.append({**row, "pipeline_status": "alphapose_failed", "error": f"{type(exc).__name__}: {exc}"})
            if opts.log_every > 0 and (idx % opts.log_every == 0 or idx == len(rows)):
                print(f"alphapose_field={idx}/{len(rows)} status={summary[-1]['pipeline_status']}", flush=True)
    return poses_2d, summary


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


def run_motionbert(opts: argparse.Namespace, rows: list[dict[str, str]], poses_2d: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[str]]:
    cfg = get_config(str(opts.motionbert_config))
    model = load_backbone(cfg)
    checkpoint = torch.load(str(opts.motionbert_checkpoint), map_location="cpu")
    state_dict = checkpoint["model_pos"]
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    grouped = {}
    row_by_id = {row["sample_id"]: row for row in rows}
    for sample_id, pose in poses_2d.items():
        row = row_by_id[sample_id]
        grouped.setdefault(row["sequence"], []).append((int(row["frame_number"]), sample_id, pose))

    pred_common = {}
    for sequence, items in sorted(grouped.items()):
        items.sort(key=lambda item: item[0])
        motion_2d = np.stack([item[2] for item in items])
        pred_h36m = predict_motion(model, cfg, motion_2d, clip_len=int(cfg.clip_len))
        for pred, (_, sample_id, _) in zip(pred_h36m, items, strict=True):
            pred_common[sample_id] = pred[MOTIONBERT_TO_COMMON, :]
        print(f"motionbert_field_sequence={sequence} rows={len(items)}", flush=True)
    return pred_common, list(pred_common.keys())


def main() -> None:
    opts = parse_args()
    rows = load_rows(opts.manifest)
    poses_2d, summary = run_alphapose(opts, rows)
    pred_common, keys = run_motionbert(opts, rows, poses_2d)

    opts.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        opts.output,
        sample_ids=np.array(keys, dtype=object),
        pred_common14=np.stack([pred_common[key] for key in keys]) if keys else np.empty((0, 14, 3)),
    )

    opts.summary_output.parent.mkdir(parents=True, exist_ok=True)
    pred_ids = set(keys)
    for row in summary:
        if row["sample_id"] in pred_ids:
            row["pipeline_status"] = "ok"
    fieldnames = list(summary[0].keys())
    with opts.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    print(f"predictions={opts.output}", flush=True)
    print(f"summary={opts.summary_output}", flush=True)
    print(f"predicted={len(keys)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
