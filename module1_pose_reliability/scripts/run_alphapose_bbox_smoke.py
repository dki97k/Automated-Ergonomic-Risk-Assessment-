#!/usr/bin/env python3
"""Smoke-test AlphaPose pose inference on one 3DPW person box.

This intentionally uses only the 3DPW 2D joints to derive a person bounding
box. The keypoints themselves are predicted from the image by AlphaPose.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
ALPHAPOSE_ROOT = PROJECT_ROOT / "vendor" / "AlphaPose"
for path in (SRC_ROOT, ALPHAPOSE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alphapose.models import builder  # noqa: E402
from alphapose.utils.config import update_config  # noqa: E402
from alphapose.utils.presets import SimpleTransform  # noqa: E402
from alphapose.utils.transforms import get_func_heatmap_to_coord  # noqa: E402
from m1.data.three_dpw import iter_frames  # noqa: E402


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
        default=ALPHAPOSE_ROOT / "configs" / "coco" / "resnet" / "256x192_res50_lr1e-3_1x.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ALPHAPOSE_ROOT / "pretrained_models" / "fast_res50_256x192.pth",
    )
    return parser.parse_args()


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


def main() -> None:
    opts = parse_args()
    annotation_root = opts.data_root / "sequenceFiles"
    image_root = opts.data_root / "imageFiles"

    frame = next(
        item
        for item in iter_frames(annotation_root, image_root, opts.split, opts.frame_stride)
        if item.camera_pose_valid and item.joints_2d_coco18 is not None and item.image_path.exists()
    )
    image_bgr = cv2.imread(str(frame.image_path))
    if image_bgr is None:
        raise FileNotFoundError(frame.image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    box = bbox_from_pose2d(frame.joints_2d_coco18, image_rgb.shape)

    cfg = update_config(str(opts.config))
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
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    state = torch.load(str(opts.checkpoint), map_location=device)
    pose_model.load_state_dict(state)
    pose_model.to(device)
    pose_model.eval()

    inp, cropped_box = transform.test_transform(image_rgb, torch.from_numpy(box))
    with torch.no_grad():
        heatmap = pose_model(inp.unsqueeze(0).to(device)).cpu()[0]

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    coords, scores = heatmap_to_coord(
        heatmap,
        cropped_box,
        hm_shape=cfg.DATA_PRESET.HEATMAP_SIZE,
        norm_type=cfg.LOSS.get("NORM_TYPE", None),
    )

    print(f"sample={frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}")
    print(f"image={frame.image_path}")
    print(f"bbox_xyxy={box.round(2).tolist()}")
    print(f"pred_shape={coords.shape}")
    print(f"mean_keypoint_score={float(np.mean(scores)):.4f}")
    print(f"first_keypoint_xy_score={[float(coords[0, 0]), float(coords[0, 1]), float(scores[0, 0])]}")


if __name__ == "__main__":
    main()
