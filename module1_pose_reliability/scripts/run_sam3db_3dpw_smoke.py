#!/usr/bin/env python3
"""Run one SAM 3D Body inference on a 3DPW frame without detector/segmentor.

This is a local Mac smoke test, not the final correctness experiment. It uses
3DPW 2D keypoints to build a person bbox and runs body-only inference on CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor" / "sam-3d-body"
for path in (SRC_ROOT, VENDOR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m1.data.three_dpw import iter_frames  # noqa: E402
from m1.occlusion.severity import coco18_xy_conf  # noqa: E402
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("<private_workspace>/data/3dpw/extracted"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "sam-3d-body-dinov3" / "model.ckpt",
    )
    parser.add_argument(
        "--mhr-path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "sam-3d-body-dinov3" / "assets" / "mhr_model.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "smoke" / "sam3db_3dpw_one_frame.npz",
    )
    parser.add_argument("--min-confidence", type=float, default=0.2)
    return parser.parse_args()


def bbox_from_coco18(pose_2d: np.ndarray, image_path: Path, min_confidence: float) -> np.ndarray:
    xy, confidence = coco18_xy_conf(pose_2d)
    valid = confidence > min_confidence
    if valid.sum() < 6:
        raise ValueError("not enough confident 2D joints to build a bbox")

    points = xy[valid]
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    width = x2 - x1
    height = y2 - y1
    pad = 0.20 * max(width, height)

    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)
    img_h, img_w = image.shape[:2]

    return np.array(
        [
            max(0.0, x1 - pad),
            max(0.0, y1 - pad),
            min(float(img_w - 1), x2 + pad),
            min(float(img_h - 1), y2 + pad),
        ],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()
    annotation_root = args.data_root / "sequenceFiles"
    image_root = args.data_root / "imageFiles"

    frame = next(
        item
        for item in iter_frames(annotation_root, image_root, "test", frame_stride=1)
        if item.camera_pose_valid and item.joints_2d_coco18 is not None
    )
    bbox = bbox_from_coco18(frame.joints_2d_coco18, frame.image_path, args.min_confidence)

    print(f"sample={frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}")
    print(f"image={frame.image_path}")
    print(f"bbox={bbox.tolist()}")
    print("loading SAM 3D Body on CPU...")
    model, model_cfg = load_sam_3d_body(
        str(args.checkpoint),
        device=torch.device("cpu"),
        mhr_path=str(args.mhr_path),
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    print("running body-only inference...")
    outputs = estimator.process_one_image(
        str(frame.image_path),
        bboxes=bbox.reshape(1, 4),
        use_mask=False,
        inference_type="body",
    )
    if not outputs:
        raise RuntimeError("SAM 3D Body returned no outputs")

    first = outputs[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_key=np.array([f"{frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}"]),
        pred_mhr70_m=first["pred_keypoints_3d"][None],
        pred_cam_t_m=first["pred_cam_t"][None],
        bbox=bbox[None],
    )
    print(f"saved={args.output}")
    print(f"pred_keypoints_3d_shape={first['pred_keypoints_3d'].shape}")
    print(f"pred_cam_t={first['pred_cam_t'].tolist()}")


if __name__ == "__main__":
    main()
