#!/usr/bin/env python3
"""Run SAM 3D Body on a temporally thinned 3DPW subset.

This runner is designed for local Mac CPU execution:

- detector, segmentor, and FOV models are disabled;
- person bboxes are built from 3DPW COCO-18 2D keypoints;
- the SAM model is loaded once and reused;
- each sample is saved as an individual ``.npz`` file so the run is resumable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

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
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--frame-stride", type=int, default=50)
    parser.add_argument("--max-samples", type=int)
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "sam3db_3dpw_test_stride50",
    )
    parser.add_argument("--min-confidence", type=float, default=0.2)
    return parser.parse_args()


def safe_sample_filename(sample_key: str) -> str:
    return sample_key.replace("/", "__") + ".npz"


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


def append_manifest_row(manifest_path: Path, row: dict[str, str | int | float]) -> None:
    fieldnames = [
        "sample_key",
        "status",
        "seconds",
        "image_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "error",
    ]
    write_header = not manifest_path.exists()
    with manifest_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "run_manifest.csv"

    annotation_root = args.data_root / "sequenceFiles"
    image_root = args.data_root / "imageFiles"
    frames = list(iter_frames(annotation_root, image_root, args.split, args.frame_stride))
    if args.max_samples is not None:
        frames = frames[: args.max_samples]

    print(f"split={args.split}", flush=True)
    print(f"frame_stride={args.frame_stride}", flush=True)
    print(f"planned_samples={len(frames)}", flush=True)
    print("loading SAM 3D Body on CPU...", flush=True)
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

    completed = 0
    skipped = 0
    failed = 0
    for idx, frame in enumerate(frames, start=1):
        sample_key = f"{frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}"
        output_path = args.output_dir / safe_sample_filename(sample_key)
        if output_path.exists():
            skipped += 1
            continue

        start = time.monotonic()
        error = ""
        bbox = np.full(4, np.nan, dtype=np.float32)
        try:
            if not frame.camera_pose_valid:
                raise ValueError("invalid camera pose")
            if frame.joints_2d_coco18 is None:
                raise ValueError("missing 2D joints")
            bbox = bbox_from_coco18(frame.joints_2d_coco18, frame.image_path, args.min_confidence)
            outputs = estimator.process_one_image(
                str(frame.image_path),
                bboxes=bbox.reshape(1, 4),
                use_mask=False,
                inference_type="body",
            )
            if not outputs:
                raise RuntimeError("SAM 3D Body returned no outputs")

            first = outputs[0]
            np.savez_compressed(
                output_path,
                sample_key=np.array([sample_key]),
                pred_mhr70_m=first["pred_keypoints_3d"][None],
                pred_cam_t_m=first["pred_cam_t"][None],
                bbox=bbox[None],
            )
            status = "ok"
            completed += 1
        except Exception as exc:  # Keep long CPU runs resumable.
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            failed += 1

        seconds = time.monotonic() - start
        append_manifest_row(
            manifest_path,
            {
                "sample_key": sample_key,
                "status": status,
                "seconds": f"{seconds:.3f}",
                "image_path": str(frame.image_path),
                "bbox_x1": f"{bbox[0]:.3f}",
                "bbox_y1": f"{bbox[1]:.3f}",
                "bbox_x2": f"{bbox[2]:.3f}",
                "bbox_y2": f"{bbox[3]:.3f}",
                "error": error,
            },
        )
        print(
            f"[{idx}/{len(frames)}] {status} {sample_key} "
            f"{seconds:.1f}s completed={completed} failed={failed} skipped={skipped}",
            flush=True,
        )

    print(f"output_dir={args.output_dir}", flush=True)
    print(f"completed={completed}", flush=True)
    print(f"failed={failed}", flush=True)
    print(f"skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
