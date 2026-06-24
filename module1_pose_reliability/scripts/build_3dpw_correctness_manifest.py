#!/usr/bin/env python3
"""Build a CSV manifest for the 3DPW correctness experiment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m1.data.three_dpw import iter_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("<private_workspace>/data/3dpw/extracted"),
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manifests" / "3dpw_test_correctness_stride10.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_root = args.data_root / "sequenceFiles"
    image_root = args.data_root / "imageFiles"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for frame in iter_frames(annotation_root, image_root, args.split, args.frame_stride):
        sample_key = f"{frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}"
        rows.append(
            {
                "sample_key": sample_key,
                "split": frame.split,
                "sequence": frame.sequence,
                "person_id": frame.person_id,
                "frame_index": frame.frame_index,
                "source_video_frame_id": frame.source_video_frame_id,
                "image_path": str(frame.image_path),
                "camera_pose_valid": int(frame.camera_pose_valid),
            }
        )

    fieldnames = [
        "sample_key",
        "split",
        "sequence",
        "person_id",
        "frame_index",
        "source_video_frame_id",
        "image_path",
        "camera_pose_valid",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest={args.output}")
    print(f"samples={len(rows)}")
    print(f"split={args.split}")
    print(f"frame_stride={args.frame_stride}")


if __name__ == "__main__":
    main()
