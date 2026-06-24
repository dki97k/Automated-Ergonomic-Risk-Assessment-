#!/usr/bin/env python3
"""Inspect the local 3DPW extraction and print split-level counts."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m1.data.three_dpw import iter_frames, sequence_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("<private_workspace>/data/3dpw/extracted"),
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--frame-stride", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_root = args.data_root / "sequenceFiles"
    image_root = args.data_root / "imageFiles"

    files = sequence_files(annotation_root, args.split)
    counter: Counter[str] = Counter()
    missing_images = 0
    invalid_camera = 0
    first = None

    for frame in iter_frames(annotation_root, image_root, args.split, args.frame_stride):
        first = first or frame
        counter["person_frames"] += 1
        counter[f"sequence:{frame.sequence}"] += 1
        if not frame.image_path.exists():
            missing_images += 1
        if not frame.camera_pose_valid:
            invalid_camera += 1

    print(f"split={args.split}")
    print(f"annotation_files={len(files)}")
    print(f"sampled_person_frames={counter['person_frames']}")
    print(f"missing_images={missing_images}")
    print(f"invalid_camera_poses={invalid_camera}")
    if first:
        print(f"first_sequence={first.sequence}")
        print(f"first_image={first.image_path}")
        print(f"joints_shape={first.joints_smpl24_world_m.shape}")
        print(f"camera_intrinsics_shape={first.camera_intrinsics.shape}")


if __name__ == "__main__":
    main()
