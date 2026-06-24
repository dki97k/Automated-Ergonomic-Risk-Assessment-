#!/usr/bin/env python3
"""Evaluate 3DPW correctness metrics for SAM 3D Body predictions.

Prediction format:
    An ``.npz`` file with:
    - ``sample_key``: array of strings matching the manifest keys
    - ``pred_mhr70_m``: array shaped ``(N, 70, 3)`` in meters

By default ``pred_mhr70_m`` is interpreted as SAM 3D Body's body/model-frame
``pred_keypoints_3d`` and ``pred_cam_t_m`` is added to convert it to the camera
frame. If predictions are already camera-frame keypoints, pass
``--prediction-coordinates camera``.

Use ``--oracle-sanity`` only to verify the metric pipeline. It copies the 3DPW
ground truth into the prediction slot and should produce near-zero errors; it
is not a model result.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m1.data.three_dpw import iter_frames  # noqa: E402
from m1.evaluation.joint_mapping import mhr70_to_common_body, smpl24_to_common_body  # noqa: E402
from m1.evaluation.metrics import mpjpe, pelvis_aligned_mpjpe, procrustes_aligned_mpjpe  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("<private_workspace>/data/3dpw/extracted"),
    )
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--prediction-coordinates",
        choices=("sam_model", "camera"),
        default="sam_model",
        help="Coordinate frame of pred_mhr70_m. sam_model adds pred_cam_t_m.",
    )
    parser.add_argument("--oracle-sanity", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def _prediction_arrays_from_npz(path: Path, coordinates: str) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keys = [str(key) for key in np.atleast_1d(data["sample_key"])]
    preds = np.asarray(data["pred_mhr70_m"], dtype=np.float64)
    if preds.ndim == 2:
        preds = preds[None]
    if preds.shape[0] != len(keys):
        raise ValueError(f"prediction count does not match sample_key count in {path}")
    if coordinates == "sam_model":
        if "pred_cam_t_m" not in data.files:
            raise ValueError(f"sam_model predictions require pred_cam_t_m in {path}")
        cam_t = np.asarray(data["pred_cam_t_m"], dtype=np.float64)
        if cam_t.ndim == 1:
            cam_t = cam_t[None]
        if cam_t.shape != (len(keys), 3):
            raise ValueError(f"pred_cam_t_m must have shape ({len(keys)}, 3), got {cam_t.shape}")
        preds = preds + cam_t[:, None, :]
    return keys, preds


def load_predictions(path: Path | None, coordinates: str) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    files = sorted(
        file_path
        for file_path in path.glob("*.npz")
        if not file_path.name.startswith("._")
    ) if path.is_dir() else [path]
    predictions: dict[str, np.ndarray] = {}
    for file_path in files:
        keys, preds = _prediction_arrays_from_npz(file_path, coordinates)
        predictions.update(dict(zip(keys, preds)))
    return predictions


def main() -> None:
    args = parse_args()
    if not args.oracle_sanity and args.predictions is None:
        raise SystemExit("Provide --predictions or use --oracle-sanity for a pipeline sanity check.")
    if args.output is None:
        if args.oracle_sanity:
            args.output = PROJECT_ROOT / "outputs" / "sanity" / "e1_3dpw_correctness_oracle.csv"
        else:
            args.output = PROJECT_ROOT / "results" / "metrics" / "e1_3dpw_correctness.csv"

    annotation_root = args.data_root / "sequenceFiles"
    image_root = args.data_root / "imageFiles"
    predictions = load_predictions(args.predictions, args.prediction_coordinates)

    rows = []
    errors = []
    pa_errors = []
    root_errors = []
    missing = 0

    for frame in iter_frames(annotation_root, image_root, args.split, args.frame_stride):
        sample_key = f"{frame.split}/{frame.sequence}/p{frame.person_id:02d}/f{frame.frame_index:05d}"
        target_common = smpl24_to_common_body(frame.joints_smpl24_camera_m)

        if args.oracle_sanity:
            pred_common = target_common.copy()
            prediction_source = "oracle_sanity"
        elif sample_key in predictions:
            pred_common = mhr70_to_common_body(predictions[sample_key])
            prediction_source = str(args.predictions)
        else:
            missing += 1
            continue

        sample_mpjpe = mpjpe(pred_common, target_common) * 1000.0
        sample_root_mpjpe = pelvis_aligned_mpjpe(pred_common, target_common) * 1000.0
        sample_pa_mpjpe = procrustes_aligned_mpjpe(pred_common, target_common) * 1000.0
        errors.append(sample_mpjpe)
        root_errors.append(sample_root_mpjpe)
        pa_errors.append(sample_pa_mpjpe)
        rows.append(
            {
                "sample_key": sample_key,
                "prediction_source": prediction_source,
                "mpjpe_mm": f"{sample_mpjpe:.6f}",
                "root_aligned_mpjpe_mm": f"{sample_root_mpjpe:.6f}",
                "pa_mpjpe_mm": f"{sample_pa_mpjpe:.6f}",
            }
        )

    if not rows:
        raise SystemExit("No evaluable samples were found.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_key",
                "prediction_source",
                "mpjpe_mm",
                "root_aligned_mpjpe_mm",
                "pa_mpjpe_mm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"output={args.output}")
    print(f"evaluated_samples={len(rows)}")
    print(f"missing_predictions={missing}")
    print(f"mean_mpjpe_mm={np.mean(errors):.3f}")
    print(f"mean_root_aligned_mpjpe_mm={np.mean(root_errors):.3f}")
    print(f"mean_pa_mpjpe_mm={np.mean(pa_errors):.3f}")
    if args.oracle_sanity:
        print("note=oracle_sanity_only_not_a_model_result")


if __name__ == "__main__":
    main()
