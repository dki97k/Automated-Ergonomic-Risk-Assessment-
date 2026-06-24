#!/usr/bin/env python3
"""Run SAM-3DBody on field frames for plausibility analysis."""

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
SAM3DB_ROOT = PROJECT_ROOT / "vendor" / "sam-3d-body"
for path in (SRC_ROOT, SAM3DB_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m1.evaluation.joint_mapping import mhr70_to_common_body  # noqa: E402
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manifests" / "field_plausibility_by_severity_balanced_manifest.csv",
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
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "field_plausibility_sam3db",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "plausibility" / "field_sam3db_summary.csv",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based row offset in the manifest.")
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


def safe_name(sample_id: str) -> str:
    return sample_id.replace("/", "__") + ".npz"


def main() -> None:
    opts = parse_args()
    rows = load_rows(opts.manifest)
    if opts.start_index:
        rows = rows[opts.start_index :]
    if opts.max_samples is not None:
        rows = rows[: opts.max_samples]
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    opts.summary_output.parent.mkdir(parents=True, exist_ok=True)

    print("loading SAM 3D Body on CPU...", flush=True)
    model, model_cfg = load_sam_3d_body(
        str(opts.checkpoint),
        device=torch.device("cpu"),
        mhr_path=str(opts.mhr_path),
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    summary = []
    for idx, row in enumerate(rows, start=1):
        output_path = opts.output_dir / safe_name(row["sample_id"])
        start = time.monotonic()
        status = "failed"
        error = ""
        if output_path.exists():
            status = "ok"
            error = ""
        elif has_bbox(row):
            try:
                bbox = bbox_from_row(row)
                outputs = estimator.process_one_image(
                    row["frame_path"],
                    bboxes=bbox.reshape(1, 4),
                    use_mask=False,
                    inference_type="body",
                )
                if not outputs:
                    raise RuntimeError("SAM 3D Body returned no outputs")
                first = outputs[0]
                pred_mhr70 = first["pred_keypoints_3d"]
                pred_common = mhr70_to_common_body(pred_mhr70)
                np.savez_compressed(
                    output_path,
                    sample_id=np.array([row["sample_id"]]),
                    pred_mhr70_m=pred_mhr70[None],
                    pred_common14_m=pred_common[None],
                    pred_cam_t_m=first["pred_cam_t"][None],
                    bbox=bbox[None],
                )
                status = "ok"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        else:
            status = "failed_no_bbox_available"
            error = "no bbox available"
        seconds = time.monotonic() - start
        summary.append({**row, "pipeline_status": status, "seconds": f"{seconds:.3f}", "prediction_path": str(output_path) if status == "ok" else "", "error": error})
        if opts.log_every > 0 and (idx % opts.log_every == 0 or idx == len(rows)):
            print(f"sam3db_field={idx}/{len(rows)} status={status} seconds={seconds:.1f}", flush=True)

    fieldnames = list(summary[0].keys())
    with opts.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)
    ok = sum(row["pipeline_status"] == "ok" for row in summary)
    print(f"summary={opts.summary_output}", flush=True)
    print(f"predicted={ok}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
