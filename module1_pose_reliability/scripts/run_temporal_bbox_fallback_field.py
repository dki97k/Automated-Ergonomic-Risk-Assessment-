#!/usr/bin/env python3
"""Apply temporal bbox fallback to field frames where YOLO failed.

For each YOLO no-person frame, the script borrows the nearest valid person
bounding box from the same sequence and runs SAM-3DB body inference. This
implements the Module 1 fallback ablation: YOLO-only vs YOLO + temporal bbox
fallback.
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
SAM3DB_ROOT = PROJECT_ROOT / "vendor" / "sam-3d-body"
for path in (SRC_ROOT, SAM3DB_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "occlusion_distribution" / "field_yolo_alphapose_occlusion_severity.csv",
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
        default=PROJECT_ROOT / "outputs" / "predictions" / "field_temporal_bbox_fallback",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "occlusion_distribution" / "field_temporal_bbox_fallback_summary.csv",
    )
    return parser.parse_args()


def safe_name(sequence: str, frame_number: int) -> str:
    return f"{sequence}__f{frame_number:05d}.npz"


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def valid_bbox_rows(rows: list[dict[str, str]], sequence: str) -> list[dict[str, str]]:
    valid = []
    for row in rows:
        if row["sequence"] != sequence or row["status"] != "ok":
            continue
        try:
            _ = [float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")]
        except ValueError:
            continue
        valid.append(row)
    return valid


def nearest_valid_bbox(rows: list[dict[str, str]], target: dict[str, str]) -> tuple[np.ndarray | None, dict[str, str] | None, int | None]:
    candidates = valid_bbox_rows(rows, target["sequence"])
    if not candidates:
        return None, None, None
    target_frame = int(target["frame_number"])
    nearest = min(candidates, key=lambda row: abs(int(row["frame_number"]) - target_frame))
    bbox = np.array(
        [
            float(nearest["bbox_x1"]),
            float(nearest["bbox_y1"]),
            float(nearest["bbox_x2"]),
            float(nearest["bbox_y2"]),
        ],
        dtype=np.float32,
    )
    return bbox, nearest, abs(int(nearest["frame_number"]) - target_frame)


def main() -> None:
    opts = parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    opts.summary_output.parent.mkdir(parents=True, exist_ok=True)

    rows = load_manifest(opts.manifest)
    failure_rows = [row for row in rows if row["status"] == "no_person_detected"]

    print(f"yolo_failure_frames={len(failure_rows)}", flush=True)
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

    summary_rows = []
    for idx, row in enumerate(failure_rows, start=1):
        frame_number = int(row["frame_number"])
        bbox, donor, frame_distance = nearest_valid_bbox(rows, row)
        output_path = opts.output_dir / safe_name(row["sequence"], frame_number)
        start = time.monotonic()
        status = "not_recovered"
        error = ""
        output_available = "no"
        if bbox is None or donor is None:
            error = "no valid bbox donor in same sequence"
        else:
            try:
                outputs = estimator.process_one_image(
                    row["frame_path"],
                    bboxes=bbox.reshape(1, 4),
                    use_mask=False,
                    inference_type="body",
                )
                if not outputs:
                    raise RuntimeError("SAM 3D Body returned no outputs")
                first = outputs[0]
                np.savez_compressed(
                    output_path,
                    sequence=np.array([row["sequence"]]),
                    frame_number=np.array([frame_number]),
                    frame_path=np.array([row["frame_path"]]),
                    fallback_bbox=bbox[None],
                    donor_sequence=np.array([donor["sequence"]]),
                    donor_frame_number=np.array([int(donor["frame_number"])]),
                    frame_distance=np.array([frame_distance]),
                    pred_mhr70_m=first["pred_keypoints_3d"][None],
                    pred_cam_t_m=first["pred_cam_t"][None],
                    pred_keypoints_2d=first["pred_keypoints_2d"][None],
                )
                status = "recovered"
                output_available = "yes"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        seconds = time.monotonic() - start
        summary_rows.append(
            {
                "sequence": row["sequence"],
                "frame_number": frame_number,
                "frame_path": row["frame_path"],
                "yolo_status": row["status"],
                "fallback_status": status,
                "output_available": output_available,
                "donor_frame_number": "" if donor is None else donor["frame_number"],
                "frame_distance": "" if frame_distance is None else frame_distance,
                "fallback_bbox_x1": "" if bbox is None else f"{bbox[0]:.2f}",
                "fallback_bbox_y1": "" if bbox is None else f"{bbox[1]:.2f}",
                "fallback_bbox_x2": "" if bbox is None else f"{bbox[2]:.2f}",
                "fallback_bbox_y2": "" if bbox is None else f"{bbox[3]:.2f}",
                "seconds": f"{seconds:.3f}",
                "prediction_path": "" if output_available == "no" else str(output_path),
                "manual_severity": "severe_or_full_review",
                "manual_source": "needs_manual_review",
                "failure_mode": "detection_failure_under_occlusion",
                "error": error,
            }
        )
        print(
            f"[{idx}/{len(failure_rows)}] {status} {row['sequence']}/{frame_number:05d} "
            f"donor={'' if donor is None else donor['frame_number']} distance={frame_distance} {seconds:.1f}s",
            flush=True,
        )

    with opts.summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    recovered = sum(row["fallback_status"] == "recovered" for row in summary_rows)
    print(f"summary={opts.summary_output}", flush=True)
    print(f"recovered={recovered}/{len(summary_rows)}", flush=True)


if __name__ == "__main__":
    main()
