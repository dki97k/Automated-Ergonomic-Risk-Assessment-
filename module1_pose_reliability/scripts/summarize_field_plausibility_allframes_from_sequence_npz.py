#!/usr/bin/env python3
"""Summarize all-frame field plausibility metrics by occlusion level.

This faster version reads SAM-3DBody predictions aggregated by sequence
instead of opening one NPY file per frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Original m1_eval.py uses COCO-17 index space.
# 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
# 5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
# 9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
# 13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
BONES = (
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
)

JAF_ANGLES = (
    ("left_elbow", 5, 7, 9, 0.0, 150.0),
    ("right_elbow", 6, 8, 10, 0.0, 150.0),
    ("left_knee", 11, 13, 15, 0.0, 150.0),
    ("right_knee", 12, 14, 16, 0.0, 150.0),
    ("left_shoulder", 7, 5, 11, 0.0, 180.0),
    ("right_shoulder", 8, 6, 12, 0.0, 180.0),
)

COMMON14_TO_COCO17 = {
    2: 5,   # left_shoulder
    3: 6,   # right_shoulder
    4: 7,   # left_elbow
    5: 8,   # right_elbow
    6: 9,   # left_wrist
    7: 10,  # right_wrist
    8: 11,  # left_hip
    9: 12,  # right_hip
    10: 13, # left_knee
    11: 14, # right_knee
    12: 15, # left_ankle
    13: 16, # right_ankle
}


def default_manifest() -> Path:
    public_manifest = (
        PROJECT_ROOT
        / "outputs"
        / "field_occlusion_plausibility"
        / "manifest"
        / "field_plausibility_allframes_manifest.csv"
    )
    if public_manifest.exists():
        return public_manifest
    return PROJECT_ROOT / "outputs" / "manifests" / "field_plausibility_by_severity_allframes_temporal_bbox_manifest.csv"


def default_sam_sequence_dir() -> Path:
    public_dir = (
        PROJECT_ROOT
        / "outputs"
        / "field_occlusion_plausibility"
        / "predictions"
        / "sam3db_allframes_temporal_bbox_by_sequence"
    )
    if public_dir.exists():
        return public_dir
    return PROJECT_ROOT / "outputs" / "predictions" / "field_plausibility_sam3db_allframes_temporal_bbox_by_sequence"


def default_motionbert_predictions() -> Path:
    public_path = (
        PROJECT_ROOT
        / "outputs"
        / "field_occlusion_plausibility"
        / "predictions"
        / "field_plausibility_alphapose_motionbert_allframes_temporal_bbox.npz"
    )
    if public_path.exists():
        return public_path
    return PROJECT_ROOT / "outputs" / "predictions" / "field_plausibility_alphapose_motionbert_allframes_temporal_bbox.npz"


def default_output_dir() -> Path:
    public_output = PROJECT_ROOT / "outputs" / "field_occlusion_plausibility" / "metrics"
    if (PROJECT_ROOT / "outputs" / "field_occlusion_plausibility").exists():
        return public_output
    return PROJECT_ROOT / "results" / "plausibility" / "allframes_temporal_bbox_3level"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_manifest(),
    )
    parser.add_argument(
        "--sam-sequence-dir",
        type=Path,
        default=default_sam_sequence_dir(),
    )
    parser.add_argument(
        "--motionbert-predictions",
        type=Path,
        default=default_motionbert_predictions(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
    )
    return parser.parse_args()


def normalize_label(label: str) -> str:
    return "severe" if label.startswith("severe") else label


def bone_lengths(pose: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(pose[i] - pose[j]) for i, j in BONES], dtype=np.float64)


def angle_deg(pose: np.ndarray, a: int, b: int, c: int) -> float:
    ba = pose[a] - pose[b]
    bc = pose[c] - pose[b]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-9:
        return np.nan
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def jaf_invalid_percent(pose: np.ndarray) -> float:
    invalid = []
    for _, a, b, c, lo, hi in JAF_ANGLES:
        angle = angle_deg(pose, a, b, c)
        invalid.append(float((not np.isfinite(angle)) or angle < lo or angle > hi))
    return float(100.0 * np.mean(invalid))


def common14_to_coco17(common14: np.ndarray) -> np.ndarray:
    common14 = np.asarray(common14, dtype=np.float64)
    coco17 = np.full((17, 3), np.nan, dtype=np.float64)
    for common_idx, coco_idx in COMMON14_TO_COCO17.items():
        coco17[coco_idx] = common14[common_idx]
    return coco17


def load_motionbert_predictions(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        str(key): common14_to_coco17(np.asarray(pose, dtype=np.float64))
        for key, pose in zip(data["sample_ids"], data["pred_common14"], strict=True)
    }


def load_sam_predictions_by_sequence(seq_dir: Path) -> dict[tuple[str, int], np.ndarray]:
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for path in sorted(seq_dir.glob("*_sam3db_mhr70_keypoints3d.npz")):
        if path.name.startswith("._"):
            continue
        data = np.load(path, allow_pickle=True)
        sequence = str(np.atleast_1d(data["sequence"])[0])
        frames = np.asarray(data["frame_index"], dtype=np.int64)
        poses = np.asarray(data["keypoints3d_mhr70_m"], dtype=np.float64)[:, :17, :]
        for frame, pose in zip(frames, poses, strict=True):
            predictions[(sequence, int(frame))] = pose
    return predictions


def summarize(per_pose: pd.DataFrame, poses: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for (pipeline, level), group in per_pose.groupby(["pipeline", "occlusion_level"], sort=False):
        ok = group[group["status"] == "ok"]
        lengths = [bone_lengths(poses[row.sample_id]) for row in ok.itertuples() if row.sample_id in poses]
        if lengths:
            arr = np.stack(lengths)
            cv = np.nanstd(arr, axis=0) / np.maximum(np.nanmean(arr, axis=0), 1e-9)
            blc_cv = float(np.nanmean(cv))
        else:
            blc_cv = np.nan
        rows.append(
            {
                "pipeline": pipeline,
                "occlusion_level": level,
                "n": int(len(group)),
                "ok": int(len(ok)),
                "failure_rate_percent": float(100.0 * (1.0 - len(ok) / len(group))) if len(group) else np.nan,
                "BLC_CV": blc_cv,
                "JAF_invalid_percent": float(ok["JAF_invalid_percent"].mean()) if len(ok) else np.nan,
            }
        )
    order = {"none": 0, "partial": 1, "severe": 2}
    pipeline_order = {"AlphaPose-MotionBERT": 0, "SAM-3DBody": 1}
    summary = pd.DataFrame(rows)
    summary["_level_order"] = summary["occlusion_level"].map(order).fillna(99)
    summary["_pipeline_order"] = summary["pipeline"].map(pipeline_order).fillna(99)
    return summary.sort_values(["_level_order", "_pipeline_order"]).drop(columns=["_level_order", "_pipeline_order"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    manifest["occlusion_level"] = manifest["field_occlusion_label"].astype(str).map(normalize_label)
    mb_preds = load_motionbert_predictions(args.motionbert_predictions)
    sam_preds = load_sam_predictions_by_sequence(args.sam_sequence_dir)

    rows = []
    poses: dict[str, np.ndarray] = {}
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        level = str(row.occlusion_level)

        mb_pose = mb_preds.get(sample_id)
        mb_key = f"AlphaPose-MotionBERT::{sample_id}"
        if mb_pose is not None:
            poses[mb_key] = mb_pose
        rows.append(
            {
                "pipeline": "AlphaPose-MotionBERT",
                "sample_id": mb_key,
                "sequence": row.sequence,
                "frame_number": int(row.frame_number),
                "occlusion_level": level,
                "source_occlusion_label": row.field_occlusion_label,
                "status": "ok" if mb_pose is not None else "missing_prediction",
                "JAF_invalid_percent": np.nan if mb_pose is None else jaf_invalid_percent(mb_pose),
            }
        )

        sam_pose = sam_preds.get((str(row.sequence), int(row.frame_number)))
        sam_key = f"SAM-3DBody::{sample_id}"
        if sam_pose is not None:
            poses[sam_key] = sam_pose
        rows.append(
            {
                "pipeline": "SAM-3DBody",
                "sample_id": sam_key,
                "sequence": row.sequence,
                "frame_number": int(row.frame_number),
                "occlusion_level": level,
                "source_occlusion_label": row.field_occlusion_label,
                "status": "ok" if sam_pose is not None else "missing_prediction",
                "JAF_invalid_percent": np.nan if sam_pose is None else jaf_invalid_percent(sam_pose),
            }
        )

    per_pose = pd.DataFrame(rows)
    per_pose.to_csv(args.output_dir / "field_plausibility_allframes_3level_per_pose_metrics.csv", index=False)

    summary = summarize(per_pose, poses)
    summary.to_csv(args.output_dir / "field_plausibility_allframes_3level_summary.csv", index=False)

    level_order = {"none": 0, "partial": 1, "severe": 2}
    label_summary = manifest.groupby("occlusion_level", sort=False).size().reset_index(name="n_frames")
    label_summary["_level_order"] = label_summary["occlusion_level"].map(level_order).fillna(99)
    label_summary = label_summary.sort_values("_level_order").drop(columns="_level_order")
    label_summary.to_csv(args.output_dir / "field_plausibility_allframes_3level_label_summary.csv", index=False)

    memo = args.output_dir / "field_plausibility_allframes_3level_memo.md"
    memo.write_text(
        "# Field Plausibility by Occlusion Level, All Frames\n\n"
        "This analysis compares SAM-3DBody and AlphaPose-MotionBERT on all field frames "
        "with temporal bounding-box fallback. `severe_detected` and "
        "`severe_detection_failure_temporal_bbox` are combined as `severe`.\n\n"
        "BLC-CV and JAF are computed using the original `m1_eval.py` definitions: "
        "COCO-17 index space, 12 bones for BLC-CV, and six joint-angle checks for "
        "JAF invalid percentage.\n\n"
        "## Label counts\n\n"
        f"{label_summary.to_string(index=False)}\n\n"
        "## Summary\n\n"
        f"{summary.to_string(index=False)}\n",
        encoding="utf-8",
    )

    print(f"summary={args.output_dir / 'field_plausibility_allframes_3level_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
