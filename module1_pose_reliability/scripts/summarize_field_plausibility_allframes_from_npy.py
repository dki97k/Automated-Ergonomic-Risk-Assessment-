#!/usr/bin/env python3
"""Summarize all-frame field plausibility metrics by occlusion level.

This version reads SAM-3DBody predictions saved as sequence/frame
`*_kpts3d.npy` files and AlphaPose-MotionBERT predictions saved as one NPZ.
It combines `severe_detected` and `severe_detection_failure_temporal_bbox`
into a single `severe` level.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("<private_workspace>/m1")
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from m1.evaluation.joint_mapping import mhr70_to_common_body  # noqa: E402


JOINTS = (
    "pelvis",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
JI = {name: idx for idx, name in enumerate(JOINTS)}

BONES = (
    ("pelvis", "neck"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
)

# Same relaxed anatomical ranges used previously; reported simply as JAF.
JAF_ANGLES = (
    ("left_elbow", "left_shoulder", "left_elbow", "left_wrist", 20.0, 175.0),
    ("right_elbow", "right_shoulder", "right_elbow", "right_wrist", 20.0, 175.0),
    ("left_knee", "left_hip", "left_knee", "left_ankle", 30.0, 175.0),
    ("right_knee", "right_hip", "right_knee", "right_ankle", 30.0, 175.0),
    ("left_shoulder", "neck", "left_shoulder", "left_elbow", 10.0, 170.0),
    ("right_shoulder", "neck", "right_shoulder", "right_elbow", 10.0, 170.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manifests" / "field_plausibility_by_severity_allframes_temporal_bbox_manifest.csv",
    )
    parser.add_argument(
        "--sam-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "field_plausibility_sam3db_allframes_temporal_bbox",
    )
    parser.add_argument(
        "--motionbert-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "field_plausibility_alphapose_motionbert_allframes_temporal_bbox.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "plausibility" / "allframes_temporal_bbox_3level",
    )
    return parser.parse_args()


def normalize_label(label: str) -> str:
    if label.startswith("severe"):
        return "severe"
    return label


def bone_lengths(pose: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(pose[JI[a]] - pose[JI[b]]) for a, b in BONES], dtype=np.float64)


def angle_deg(pose: np.ndarray, a: str, b: str, c: str) -> float:
    ba = pose[JI[a]] - pose[JI[b]]
    bc = pose[JI[c]] - pose[JI[b]]
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


def load_motionbert_predictions(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(key): np.asarray(pose, dtype=np.float64) for key, pose in zip(data["sample_ids"], data["pred_common14"], strict=True)}


def load_sam_common14(sam_dir: Path, sequence: str, frame_number: int) -> np.ndarray | None:
    path = sam_dir / sequence / f"{frame_number:05d}_kpts3d.npy"
    if not path.exists():
        return None
    return mhr70_to_common_body(np.load(path).astype(np.float64))


def summarize_pipeline(per_pose: pd.DataFrame, poses_by_id: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for (pipeline, level), group in per_pose.groupby(["pipeline", "occlusion_level"], sort=False):
        ok_group = group[group["status"] == "ok"]
        lengths = [bone_lengths(poses_by_id[row.sample_id]) for row in ok_group.itertuples() if row.sample_id in poses_by_id]
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
                "ok": int(len(ok_group)),
                "failure_rate_percent": float(100.0 * (1.0 - len(ok_group) / len(group))) if len(group) else np.nan,
                "BLC_CV": blc_cv,
                "JAF_invalid_percent": float(ok_group["JAF_invalid_percent"].mean()) if len(ok_group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    manifest["occlusion_level"] = manifest["field_occlusion_label"].astype(str).map(normalize_label)
    mb_preds = load_motionbert_predictions(args.motionbert_predictions)

    rows = []
    poses: dict[str, np.ndarray] = {}
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        level = str(row.occlusion_level)

        mb_pose = mb_preds.get(sample_id)
        mb_status = "ok" if mb_pose is not None else "missing_prediction"
        if mb_pose is not None:
            poses[f"AlphaPose-MotionBERT::{sample_id}"] = mb_pose
        rows.append(
            {
                "pipeline": "AlphaPose-MotionBERT",
                "sample_id": f"AlphaPose-MotionBERT::{sample_id}",
                "sequence": row.sequence,
                "frame_number": int(row.frame_number),
                "occlusion_level": level,
                "source_occlusion_label": row.field_occlusion_label,
                "status": mb_status,
                "JAF_invalid_percent": np.nan if mb_pose is None else jaf_invalid_percent(mb_pose),
            }
        )

        sam_pose = load_sam_common14(args.sam_dir, str(row.sequence), int(row.frame_number))
        sam_status = "ok" if sam_pose is not None else "missing_prediction"
        if sam_pose is not None:
            poses[f"SAM-3DBody::{sample_id}"] = sam_pose
        rows.append(
            {
                "pipeline": "SAM-3DBody",
                "sample_id": f"SAM-3DBody::{sample_id}",
                "sequence": row.sequence,
                "frame_number": int(row.frame_number),
                "occlusion_level": level,
                "source_occlusion_label": row.field_occlusion_label,
                "status": sam_status,
                "JAF_invalid_percent": np.nan if sam_pose is None else jaf_invalid_percent(sam_pose),
            }
        )

    per_pose = pd.DataFrame(rows)
    per_pose.to_csv(args.output_dir / "field_plausibility_allframes_3level_per_pose_metrics.csv", index=False)
    summary = summarize_pipeline(per_pose, poses)
    summary.to_csv(args.output_dir / "field_plausibility_allframes_3level_summary.csv", index=False)

    label_summary = (
        manifest.groupby("occlusion_level", sort=False)
        .size()
        .reset_index(name="n_frames")
    )
    label_summary.to_csv(args.output_dir / "field_plausibility_allframes_3level_label_summary.csv", index=False)

    memo = args.output_dir / "field_plausibility_allframes_3level_memo.md"
    memo.write_text(
        "# Field Plausibility by Occlusion Level, All Frames\n\n"
        "This analysis compares SAM-3DBody and AlphaPose-MotionBERT on all field frames "
        "with temporal bounding-box fallback. `severe_detected` and "
        "`severe_detection_failure_temporal_bbox` are combined as `severe`.\n\n"
        "The JAF column uses the same anatomical feasibility ranges used in the prior "
        "Module 1 plausibility analyses; the term `relaxed` is omitted in reporting.\n\n"
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
