#!/usr/bin/env python3
"""Summarize synthetic occlusion plausibility metrics for Module 1."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

RELAXED_ANGLES = (
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
        default=PROJECT_ROOT / "outputs" / "manifests" / "synthetic_c2_plausibility_balanced30_manifest.csv",
    )
    parser.add_argument(
        "--sam-summary",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "synthetic_occlusion"
        / "video_cases_c2"
        / "synthetic_c2_sam3db_balanced30_summary.csv",
    )
    parser.add_argument(
        "--motionbert-summary",
        type=Path,
        default=PROJECT_ROOT
        / "results"
        / "synthetic_occlusion"
        / "video_cases_c2"
        / "synthetic_c2_alphapose_motionbert_balanced100_summary.csv",
    )
    parser.add_argument(
        "--motionbert-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "synthetic_c2_alphapose_motionbert_balanced100.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "synthetic_occlusion" / "video_cases_c2" / "plausibility_balanced30",
    )
    return parser.parse_args()


def bone_lengths(pose: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(pose[JI[a]] - pose[JI[b]]) for a, b in BONES], dtype=np.float64)


def angle_deg(pose: np.ndarray, a: str, b: str, c: str) -> float:
    ba = pose[JI[a]] - pose[JI[b]]
    bc = pose[JI[c]] - pose[JI[b]]
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-9:
        return np.nan
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def relaxed_jaf_invalid_percent(pose: np.ndarray) -> float:
    invalid = []
    for _, a, b, c, lo, hi in RELAXED_ANGLES:
        angle = angle_deg(pose, a, b, c)
        invalid.append(float((not np.isfinite(angle)) or angle < lo or angle > hi))
    return float(100.0 * np.mean(invalid))


def load_motionbert_predictions(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(key): pose for key, pose in zip(data["sample_ids"], data["pred_common14"], strict=True)}


def load_sam_prediction(path: str) -> np.ndarray | None:
    if not isinstance(path, str) or not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return data["pred_common14_m"][0]


def safe_mean(values: list[float]) -> float:
    return float(np.nanmean(values)) if values else np.nan


def summarize_pipeline(
    manifest: pd.DataFrame,
    pipeline: str,
    poses: dict[str, np.ndarray],
    status_by_id: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_rows = []
    for row in manifest.to_dict("records"):
        sample_id = row["sample_id"]
        status = status_by_id.get(sample_id, "missing")
        pose = poses.get(sample_id)
        out = {
            "pipeline": pipeline,
            "sample_id": sample_id,
            "analysis_severity": row["analysis_severity"],
            "intended_level": row["intended_level"],
            "status": status,
            "visible_joint_ratio_body12": row["visible_joint_ratio_body12"],
        }
        if status == "ok" and pose is not None:
            lengths = bone_lengths(pose)
            out["mean_bone_length"] = float(np.nanmean(lengths))
            out["relaxed_JAF_invalid_percent"] = relaxed_jaf_invalid_percent(pose)
        else:
            out["mean_bone_length"] = np.nan
            out["relaxed_JAF_invalid_percent"] = np.nan
        per_rows.append(out)
    per_pose = pd.DataFrame(per_rows)

    summary_rows = []
    for severity, group in per_pose.groupby("analysis_severity", sort=False):
        ok_group = group[group["status"].eq("ok")]
        lengths = []
        for sample_id in ok_group["sample_id"]:
            pose = poses.get(sample_id)
            if pose is not None:
                lengths.append(bone_lengths(pose))
        if lengths:
            arr = np.stack(lengths)
            cv = np.nanstd(arr, axis=0) / np.maximum(np.nanmean(arr, axis=0), 1e-9)
            blc_cv = float(np.nanmean(cv))
        else:
            blc_cv = np.nan
        summary_rows.append(
            {
                "pipeline": pipeline,
                "analysis_severity": severity,
                "n": len(group),
                "ok": int(len(ok_group)),
                "failure_rate_percent": round(100.0 * (1.0 - len(ok_group) / len(group)), 2),
                "BLC_CV": blc_cv,
                "relaxed_JAF_invalid_percent": safe_mean(ok_group["relaxed_JAF_invalid_percent"].tolist()),
            }
        )
    return per_pose, pd.DataFrame(summary_rows)


def main() -> None:
    opts = parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(opts.manifest)

    sam_summary = pd.read_csv(opts.sam_summary)
    sam_poses = {}
    sam_status = {}
    for row in sam_summary.to_dict("records"):
        sam_status[row["sample_id"]] = row["pipeline_status"]
        pose = load_sam_prediction(row.get("prediction_path", ""))
        if pose is not None:
            sam_poses[row["sample_id"]] = pose

    mb_summary = pd.read_csv(opts.motionbert_summary)
    mb_status = {row["sample_id"]: row["pipeline_status"] for row in mb_summary.to_dict("records")}
    mb_poses_all = load_motionbert_predictions(opts.motionbert_predictions)
    mb_poses = {sample_id: mb_poses_all[sample_id] for sample_id in manifest["sample_id"] if sample_id in mb_poses_all}

    sam_per, sam_sum = summarize_pipeline(manifest, "SAM-3DBody", sam_poses, sam_status)
    mb_per, mb_sum = summarize_pipeline(manifest, "AlphaPose-MotionBERT", mb_poses, mb_status)
    per_pose = pd.concat([sam_per, mb_per], ignore_index=True)
    summary = pd.concat([sam_sum, mb_sum], ignore_index=True)

    per_pose.to_csv(opts.output_dir / "synthetic_c2_plausibility_per_pose_metrics.csv", index=False)
    summary.to_csv(opts.output_dir / "synthetic_c2_plausibility_by_measured_severity_summary.csv", index=False)
    memo = opts.output_dir / "synthetic_c2_plausibility_memo.md"
    memo.write_text(
        "Synthetic C2 plausibility stress test summarized by measured AlphaPose visibility severity. "
        "BLC-CV and relaxed JAF invalid percentage are plausibility metrics, not ground-truth correctness metrics.\n\n"
        + summary.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"summary={opts.output_dir / 'synthetic_c2_plausibility_by_measured_severity_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
