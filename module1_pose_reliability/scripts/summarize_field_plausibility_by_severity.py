#!/usr/bin/env python3
"""Summarize field plausibility metrics by occlusion severity."""

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

SYMMETRIC_BONES = (
    (("neck", "left_shoulder"), ("neck", "right_shoulder")),
    (("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")),
    (("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")),
    (("pelvis", "left_hip"), ("pelvis", "right_hip")),
    (("left_hip", "left_knee"), ("right_hip", "right_knee")),
    (("left_knee", "left_ankle"), ("right_knee", "right_ankle")),
)

ANGLES = (
    ("left_elbow", "left_shoulder", "left_elbow", "left_wrist", 0, 180),
    ("right_elbow", "right_shoulder", "right_elbow", "right_wrist", 0, 180),
    ("left_knee", "left_hip", "left_knee", "left_ankle", 0, 180),
    ("right_knee", "right_hip", "right_knee", "right_ankle", 0, 180),
    ("left_shoulder", "neck", "left_shoulder", "left_elbow", 0, 180),
    ("right_shoulder", "neck", "right_shoulder", "right_elbow", 0, 180),
    ("left_hip", "pelvis", "left_hip", "left_knee", 0, 180),
    ("right_hip", "pelvis", "right_hip", "right_knee", 0, 180),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manifests" / "field_plausibility_by_severity_balanced_manifest.csv",
    )
    parser.add_argument(
        "--sam-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "plausibility" / "field_sam3db_summary.csv",
    )
    parser.add_argument(
        "--motionbert-summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "plausibility" / "field_alphapose_motionbert_summary.csv",
    )
    parser.add_argument(
        "--motionbert-predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "field_plausibility_alphapose_motionbert.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "plausibility",
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
    cos = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def pose_metrics(pose: np.ndarray) -> dict[str, float]:
    lengths = bone_lengths(pose)
    sym_errors = []
    for left, right in SYMMETRIC_BONES:
        l_len = np.linalg.norm(pose[JI[left[0]]] - pose[JI[left[1]]])
        r_len = np.linalg.norm(pose[JI[right[0]]] - pose[JI[right[1]]])
        denom = max((l_len + r_len) / 2.0, 1e-9)
        sym_errors.append(abs(l_len - r_len) / denom)

    feasible = []
    for _, a, b, c, lo, hi in ANGLES:
        angle = angle_deg(pose, a, b, c)
        feasible.append(float(np.isfinite(angle) and lo <= angle <= hi))

    return {
        "mean_bone_length": float(np.nanmean(lengths)),
        "left_right_bone_symmetry_error": float(np.nanmean(sym_errors)),
        "joint_angle_feasibility": float(np.nanmean(feasible)),
    }


def load_motionbert_predictions(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {str(key): pose for key, pose in zip(data["sample_ids"], data["pred_common14"], strict=True)}


def load_sam_prediction(path: str) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return data["pred_common14_m"][0]


def add_prediction_metrics(summary: pd.DataFrame, pipeline: str, predictions: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    rows = []
    for row in summary.to_dict("records"):
        out = {
            "pipeline": pipeline,
            "sample_id": row["sample_id"],
            "sequence": row["sequence"],
            "frame_number": row["frame_number"],
            "field_occlusion_label": row["field_occlusion_label"],
            "auto_severity": row["auto_severity"],
            "status": row["pipeline_status"],
        }
        pose = None
        if pipeline == "AlphaPose-MotionBERT" and predictions is not None:
            pose = predictions.get(row["sample_id"])
        elif pipeline == "SAM-3DBody":
            pose = load_sam_prediction(row.get("prediction_path", ""))

        if pose is not None and row["pipeline_status"] == "ok":
            out.update(pose_metrics(np.asarray(pose, dtype=np.float64)))
        else:
            out.update(
                {
                    "mean_bone_length": np.nan,
                    "left_right_bone_symmetry_error": np.nan,
                    "joint_angle_feasibility": np.nan,
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def group_bone_length_cv(per_pose: pd.DataFrame, poses_by_id: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for (pipeline, label), group in per_pose[per_pose["status"] == "ok"].groupby(["pipeline", "field_occlusion_label"]):
        lengths = []
        for sample_id in group["sample_id"]:
            pose = poses_by_id.get((pipeline, sample_id))
            if pose is not None:
                lengths.append(bone_lengths(pose))
        if not lengths:
            continue
        arr = np.stack(lengths)
        cv = np.nanstd(arr, axis=0) / np.maximum(np.nanmean(arr, axis=0), 1e-9)
        rows.append(
            {
                "pipeline": pipeline,
                "field_occlusion_label": label,
                "bone_length_cv_mean": float(np.nanmean(cv)),
                "bone_length_cv_median": float(np.nanmedian(cv)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    opts = parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    sam_summary = pd.read_csv(opts.sam_summary)
    mb_summary = pd.read_csv(opts.motionbert_summary)
    mb_preds = load_motionbert_predictions(opts.motionbert_predictions)

    sam_metrics = add_prediction_metrics(sam_summary, "SAM-3DBody")
    mb_metrics = add_prediction_metrics(mb_summary, "AlphaPose-MotionBERT", mb_preds)
    per_pose = pd.concat([sam_metrics, mb_metrics], ignore_index=True)
    per_pose.to_csv(opts.output_dir / "field_plausibility_per_pose_metrics.csv", index=False)

    poses_by_id = {}
    for row in sam_summary.to_dict("records"):
        pose = load_sam_prediction(row.get("prediction_path", ""))
        if pose is not None and row["pipeline_status"] == "ok":
            poses_by_id[("SAM-3DBody", row["sample_id"])] = pose
    for sample_id, pose in mb_preds.items():
        poses_by_id[("AlphaPose-MotionBERT", sample_id)] = pose
    blc = group_bone_length_cv(per_pose, poses_by_id)

    summary_rows = []
    for (pipeline, label), group in per_pose.groupby(["pipeline", "field_occlusion_label"]):
        n = len(group)
        ok = int((group["status"] == "ok").sum())
        metrics_group = group[group["status"] == "ok"]
        row = {
            "pipeline": pipeline,
            "field_occlusion_label": label,
            "n": n,
            "ok": ok,
            "failure_rate_percent": round((1.0 - ok / n) * 100.0, 2) if n else np.nan,
            "left_right_bone_symmetry_error_mean": metrics_group["left_right_bone_symmetry_error"].mean(),
            "joint_angle_feasibility_mean": metrics_group["joint_angle_feasibility"].mean(),
        }
        match = blc[(blc["pipeline"] == pipeline) & (blc["field_occlusion_label"] == label)]
        row["bone_length_cv_mean"] = np.nan if match.empty else float(match.iloc[0]["bone_length_cv_mean"])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(opts.output_dir / "field_plausibility_by_severity_summary.csv", index=False)

    memo = opts.output_dir / "field_plausibility_by_severity_memo.md"
    memo.write_text(
        f"""# Field Plausibility by Occlusion Severity

This analysis compares SAM-3DBody and AlphaPose-MotionBERT on a balanced field-frame subset sampled by occlusion label.

Important caveat: these are plausibility and operational robustness metrics, not 3D ground-truth correctness metrics.

## Outputs

- `field_plausibility_per_pose_metrics.csv`
- `field_plausibility_by_severity_summary.csv`

## Summary

{summary.to_string(index=False)}
""",
        encoding="utf-8",
    )
    print(f"summary={opts.output_dir / 'field_plausibility_by_severity_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
