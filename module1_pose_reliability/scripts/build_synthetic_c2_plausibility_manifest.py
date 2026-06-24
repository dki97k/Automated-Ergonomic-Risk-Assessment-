#!/usr/bin/env python3
"""Build SAM/MotionBERT-ready manifests for synthetic occlusion C2."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "synthetic_occlusion"
        / "video_cases_c2"
        / "synthetic_video_case_analysis_manifest.csv",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manifests" / "synthetic_c2_plausibility_full_manifest.csv",
    )
    parser.add_argument(
        "--subset-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "manifests" / "synthetic_c2_plausibility_balanced100_manifest.csv",
    )
    parser.add_argument("--per-severity", type=int, default=100)
    return parser.parse_args()


def evenly_sample(group: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(group) <= n:
        return group
    if n == 1:
        return group.iloc[[0]]
    indices = [round(i * (len(group) - 1) / (n - 1)) for i in range(n)]
    return group.iloc[indices]


def main() -> None:
    opts = parse_args()
    df = pd.read_csv(opts.analysis_manifest)

    exp = pd.DataFrame()
    exp["sample_id"] = df.apply(
        lambda row: f"synthetic_c2/{row['case_id']}/{int(row['frame_offset']):03d}/{row['intended_level']}",
        axis=1,
    )
    exp["sequence"] = df["case_id"].astype(str) + "__" + df["intended_level"].astype(str)
    exp["frame_number"] = df["frame_offset"].astype(int)
    exp["frame_path"] = df["synthetic_image_path"]
    exp["bbox_x1"] = df["base_bbox_x1"]
    exp["bbox_y1"] = df["base_bbox_y1"]
    exp["bbox_x2"] = df["base_bbox_x2"]
    exp["bbox_y2"] = df["base_bbox_y2"]
    exp["intended_level"] = df["intended_level"]
    exp["analysis_severity"] = df["analysis_severity"]
    exp["visible_joint_ratio_body12"] = df["visible_joint_ratio_body12"]
    exp["detection_status"] = df["status"]
    exp["base_sequence"] = df["sequence"]
    exp["base_frame_number"] = df["base_frame_number"]
    exp["synthetic_image_path"] = df["synthetic_image_path"]

    opts.full_output.parent.mkdir(parents=True, exist_ok=True)
    exp.to_csv(opts.full_output, index=False)

    parts = []
    for severity in ["none", "partial", "severe"]:
        group = exp[exp["analysis_severity"].eq(severity)].copy()
        group = group.sort_values(["base_sequence", "frame_number", "intended_level"]).reset_index(drop=True)
        parts.append(evenly_sample(group, opts.per_severity))
    subset = pd.concat(parts, ignore_index=True)
    opts.subset_output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(opts.subset_output, index=False)

    print(f"full={opts.full_output} rows={len(exp)}")
    print(f"subset={opts.subset_output} rows={len(subset)}")
    print(subset["analysis_severity"].value_counts().to_string())
    print(pd.crosstab(subset["intended_level"], subset["analysis_severity"]).to_string())


if __name__ == "__main__":
    main()
