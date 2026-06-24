#!/usr/bin/env python3
"""Compute Module 1 configuration-analysis metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = ("alphapose_motionbert", "sam3db")
REGIONS = ("trunk", "upper_arm", "lower_arm", "leg")
ORDER = {"Low": 0, "Medium": 1, "High": 2}


def score_trunk(flex: float, bend: float, twist: float) -> int:
    if flex < -5:
        s_flex = 2
    elif flex <= 5:
        s_flex = 1
    elif flex <= 20:
        s_flex = 2
    elif flex <= 60:
        s_flex = 3
    else:
        s_flex = 4
    return s_flex + (1 if abs(bend) >= 20 or abs(twist) >= 20 else 0)


def score_upper_arm(flex: float, abduction: float) -> int:
    if flex < -20:
        s_flex = 2
    elif flex <= 20:
        s_flex = 1
    elif flex <= 45:
        s_flex = 2
    elif flex <= 90:
        s_flex = 3
    else:
        s_flex = 4
    return s_flex + (1 if abduction > 45 else 0)


def score_lower_arm(flex: float) -> int:
    phi = 90.0 - flex
    return 1 if 60.0 <= phi <= 100.0 else 2


def score_leg(knee_angle: float, support_ratio: float) -> int:
    if knee_angle >= 60:
        s_flex = 2
    elif knee_angle >= 30:
        s_flex = 1
    else:
        s_flex = 0
    s_balance = 2 if support_ratio < 0.48 or support_ratio > 0.52 else 1
    return s_flex + s_balance


def region_bin(region: str, score: float) -> str:
    if region == "lower_arm":
        return "High" if score >= 2 else "Low"
    if score <= 1:
        return "Low"
    if score == 2:
        return "Medium"
    return "High"


def direction(a: str, b: str) -> str:
    if ORDER[b] > ORDER[a]:
        return "higher-risk shift"
    if ORDER[b] < ORDER[a]:
        return "lower-risk shift"
    return "unchanged"


def per_frame_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["trunk"] = [
        score_trunk(row[("trunk", "flexion")], row[("trunk", "bending")], row[("trunk", "twisting")])
        for _, row in df.iterrows()
    ]
    out["upper_arm"] = [
        max(
            score_upper_arm(row[("upperarm", "left_flexion")], row[("upperarm", "left_abduction")]),
            score_upper_arm(row[("upperarm", "right_flexion")], row[("upperarm", "right_abduction")]),
        )
        for _, row in df.iterrows()
    ]
    out["lower_arm"] = [
        max(score_lower_arm(row[("lower arm", "left_flexion")]), score_lower_arm(row[("lower arm", "right_flexion")]))
        for _, row in df.iterrows()
    ]
    out["leg"] = [
        score_leg(max(row[("knee", "left_flexion")], row[("knee", "right_flexion")]), row[("leg_support", "ratio")])
        for _, row in df.iterrows()
    ]
    return out


def compute_region_metrics(root: Path) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        angle_dir = root / "inputs/shared_angle_csv" / condition
        for path in sorted(angle_dir.glob("*_angle.csv")):
            if path.name.startswith("._"):
                continue
            clip = path.name.replace("_angle.csv", "")
            df = pd.read_csv(path, header=[0, 1], index_col=0)
            scores = per_frame_scores(df)
            for region in REGIONS:
                p90 = float(np.percentile(scores[region], 90))
                rows.append(
                    {
                        "condition": condition,
                        "clip": clip,
                        "region": region,
                        "mean_score": float(scores[region].mean()),
                        "p90_score": p90,
                        "peak_score": int(scores[region].max()),
                        "risk_bin": region_bin(region, round(p90)),
                        "n_frames": int(len(scores)),
                    }
                )
    return pd.DataFrame(rows)


def compare_region_metrics(region_df: pd.DataFrame) -> pd.DataFrame:
    alpha = region_df[region_df["condition"] == "alphapose_motionbert"].set_index(["clip", "region"])
    sam = region_df[region_df["condition"] == "sam3db"].set_index(["clip", "region"])
    rows = []
    for idx in sorted(set(alpha.index) & set(sam.index)):
        a = alpha.loc[idx]
        b = sam.loc[idx]
        rows.append(
            {
                "clip": idx[0],
                "region": idx[1],
                "alpha_risk_bin": a["risk_bin"],
                "sam_risk_bin": b["risk_bin"],
                "risk_direction_alpha_to_sam": direction(a["risk_bin"], b["risk_bin"]),
                "risk_bin_changed": a["risk_bin"] != b["risk_bin"],
                "p90_score_alpha": a["p90_score"],
                "p90_score_sam": b["p90_score"],
                "p90_score_diff_sam_minus_alpha": b["p90_score"] - a["p90_score"],
                "abs_p90_score_diff": abs(b["p90_score"] - a["p90_score"]),
            }
        )
    return pd.DataFrame(rows)


def evidence_availability() -> pd.DataFrame:
    rows = [
        ("alphapose_motionbert", "shared_body_posture", "available", "pelvis, neck, shoulders, elbows, wrists, hips, knees, and ankles are present"),
        ("alphapose_motionbert", "full_reba_final", "not_estimable", "head/ear and hand landmarks for neck and wrist scoring are unavailable"),
        ("alphapose_motionbert", "wrist_flexion_twisting", "not_estimable", "hand landmarks required by Module 2 are unavailable"),
        ("alphapose_motionbert", "neck_flexion", "not_estimable", "head/ear landmarks are unavailable"),
        ("alphapose_motionbert", "shared_duration", "available", "duration can be estimated on shared trunk/arm/leg angle signals"),
        ("alphapose_motionbert", "repetition", "available_after_rep_run", "REP++ accepts normalized major-body-joint JSONL"),
        ("sam3db", "shared_body_posture", "available", "shared body joints are available"),
        ("sam3db", "full_reba_final", "available", "MHR-70 keypoints include additional neck/wrist/hand landmarks expected by Module 2"),
        ("sam3db", "wrist_flexion_twisting", "available", "MHR-70 hand/wrist landmarks are available"),
        ("sam3db", "neck_flexion", "available", "MHR-70 head/neck landmarks are available"),
        ("sam3db", "shared_duration", "available", "duration can be estimated on shared trunk/arm/leg angle signals"),
        ("sam3db", "repetition", "available_after_rep_run", "REP++ accepts normalized major-body-joint JSONL"),
    ]
    return pd.DataFrame(rows, columns=["condition", "feature", "availability", "reason"])


def compare_duration(root: Path) -> tuple[pd.DataFrame | None, dict]:
    path = root / "results/shared_duration/shared_duration_summary.csv"
    if not path.exists():
        return None, {"shared_duration_status": "not_run"}
    df = pd.read_csv(path)
    alpha = df[df["condition"] == "alphapose_motionbert"].set_index("VIDEO_NM")
    sam = df[df["condition"] == "sam3db"].set_index("VIDEO_NM")
    rows = []
    for clip in sorted(set(alpha.index) & set(sam.index)):
        a = alpha.loc[clip]
        b = sam.loc[clip]
        rows.append(
            {
                "clip": clip,
                "alpha_static_duration_sec": a["Total static duration (s)"],
                "sam_static_duration_sec": b["Total static duration (s)"],
                "static_duration_diff_sam_minus_alpha": b["Total static duration (s)"] - a["Total static duration (s)"],
                "abs_static_duration_diff": abs(b["Total static duration (s)"] - a["Total static duration (s)"]),
                "alpha_static_ratio_percent": a["Static posture ratio (%)"],
                "sam_static_ratio_percent": b["Static posture ratio (%)"],
                "static_presence_changed": (a["Total static duration (s)"] > 0) != (b["Total static duration (s)"] > 0),
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "shared_duration_status": "ok",
        "shared_duration_presence_transition_rate": float(out["static_presence_changed"].mean()) if len(out) else None,
        "median_abs_static_duration_diff_sec": float(out["abs_static_duration_diff"].median()) if len(out) else None,
    }
    return out, summary


def freq_band(rpm: float) -> str:
    if rpm >= 10:
        return "very frequent"
    if rpm >= 5:
        return "frequent"
    if rpm >= 2:
        return "moderate"
    return "low"


def compare_repetition(root: Path) -> tuple[pd.DataFrame | None, dict]:
    paths = {
        condition: root / "results/repetition" / condition / "repetition_case_summary.csv"
        for condition in CONDITIONS
    }
    if not all(path.exists() for path in paths.values()):
        return None, {"repetition_status": "not_run_for_both_conditions"}
    alpha = pd.read_csv(paths["alphapose_motionbert"]).set_index("clip")
    sam = pd.read_csv(paths["sam3db"]).set_index("clip")
    rows = []
    for clip in sorted(set(alpha.index) & set(sam.index)):
        a = alpha.loc[clip]
        b = sam.loc[clip]
        a_band = freq_band(float(a["rpm_mean"]))
        b_band = freq_band(float(b["rpm_mean"]))
        rows.append(
            {
                "clip": clip,
                "alpha_repetitions_total_peaks": a["repetitions_total_peaks"],
                "sam_repetitions_total_peaks": b["repetitions_total_peaks"],
                "peak_count_diff_sam_minus_alpha": b["repetitions_total_peaks"] - a["repetitions_total_peaks"],
                "alpha_mean_period_sec": a["mean_period_sec"],
                "sam_mean_period_sec": b["mean_period_sec"],
                "abs_mean_period_diff_sec": abs(b["mean_period_sec"] - a["mean_period_sec"]),
                "alpha_rpm_band": a_band,
                "sam_rpm_band": b_band,
                "rpm_band_changed": a_band != b_band,
                "alpha_quality_flag": a.get("quality_flag", ""),
                "sam_quality_flag": b.get("quality_flag", ""),
            }
        )
    out = pd.DataFrame(rows)
    summary = {
        "repetition_status": "ok",
        "repetition_frequency_band_change_rate": float(out["rpm_band_changed"].mean()) if len(out) else None,
        "median_abs_mean_period_diff_sec": float(out["abs_mean_period_diff_sec"].median()) if len(out) else None,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("<private_workspace>/configuration_analysis/m1"))
    args = parser.parse_args()
    root = args.root
    out_dir = root / "results/metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence = evidence_availability()
    evidence.to_csv(out_dir / "evidence_availability.csv", index=False)

    region = compute_region_metrics(root)
    region.to_csv(out_dir / "shared_region_case_metrics.csv", index=False)
    region_changes = compare_region_metrics(region)
    region_changes.to_csv(out_dir / "shared_region_pairwise_changes.csv", index=False)

    duration_changes, duration_summary = compare_duration(root)
    if duration_changes is not None:
        duration_changes.to_csv(out_dir / "shared_duration_pairwise_changes.csv", index=False)

    repetition_changes, repetition_summary = compare_repetition(root)
    if repetition_changes is not None:
        repetition_changes.to_csv(out_dir / "repetition_pairwise_changes.csv", index=False)

    summary = {
        "n_shared_region_case_pairs": int(len(region_changes)),
        "shared_region_risk_transition_rate": float(region_changes["risk_bin_changed"].mean()) if len(region_changes) else None,
        "higher_risk_shift_rate_alpha_to_sam": float((region_changes["risk_direction_alpha_to_sam"] == "higher-risk shift").mean()) if len(region_changes) else None,
        "lower_risk_shift_rate_alpha_to_sam": float((region_changes["risk_direction_alpha_to_sam"] == "lower-risk shift").mean()) if len(region_changes) else None,
        "median_abs_shared_region_p90_score_diff": float(region_changes["abs_p90_score_diff"].median()) if len(region_changes) else None,
        **duration_summary,
        **repetition_summary,
    }
    (out_dir / "module1_configuration_metric_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[ok] wrote metrics under {out_dir}")


if __name__ == "__main__":
    main()
