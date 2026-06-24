#!/usr/bin/env python3
"""Summarize representative Module 1 configuration metrics from Module 2 schemas."""

from __future__ import annotations

import argparse
import json
from itertools import zip_longest
from pathlib import Path

import pandas as pd


ORDER = ["Negligible", "Low", "Medium", "High", "VeryHigh"]
RANK = {name: idx for idx, name in enumerate(ORDER)}


def normalize_risk(value) -> str:
    if value is None:
        return ""
    value = str(value)
    if value == "Low/Acceptable":
        return "Low"
    if value.startswith("High repetition"):
        return "High"
    return value


def risk_direction(a: str, b: str) -> str:
    a = normalize_risk(a)
    b = normalize_risk(b)
    if a not in RANK or b not in RANK:
        return "not_comparable"
    if RANK[b] > RANK[a]:
        return "higher-risk shift"
    if RANK[b] < RANK[a]:
        return "lower-risk shift"
    return "unchanged"


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "clip" not in df.columns:
        raise ValueError(f"Missing clip column: {path}")
    return df.set_index("clip")


def jaccard(a, b) -> float:
    sa = set(x for x in a if x)
    sb = set(x for x in b if x)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def key_factors_from_schema(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    factors = set()
    pose = data.get("pose") or {}
    duration = data.get("duration") or {}
    repetition = data.get("repetition") or {}
    if normalize_risk(pose.get("risk_level")) in {"High", "VeryHigh"}:
        factors.add(f"posture:{pose.get('provenance_dominant_part', '')}")
    if normalize_risk(duration.get("risk_level")) in {"High", "VeryHigh"}:
        factors.add("duration:static")
    if normalize_risk(repetition.get("risk_level")) in {"High", "VeryHigh"}:
        factors.add(f"repetition:{repetition.get('freq_band', '')}")
    if (data.get("cooccurrence") or {}).get("flag"):
        factors.add("cooccurrence")
    return factors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-summary", type=Path, required=True)
    parser.add_argument("--sam-summary", type=Path, required=True)
    parser.add_argument("--alpha-schema-dir", type=Path)
    parser.add_argument("--sam-schema-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    alpha = load_summary(args.alpha_summary)
    sam = load_summary(args.sam_summary)
    clips = sorted(set(alpha.index) & set(sam.index))
    rows = []
    for clip in clips:
        a = alpha.loc[clip]
        b = sam.loc[clip]
        rows.append(
            {
                "clip": clip,
                "pose_risk_alpha": normalize_risk(a.get("pose_risk")),
                "pose_risk_sam": normalize_risk(b.get("pose_risk")),
                "pose_risk_direction": risk_direction(a.get("pose_risk"), b.get("pose_risk")),
                "duration_risk_alpha": normalize_risk(a.get("dur_risk")),
                "duration_risk_sam": normalize_risk(b.get("dur_risk")),
                "duration_risk_direction": risk_direction(a.get("dur_risk"), b.get("dur_risk")),
                "static_sec_abs_diff": abs(float(a.get("dur_static_sec", 0)) - float(b.get("dur_static_sec", 0))),
                "repetition_band_alpha": a.get("rep_freq_band", ""),
                "repetition_band_sam": b.get("rep_freq_band", ""),
                "repetition_band_changed": a.get("rep_freq_band", "") != b.get("rep_freq_band", ""),
            }
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "module1_config_case_changes.csv", index=False)

    summary = {
        "n_cases": len(detail),
        "reba_risk_bin_transition_rate": float((detail["pose_risk_direction"] != "unchanged").mean()) if len(detail) else None,
        "higher_risk_shift_rate": float((detail["pose_risk_direction"] == "higher-risk shift").mean()) if len(detail) else None,
        "lower_risk_shift_rate": float((detail["pose_risk_direction"] == "lower-risk shift").mean()) if len(detail) else None,
        "duration_risk_transition_rate": float((detail["duration_risk_direction"] != "unchanged").mean()) if len(detail) else None,
        "median_static_sec_abs_diff": float(detail["static_sec_abs_diff"].median()) if len(detail) else None,
        "repetition_frequency_band_change_rate": float(detail["repetition_band_changed"].mean()) if len(detail) else None,
    }

    if args.alpha_schema_dir and args.sam_schema_dir:
        overlaps = []
        for clip in clips:
            ap = args.alpha_schema_dir / f"{clip}.json"
            sp = args.sam_schema_dir / f"{clip}.json"
            if ap.exists() and sp.exists():
                overlaps.append(
                    {
                        "clip": clip,
                        "module3_key_factor_proxy_jaccard": jaccard(
                            key_factors_from_schema(ap),
                            key_factors_from_schema(sp),
                        ),
                    }
                )
        if overlaps:
            overlap_df = pd.DataFrame(overlaps)
            overlap_df.to_csv(out_dir / "module1_config_key_factor_overlap.csv", index=False)
            summary["mean_key_factor_proxy_jaccard"] = float(
                overlap_df["module3_key_factor_proxy_jaccard"].mean()
            )

    (out_dir / "module1_config_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
