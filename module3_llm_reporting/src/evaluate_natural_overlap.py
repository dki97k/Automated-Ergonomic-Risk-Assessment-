#!/usr/bin/env python3
"""Compute deterministic overlap metrics for natural-language report reproducibility."""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path

from natural_common import project_root, report_json_files, split_claims
from structured_common import read_json, write_json


KEY_FACTOR_KEYWORDS = {
    "trunk_overflexion": (
        "trunk",
        "lumbar",
        "back",
        "spine",
        "forward flexion",
        "bending",
        "stoop",
    ),
    "neck_overflexion_or_extension": (
        "neck",
        "cervical",
    ),
    "upper_arm_elevation": (
        "upper arm",
        "arm elevation",
        "shoulder",
        "overhead",
        "reaching",
        "reach",
    ),
    "wrist_deviation": (
        "wrist",
        "hand deviation",
        "forearm rotation",
    ),
    "knee_overflexion": (
        "knee",
        "kneeling",
        "squat",
        "leg flexion",
    ),
    "prolonged_static_posture": (
        "static",
        "sustained",
        "prolonged",
        "held posture",
        "duration",
    ),
    "repetitive_work": (
        "repetition",
        "repetitive",
        "repeated",
        "cycle",
        "cycles",
    ),
}


RECOMMENDATION_KEYWORDS = {
    "neutral_trunk_neck_posture": (
        "neutral spine",
        "neutral posture",
        "trunk flexion",
        "neck flexion",
        "reduce bending",
        "posture modification",
    ),
    "adjust_work_height_or_material_position": (
        "work height",
        "working height",
        "material height",
        "position materials",
        "place materials",
        "closer",
        "raise",
        "lower",
    ),
    "reduce_reaching_or_arm_elevation": (
        "reach",
        "reaching",
        "shoulder",
        "upper arm",
        "arm elevation",
        "overhead",
    ),
    "wrist_hand_tool_adjustment": (
        "wrist",
        "grip",
        "handle",
        "tool",
        "hand",
    ),
    "reduce_knee_flexion_or_kneeling": (
        "knee",
        "kneeling",
        "squat",
        "leg flexion",
    ),
    "reduce_static_exposure": (
        "static",
        "sustained",
        "prolonged",
        "posture variation",
        "change posture",
    ),
    "reduce_repetition_or_add_recovery": (
        "repetition",
        "repetitive",
        "cycle",
        "pace",
        "rotation",
        "break",
        "microbreak",
        "recovery",
    ),
    "mechanical_assistance_or_support": (
        "mechanical",
        "assist",
        "aid",
        "support",
        "fixture",
        "jig",
    ),
    "monitoring_feedback_or_training": (
        "training",
        "feedback",
        "monitoring",
        "instruction",
    ),
}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only"
        / "reports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only"
        / "evaluation",
    )
    return parser.parse_args()


def contains_keyword(text: str, keyword: str) -> bool:
    pattern = r"(?<![a-z])" + re.escape(keyword.lower()) + r"(?![a-z])"
    return re.search(pattern, text.lower()) is not None


def tag_text(text: str, vocabulary: dict[str, tuple[str, ...]]) -> set[str]:
    tags = set()
    for tag, keywords in vocabulary.items():
        if any(contains_keyword(text, keyword) for keyword in keywords):
            tags.add(tag)
    return tags


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def main() -> None:
    args = parse_args()
    report_files = report_json_files(args.reports_dir)
    if not report_files:
        raise SystemExit(f"No report JSON files found: {args.reports_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []
    by_sample: dict[str, list[dict]] = defaultdict(list)

    for report_file in report_files:
        payload = read_json(report_file)
        if payload.get("error") or not payload.get("report_text"):
            continue
        sections = payload.get("sections", {})
        key_text = sections.get("Key Contributing Factors", "")
        rec_text = sections.get("Recommendations", "")

        key_tags = tag_text(key_text, KEY_FACTOR_KEYWORDS)
        recommendation_claims = split_claims(rec_text)
        recommendation_tags = set()
        for claim in recommendation_claims:
            recommendation_tags.update(tag_text(claim, RECOMMENDATION_KEYWORDS))

        row = {
            "report_id": f"{payload['sample_id']}__run_{int(payload['run_index']):02d}",
            "sample_id": payload["sample_id"],
            "run_index": int(payload["run_index"]),
            "key_factor_tags": sorted(key_tags),
            "recommendation_tags": sorted(recommendation_tags),
            "key_factor_section": key_text,
            "recommendations_section": rec_text,
            "report_file": str(report_file),
            "method": "deterministic_keyword_extraction_preview",
        }
        report_rows.append(row)
        by_sample[payload["sample_id"]].append(row)

    pair_rows = []
    for sample_id, rows in sorted(by_sample.items()):
        for left, right in itertools.combinations(sorted(rows, key=lambda r: r["run_index"]), 2):
            key_score = jaccard(set(left["key_factor_tags"]), set(right["key_factor_tags"]))
            rec_score = jaccard(set(left["recommendation_tags"]), set(right["recommendation_tags"]))
            pair_rows.append(
                {
                    "sample_id": sample_id,
                    "left_run": left["run_index"],
                    "right_run": right["run_index"],
                    "key_factor_jaccard": round(key_score, 4),
                    "recommendation_jaccard": round(rec_score, 4),
                    "left_key_factor_tags": "|".join(left["key_factor_tags"]),
                    "right_key_factor_tags": "|".join(right["key_factor_tags"]),
                    "left_recommendation_tags": "|".join(left["recommendation_tags"]),
                    "right_recommendation_tags": "|".join(right["recommendation_tags"]),
                }
            )

    sample_rows = []
    for sample_id in sorted(by_sample):
        sample_pairs = [row for row in pair_rows if row["sample_id"] == sample_id]
        sample_rows.append(
            {
                "sample_id": sample_id,
                "report_count": len(by_sample[sample_id]),
                "pair_count": len(sample_pairs),
                "key_factor_jaccard_mean": mean(
                    [float(row["key_factor_jaccard"]) for row in sample_pairs]
                ),
                "recommendation_jaccard_mean": mean(
                    [float(row["recommendation_jaccard"]) for row in sample_pairs]
                ),
            }
        )

    metrics = {
        "method": "deterministic_keyword_extraction_preview",
        "report_count": len(report_rows),
        "sample_count": len(by_sample),
        "pair_count": len(pair_rows),
        "overall": {
            "key_factor_overlap_jaccard": mean(
                [float(row["key_factor_jaccard"]) for row in pair_rows]
            ),
            "recommendation_overlap_jaccard": mean(
                [float(row["recommendation_jaccard"]) for row in pair_rows]
            ),
        },
        "by_sample": sample_rows,
    }

    with (args.output_dir / "overlap_items.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "report_id",
            "sample_id",
            "run_index",
            "key_factor_tags",
            "recommendation_tags",
            "key_factor_section",
            "recommendations_section",
            "report_file",
            "method",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in report_rows:
            csv_row = row.copy()
            csv_row["key_factor_tags"] = "|".join(row["key_factor_tags"])
            csv_row["recommendation_tags"] = "|".join(row["recommendation_tags"])
            writer.writerow(csv_row)

    with (args.output_dir / "overlap_pairwise.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "sample_id",
            "left_run",
            "right_run",
            "key_factor_jaccard",
            "recommendation_jaccard",
            "left_key_factor_tags",
            "right_key_factor_tags",
            "left_recommendation_tags",
            "right_recommendation_tags",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pair_rows)

    with (args.output_dir / "overlap_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "sample_id",
            "report_count",
            "pair_count",
            "key_factor_jaccard_mean",
            "recommendation_jaccard_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)

    write_json(args.output_dir / "overlap_metrics.json", metrics)
    print(f"Reports: {len(report_rows)}")
    print(f"Pairs: {len(pair_rows)}")
    print(f"Key-factor overlap Jaccard: {metrics['overall']['key_factor_overlap_jaccard']}")
    print(f"Recommendation overlap Jaccard: {metrics['overall']['recommendation_overlap_jaccard']}")
    print(args.output_dir / "overlap_metrics.json")


if __name__ == "__main__":
    main()

