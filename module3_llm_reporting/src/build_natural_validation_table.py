#!/usr/bin/env python3
"""Build the natural-language validation summary table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from natural_common import project_root
from structured_common import read_json, write_json


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only_m2_current_remapped"
        / "evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only_m2_current_remapped"
        / "validation_table",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    claim_metrics = read_json(args.evaluation_dir / "claim_support_metrics.json")
    overlap_metrics = read_json(args.evaluation_dir / "overlap_metrics.json")
    row = {
        "validation_condition": "natural evidence-grounded prompt, numerical Module 2 input only",
        "report_count": overlap_metrics["report_count"],
        "labeled_claims": claim_metrics["overall"]["labeled_claims"],
        "supported_claim_rate": claim_metrics["overall"]["supported_claim_rate"],
        "unsupported_claim_rate": claim_metrics["overall"]["unsupported_claim_rate"],
        "contradiction_rate": claim_metrics["overall"]["contradiction_rate"],
        "key_factor_overlap_jaccard": overlap_metrics["overall"]["key_factor_overlap_jaccard"],
        "recommendation_overlap_jaccard": overlap_metrics["overall"]["recommendation_overlap_jaccard"],
        "annotation_status": "Manually reviewed claim labels",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "validation_table.json", [row])
    with (args.output_dir / "validation_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(args.output_dir / "validation_table.csv")


if __name__ == "__main__":
    main()
