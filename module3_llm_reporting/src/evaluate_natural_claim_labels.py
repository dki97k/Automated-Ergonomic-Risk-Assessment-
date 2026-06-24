#!/usr/bin/env python3
"""Compute supported/unsupported/contradiction rates from a labeled claim sheet."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from natural_common import project_root
from structured_common import write_json


VALID_LABELS = {"supported", "unsupported", "contradiction"}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only"
        / "evaluation"
        / "claim_annotation_sheet.csv",
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


def rates(counter: Counter[str]) -> dict[str, float | int]:
    total = sum(counter[label] for label in VALID_LABELS)
    if total == 0:
        return {
            "labeled_claims": 0,
            "supported_claim_rate": None,
            "unsupported_claim_rate": None,
            "contradiction_rate": None,
        }
    return {
        "labeled_claims": total,
        "supported_claim_rate": round(counter["supported"] / total, 4),
        "unsupported_claim_rate": round(counter["unsupported"] / total, 4),
        "contradiction_rate": round(counter["contradiction"] / total, 4),
    }


def main() -> None:
    args = parse_args()
    if not args.annotation_csv.exists():
        raise SystemExit(f"Annotation CSV not found: {args.annotation_csv}")

    overall = Counter()
    by_section: dict[str, Counter[str]] = defaultdict(Counter)
    by_sample: dict[str, Counter[str]] = defaultdict(Counter)
    unlabeled = 0

    with args.annotation_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("support_label") or "").strip().lower()
            if not label:
                unlabeled += 1
                continue
            if label not in VALID_LABELS:
                raise SystemExit(
                    f"Invalid support_label '{label}' in claim_id={row.get('claim_id')}"
                )
            overall[label] += 1
            by_section[row["section"]][label] += 1
            by_sample[row["sample_id"]][label] += 1

    metrics = {
        "label_set": sorted(VALID_LABELS),
        "unlabeled_claims": unlabeled,
        "overall": rates(overall),
        "by_section": {section: rates(counter) for section, counter in sorted(by_section.items())},
        "by_sample": {sample: rates(counter) for sample, counter in sorted(by_sample.items())},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "claim_support_metrics.json", metrics)

    with (args.output_dir / "claim_support_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "group_type",
            "group",
            "labeled_claims",
            "supported_claim_rate",
            "unsupported_claim_rate",
            "contradiction_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"group_type": "overall", "group": "overall", **metrics["overall"]})
        for section, section_rates in metrics["by_section"].items():
            writer.writerow({"group_type": "section", "group": section, **section_rates})
        for sample, sample_rates in metrics["by_sample"].items():
            writer.writerow({"group_type": "sample", "group": sample, **sample_rates})

    print(f"Labeled claims: {metrics['overall']['labeled_claims']}")
    print(f"Unlabeled claims: {unlabeled}")
    print(args.output_dir / "claim_support_metrics.json")


if __name__ == "__main__":
    main()

