#!/usr/bin/env python3
"""Create a claim-level annotation sheet for natural-language validation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from natural_common import NATURAL_SECTIONS, project_root, report_json_files, split_claims
from structured_common import read_json


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
        "--output",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only"
        / "evaluation"
        / "claim_annotation_sheet.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_files = report_json_files(args.reports_dir)
    if not report_files:
        raise SystemExit(f"No report JSON files found: {args.reports_dir}")

    rows = []
    for report_file in report_files:
        payload = read_json(report_file)
        if payload.get("error") or not payload.get("report_text"):
            continue
        sample_id = payload["sample_id"]
        run_index = int(payload["run_index"])
        sections = payload.get("sections", {})
        for section in NATURAL_SECTIONS:
            claims = split_claims(sections.get(section, ""))
            for claim_index, claim_text in enumerate(claims, start=1):
                claim_id = (
                    f"{sample_id}__run_{run_index:02d}__"
                    f"{section.lower().replace(' ', '_')}__claim_{claim_index:02d}"
                )
                rows.append(
                    {
                        "claim_id": claim_id,
                        "sample_id": sample_id,
                        "run_index": run_index,
                        "section": section,
                        "claim_index": claim_index,
                        "claim_text": claim_text,
                        "support_label": "",
                        "allowed_labels": "supported|unsupported|contradiction",
                        "evidence_note": "",
                        "reviewer_note": "",
                        "report_file": str(report_file),
                        "input_file": payload.get("input_file", ""),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "claim_id",
        "sample_id",
        "run_index",
        "section",
        "claim_index",
        "claim_text",
        "support_label",
        "allowed_labels",
        "evidence_note",
        "reviewer_note",
        "report_file",
        "input_file",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} claim rows")
    print(args.output)


if __name__ == "__main__":
    main()

