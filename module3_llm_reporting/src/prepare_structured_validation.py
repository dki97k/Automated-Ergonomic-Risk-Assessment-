#!/usr/bin/env python3
"""Prepare structured validation inputs and rule-based references."""

from __future__ import annotations

import argparse
from pathlib import Path

from structured_common import (
    build_input_summary,
    build_key_factor_reference,
    project_root,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "module2_processed" / "processed",
    )
    parser.add_argument(
        "--input-output-dir",
        type=Path,
        default=root / "data" / "structured_validation" / "inputs" / "numerical_only",
    )
    parser.add_argument(
        "--derived-reference-output",
        type=Path,
        default=root
        / "data"
        / "structured_validation"
        / "reference"
        / "derived_reference_preview.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=root / "data" / "structured_validation" / "manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    references = []

    input_files = sorted(args.processed_dir.glob("*.json"))
    if not input_files:
        raise SystemExit(f"No processed JSON files found: {args.processed_dir}")

    args.input_output_dir.mkdir(parents=True, exist_ok=True)
    for path in input_files:
        processed = read_json(path)
        summary = build_input_summary(processed)
        reference = build_key_factor_reference(summary)
        sample_id = summary["sample_id"]

        write_json(args.input_output_dir / f"{sample_id}.json", summary)
        summaries.append(summary)
        references.append(reference)

    reference_payload = {
        "reference_type": "derived_reference_preview",
        "source": "Module 2 quantitative summaries",
        "note": (
            "Preview only. The validation GT is imported from origin "
            "results_structured_key.csv by import_structured_gt_from_origin.py."
        ),
        "risk_summary_reference": None,
        "samples": references,
    }
    write_json(args.derived_reference_output, reference_payload)

    manifest = {
        "condition": "structured_validation_evidence_based_numerical_only",
        "sample_count": len(summaries),
        "samples": [summary["sample_id"] for summary in summaries],
        "input_dir": str(args.input_output_dir),
        "derived_reference_preview_file": str(args.derived_reference_output),
    }
    write_json(args.manifest_output, manifest)

    print(f"Prepared {len(summaries)} structured validation inputs")
    print(f"Inputs: {args.input_output_dir}")
    print(f"Derived reference preview: {args.derived_reference_output}")
    print(f"Manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
