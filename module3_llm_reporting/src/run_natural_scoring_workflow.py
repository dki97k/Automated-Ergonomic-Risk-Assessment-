#!/usr/bin/env python3
"""Run the natural-language validation scoring workflow.

This script assumes natural-language reports already exist. It regenerates the
claim annotation sheet, backs up the unlabeled sheet, applies draft labels,
computes claim-support metrics, computes overlap metrics, and builds
the validation summary table.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from natural_common import project_root


def parse_args() -> argparse.Namespace:
    root = project_root()
    condition_dir = (
        root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only_m2_current_remapped"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition-dir",
        type=Path,
        default=condition_dir,
        help="Natural-validation condition directory containing reports/.",
    )
    parser.add_argument(
        "--keep-existing-template",
        action="store_true",
        help="Do not overwrite claim_annotation_sheet_unlabeled_template.csv.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    condition_dir = args.condition_dir
    reports_dir = condition_dir / "reports"
    evaluation_dir = condition_dir / "evaluation"
    validation_table_dir = condition_dir / "validation_table"
    annotation_csv = evaluation_dir / "claim_annotation_sheet.csv"
    unlabeled_template_csv = evaluation_dir / "claim_annotation_sheet_unlabeled_template.csv"
    draft_csv = evaluation_dir / "claim_annotation_sheet_labeled_draft.csv"

    if not reports_dir.exists():
        raise SystemExit(f"Reports directory not found: {reports_dir}")

    python = sys.executable
    root = project_root()

    run(
        [
            python,
            str(root / "src" / "prepare_natural_claim_annotation_sheet.py"),
            "--reports-dir",
            str(reports_dir),
            "--output",
            str(annotation_csv),
        ]
    )

    if not unlabeled_template_csv.exists() or not args.keep_existing_template:
        shutil.copyfile(annotation_csv, unlabeled_template_csv)
        print(f"Backed up unlabeled template: {unlabeled_template_csv}")

    run(
        [
            python,
            str(root / "src" / "draft_label_natural_claims.py"),
            "--annotation-csv",
            str(annotation_csv),
            "--output-csv",
            str(draft_csv),
        ]
    )

    shutil.copyfile(draft_csv, annotation_csv)
    print(f"Applied draft labels to: {annotation_csv}")

    run(
        [
            python,
            str(root / "src" / "evaluate_natural_claim_labels.py"),
            "--annotation-csv",
            str(annotation_csv),
            "--output-dir",
            str(evaluation_dir),
        ]
    )
    run(
        [
            python,
            str(root / "src" / "evaluate_natural_overlap.py"),
            "--reports-dir",
            str(reports_dir),
            "--output-dir",
            str(evaluation_dir),
        ]
    )
    run(
        [
            python,
            str(root / "src" / "build_natural_validation_table.py"),
            "--evaluation-dir",
            str(evaluation_dir),
            "--output-dir",
            str(validation_table_dir),
        ]
    )

    print("Natural-language scoring workflow complete.")
    print(f"Annotation sheet: {annotation_csv}")
    print(f"Validation table: {validation_table_dir / 'validation_table.csv'}")


if __name__ == "__main__":
    main()
