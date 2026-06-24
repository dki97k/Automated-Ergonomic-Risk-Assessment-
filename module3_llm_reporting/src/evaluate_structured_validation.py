#!/usr/bin/env python3
"""Evaluate structured validation reports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from structured_common import (
    KEY_FACTOR_FIELDS,
    RISK_SUMMARY_FIELDS,
    project_root,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=root
        / "results"
        / "structured_validation"
        / "evidence_based_numerical_only"
        / "reports",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=root
        / "data"
        / "structured_validation"
        / "reference"
        / "key_factor_reference.json",
    )
    parser.add_argument(
        "--risk-reference-csv",
        type=Path,
        default=root
        / "data"
        / "structured_validation"
        / "reference"
        / "risk_summary_reference.csv",
        help="Optional risk-summary High/Low reference CSV.",
    )
    parser.add_argument(
        "--risk-labels",
        default="High,Low",
        help="Comma-separated risk-summary labels used for consistency, e.g. High,Moderate,Low.",
    )
    parser.add_argument(
        "--ignore-risk-reference",
        action="store_true",
        help="Compute risk-summary consistency only, without GT-style risk-summary metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "results"
        / "structured_validation"
        / "evidence_based_numerical_only"
        / "evaluation",
    )
    return parser.parse_args()


def normalize_binary(value: Any, allowed: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    mapping = {label.lower(): label for label in allowed}
    return mapping.get(normalized)


def get_label(report: dict[str, Any], section: str, field: str, allowed: tuple[str, ...]) -> str | None:
    try:
        value = report[section][field]["label"]
    except (KeyError, TypeError):
        return None
    return normalize_binary(value, allowed)


def metric_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": total,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def fleiss_kappa(label_sets: list[list[str]], categories: tuple[str, ...]) -> float | None:
    usable = [labels for labels in label_sets if len(labels) >= 2]
    if not usable:
        return None
    n_values = {len(labels) for labels in usable}
    if len(n_values) != 1:
        return None
    n_raters = n_values.pop()
    n_items = len(usable)
    category_totals = Counter()
    p_i_values = []
    for labels in usable:
        counts = Counter(labels)
        category_totals.update(counts)
        p_i = (sum(counts[cat] ** 2 for cat in categories) - n_raters) / (
            n_raters * (n_raters - 1)
        )
        p_i_values.append(p_i)
    p_bar = sum(p_i_values) / n_items
    p_e = sum((category_totals[cat] / (n_items * n_raters)) ** 2 for cat in categories)
    if p_e == 1:
        return None
    return round((p_bar - p_e) / (1 - p_e), 4)


def load_references(path: Path) -> dict[str, dict[str, str]]:
    payload = read_json(path)
    references: dict[str, dict[str, str]] = {}
    if "samples" in payload:
        for sample in payload["samples"]:
            references[sample["sample_id"]] = {
                field: sample["key_risk_factors"][field]["label"]
                for field in KEY_FACTOR_FIELDS
            }
    else:
        for sample_id, sample_ref in payload.items():
            references[sample_id] = {
                field: sample_ref[field]["label"] for field in KEY_FACTOR_FIELDS
            }
    return references


def load_risk_references(path: Path, risk_labels: tuple[str, ...]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    references: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sample_id = row["sample_id"].strip()
            references[sample_id] = {
                field: normalize_binary(row[field], risk_labels) or row[field].strip()
                for field in RISK_SUMMARY_FIELDS
            }
    return references


def main() -> None:
    args = parse_args()
    risk_labels = tuple(label.strip() for label in args.risk_labels.split(",") if label.strip())
    if len(risk_labels) < 2:
        raise SystemExit("--risk-labels must contain at least two labels")
    references = load_references(args.reference_file)
    risk_references = (
        {}
        if args.ignore_risk_reference
        else load_risk_references(args.risk_reference_csv, risk_labels)
    )
    report_files = sorted(
        path for path in args.reports_dir.glob("*/*.json") if not path.name.startswith("._")
    )
    if not report_files:
        raise SystemExit(f"No report files found: {args.reports_dir}")

    rows = []
    counts_by_factor = defaultdict(lambda: Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0}))
    counts_total = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    risk_counts_by_field = defaultdict(lambda: Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0}))
    risk_counts_total = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    risk_labels_by_item = defaultdict(list)
    risk_rows = []
    invalid_reports = []

    for path in report_files:
        payload = read_json(path)
        sample_id = payload.get("sample_id") or path.parent.name
        run_index = payload.get("run_index")
        report = payload.get("report")
        if not isinstance(report, dict):
            invalid_reports.append(str(path))
            continue

        for risk_field in RISK_SUMMARY_FIELDS:
            label = get_label(report, "risk_summary", risk_field, risk_labels)
            if label is not None:
                risk_labels_by_item[(sample_id, risk_field)].append(label)
                risk_reference = risk_references.get(sample_id, {}).get(risk_field)
                if risk_reference and risk_labels == ("High", "Low"):
                    if label == "High" and risk_reference == "High":
                        outcome = "tp"
                    elif label == "High" and risk_reference == "Low":
                        outcome = "fp"
                    elif label == "Low" and risk_reference == "Low":
                        outcome = "tn"
                    else:
                        outcome = "fn"
                    risk_counts_total[outcome] += 1
                    risk_counts_by_field[risk_field][outcome] += 1
                    risk_rows.append(
                        {
                            "sample_id": sample_id,
                            "run_index": run_index,
                            "risk_field": risk_field,
                            "prediction": label,
                            "reference": risk_reference,
                            "outcome": outcome,
                            "file": str(path),
                        }
                    )

        reference = references.get(sample_id)
        if not reference:
            continue
        for factor in KEY_FACTOR_FIELDS:
            pred = get_label(report, "key_risk_factors", factor, ("Yes", "No"))
            ref = reference[factor]
            if pred is None:
                invalid_reports.append(f"{path}:{factor}")
                continue
            if pred == "Yes" and ref == "Yes":
                outcome = "tp"
            elif pred == "Yes" and ref == "No":
                outcome = "fp"
            elif pred == "No" and ref == "No":
                outcome = "tn"
            else:
                outcome = "fn"
            counts_total[outcome] += 1
            counts_by_factor[factor][outcome] += 1
            rows.append(
                {
                    "sample_id": sample_id,
                    "run_index": run_index,
                    "factor": factor,
                    "prediction": pred,
                    "reference": ref,
                    "outcome": outcome,
                    "file": str(path),
                }
            )

    overall_metrics = metric_from_counts(
        counts_total["tp"], counts_total["fp"], counts_total["tn"], counts_total["fn"]
    )
    risk_overall_metrics = metric_from_counts(
        risk_counts_total["tp"],
        risk_counts_total["fp"],
        risk_counts_total["tn"],
        risk_counts_total["fn"],
    )
    by_factor = {
        factor: metric_from_counts(
            counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        )
        for factor, counts in counts_by_factor.items()
    }
    fleiss_overall = fleiss_kappa(list(risk_labels_by_item.values()), risk_labels)
    fleiss_by_risk = {
        risk_field: fleiss_kappa(
            [
                labels
                for (sample_id, field), labels in risk_labels_by_item.items()
                if field == risk_field
            ],
            risk_labels,
        )
        for risk_field in RISK_SUMMARY_FIELDS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "condition": "evidence_based_numerical_only",
        "report_file_count": len(report_files),
        "valid_key_factor_rows": len(rows),
        "invalid_entries": invalid_reports,
        "risk_summary_fleiss_kappa": {
            "overall": fleiss_overall,
            "by_risk_item": fleiss_by_risk,
        },
        "risk_summary_classification": {
            "reference_file": str(args.risk_reference_csv) if risk_references else None,
            "overall": risk_overall_metrics if risk_rows else None,
            "by_risk_item": {
                risk_field: metric_from_counts(
                    counts["tp"], counts["fp"], counts["tn"], counts["fn"]
                )
                for risk_field, counts in risk_counts_by_field.items()
            }
            if risk_rows
            else {},
        },
        "key_factor_classification": {
            "overall": overall_metrics,
            "by_factor": by_factor,
        },
    }
    write_json(args.output_dir / "metrics.json", metrics_payload)

    with (args.output_dir / "key_factor_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "run_index",
                "factor",
                "prediction",
                "reference",
                "outcome",
                "file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if risk_rows:
        with (args.output_dir / "risk_summary_rows.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_id",
                    "run_index",
                    "risk_field",
                    "prediction",
                    "reference",
                    "outcome",
                    "file",
                ],
            )
            writer.writeheader()
            writer.writerows(risk_rows)

    print("Structured validation metrics")
    print(f"Risk-summary Fleiss kappa: {fleiss_overall}")
    if risk_rows:
        print(
            "Risk-summary GT comparison: "
            f"accuracy={risk_overall_metrics['accuracy']}, "
            f"precision={risk_overall_metrics['precision']}, "
            f"recall={risk_overall_metrics['recall']}, "
            f"f1={risk_overall_metrics['f1']}"
        )
    print(
        "Key-factor overall: "
        f"accuracy={overall_metrics['accuracy']}, "
        f"precision={overall_metrics['precision']}, "
        f"recall={overall_metrics['recall']}, "
        f"f1={overall_metrics['f1']}"
    )
    print(f"Metrics: {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
