#!/usr/bin/env python3
"""Evaluate structured Module #3 configuration reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M3_SRC = PROJECT_ROOT / "m3" / "src"
if M3_SRC.exists():
    sys.path.insert(0, str(M3_SRC))
else:
    sys.path.insert(0, str(Path("<private_workspace>/m3/src")))

from structured_common import KEY_FACTOR_FIELDS, RISK_SUMMARY_FIELDS, read_json, write_json  # noqa: E402


INPUT_CONDITIONS = ("module2_only", "module2_rgb")
VIDEO_ID_MAP = {
    "V1": "MansoryBrickLaying_00",
    "V2": "MansoryBrickLaying_01",
    "V3": "MansoryBrickLaying_02",
    "V4": "MansoryCement_02",
    "V5": "RebarPlacement_00",
    "V6": "RebarTying_01",
    "V7": "RebarTying_02",
    "V8": "WallPlacement_00",
}
PART_TO_FACTOR = {
    "Trunk": "trunk_overflexion",
    "Neck": "neck_overflexion_or_extension",
    "Upper_arm": "upper_arm_elevation",
    "Wrist": "wrist_deviation",
    "Knee": "knee_overflexion",
    "Static_posture": "prolonged_static_posture",
    "Repetitive_activity": "repetitive_work",
}


def parse_args() -> argparse.Namespace:
    default_m3_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path, default=default_m3_root)
    parser.add_argument(
        "--reports-root",
        type=Path,
        help="Defaults to <m3-root>/results/generated_reports/structured.",
    )
    parser.add_argument(
        "--key-gt-csv",
        type=Path,
        default=Path("<private_workspace>/m3/results_structured_key.csv"),
        help="Structured key-factor GT CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <m3-root>/results/structured_evaluation.",
    )
    return parser.parse_args()


def normalize_binary(value: Any, allowed: tuple[str, str]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    mapping = {allowed[0].lower(): allowed[0], allowed[1].lower(): allowed[1]}
    return mapping.get(normalized)


def csv_binary(value: Any, positive: str, negative: str) -> str:
    normalized = str(value).strip()
    if normalized in {"1", "1.0", positive, positive.lower(), "Y", "y", "Yes", "yes"}:
        return positive
    if normalized in {"0", "0.0", negative, negative.lower(), "N", "n", "No", "no"}:
        return negative
    raise ValueError(f"Unsupported binary value: {value!r}")


def get_label(report: dict[str, Any], section: str, field: str, allowed: tuple[str, str]) -> str | None:
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


def fleiss_kappa(label_sets: list[list[str]], categories: tuple[str, str]) -> float | None:
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


def load_key_gt(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {sample_id: {} for sample_id in VIDEO_ID_MAP.values()}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            video_code = row["VIDEO"].strip()
            part = row["Part"].strip()
            if video_code not in VIDEO_ID_MAP or part not in PART_TO_FACTOR:
                continue
            sample_id = VIDEO_ID_MAP[video_code]
            factor = PART_TO_FACTOR[part]
            labels[sample_id][factor] = csv_binary(row["GT"], "Yes", "No")

    for sample_id, sample_labels in labels.items():
        missing = [factor for factor in KEY_FACTOR_FIELDS if factor not in sample_labels]
        if missing:
            raise ValueError(f"Missing GT labels for {sample_id}: {missing}")
    return labels


def load_case_mapping(m3_root: Path) -> dict[str, str]:
    manifest_path = m3_root / "payloads" / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    mapping: dict[str, str] = {}
    for case in manifest.get("cases", []):
        case_id = case.get("case_id")
        source_sample_id = case.get("source_sample_id")
        if isinstance(case_id, str) and isinstance(source_sample_id, str):
            mapping[case_id] = source_sample_id
    return mapping


def report_files(reports_root: Path, input_condition: str) -> list[Path]:
    reports_dir = reports_root / input_condition / "reports"
    return sorted(
        path
        for path in reports_dir.glob("*/*.json")
        if path.is_file() and not path.name.startswith("._")
    )


def count_prediction(counts: Counter[str], pred: str, ref: str) -> str:
    if pred == "Yes" and ref == "Yes":
        outcome = "tp"
    elif pred == "Yes" and ref == "No":
        outcome = "fp"
    elif pred == "No" and ref == "No":
        outcome = "tn"
    else:
        outcome = "fn"
    counts[outcome] += 1
    return outcome


def evaluate_condition(
    *,
    input_condition: str,
    files: list[Path],
    references: dict[str, dict[str, str]],
    case_mapping: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    risk_rows = []
    counts_by_factor = defaultdict(lambda: Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0}))
    counts_total = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    risk_labels_by_item = defaultdict(list)
    invalid_entries = []

    for path in files:
        payload = read_json(path)
        case_id = payload.get("case_id") or path.parent.name
        sample_id = payload.get("sample_id") or case_mapping.get(case_id, case_id)
        run_index = payload.get("run_index")
        report = payload.get("report")
        if not isinstance(report, dict):
            invalid_entries.append(str(path))
            continue

        for risk_field in RISK_SUMMARY_FIELDS:
            label = get_label(report, "risk_summary", risk_field, ("High", "Low"))
            if label is None:
                invalid_entries.append(f"{path}:{risk_field}")
                continue
            risk_labels_by_item[(sample_id, risk_field)].append(label)
            risk_rows.append(
                {
                    "input_condition": input_condition,
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "run_index": run_index,
                    "risk_field": risk_field,
                    "prediction": label,
                    "file": str(path),
                }
            )

        reference = references.get(sample_id)
        if not reference:
            invalid_entries.append(f"{path}:missing_reference:{sample_id}")
            continue
        for factor in KEY_FACTOR_FIELDS:
            pred = get_label(report, "key_risk_factors", factor, ("Yes", "No"))
            ref = reference[factor]
            if pred is None:
                invalid_entries.append(f"{path}:{factor}")
                continue
            outcome = count_prediction(counts_total, pred, ref)
            count_prediction(counts_by_factor[factor], pred, ref)
            rows.append(
                {
                    "input_condition": input_condition,
                    "case_id": case_id,
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
    by_factor = {
        factor: metric_from_counts(counts["tp"], counts["fp"], counts["tn"], counts["fn"])
        for factor, counts in counts_by_factor.items()
    }
    fleiss_overall = fleiss_kappa(list(risk_labels_by_item.values()), ("High", "Low"))
    fleiss_by_risk = {
        risk_field: fleiss_kappa(
            [
                labels
                for (_sample_id, field), labels in risk_labels_by_item.items()
                if field == risk_field
            ],
            ("High", "Low"),
        )
        for risk_field in RISK_SUMMARY_FIELDS
    }

    metrics = {
        "input_condition": input_condition,
        "report_file_count": len(files),
        "valid_key_factor_rows": len(rows),
        "invalid_entries": invalid_entries,
        "risk_summary_consistency": {
            "fleiss_kappa_overall": fleiss_overall,
            "fleiss_kappa_by_risk_item": fleiss_by_risk,
        },
        "key_factor_classification": {
            "overall": overall_metrics,
            "by_factor": by_factor,
        },
    }
    return metrics, rows, risk_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    reports_root = args.reports_root or args.m3_root / "results" / "generated_reports" / "structured"
    output_dir = args.output_dir or args.m3_root / "results" / "structured_evaluation"
    references = load_key_gt(args.key_gt_csv)
    case_mapping = load_case_mapping(args.m3_root)

    condition_metrics = []
    all_key_rows = []
    all_risk_rows = []
    for input_condition in INPUT_CONDITIONS:
        files = report_files(reports_root, input_condition)
        if not files:
            raise SystemExit(f"No structured report files found for {input_condition}: {reports_root}")
        metrics, key_rows, risk_rows = evaluate_condition(
            input_condition=input_condition,
            files=files,
            references=references,
            case_mapping=case_mapping,
        )
        condition_metrics.append(metrics)
        all_key_rows.extend(key_rows)
        all_risk_rows.extend(risk_rows)

    summary_rows = []
    for metrics in condition_metrics:
        key = metrics["key_factor_classification"]["overall"]
        risk = metrics["risk_summary_consistency"]
        summary_rows.append(
            {
                "input_condition": metrics["input_condition"],
                "report_file_count": metrics["report_file_count"],
                "risk_summary_fleiss_kappa": risk["fleiss_kappa_overall"],
                "posture_risk_fleiss_kappa": risk["fleiss_kappa_by_risk_item"].get("posture_risk"),
                "duration_risk_fleiss_kappa": risk["fleiss_kappa_by_risk_item"].get("duration_risk"),
                "repetition_risk_fleiss_kappa": risk["fleiss_kappa_by_risk_item"].get("repetition_risk"),
                "key_factor_accuracy": key["accuracy"],
                "key_factor_precision": key["precision"],
                "key_factor_recall": key["recall"],
                "key_factor_f1": key["f1"],
                "key_factor_n": key["n"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "structured_configuration_metrics.json",
        {
            "analysis": "module3_structured_configuration",
            "reports_root": str(reports_root),
            "key_gt_csv": str(args.key_gt_csv),
            "metrics_by_input_condition": condition_metrics,
            "summary_table": summary_rows,
        },
    )
    write_csv(
        output_dir / "structured_configuration_summary.csv",
        summary_rows,
        [
            "input_condition",
            "report_file_count",
            "risk_summary_fleiss_kappa",
            "posture_risk_fleiss_kappa",
            "duration_risk_fleiss_kappa",
            "repetition_risk_fleiss_kappa",
            "key_factor_accuracy",
            "key_factor_precision",
            "key_factor_recall",
            "key_factor_f1",
            "key_factor_n",
        ],
    )
    write_csv(
        output_dir / "structured_key_factor_rows.csv",
        all_key_rows,
        [
            "input_condition",
            "case_id",
            "sample_id",
            "run_index",
            "factor",
            "prediction",
            "reference",
            "outcome",
            "file",
        ],
    )
    write_csv(
        output_dir / "structured_risk_summary_rows.csv",
        all_risk_rows,
        [
            "input_condition",
            "case_id",
            "sample_id",
            "run_index",
            "risk_field",
            "prediction",
            "file",
        ],
    )

    print("Structured configuration metrics complete.")
    for row in summary_rows:
        print(
            f"{row['input_condition']}: "
            f"kappa={row['risk_summary_fleiss_kappa']}, "
            f"accuracy={row['key_factor_accuracy']}, "
            f"precision={row['key_factor_precision']}, "
            f"recall={row['key_factor_recall']}, "
            f"f1={row['key_factor_f1']}"
        )
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
