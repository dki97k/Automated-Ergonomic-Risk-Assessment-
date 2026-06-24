#!/usr/bin/env python3
"""Evaluate deterministic metrics for Module #2 configuration analysis."""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M3_SRC = PROJECT_ROOT / "m3" / "src"
sys.path.insert(0, str(M3_SRC))

from evaluate_natural_overlap import (  # noqa: E402
    KEY_FACTOR_KEYWORDS,
    RECOMMENDATION_KEYWORDS,
    jaccard,
    tag_text,
)
from natural_common import NATURAL_SECTIONS, parse_sections, split_claims  # noqa: E402
from structured_common import (  # noqa: E402
    KEY_FACTOR_FIELDS,
    RISK_SUMMARY_FIELDS,
    build_key_factor_reference,
    read_json,
    write_json,
)


CONDITIONS = ("rgb_only", "reba_only", "full_module2")


def parse_args() -> argparse.Namespace:
    m2_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=m2_root)
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=m2_root / "results" / "generated_reports",
    )
    parser.add_argument(
        "--payload-root",
        type=Path,
        default=m2_root / "payloads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=m2_root / "results" / "evaluation",
    )
    parser.add_argument(
        "--force-annotation",
        action="store_true",
        help="Overwrite claim_annotation_sheet.csv if it already exists.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def report_json_files(generated_root: Path, report_type: str, condition: str) -> list[Path]:
    reports_dir = generated_root / report_type / condition / "reports"
    return sorted(
        path
        for path in reports_dir.glob("*/*.json")
        if path.is_file() and not path.name.startswith("._")
    )


def normalize_label(value: Any, allowed: set[str]) -> str | None:
    if not isinstance(value, str):
        return None
    mapping = {label.lower(): label for label in allowed}
    return mapping.get(value.strip().lower())


def nested_label(report: dict[str, Any], section: str, field: str, allowed: set[str]) -> str | None:
    try:
        value = report[section][field]["label"]
    except (KeyError, TypeError):
        return None
    return normalize_label(value, allowed)


def reference_labels_from_full_payloads(payload_root: Path) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    for path in sorted((payload_root / "full_module2").glob("*.json")):
        if path.name.startswith("._"):
            continue
        summary = read_json(path)
        case_id = summary.get("case_id") or path.stem
        reference_input = dict(summary)
        reference_input["sample_id"] = case_id
        reference = build_key_factor_reference(reference_input)
        references[case_id] = {
            field: reference["key_risk_factors"][field]["label"]
            for field in KEY_FACTOR_FIELDS
        }
    return references


def load_structured_reports(generated_root: Path) -> list[dict[str, Any]]:
    reports = []
    for condition in CONDITIONS:
        for path in report_json_files(generated_root, "structured", condition):
            payload = read_json(path)
            report = payload.get("report")
            if not isinstance(report, dict):
                continue
            case_id = payload.get("case_id") or payload.get("sample_id") or path.parent.name
            reports.append(
                {
                    "condition": condition,
                    "case_id": case_id,
                    "run_index": int(payload.get("run_index") or 0),
                    "report": report,
                    "file": str(path),
                }
            )
    return reports


def binary_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    answered = 0
    insufficient = 0
    correct = 0
    if not rows:
        return {
            "n_items": 0,
            "answered_items": 0,
            "insufficient_items": 0,
            "insufficient_evidence_rate": None,
            "answered_accuracy": None,
            "strict_accuracy": None,
            "precision_pred_yes": None,
            "recall_pred_yes": None,
            "f1_pred_yes": None,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }
    for row in rows:
        pred = row["prediction"]
        ref = row["reference"]
        if pred == "Insufficient evidence":
            insufficient += 1
        else:
            answered += 1
            if pred == ref:
                correct += 1

        pred_positive = pred == "Yes"
        ref_positive = ref == "Yes"
        if pred_positive and ref_positive:
            counts["tp"] += 1
        elif pred_positive and not ref_positive:
            counts["fp"] += 1
        elif not pred_positive and not ref_positive:
            counts["tn"] += 1
        else:
            counts["fn"] += 1

    total = len(rows)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_items": total,
        "answered_items": answered,
        "insufficient_items": insufficient,
        "insufficient_evidence_rate": round(insufficient / total, 4) if total else None,
        "answered_accuracy": round(correct / answered, 4) if answered else None,
        "strict_accuracy": round(correct / total, 4) if total else None,
        "precision_pred_yes": round(precision, 4),
        "recall_pred_yes": round(recall, 4),
        "f1_pred_yes": round(f1, 4),
        "tp": counts["tp"],
        "fp": counts["fp"],
        "tn": counts["tn"],
        "fn": counts["fn"],
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


def evaluate_structured(
    structured_reports: list[dict[str, Any]],
    references: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    key_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    label_maps: dict[tuple[str, str, int], dict[str, str]] = {}
    risk_labels_by_item: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    for item in structured_reports:
        condition = item["condition"]
        case_id = item["case_id"]
        run_index = item["run_index"]
        report = item["report"]
        reference = references.get(case_id)
        factor_map: dict[str, str] = {}

        for field in RISK_SUMMARY_FIELDS:
            label = nested_label(
                report,
                "risk_summary",
                field,
                {"High", "Low", "Insufficient evidence"},
            )
            risk_rows.append(
                {
                    "condition": condition,
                    "case_id": case_id,
                    "run_index": run_index,
                    "risk_item": field,
                    "prediction": label or "invalid",
                    "file": item["file"],
                }
            )
            if label is not None:
                risk_labels_by_item[(condition, case_id, field)].append(label)

        if reference is None:
            continue
        for field in KEY_FACTOR_FIELDS:
            pred = nested_label(
                report,
                "key_risk_factors",
                field,
                {"Yes", "No", "Insufficient evidence"},
            )
            pred = pred or "invalid"
            factor_map[field] = pred
            key_rows.append(
                {
                    "condition": condition,
                    "case_id": case_id,
                    "run_index": run_index,
                    "factor": field,
                    "prediction": pred,
                    "reference": reference[field],
                    "correct": pred == reference[field],
                    "insufficient": pred == "Insufficient evidence",
                    "file": item["file"],
                }
            )
        label_maps[(condition, case_id, run_index)] = factor_map

    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = [row for row in key_rows if row["condition"] == condition]
        condition_metrics[condition] = binary_metrics(rows)
        risk_condition_rows = [row for row in risk_rows if row["condition"] == condition]
        risk_total = len(risk_condition_rows)
        risk_insufficient = sum(
            1 for row in risk_condition_rows if row["prediction"] == "Insufficient evidence"
        )
        condition_metrics[condition]["risk_summary_insufficient_evidence_rate"] = (
            round(risk_insufficient / risk_total, 4) if risk_total else None
        )
        condition_metrics[condition]["risk_summary_fleiss_kappa"] = fleiss_kappa(
            [
                labels
                for (item_condition, _case_id, _field), labels in risk_labels_by_item.items()
                if item_condition == condition
            ],
            ("High", "Low", "Insufficient evidence"),
        )
        condition_metrics[condition]["structured_report_count"] = len(
            {(
                row["case_id"],
                row["run_index"],
            ) for row in rows}
        )

    change_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for source_condition in ("rgb_only", "reba_only"):
        for (condition, case_id, run_index), source_map in sorted(label_maps.items()):
            if condition != source_condition:
                continue
            full_map = label_maps.get(("full_module2", case_id, run_index))
            if full_map is None:
                continue
            source_yes = {field for field, label in source_map.items() if label == "Yes"}
            full_yes = {field for field, label in full_map.items() if label == "Yes"}
            overlap_rows.append(
                {
                    "comparison": f"{source_condition}_vs_full_module2",
                    "case_id": case_id,
                    "run_index": run_index,
                    "key_factor_positive_jaccard": round(jaccard(source_yes, full_yes), 4),
                    "source_yes": "|".join(sorted(source_yes)),
                    "full_yes": "|".join(sorted(full_yes)),
                }
            )
            for factor in KEY_FACTOR_FIELDS:
                source_label = source_map.get(factor, "invalid")
                full_label = full_map.get(factor, "invalid")
                change_rows.append(
                    {
                        "comparison": f"{source_condition}_vs_full_module2",
                        "case_id": case_id,
                        "run_index": run_index,
                        "factor": factor,
                        "source_label": source_label,
                        "full_module2_label": full_label,
                        "changed": source_label != full_label,
                        "source_insufficient": source_label == "Insufficient evidence",
                        "full_insufficient": full_label == "Insufficient evidence",
                    }
                )

    csv_write(
        output_dir / "structured_key_factor_rows.csv",
        key_rows,
        [
            "condition",
            "case_id",
            "run_index",
            "factor",
            "prediction",
            "reference",
            "correct",
            "insufficient",
            "file",
        ],
    )
    csv_write(
        output_dir / "structured_risk_summary_rows.csv",
        risk_rows,
        ["condition", "case_id", "run_index", "risk_item", "prediction", "file"],
    )
    csv_write(
        output_dir / "structured_label_changes_vs_full.csv",
        change_rows,
        [
            "comparison",
            "case_id",
            "run_index",
            "factor",
            "source_label",
            "full_module2_label",
            "changed",
            "source_insufficient",
            "full_insufficient",
        ],
    )
    csv_write(
        output_dir / "structured_positive_overlap_vs_full.csv",
        overlap_rows,
        ["comparison", "case_id", "run_index", "key_factor_positive_jaccard", "source_yes", "full_yes"],
    )
    write_json(output_dir / "structured_metrics.json", condition_metrics)
    return condition_metrics


def load_natural_reports(generated_root: Path) -> list[dict[str, Any]]:
    reports = []
    for condition in CONDITIONS:
        for path in report_json_files(generated_root, "natural", condition):
            payload = read_json(path)
            report_text = payload.get("report_text")
            if not isinstance(report_text, str) or not report_text.strip():
                continue
            case_id = payload.get("case_id") or payload.get("sample_id") or path.parent.name
            sections = payload.get("sections")
            if not isinstance(sections, dict):
                sections = parse_sections(report_text)
            reports.append(
                {
                    "condition": condition,
                    "case_id": case_id,
                    "run_index": int(payload.get("run_index") or 0),
                    "report_text": report_text,
                    "sections": sections,
                    "file": str(path),
                    "input_file": payload.get("input_file", ""),
                }
            )
    return reports


def natural_tags(report: dict[str, Any]) -> dict[str, Any]:
    sections = report["sections"]
    key_text = sections.get("Key Contributing Factors", "")
    rec_text = sections.get("Recommendations", "")
    recommendation_tags: set[str] = set()
    for claim in split_claims(rec_text):
        recommendation_tags.update(tag_text(claim, RECOMMENDATION_KEYWORDS))
    return {
        "key_factor_tags": tag_text(key_text, KEY_FACTOR_KEYWORDS),
        "recommendation_tags": recommendation_tags,
    }


def evaluate_natural(natural_reports: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    tagged_reports = []
    for report in natural_reports:
        tags = natural_tags(report)
        tagged_reports.append({**report, **tags})

    report_tag_rows = [
        {
            "condition": report["condition"],
            "case_id": report["case_id"],
            "run_index": report["run_index"],
            "key_factor_tags": "|".join(sorted(report["key_factor_tags"])),
            "recommendation_tags": "|".join(sorted(report["recommendation_tags"])),
            "file": report["file"],
        }
        for report in tagged_reports
    ]

    within_rows: list[dict[str, Any]] = []
    by_condition_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in tagged_reports:
        by_condition_case[(report["condition"], report["case_id"])].append(report)
    for (condition, case_id), rows in sorted(by_condition_case.items()):
        for left, right in itertools.combinations(sorted(rows, key=lambda r: r["run_index"]), 2):
            within_rows.append(
                {
                    "condition": condition,
                    "case_id": case_id,
                    "left_run": left["run_index"],
                    "right_run": right["run_index"],
                    "key_factor_jaccard": round(
                        jaccard(left["key_factor_tags"], right["key_factor_tags"]), 4
                    ),
                    "recommendation_jaccard": round(
                        jaccard(left["recommendation_tags"], right["recommendation_tags"]), 4
                    ),
                }
            )

    between_rows: list[dict[str, Any]] = []
    by_case_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in tagged_reports:
        by_case_condition[(report["case_id"], report["condition"])].append(report)
    for source_condition in ("rgb_only", "reba_only"):
        for case_id in sorted({report["case_id"] for report in tagged_reports}):
            source_reports = by_case_condition.get((case_id, source_condition), [])
            full_reports = by_case_condition.get((case_id, "full_module2"), [])
            for source, full in itertools.product(source_reports, full_reports):
                between_rows.append(
                    {
                        "comparison": f"{source_condition}_vs_full_module2",
                        "case_id": case_id,
                        "source_run": source["run_index"],
                        "full_run": full["run_index"],
                        "key_factor_jaccard": round(
                            jaccard(source["key_factor_tags"], full["key_factor_tags"]), 4
                        ),
                        "recommendation_jaccard": round(
                            jaccard(source["recommendation_tags"], full["recommendation_tags"]), 4
                        ),
                        "source_key_tags": "|".join(sorted(source["key_factor_tags"])),
                        "full_key_tags": "|".join(sorted(full["key_factor_tags"])),
                        "source_recommendation_tags": "|".join(sorted(source["recommendation_tags"])),
                        "full_recommendation_tags": "|".join(sorted(full["recommendation_tags"])),
                    }
                )

    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = [row for row in within_rows if row["condition"] == condition]
        condition_reports = [row for row in tagged_reports if row["condition"] == condition]
        condition_metrics[condition] = {
            "natural_report_count": len(condition_reports),
            "within_condition_pair_count": len(rows),
            "within_key_factor_jaccard_mean": mean(
                [float(row["key_factor_jaccard"]) for row in rows]
            ),
            "within_recommendation_jaccard_mean": mean(
                [float(row["recommendation_jaccard"]) for row in rows]
            ),
        }

    comparison_metrics: dict[str, dict[str, Any]] = {}
    for comparison in sorted({row["comparison"] for row in between_rows}):
        rows = [row for row in between_rows if row["comparison"] == comparison]
        comparison_metrics[comparison] = {
            "pair_count": len(rows),
            "key_factor_jaccard_vs_full_mean": mean(
                [float(row["key_factor_jaccard"]) for row in rows]
            ),
            "recommendation_jaccard_vs_full_mean": mean(
                [float(row["recommendation_jaccard"]) for row in rows]
            ),
        }

    csv_write(
        output_dir / "natural_report_tags.csv",
        report_tag_rows,
        ["condition", "case_id", "run_index", "key_factor_tags", "recommendation_tags", "file"],
    )
    csv_write(
        output_dir / "natural_within_condition_overlap.csv",
        within_rows,
        [
            "condition",
            "case_id",
            "left_run",
            "right_run",
            "key_factor_jaccard",
            "recommendation_jaccard",
        ],
    )
    csv_write(
        output_dir / "natural_overlap_vs_full.csv",
        between_rows,
        [
            "comparison",
            "case_id",
            "source_run",
            "full_run",
            "key_factor_jaccard",
            "recommendation_jaccard",
            "source_key_tags",
            "full_key_tags",
            "source_recommendation_tags",
            "full_recommendation_tags",
        ],
    )
    payload = {
        "by_condition": condition_metrics,
        "vs_full": comparison_metrics,
    }
    write_json(output_dir / "natural_overlap_metrics.json", payload)
    return payload


def build_claim_annotation_sheet(
    natural_reports: list[dict[str, Any]],
    output_dir: Path,
    force_annotation: bool,
) -> Path:
    annotation_csv = output_dir / "claim_annotation_sheet.csv"
    if annotation_csv.exists() and not force_annotation:
        return annotation_csv

    rows: list[dict[str, Any]] = []
    for report in natural_reports:
        for section in NATURAL_SECTIONS:
            claims = split_claims(report["sections"].get(section, ""))
            for claim_index, claim_text in enumerate(claims, start=1):
                claim_id = (
                    f"{report['condition']}__{report['case_id']}__"
                    f"run_{report['run_index']:02d}__"
                    f"{section.lower().replace(' ', '_')}__claim_{claim_index:02d}"
                )
                rows.append(
                    {
                        "claim_id": claim_id,
                        "condition": report["condition"],
                        "case_id": report["case_id"],
                        "run_index": report["run_index"],
                        "section": section,
                        "claim_index": claim_index,
                        "claim_text": claim_text,
                        "support_label": "",
                        "allowed_labels": "supported|unsupported|contradiction",
                        "evidence_note": "",
                        "reviewer_note": "",
                        "report_file": report["file"],
                        "input_file": report["input_file"],
                    }
                )
    csv_write(
        annotation_csv,
        rows,
        [
            "claim_id",
            "condition",
            "case_id",
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
        ],
    )
    return annotation_csv


def claim_support_metrics(annotation_csv: Path, output_dir: Path) -> dict[str, Any]:
    valid_labels = {"supported", "unsupported", "contradiction"}
    overall = Counter()
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    unlabeled = 0
    if not annotation_csv.exists():
        return {}
    with annotation_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = (row.get("support_label") or "").strip().lower()
            if not label:
                unlabeled += 1
                continue
            if label not in valid_labels:
                continue
            overall[label] += 1
            by_condition[row["condition"]][label] += 1

    def rates(counter: Counter[str]) -> dict[str, Any]:
        total = sum(counter[label] for label in valid_labels)
        return {
            "labeled_claims": total,
            "supported_claim_rate": round(counter["supported"] / total, 4) if total else None,
            "unsupported_claim_rate": round(counter["unsupported"] / total, 4) if total else None,
            "contradiction_rate": round(counter["contradiction"] / total, 4) if total else None,
        }

    metrics = {
        "unlabeled_claims": unlabeled,
        "overall": rates(overall),
        "by_condition": {
            condition: rates(by_condition[condition]) for condition in CONDITIONS
        },
    }
    write_json(output_dir / "claim_support_metrics.json", metrics)
    rows = [
        {"group_type": "overall", "group": "overall", **metrics["overall"]},
        *[
            {"group_type": "condition", "group": condition, **values}
            for condition, values in metrics["by_condition"].items()
        ],
    ]
    csv_write(
        output_dir / "claim_support_metrics.csv",
        rows,
        [
            "group_type",
            "group",
            "labeled_claims",
            "supported_claim_rate",
            "unsupported_claim_rate",
            "contradiction_rate",
        ],
    )
    return metrics


def build_summary_table(
    structured_metrics: dict[str, dict[str, Any]],
    natural_metrics: dict[str, Any],
    claim_metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    natural_by_condition = natural_metrics.get("by_condition", {})
    claim_by_condition = claim_metrics.get("by_condition", {})
    for condition in CONDITIONS:
        structured = structured_metrics.get(condition, {})
        natural = natural_by_condition.get(condition, {})
        claims = claim_by_condition.get(condition, {})
        rows.append(
            {
                "condition": condition,
                "structured_report_count": structured.get("structured_report_count"),
                "risk_summary_fleiss_kappa": structured.get("risk_summary_fleiss_kappa"),
                "key_factor_strict_accuracy": structured.get("strict_accuracy"),
                "key_factor_answered_accuracy": structured.get("answered_accuracy"),
                "key_factor_insufficient_evidence_rate": structured.get(
                    "insufficient_evidence_rate"
                ),
                "risk_summary_insufficient_evidence_rate": structured.get(
                    "risk_summary_insufficient_evidence_rate"
                ),
                "key_factor_recall_pred_yes": structured.get("recall_pred_yes"),
                "key_factor_f1_pred_yes": structured.get("f1_pred_yes"),
                "natural_report_count": natural.get("natural_report_count"),
                "within_key_factor_jaccard_mean": natural.get(
                    "within_key_factor_jaccard_mean"
                ),
                "within_recommendation_jaccard_mean": natural.get(
                    "within_recommendation_jaccard_mean"
                ),
                "labeled_claims": claims.get("labeled_claims"),
                "supported_claim_rate": claims.get("supported_claim_rate"),
                "unsupported_claim_rate": claims.get("unsupported_claim_rate"),
                "contradiction_rate": claims.get("contradiction_rate"),
            }
        )
    csv_write(
        output_dir / "configuration_summary.csv",
        rows,
        [
            "condition",
            "structured_report_count",
            "risk_summary_fleiss_kappa",
            "key_factor_strict_accuracy",
            "key_factor_answered_accuracy",
            "key_factor_insufficient_evidence_rate",
            "risk_summary_insufficient_evidence_rate",
            "key_factor_recall_pred_yes",
            "key_factor_f1_pred_yes",
            "natural_report_count",
            "within_key_factor_jaccard_mean",
            "within_recommendation_jaccard_mean",
            "labeled_claims",
            "supported_claim_rate",
            "unsupported_claim_rate",
            "contradiction_rate",
        ],
    )
    write_json(output_dir / "configuration_summary.json", rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    references = reference_labels_from_full_payloads(args.payload_root)
    structured_reports = load_structured_reports(args.generated_root)
    natural_reports = load_natural_reports(args.generated_root)

    structured_metrics = evaluate_structured(
        structured_reports,
        references,
        args.output_dir,
    )
    natural_metrics = evaluate_natural(natural_reports, args.output_dir)
    annotation_csv = build_claim_annotation_sheet(
        natural_reports,
        args.output_dir,
        args.force_annotation,
    )
    claim_metrics = claim_support_metrics(annotation_csv, args.output_dir)
    build_summary_table(structured_metrics, natural_metrics, claim_metrics, args.output_dir)

    print("Configuration deterministic evaluation complete.")
    print(f"Structured reports: {len(structured_reports)}")
    print(f"Natural reports: {len(natural_reports)}")
    print(f"Evaluation directory: {args.output_dir}")
    print(f"Claim annotation sheet: {annotation_csv}")


if __name__ == "__main__":
    main()
