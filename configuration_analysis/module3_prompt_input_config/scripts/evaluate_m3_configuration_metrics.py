#!/usr/bin/env python3
"""Evaluate deterministic metrics for Module #3 configuration analysis."""

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
if M3_SRC.exists():
    sys.path.insert(0, str(M3_SRC))
else:
    sys.path.insert(0, str(Path("<private_workspace>/m3/src")))

from evaluate_natural_overlap import (  # noqa: E402
    KEY_FACTOR_KEYWORDS,
    RECOMMENDATION_KEYWORDS,
    jaccard,
    tag_text,
)
from natural_common import NATURAL_SECTIONS, parse_sections, split_claims  # noqa: E402
from structured_common import read_json, write_json  # noqa: E402


INPUT_CONDITIONS = ("module2_only", "module2_rgb")
PROMPT_CONDITIONS = (
    "neutral",
    "original_rgb_compatible",
    "bounded_context_augmented",
)


def parse_args() -> argparse.Namespace:
    default_m3_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path, default=default_m3_root)
    parser.add_argument(
        "--generated-root",
        type=Path,
        help="Defaults to <m3-root>/results/generated_reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <m3-root>/results/evaluation.",
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


def report_json_files(generated_root: Path, input_condition: str, prompt_condition: str) -> list[Path]:
    reports_dir = generated_root / "natural" / input_condition / prompt_condition / "reports"
    return sorted(
        path
        for path in reports_dir.glob("*/*.json")
        if path.is_file() and not path.name.startswith("._")
    )


def load_natural_reports(generated_root: Path) -> list[dict[str, Any]]:
    reports = []
    for input_condition in INPUT_CONDITIONS:
        for prompt_condition in PROMPT_CONDITIONS:
            for path in report_json_files(generated_root, input_condition, prompt_condition):
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
                        "condition": f"{input_condition}__{prompt_condition}",
                        "input_condition": input_condition,
                        "prompt_condition": prompt_condition,
                        "case_id": case_id,
                        "run_index": int(payload.get("run_index") or 0),
                        "report_text": report_text,
                        "sections": sections,
                        "file": str(path),
                        "input_file": payload.get("input_file", ""),
                        "image_file": payload.get("image_file", ""),
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


def evaluate_overlap(natural_reports: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    tagged_reports = []
    for report in natural_reports:
        tagged_reports.append({**report, **natural_tags(report)})

    tag_rows = [
        {
            "condition": report["condition"],
            "input_condition": report["input_condition"],
            "prompt_condition": report["prompt_condition"],
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
                    "input_condition": left["input_condition"],
                    "prompt_condition": left["prompt_condition"],
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

    case_ids = sorted({report["case_id"] for report in tagged_reports})
    for prompt_condition in PROMPT_CONDITIONS:
        left_condition = f"module2_only__{prompt_condition}"
        right_condition = f"module2_rgb__{prompt_condition}"
        for case_id in case_ids:
            for left, right in itertools.product(
                by_case_condition.get((case_id, left_condition), []),
                by_case_condition.get((case_id, right_condition), []),
            ):
                between_rows.append(
                    {
                        "comparison_type": "input_effect",
                        "comparison": f"{left_condition}_with_{right_condition}",
                        "case_id": case_id,
                        "left_run": left["run_index"],
                        "right_run": right["run_index"],
                        "key_factor_jaccard": round(
                            jaccard(left["key_factor_tags"], right["key_factor_tags"]), 4
                        ),
                        "recommendation_jaccard": round(
                            jaccard(left["recommendation_tags"], right["recommendation_tags"]), 4
                        ),
                        "left_key_tags": "|".join(sorted(left["key_factor_tags"])),
                        "right_key_tags": "|".join(sorted(right["key_factor_tags"])),
                        "left_recommendation_tags": "|".join(sorted(left["recommendation_tags"])),
                        "right_recommendation_tags": "|".join(sorted(right["recommendation_tags"])),
                    }
                )

    prompt_pairs = (
        ("neutral", "original_rgb_compatible"),
        ("original_rgb_compatible", "bounded_context_augmented"),
        ("neutral", "bounded_context_augmented"),
    )
    for input_condition in INPUT_CONDITIONS:
        for left_prompt, right_prompt in prompt_pairs:
            left_condition = f"{input_condition}__{left_prompt}"
            right_condition = f"{input_condition}__{right_prompt}"
            for case_id in case_ids:
                for left, right in itertools.product(
                    by_case_condition.get((case_id, left_condition), []),
                    by_case_condition.get((case_id, right_condition), []),
                ):
                    between_rows.append(
                        {
                            "comparison_type": "prompt_effect",
                            "comparison": f"{left_condition}_with_{right_condition}",
                            "case_id": case_id,
                            "left_run": left["run_index"],
                            "right_run": right["run_index"],
                            "key_factor_jaccard": round(
                                jaccard(left["key_factor_tags"], right["key_factor_tags"]), 4
                            ),
                            "recommendation_jaccard": round(
                                jaccard(left["recommendation_tags"], right["recommendation_tags"]), 4
                            ),
                            "left_key_tags": "|".join(sorted(left["key_factor_tags"])),
                            "right_key_tags": "|".join(sorted(right["key_factor_tags"])),
                            "left_recommendation_tags": "|".join(sorted(left["recommendation_tags"])),
                            "right_recommendation_tags": "|".join(sorted(right["recommendation_tags"])),
                        }
                    )

    condition_metrics: dict[str, dict[str, Any]] = {}
    for input_condition in INPUT_CONDITIONS:
        for prompt_condition in PROMPT_CONDITIONS:
            condition = f"{input_condition}__{prompt_condition}"
            rows = [row for row in within_rows if row["condition"] == condition]
            reports = [row for row in tagged_reports if row["condition"] == condition]
            condition_metrics[condition] = {
                "input_condition": input_condition,
                "prompt_condition": prompt_condition,
                "natural_report_count": len(reports),
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
            "comparison_type": rows[0]["comparison_type"] if rows else "",
            "pair_count": len(rows),
            "key_factor_jaccard_mean": mean(
                [float(row["key_factor_jaccard"]) for row in rows]
            ),
            "recommendation_jaccard_mean": mean(
                [float(row["recommendation_jaccard"]) for row in rows]
            ),
        }

    csv_write(
        output_dir / "natural_report_tags.csv",
        tag_rows,
        [
            "condition",
            "input_condition",
            "prompt_condition",
            "case_id",
            "run_index",
            "key_factor_tags",
            "recommendation_tags",
            "file",
        ],
    )
    csv_write(
        output_dir / "natural_within_condition_overlap.csv",
        within_rows,
        [
            "condition",
            "input_condition",
            "prompt_condition",
            "case_id",
            "left_run",
            "right_run",
            "key_factor_jaccard",
            "recommendation_jaccard",
        ],
    )
    csv_write(
        output_dir / "natural_cross_condition_overlap.csv",
        between_rows,
        [
            "comparison_type",
            "comparison",
            "case_id",
            "left_run",
            "right_run",
            "key_factor_jaccard",
            "recommendation_jaccard",
            "left_key_tags",
            "right_key_tags",
            "left_recommendation_tags",
            "right_recommendation_tags",
        ],
    )

    metrics = {
        "by_condition": condition_metrics,
        "cross_condition": comparison_metrics,
    }
    write_json(output_dir / "natural_overlap_metrics.json", metrics)
    return metrics


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
                        "input_condition": report["input_condition"],
                        "prompt_condition": report["prompt_condition"],
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
                        "image_file": report["image_file"],
                    }
                )

    csv_write(
        annotation_csv,
        rows,
        [
            "claim_id",
            "condition",
            "input_condition",
            "prompt_condition",
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
            "image_file",
        ],
    )
    return annotation_csv


def claim_support_metrics(annotation_csv: Path, output_dir: Path) -> dict[str, Any]:
    valid_labels = {"supported", "unsupported", "contradiction"}
    overall = Counter()
    by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    by_input: dict[str, Counter[str]] = defaultdict(Counter)
    by_prompt: dict[str, Counter[str]] = defaultdict(Counter)
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
            by_input[row["input_condition"]][label] += 1
            by_prompt[row["prompt_condition"]][label] += 1

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
            condition: rates(by_condition[condition])
            for condition in sorted(by_condition)
        },
        "by_input_condition": {
            condition: rates(by_input[condition])
            for condition in sorted(by_input)
        },
        "by_prompt_condition": {
            condition: rates(by_prompt[condition])
            for condition in sorted(by_prompt)
        },
    }
    write_json(output_dir / "claim_support_metrics.json", metrics)

    rows = [{"group_type": "overall", "group": "overall", **metrics["overall"]}]
    rows.extend(
        {"group_type": "condition", "group": condition, **values}
        for condition, values in metrics["by_condition"].items()
    )
    rows.extend(
        {"group_type": "input_condition", "group": condition, **values}
        for condition, values in metrics["by_input_condition"].items()
    )
    rows.extend(
        {"group_type": "prompt_condition", "group": condition, **values}
        for condition, values in metrics["by_prompt_condition"].items()
    )
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
    natural_metrics: dict[str, Any],
    claim_metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    by_condition = natural_metrics.get("by_condition", {})
    claim_by_condition = claim_metrics.get("by_condition", {})
    for input_condition in INPUT_CONDITIONS:
        for prompt_condition in PROMPT_CONDITIONS:
            condition = f"{input_condition}__{prompt_condition}"
            natural = by_condition.get(condition, {})
            claims = claim_by_condition.get(condition, {})
            rows.append(
                {
                    "condition": condition,
                    "input_condition": input_condition,
                    "prompt_condition": prompt_condition,
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
            "input_condition",
            "prompt_condition",
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
    generated_root = args.generated_root or args.m3_root / "results" / "generated_reports"
    output_dir = args.output_dir or args.m3_root / "results" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    natural_reports = load_natural_reports(generated_root)
    natural_metrics = evaluate_overlap(natural_reports, output_dir)
    annotation_csv = build_claim_annotation_sheet(
        natural_reports,
        output_dir,
        args.force_annotation,
    )
    claim_metrics = claim_support_metrics(annotation_csv, output_dir)
    build_summary_table(natural_metrics, claim_metrics, output_dir)

    print("Module #3 configuration deterministic evaluation complete.")
    print(f"Natural reports: {len(natural_reports)}")
    print(f"Evaluation directory: {output_dir}")
    print(f"Claim annotation sheet: {annotation_csv}")


if __name__ == "__main__":
    main()
