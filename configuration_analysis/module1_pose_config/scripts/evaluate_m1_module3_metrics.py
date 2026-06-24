#!/usr/bin/env python3
"""Evaluate Module 3 downstream changes for Module 1 configuration analysis."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


CONDITIONS = ("alphapose_motionbert", "sam3db")
RISK_SUMMARY_FIELDS = ("posture_risk", "duration_risk", "repetition_risk")
KEY_FACTOR_FIELDS = (
    "trunk_overflexion",
    "neck_overflexion_or_extension",
    "upper_arm_elevation",
    "wrist_deviation",
    "knee_overflexion",
    "prolonged_static_posture",
    "repetitive_work",
)
NATURAL_SECTIONS = ("Risk Interpretation", "Key Contributing Factors", "Recommendations")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
M3_SRC = PROJECT_ROOT / "m3" / "src"
sys.path.insert(0, str(M3_SRC))

try:
    from evaluate_natural_overlap import KEY_FACTOR_KEYWORDS, RECOMMENDATION_KEYWORDS, tag_text
    from natural_common import parse_sections, split_claims
except Exception:  # pragma: no cover - fallback keeps the metric script self-contained.
    KEY_FACTOR_KEYWORDS = {
        "trunk": ("trunk", "back", "torso"),
        "neck": ("neck",),
        "upper_arm": ("upper arm", "shoulder", "arm elevation"),
        "wrist": ("wrist",),
        "knee": ("knee", "leg"),
        "duration": ("static", "duration", "sustained"),
        "repetition": ("repetition", "cycle", "repetitive"),
    }
    RECOMMENDATION_KEYWORDS = {
        "reduce_trunk_flexion": ("trunk", "back", "torso"),
        "reduce_neck_demand": ("neck",),
        "reduce_arm_elevation": ("upper arm", "shoulder"),
        "reduce_wrist_deviation": ("wrist",),
        "reduce_knee_flexion": ("knee",),
        "reduce_static_exposure": ("static", "duration", "sustained"),
        "reduce_repetition": ("repetition", "cycle", "repetitive"),
    }

    def tag_text(text: str, keyword_map: dict[str, tuple[str, ...]]) -> set[str]:
        lowered = text.lower()
        return {tag for tag, words in keyword_map.items() if any(word in lowered for word in words)}

    def parse_sections(text: str) -> dict[str, str]:
        sections = {section: "" for section in NATURAL_SECTIONS}
        current = None
        for line in text.splitlines():
            clean = line.strip().strip("*").strip()
            if clean in sections:
                current = clean
                continue
            if current:
                sections[current] += line + "\n"
        return {key: value.strip() for key, value in sections.items()}

    def split_claims(text: str) -> list[str]:
        claims = []
        for line in text.splitlines():
            clean = line.strip().lstrip("-").strip()
            if clean:
                claims.append(clean)
        return claims


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=root / "results" / "generated_reports",
        help="Directory containing structured/ and natural/ report outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "m3_downstream_metrics",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def report_json_files(generated_root: Path, report_type: str, condition: str) -> list[Path]:
    reports_dir = generated_root / report_type / condition / "reports"
    if not reports_dir.exists():
        return []
    return sorted(
        path
        for path in reports_dir.glob("*/*.json")
        if path.is_file() and not path.name.startswith("._")
    )


def nested_label(report: dict[str, Any], section: str, field: str) -> str:
    try:
        value = report[section][field]["label"]
    except (KeyError, TypeError):
        return "invalid"
    return value.strip() if isinstance(value, str) else "invalid"


def load_structured_reports(generated_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for path in report_json_files(generated_root, "structured", condition):
            payload = read_json(path)
            report = payload.get("report")
            if not isinstance(report, dict):
                continue
            reports.append(
                {
                    "condition": condition,
                    "case_id": payload.get("case_id") or payload.get("sample_id") or path.parent.name,
                    "run_index": int(payload.get("run_index") or 0),
                    "report": report,
                    "file": str(path),
                }
            )
    return reports


def evaluate_structured(reports: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for item in reports:
        by_key[(item["condition"], item["case_id"], item["run_index"])] = item

    paired_keys = []
    for case_id, run_index in sorted({(item["case_id"], item["run_index"]) for item in reports}):
        left = by_key.get(("alphapose_motionbert", case_id, run_index))
        right = by_key.get(("sam3db", case_id, run_index))
        if left and right:
            paired_keys.append((case_id, run_index, left, right))

    risk_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    insufficient_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for case_id, run_index, alpha, sam in paired_keys:
        alpha_report = alpha["report"]
        sam_report = sam["report"]
        alpha_yes = set()
        sam_yes = set()

        for field in RISK_SUMMARY_FIELDS:
            alpha_label = nested_label(alpha_report, "risk_summary", field)
            sam_label = nested_label(sam_report, "risk_summary", field)
            risk_rows.append(
                {
                    "case_id": case_id,
                    "run_index": run_index,
                    "risk_item": field,
                    "alpha_label": alpha_label,
                    "sam_label": sam_label,
                    "changed": alpha_label != sam_label,
                }
            )
            for condition, label in (("alphapose_motionbert", alpha_label), ("sam3db", sam_label)):
                label_counts[(condition, "risk_summary")] += 1
                if label == "Insufficient evidence":
                    insufficient_counts[(condition, "risk_summary")] += 1

        for field in KEY_FACTOR_FIELDS:
            alpha_label = nested_label(alpha_report, "key_risk_factors", field)
            sam_label = nested_label(sam_report, "key_risk_factors", field)
            if alpha_label == "Yes":
                alpha_yes.add(field)
            if sam_label == "Yes":
                sam_yes.add(field)
            factor_rows.append(
                {
                    "case_id": case_id,
                    "run_index": run_index,
                    "factor": field,
                    "alpha_label": alpha_label,
                    "sam_label": sam_label,
                    "changed": alpha_label != sam_label,
                    "alpha_insufficient": alpha_label == "Insufficient evidence",
                    "sam_insufficient": sam_label == "Insufficient evidence",
                }
            )
            for condition, label in (("alphapose_motionbert", alpha_label), ("sam3db", sam_label)):
                label_counts[(condition, "key_factors")] += 1
                if label == "Insufficient evidence":
                    insufficient_counts[(condition, "key_factors")] += 1

        overlap_rows.append(
            {
                "case_id": case_id,
                "run_index": run_index,
                "key_factor_positive_jaccard": round(jaccard(alpha_yes, sam_yes), 4),
                "alpha_yes": "|".join(sorted(alpha_yes)),
                "sam_yes": "|".join(sorted(sam_yes)),
            }
        )

    csv_write(
        output_dir / "structured_risk_summary_label_changes.csv",
        risk_rows,
        ["case_id", "run_index", "risk_item", "alpha_label", "sam_label", "changed"],
    )
    csv_write(
        output_dir / "structured_key_factor_label_changes.csv",
        factor_rows,
        [
            "case_id",
            "run_index",
            "factor",
            "alpha_label",
            "sam_label",
            "changed",
            "alpha_insufficient",
            "sam_insufficient",
        ],
    )
    csv_write(
        output_dir / "structured_key_factor_positive_overlap.csv",
        overlap_rows,
        ["case_id", "run_index", "key_factor_positive_jaccard", "alpha_yes", "sam_yes"],
    )

    summary = {
        "structured_status": "ok" if paired_keys else "not_run_or_unpaired",
        "structured_condition_report_counts": {
            condition: len([item for item in reports if item["condition"] == condition])
            for condition in CONDITIONS
        },
        "structured_paired_report_count": len(paired_keys),
        "risk_summary_label_change_rate": mean([float(row["changed"]) for row in risk_rows]),
        "key_factor_label_change_rate": mean([float(row["changed"]) for row in factor_rows]),
        "key_factor_positive_jaccard_mean": mean(
            [float(row["key_factor_positive_jaccard"]) for row in overlap_rows]
        ),
        "insufficient_evidence_rate": {
            f"{condition}_{group}": (
                round(insufficient_counts[(condition, group)] / label_counts[(condition, group)], 4)
                if label_counts[(condition, group)]
                else None
            )
            for condition in CONDITIONS
            for group in ("risk_summary", "key_factors")
        },
    }
    return summary


def load_natural_reports(generated_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        for path in report_json_files(generated_root, "natural", condition):
            payload = read_json(path)
            report_text = payload.get("report_text")
            if not isinstance(report_text, str) or not report_text.strip():
                continue
            sections = payload.get("sections")
            if not isinstance(sections, dict):
                sections = parse_sections(report_text)
            reports.append(
                {
                    "condition": condition,
                    "case_id": payload.get("case_id") or payload.get("sample_id") or path.parent.name,
                    "run_index": int(payload.get("run_index") or 0),
                    "sections": sections,
                    "file": str(path),
                }
            )
    return reports


def natural_tags(report: dict[str, Any]) -> dict[str, set[str]]:
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


def evaluate_natural(reports: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    tagged_reports = []
    for report in reports:
        tagged_reports.append({**report, **natural_tags(report)})

    tag_rows = [
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

    by_case_condition: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in tagged_reports:
        by_case_condition[(report["case_id"], report["condition"])].append(report)

    overlap_rows: list[dict[str, Any]] = []
    for case_id in sorted({report["case_id"] for report in tagged_reports}):
        alpha_reports = by_case_condition.get((case_id, "alphapose_motionbert"), [])
        sam_reports = by_case_condition.get((case_id, "sam3db"), [])
        for alpha, sam in itertools.product(alpha_reports, sam_reports):
            overlap_rows.append(
                {
                    "case_id": case_id,
                    "alpha_run": alpha["run_index"],
                    "sam_run": sam["run_index"],
                    "key_factor_jaccard": round(
                        jaccard(alpha["key_factor_tags"], sam["key_factor_tags"]), 4
                    ),
                    "recommendation_jaccard": round(
                        jaccard(alpha["recommendation_tags"], sam["recommendation_tags"]), 4
                    ),
                    "alpha_key_tags": "|".join(sorted(alpha["key_factor_tags"])),
                    "sam_key_tags": "|".join(sorted(sam["key_factor_tags"])),
                    "alpha_recommendation_tags": "|".join(sorted(alpha["recommendation_tags"])),
                    "sam_recommendation_tags": "|".join(sorted(sam["recommendation_tags"])),
                }
            )

    csv_write(
        output_dir / "natural_report_tags.csv",
        tag_rows,
        ["condition", "case_id", "run_index", "key_factor_tags", "recommendation_tags", "file"],
    )
    csv_write(
        output_dir / "natural_cross_condition_overlap.csv",
        overlap_rows,
        [
            "case_id",
            "alpha_run",
            "sam_run",
            "key_factor_jaccard",
            "recommendation_jaccard",
            "alpha_key_tags",
            "sam_key_tags",
            "alpha_recommendation_tags",
            "sam_recommendation_tags",
        ],
    )

    return {
        "natural_status": "ok" if overlap_rows else "not_run_or_unpaired",
        "natural_condition_report_counts": {
            condition: len([item for item in reports if item["condition"] == condition])
            for condition in CONDITIONS
        },
        "natural_cross_condition_pair_count": len(overlap_rows),
        "natural_key_factor_jaccard_mean": mean(
            [float(row["key_factor_jaccard"]) for row in overlap_rows]
        ),
        "natural_recommendation_jaccard_mean": mean(
            [float(row["recommendation_jaccard"]) for row in overlap_rows]
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structured_reports = load_structured_reports(args.generated_root)
    natural_reports = load_natural_reports(args.generated_root)
    structured_summary = evaluate_structured(structured_reports, args.output_dir)
    natural_summary = evaluate_natural(natural_reports, args.output_dir)
    summary = {**structured_summary, **natural_summary}
    write_json(args.output_dir / "module1_module3_downstream_metric_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"[ok] wrote Module 3 downstream metrics under {args.output_dir}")


if __name__ == "__main__":
    main()
