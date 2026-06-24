#!/usr/bin/env python3
"""Build a structured validation table including human and LLM results."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluate_structured_validation import fleiss_kappa, metric_from_counts
from structured_common import project_root, read_json, write_json


VIDEO_CODE_TO_SAMPLE_ID = {
    "V1": "MansoryBrickLaying_00",
    "V2": "MansoryBrickLaying_01",
    "V3": "MansoryBrickLaying_02",
    "V4": "MansoryCement_02",
    "V5": "RebarPlacement_00",
    "V6": "RebarTying_01",
    "V7": "RebarTying_02",
    "V8": "WallPlacement_00",
}

KEY_FACTOR_MAP = {
    "Trunk": "trunk_overflexion",
    "Neck": "neck_overflexion_or_extension",
    "Upper_arm": "upper_arm_elevation",
    "Wrist": "wrist_deviation",
    "Knee": "knee_overflexion",
    "Static_posture": "prolonged_static_posture",
    "Repetitive_activity": "repetitive_work",
}

RISK_MAP = {
    "Posture": "posture_risk",
    "Duration": "duration_risk",
    "Repetition": "repetition_risk",
}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm-metrics",
        type=Path,
        default=root
        / "results"
        / "structured_validation"
        / "evidence_based_numerical_only"
        / "evaluation"
        / "metrics.json",
    )
    parser.add_argument(
        "--human-key-csv",
        type=Path,
        default=Path("<private_workspace>/m3_origin/results_structured_key.csv"),
    )
    parser.add_argument(
        "--human-level-csv",
        type=Path,
        default=Path("<private_workspace>/m3_origin/results_structured_level.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "results"
        / "structured_validation"
        / "evidence_based_numerical_only"
        / "validation_table",
    )
    parser.add_argument(
        "--skip-human",
        action="store_true",
        help="Build an LLM-only validation table without human evaluator rows.",
    )
    return parser.parse_args()


def to_label(value: str, positive: str, negative: str) -> str:
    normalized = str(value).strip()
    if normalized in {"1", "1.0", positive}:
        return positive
    if normalized in {"0", "0.0", negative}:
        return negative
    raise ValueError(f"Unsupported binary value: {value!r}")


def count_prediction(counts: Counter[str], pred: str, ref: str) -> None:
    if pred == "Yes" and ref == "Yes":
        counts["tp"] += 1
    elif pred == "Yes" and ref == "No":
        counts["fp"] += 1
    elif pred == "No" and ref == "No":
        counts["tn"] += 1
    else:
        counts["fn"] += 1


def load_human_key_metrics(path: Path) -> dict[str, Any]:
    human_cols = [f"H{i}" for i in range(1, 6)]
    counts_total: Counter[str] = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    counts_by_factor: dict[str, Counter[str]] = defaultdict(
        lambda: Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    )
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            part = row["Part"].strip()
            if part not in KEY_FACTOR_MAP:
                continue
            factor = KEY_FACTOR_MAP[part]
            ref = to_label(row["GT"], "Yes", "No")
            for col in human_cols:
                pred = to_label(row[col], "Yes", "No")
                count_prediction(counts_total, pred, ref)
                count_prediction(counts_by_factor[factor], pred, ref)
                rows.append(
                    {
                        "video": row["VIDEO"],
                        "factor": factor,
                        "rater": col,
                        "prediction": pred,
                        "reference": ref,
                    }
                )

    return {
        "overall": metric_from_counts(
            counts_total["tp"],
            counts_total["fp"],
            counts_total["tn"],
            counts_total["fn"],
        ),
        "by_factor": {
            factor: metric_from_counts(
                counts["tp"], counts["fp"], counts["tn"], counts["fn"]
            )
            for factor, counts in counts_by_factor.items()
        },
        "rows": rows,
    }


def load_human_risk_kappa(path: Path) -> dict[str, Any]:
    human_cols = [f"H{i}" for i in range(1, 6)]
    label_sets_by_item: dict[tuple[str, str], list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            risk = row["RISK"].strip()
            if risk not in RISK_MAP:
                continue
            key = (VIDEO_CODE_TO_SAMPLE_ID[row["VIDEO"].strip()], RISK_MAP[risk])
            label_sets_by_item[key] = [
                to_label(row[col], "High", "Low") for col in human_cols
            ]

    by_risk = {}
    for risk_name in RISK_MAP.values():
        by_risk[risk_name] = fleiss_kappa(
            [labels for (_, field), labels in label_sets_by_item.items() if field == risk_name],
            ("High", "Low"),
        )
    overall = fleiss_kappa(list(label_sets_by_item.values()), ("High", "Low"))
    return {"overall": overall, "by_risk_item": by_risk}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    llm_metrics = read_json(args.llm_metrics)

    llm_key = llm_metrics["key_factor_classification"]["overall"]
    llm_kappa = llm_metrics["risk_summary_fleiss_kappa"]["overall"]
    llm_risk = (llm_metrics.get("risk_summary_classification") or {}).get("overall")

    table_rows = []
    human_key = None
    human_kappa = None
    if not args.skip_human:
        human_key = load_human_key_metrics(args.human_key_csv)
        human_kappa = load_human_risk_kappa(args.human_level_csv)
        human_key_overall = human_key["overall"]
        table_rows.append(
            {
                "evaluator": "Human evaluators",
                "risk_summary_fleiss_kappa": human_kappa["overall"],
                "risk_summary_accuracy": "",
                "risk_summary_precision": "",
                "risk_summary_recall": "",
                "risk_summary_f1": "",
                "key_factor_accuracy": human_key_overall["accuracy"],
                "key_factor_precision": human_key_overall["precision"],
                "key_factor_recall": human_key_overall["recall"],
                "key_factor_f1": human_key_overall["f1"],
                "key_factor_n": human_key_overall["n"],
            }
        )

    table_rows.append(
        {
            "evaluator": "LLM Module 3",
            "risk_summary_fleiss_kappa": llm_kappa,
            "risk_summary_accuracy": llm_risk["accuracy"] if llm_risk else "",
            "risk_summary_precision": llm_risk["precision"] if llm_risk else "",
            "risk_summary_recall": llm_risk["recall"] if llm_risk else "",
            "risk_summary_f1": llm_risk["f1"] if llm_risk else "",
            "key_factor_accuracy": llm_key["accuracy"],
            "key_factor_precision": llm_key["precision"],
            "key_factor_recall": llm_key["recall"],
            "key_factor_f1": llm_key["f1"],
            "key_factor_n": llm_key["n"],
        }
    )

    payload = {
        "validation_condition": "structured evidence-based prompt, numerical Module 2 input only",
        "human_sources": {
            "risk_summary": None if args.skip_human else str(args.human_level_csv),
            "key_factors": None if args.skip_human else str(args.human_key_csv),
        },
        "llm_metrics_source": str(args.llm_metrics),
        "summary_table": table_rows,
        "human_risk_summary_fleiss_kappa": human_kappa,
        "human_key_factor_classification": (
            {
                "overall": human_key["overall"],
                "by_factor": human_key["by_factor"],
            }
            if human_key
            else None
        ),
        "llm_risk_summary_fleiss_kappa": llm_metrics["risk_summary_fleiss_kappa"],
        "llm_key_factor_classification": llm_metrics["key_factor_classification"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "validation_table.json", payload)
    write_csv(args.output_dir / "validation_table.csv", table_rows)
    if human_key:
        write_csv(args.output_dir / "human_key_factor_rows.csv", human_key["rows"])

    print("Structured validation summary")
    for row in table_rows:
        print(
            f"{row['evaluator']}: kappa={row['risk_summary_fleiss_kappa']}, "
            f"accuracy={row['key_factor_accuracy']}, "
            f"precision={row['key_factor_precision']}, "
            f"recall={row['key_factor_recall']}, f1={row['key_factor_f1']}"
        )
    print(args.output_dir / "validation_table.csv")


if __name__ == "__main__":
    main()
