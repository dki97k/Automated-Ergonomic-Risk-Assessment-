#!/usr/bin/env python3
"""Analyze Module #2 configuration LLM-as-judge pairwise results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CRITERIA = (
    "clarity",
    "coherence",
    "relevance",
    "usefulness",
    "professionalism",
    "evidence_grounding",
    "overall",
)


def parse_args() -> argparse.Namespace:
    m2_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-file",
        type=Path,
        help="pairwise_results_*.jsonl from run_pairwise_judge.py.",
    )
    parser.add_argument(
        "--pairwise-dir",
        type=Path,
        default=m2_root / "results" / "pairwise_judge",
        help="Used to find latest pairwise_results_*.jsonl when --results-file is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=m2_root / "results" / "pairwise_judge_analysis",
    )
    return parser.parse_args()


def latest_results_file(pairwise_dir: Path) -> Path:
    files = sorted(pairwise_dir.glob("pairwise_results_*.jsonl"))
    if not files:
        raise SystemExit(f"No pairwise_results_*.jsonl files found in {pairwise_dir}")
    return files[-1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


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


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    variance = sum((value - m) ** 2 for value in values) / (len(values) - 1)
    return round(variance**0.5, 4)


def normalize_preferred(value: Any, strength: int) -> str:
    if isinstance(value, str):
        preferred = value.strip()
        if preferred in {"A", "B", "Tie"}:
            return preferred
    if strength > 0:
        return "A"
    if strength < 0:
        return "B"
    return "Tie"


def normalize_strength(preferred: str, raw_strength: int) -> int:
    if preferred == "Tie":
        return 0
    magnitude = min(abs(raw_strength), 2)
    if magnitude == 0:
        magnitude = 1
    return magnitude if preferred == "A" else -magnitude


def criterion_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        decision = item.get("decision")
        if not isinstance(decision, dict):
            continue
        report_a = item.get("report_A", {})
        report_b = item.get("report_B", {})
        condition_a = report_a.get("condition")
        condition_b = report_b.get("condition")
        for criterion in CRITERIA:
            criterion_payload = decision.get(criterion)
            if not isinstance(criterion_payload, dict):
                continue
            try:
                raw_strength = int(criterion_payload.get("preference_strength"))
            except (TypeError, ValueError):
                continue
            preferred = normalize_preferred(
                criterion_payload.get("preferred_report"),
                raw_strength,
            )
            strength = normalize_strength(preferred, raw_strength)
            rows.append(
                {
                    "comparison_id": item.get("comparison_id"),
                    "condition_pair": item.get("condition_pair"),
                    "case_id": item.get("case_id"),
                    "order": item.get("order"),
                    "criterion": criterion,
                    "condition_A": condition_a,
                    "condition_B": condition_b,
                    "run_index_A": report_a.get("run_index"),
                    "run_index_B": report_b.get("run_index"),
                    "preferred_report": preferred,
                    "preference_strength": strength,
                    "raw_preference_strength": raw_strength,
                    "preferred_condition": (
                        condition_a if preferred == "A" else condition_b if preferred == "B" else "Tie"
                    ),
                    "justification": criterion_payload.get("justification", ""),
                }
            )
    return rows


def expand_condition_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        strength = int(row["preference_strength"])
        preferred = row["preferred_report"]
        for side in ("A", "B"):
            condition = row[f"condition_{side}"]
            signed_strength = strength if side == "A" else -strength
            expanded.append(
                {
                    "condition": condition,
                    "criterion": row["criterion"],
                    "condition_pair": row["condition_pair"],
                    "case_id": row["case_id"],
                    "comparison_id": row["comparison_id"],
                    "order": row["order"],
                    "win": 1 if preferred == side else 0,
                    "loss": 1 if preferred in {"A", "B"} and preferred != side else 0,
                    "tie": 1 if preferred == "Tie" or strength == 0 else 0,
                    "signed_preference_strength": 0 if preferred == "Tie" else signed_strength,
                }
            )
    return expanded


def aggregate_condition_stats(expanded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        groups[(row["condition"], row["criterion"])].append(row)

    stats_rows = []
    for (condition, criterion), rows in sorted(groups.items()):
        count = len(rows)
        wins = sum(int(row["win"]) for row in rows)
        losses = sum(int(row["loss"]) for row in rows)
        ties = sum(int(row["tie"]) for row in rows)
        strengths = [float(row["signed_preference_strength"]) for row in rows]
        stats_rows.append(
            {
                "condition": condition,
                "criterion": criterion,
                "n_comparisons": count,
                "win_rate": round(wins / count, 4) if count else None,
                "loss_rate": round(losses / count, 4) if count else None,
                "tie_rate": round(ties / count, 4) if count else None,
                "mean_signed_preference_strength": mean(strengths),
                "std_signed_preference_strength": std(strengths),
            }
        )
    return stats_rows


def aggregate_pair_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition_pair"], row["criterion"])].append(row)

    output = []
    for (condition_pair, criterion), group_rows in sorted(groups.items()):
        try:
            left_condition, right_condition = condition_pair.split("_vs_", 1)
        except ValueError:
            left_condition = str(group_rows[0]["condition_A"])
            right_condition = str(group_rows[0]["condition_B"])
        normalized_strengths = []
        preferred_counts: dict[str, int] = defaultdict(int)
        for row in group_rows:
            strength = float(row["preference_strength"])
            if row["condition_A"] == left_condition:
                normalized_strengths.append(strength)
            else:
                normalized_strengths.append(-strength)
            preferred_counts[str(row["preferred_condition"])] += 1
        output.append(
            {
                "condition_pair": condition_pair,
                "criterion": criterion,
                "n_comparisons": len(group_rows),
                "mean_left_minus_right_strength": mean(normalized_strengths),
                "std_left_minus_right_strength": std(normalized_strengths),
                "preferred_left_count": preferred_counts.get(left_condition, 0),
                "preferred_right_count": preferred_counts.get(right_condition, 0),
                "tie_count": preferred_counts.get("Tie", 0),
                "left_condition": left_condition,
                "right_condition": right_condition,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    results_file = args.results_file or latest_results_file(args.pairwise_dir)
    items = load_jsonl(results_file)
    rows = criterion_rows(items)
    expanded = expand_condition_records(rows)
    condition_stats = aggregate_condition_stats(expanded)
    pair_stats = aggregate_pair_stats(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_write(
        args.output_dir / "pairwise_criterion_rows.csv",
        rows,
        [
            "comparison_id",
            "condition_pair",
            "case_id",
            "order",
            "criterion",
            "condition_A",
            "condition_B",
            "run_index_A",
            "run_index_B",
            "preferred_report",
            "preference_strength",
            "raw_preference_strength",
            "preferred_condition",
            "justification",
        ],
    )
    csv_write(
        args.output_dir / "pairwise_condition_stats.csv",
        condition_stats,
        [
            "condition",
            "criterion",
            "n_comparisons",
            "win_rate",
            "loss_rate",
            "tie_rate",
            "mean_signed_preference_strength",
            "std_signed_preference_strength",
        ],
    )
    csv_write(
        args.output_dir / "pairwise_by_condition_pair.csv",
        pair_stats,
        [
            "condition_pair",
            "criterion",
            "n_comparisons",
            "mean_left_minus_right_strength",
            "std_left_minus_right_strength",
            "preferred_left_count",
            "preferred_right_count",
            "tie_count",
            "left_condition",
            "right_condition",
        ],
    )

    overall_condition_stats = [
        row for row in condition_stats if row["criterion"] == "overall"
    ]
    csv_write(
        args.output_dir / "pairwise_overall_win_rate.csv",
        overall_condition_stats,
        [
            "condition",
            "criterion",
            "n_comparisons",
            "win_rate",
            "loss_rate",
            "tie_rate",
            "mean_signed_preference_strength",
            "std_signed_preference_strength",
        ],
    )
    with (args.output_dir / "pairwise_analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "results_file": str(results_file),
                "raw_item_count": len(items),
                "criterion_row_count": len(rows),
                "condition_stats": condition_stats,
                "pair_stats": pair_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print("Pairwise judge analysis complete.")
    print(f"Input: {results_file}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
