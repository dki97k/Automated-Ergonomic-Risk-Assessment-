#!/usr/bin/env python3
"""Run the full Module #2 configuration-analysis workflow.

Expected workflow after replacing Module #2 inputs:

1. Build anonymized payloads.
2. Generate structured and natural reports.
3. Build deterministic metric tables and a blank claim sheet.
4. Apply initial evidence-audit labels.
5. Recompute metrics with the initial labels.
6. Run and analyze LLM-as-Judge pairwise comparisons.

The claim labels produced by this workflow are initial labels. They are intended
to be reviewed and finalized manually before manuscript reporting.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    m2_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=m2_root)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--api-delay-sec", type=float, default=2.0)
    parser.add_argument("--rate-limit-retry-sec", type=float, default=45.0)
    parser.add_argument(
        "--force-reports",
        action="store_true",
        help="Overwrite all report outputs instead of only retrying failed/dry-run outputs.",
    )
    parser.add_argument(
        "--skip-report-generation",
        action="store_true",
        help="Reuse existing structured/natural report outputs.",
    )
    parser.add_argument(
        "--skip-pairwise",
        action="store_true",
        help="Skip LLM-as-Judge pairwise generation and analysis.",
    )
    parser.add_argument(
        "--skip-initial-labeling",
        action="store_true",
        help="Leave the claim annotation sheet blank for manual labeling.",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def newest_pairwise_file(pairwise_dir: Path) -> Path:
    files = sorted(pairwise_dir.glob("pairwise_results_*.jsonl"))
    if not files:
        raise SystemExit(f"No pairwise result files found: {pairwise_dir}")
    return files[-1]


def main() -> None:
    args = parse_args()
    root = args.m2_root
    scripts = root / "scripts"
    python = sys.executable

    run(
        [
            python,
            str(scripts / "build_condition_payloads.py"),
            "--m2-root",
            str(root),
            "--force",
        ]
    )

    if not args.skip_report_generation:
        report_cmd = [
            python,
            "-u",
            str(scripts / "run_configuration_reports.py"),
            "--m2-root",
            str(root),
            "--report-type",
            "both",
            "--runs",
            str(args.runs),
            "--api-delay-sec",
            str(args.api_delay_sec),
            "--rate-limit-retry-sec",
            str(args.rate_limit_retry_sec),
        ]
        if args.model:
            report_cmd.extend(["--model", args.model])
        if args.force_reports:
            report_cmd.append("--force")
        else:
            report_cmd.append("--retry-errors")
        run(report_cmd)

    run(
        [
            python,
            str(scripts / "evaluate_configuration_metrics.py"),
            "--m2-root",
            str(root),
            "--force-annotation",
        ]
    )

    if not args.skip_initial_labeling:
        run(
            [
                python,
                str(scripts / "label_configuration_claims.py"),
                "--annotation-csv",
                str(root / "results" / "evaluation" / "claim_annotation_sheet.csv"),
                "--output-csv",
                str(root / "results" / "evaluation" / "claim_annotation_sheet.csv"),
            ]
        )
        run(
            [
                python,
                str(scripts / "evaluate_configuration_metrics.py"),
                "--m2-root",
                str(root),
            ]
        )

    if not args.skip_pairwise:
        judge_cmd = [
            python,
            "-u",
            str(scripts / "run_pairwise_judge.py"),
            "--m2-root",
            str(root),
            "--pairing",
            "same-run",
            "--api-delay-sec",
            str(args.api_delay_sec),
            "--rate-limit-retry-sec",
            str(args.rate_limit_retry_sec),
        ]
        if args.judge_model:
            judge_cmd.extend(["--model", args.judge_model])
        run(judge_cmd)
        pairwise_file = newest_pairwise_file(root / "results" / "pairwise_judge")
        run(
            [
                python,
                str(scripts / "analyze_pairwise_judge.py"),
                "--results-file",
                str(pairwise_file),
                "--output-dir",
                str(root / "results" / "pairwise_judge_analysis"),
            ]
        )

    print("Full Module #2 configuration-analysis workflow complete.", flush=True)
    print(root / "results" / "evaluation" / "configuration_summary.csv", flush=True)
    print(root / "results" / "evaluation" / "claim_annotation_sheet.csv", flush=True)
    if not args.skip_pairwise:
        print(root / "results" / "pairwise_judge_analysis" / "pairwise_overall_win_rate.csv", flush=True)


if __name__ == "__main__":
    main()
