#!/usr/bin/env python3
"""Run Module 2 downstream generation without the sklearn-dependent final eval."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(message: str) -> None:
    print(f"\n=== {message} ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module2-root",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1/module2_runner/ergonomic-risk-module2-main"),
    )
    args = parser.parse_args()

    root = args.module2_root.resolve()
    os.chdir(root)

    step("1) REBA: angle CSV -> SCORE")
    reba = load_module("03_1_REBA_table/REBA_table.py", "reba")
    Path("03_1_REBA_table/results_v2").mkdir(parents=True, exist_ok=True)
    summaries = [
        reba.process_file(path, "03_1_REBA_table/results_v2")
        for path in glob.glob("02_joint_angle/*_angle.csv")
    ]
    pd.DataFrame([item for item in summaries if item]).to_csv(
        "03_1_REBA_table/results_v2/TOTAL_SUMMARY_REPORT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    step("2) Duration: angle CSV -> segments/integrated")
    dur = load_module("03_2_Duration/duration_SD_제미나이 수정.py", "dur")
    out_dir = Path("03_2_Duration/_selffix_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    analyzer = dur.StaticPostureAnalyzer(dur.AnalysisConfig())
    for path in glob.glob("02_joint_angle/*_angle.csv"):
        analyzer.process_file(Path(path), out_dir)

    for script in [
        "_pipeline/iso_duration.py",
        "_pipeline/rep_risk.py",
        "_pipeline/build_schema.py",
    ]:
        step(f"3) {Path(script).name}")
        subprocess.run([sys.executable, script], check=True)

    print("\n[ok] Module 2 schema generation completed without final sklearn eval.")


if __name__ == "__main__":
    main()
