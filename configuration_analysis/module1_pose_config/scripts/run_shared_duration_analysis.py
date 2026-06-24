#!/usr/bin/env python3
"""Run Module 2 duration analyzer on shared-angle CSVs for each M1 condition."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("<private_workspace>/configuration_analysis/m1"))
    parser.add_argument(
        "--duration-script",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1/module2_runner/ergonomic-risk-module2-main/03_2_Duration/duration_SD_제미나이 수정.py"),
    )
    args = parser.parse_args()

    root = args.root
    dur = load_module(args.duration_script, "shared_duration")
    analyzer = dur.StaticPostureAnalyzer(dur.AnalysisConfig())
    rows = []
    for condition_dir in sorted((root / "inputs/shared_angle_csv").iterdir()):
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        out_dir = root / "results/shared_duration" / condition
        out_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(condition_dir.glob("*_angle.csv")):
            if csv_path.name.startswith("._"):
                continue
            result = analyzer.process_file(csv_path, out_dir)
            result["condition"] = condition
            rows.append(result)

    out = pd.DataFrame(rows)
    out_dir = root / "results/shared_duration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "shared_duration_summary.csv", index=False)
    print(out.to_string(index=False))
    print(f"[ok] wrote {out_dir / 'shared_duration_summary.csv'}")


if __name__ == "__main__":
    main()
