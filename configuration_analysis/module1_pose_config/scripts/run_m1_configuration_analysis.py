#!/usr/bin/env python3
"""Prepare and evaluate Module 1 configuration-analysis metrics.

This runner does not run REP++ for all cases by default because full repetition
estimation can take several minutes. Use --include-repetition when repetition
metrics should be regenerated from both pose configurations.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("<private_workspace>/configuration_analysis/m1"))
    parser.add_argument("--include-repetition", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Forward --cpu to REP++ when repetition is included.")
    parser.add_argument("--include-module3", action="store_true", help="Evaluate Module 3 downstream metrics if reports exist.")
    args = parser.parse_args()

    scripts = args.root / "scripts"
    py = sys.executable
    run([py, str(scripts / "build_m1_configuration_inputs.py"), "--root", str(args.root)])
    run([py, str(scripts / "run_shared_duration_analysis.py"), "--root", str(args.root)])
    if args.include_repetition:
        for condition in ("alphapose_motionbert", "sam3db"):
            cmd = [
                py,
                str(scripts / "run_repetition_all_cases.py"),
                "--jsonl-dir",
                str(args.root / "inputs/repetition_jsonl" / condition),
                "--out-dir",
                str(args.root / "results/repetition" / condition),
            ]
            if args.cpu:
                cmd.append("--cpu")
            run(cmd)
    run([py, str(scripts / "evaluate_m1_configuration_metrics.py"), "--root", str(args.root)])
    if args.include_module3:
        run([py, str(scripts / "evaluate_m1_module3_metrics.py"), "--root", str(args.root)])


if __name__ == "__main__":
    main()
