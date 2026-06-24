#!/usr/bin/env python3
"""Run REP++ repetition counting for each normalized JSONL case.

The upstream REP++ script writes to ``<script_dir>/output``. This wrapper copies
the script into a temporary per-case directory, runs it there, and then stores
each case's output under the configured result directory.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


def run_case(
    *,
    python_bin: Path,
    pipeline_script: Path,
    checkpoint: Path,
    jsonl: Path,
    out_dir: Path,
    fps: float,
    cpu: bool,
) -> dict:
    case_id = jsonl.stem
    case_out = out_dir / case_id
    case_out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"rep_{case_id}_") as temp:
        temp_dir = Path(temp)
        temp_script = temp_dir / "pipeline_reps.py"
        shutil.copy2(pipeline_script, temp_script)

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(Path("/private/tmp/mplconfig")))
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        cmd = [
            str(python_bin),
            str(temp_script),
            "--jsonl",
            str(jsonl),
            "--ckpt",
            str(checkpoint),
            "--fps",
            str(fps),
        ]
        if cpu:
            cmd.append("--cpu")

        proc = subprocess.run(cmd, cwd=str(temp_dir), env=env, text=True, capture_output=True)
        (case_out / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        (case_out / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {"clip": case_id, "status": "failed", "returncode": proc.returncode}

        temp_output = temp_dir / "output"
        if case_out.exists():
            for child in case_out.iterdir():
                if child.name not in {"stdout.txt", "stderr.txt"}:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
        for item in temp_output.iterdir():
            target = case_out / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)

    summary_path = case_out / "summary_reps_integrated.csv"
    if not summary_path.exists():
        return {"clip": case_id, "status": "missing_summary", "returncode": 0}
    summary = pd.read_csv(summary_path).iloc[0].to_dict()
    return {
        "clip": case_id,
        "status": "ok",
        "returncode": 0,
        "repetitions_total_peaks": summary.get("repetitions_total_peaks"),
        "mean_period_sec": summary.get("mean_period_sec"),
        "rpm_mean": summary.get("rpm_mean"),
        "quality_flag": summary.get("quality_flag"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1/inputs/repetition_jsonl/normalized"),
    )
    parser.add_argument(
        "--pipeline-script",
        type=Path,
        default=Path("<private_workspace>/m2/REP++/pipeline_reps.py"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("<private_workspace>/m2/REP++/pytorch_weights.pth"),
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("<private_workspace>/m1/.venv-sam/bin/python"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1/results/repetition"),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--case", action="append", help="Run only this case id; may be repeated.")
    args = parser.parse_args()

    jsonls = sorted(p for p in args.jsonl_dir.glob("*.jsonl") if not p.name.startswith("._"))
    if args.case:
        requested = set(args.case)
        jsonls = [p for p in jsonls if p.stem in requested]
    if not jsonls:
        raise SystemExit("No JSONL files found.")

    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for jsonl in jsonls:
        print(f"[run] {jsonl.stem}", flush=True)
        row = run_case(
            python_bin=args.python_bin,
            pipeline_script=args.pipeline_script,
            checkpoint=args.checkpoint,
            jsonl=jsonl,
            out_dir=args.out_dir,
            fps=args.fps,
            cpu=args.cpu,
        )
        rows.append(row)
        print(row, flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "repetition_case_summary.csv", index=False)
    print(f"[ok] wrote {args.out_dir / 'repetition_case_summary.csv'}")


if __name__ == "__main__":
    main()
