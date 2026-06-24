#!/usr/bin/env python3
"""Inspect Module 1 pose outputs and Module 2 runner compatibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def npz_summary(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "path": str(path),
        "keys": [
            {"name": key, "shape": list(data[key].shape), "dtype": str(data[key].dtype)}
            for key in data.files
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1"),
    )
    args = parser.parse_args()

    root = args.root
    alpha = root / "inputs/m1_pose/alphapose_motionbert/field_plausibility_alphapose_motionbert_allframes_temporal_bbox.npz"
    sam_dir = root / "inputs/m1_pose/sam3db_mhr70_by_sequence"
    manifest = root / "inputs/m1_pose/manifest/field_plausibility_by_severity_allframes_temporal_bbox_manifest.csv"
    module2 = root / "module2_runner/ergonomic-risk-module2-main"

    report = {
        "alphapose_motionbert": npz_summary(alpha) if alpha.exists() else None,
        "sam3db": [
            npz_summary(p)
            for p in sorted(sam_dir.glob("*.npz"))
            if not p.name.startswith("._")
        ],
        "manifest_exists": manifest.exists(),
        "module2_runner_exists": module2.exists(),
        "module2_run_script_exists": (module2 / "run_module2.py").exists(),
        "module2_angle_script_exists": (module2 / "02_joint_angle/joint_angle.py").exists(),
        "compatibility": {
            "sam3db": "compatible with Module 2 raw angle extraction shape expectation (70 keypoints)",
            "alphapose_motionbert": "requires adapter because available output is pred_common14",
        },
    }

    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            counts: dict[str, int] = {}
            for row in reader:
                seq = row.get("sequence", "")
                counts[seq] = counts.get(seq, 0) + 1
        report["manifest_sequence_counts"] = counts

    out_dir = root / "results/compatibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "m1_module2_compatibility.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
