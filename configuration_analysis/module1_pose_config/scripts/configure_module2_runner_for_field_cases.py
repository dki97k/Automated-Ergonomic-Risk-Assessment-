#!/usr/bin/env python3
"""Configure the copied Module 2 runner to use field-case RebarTying IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TMP = "__REBAR_TYING_TMP__"


def map_text(text: str) -> str:
    text = text.replace("RebarTying_01", TMP)
    text = text.replace("RebarTying_00", "RebarTying_01")
    text = text.replace(TMP, "RebarTying_02")
    text = text.replace(
        "clip-name unified to angle-CSV basis (_01 jsonl -> _00); frozen Tier-2 value",
        "clip-name corrected to field-case basis - frozen Tier-2 value",
    )
    text = text.replace(
        "clip-name unified to angle-CSV basis (_02 jsonl -> _01); frozen Tier-2 value",
        "clip-name corrected to field-case basis - frozen Tier-2 value",
    )
    return text


def patch_text_file(path: Path) -> None:
    if path.exists():
        path.write_text(map_text(path.read_text(encoding="utf-8-sig")), encoding="utf-8")


def rename_pair(directory: Path, suffix: str) -> None:
    if not directory.exists():
        return
    old_00 = directory / f"RebarTying_00{suffix}"
    old_01 = directory / f"RebarTying_01{suffix}"
    tmp = directory / f"{TMP}{suffix}"
    new_01 = directory / f"RebarTying_01{suffix}"
    new_02 = directory / f"RebarTying_02{suffix}"

    if old_01.exists() and not tmp.exists():
        old_01.rename(tmp)
    if old_00.exists() and old_00 != new_01:
        old_00.rename(new_01)
    if tmp.exists():
        tmp.rename(new_02)


def patch_schema_json(schema_dir: Path) -> None:
    if not schema_dir.exists():
        return
    for path in schema_dir.glob("RebarTying_0*.json"):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and "clip" in data:
            data["clip"] = map_text(data["clip"])
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    rename_pair(schema_dir, ".json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module2-root",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1/module2_runner/ergonomic-risk-module2-main"),
    )
    args = parser.parse_args()
    root = args.module2_root.resolve()

    for path in [
        root / "_pipeline/build_schema.py",
        root / "_pipeline/iso_duration.py",
        root / "_pipeline/rep_risk.py",
        root / "_pipeline/rep_period_frozen.csv",
        root / "_pipeline/iso_duration_trunk.csv",
        root / "_pipeline/schema_summary.csv",
        root / "03_1_REBA_table/results/TOTAL_SUMMARY_REPORT.csv",
        root / "03_1_REBA_table/results_v2/TOTAL_SUMMARY_REPORT.csv",
        root / "03_2_Duration/TOTAL_SUMMARY_REPORT.csv",
        root / "03_2_Duration/_r2_delta_result.csv",
    ]:
        patch_text_file(path)

    for directory, suffix in [
        (root / "02_joint_angle", "_angle.csv"),
        (root / "03_1_REBA_table/results", "_angle_SCORE.csv"),
        (root / "03_1_REBA_table/results_v2", "_angle_SCORE.csv"),
        (root / "03_2_Duration/duration", "_angle_integrated_analysis.csv"),
        (root / "03_2_Duration/_selffix_out/duration", "_angle_integrated_analysis.csv"),
        (root / "03_2_Duration/segments", "_angle_segments_wide.csv"),
        (root / "03_2_Duration/_selffix_out/segments", "_angle_segments_wide.csv"),
        (root / "03_1_REBA_table/parsing", "_hybrid.json"),
    ]:
        rename_pair(directory, suffix)

    patch_schema_json(root / "_pipeline/schema")

    print(f"[ok] Module 2 runner configured for field case IDs under {root}")


if __name__ == "__main__":
    main()
