#!/usr/bin/env python3
"""Normalize Module 2 RebarTying IDs from 00/01 to field-case 01/02."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MAPPING = {
    "RebarTying_00": "RebarTying_01",
    "RebarTying_01": "RebarTying_02",
}
TMP = "__REBAR_TYING_TMP__"


def map_text(value: str) -> str:
    value = value.replace("RebarTying_01", TMP)
    value = value.replace("RebarTying_00", "RebarTying_01")
    value = value.replace(TMP, "RebarTying_02")
    value = value.replace(
        "clip-name unified to angle-CSV basis (_01 jsonl -> _00); frozen Tier-2 value",
        "clip-name corrected to field-case basis - frozen Tier-2 value",
    )
    value = value.replace(
        "clip-name unified to angle-CSV basis (_02 jsonl -> _01); frozen Tier-2 value",
        "clip-name corrected to field-case basis - frozen Tier-2 value",
    )
    return value


def normalize_csv(path: Path) -> None:
    text = map_text(path.read_text(encoding="utf-8-sig"))
    path.write_text(text, encoding="utf-8")


def normalize_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and data.get("clip") in MAPPING:
        data["clip"] = MAPPING[data["clip"]]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rename_pair(directory: Path, suffix: str) -> None:
    old_00 = directory / f"RebarTying_00{suffix}"
    old_01 = directory / f"RebarTying_01{suffix}"
    tmp = directory / f"{TMP}{suffix}"
    new_01 = directory / f"RebarTying_01{suffix}"
    new_02 = directory / f"RebarTying_02{suffix}"

    if old_01.exists():
        old_01.rename(tmp)
    if old_00.exists():
        old_00.rename(new_01)
    if tmp.exists():
        tmp.rename(new_02)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module2-root",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1/module2_runner/ergonomic-risk-module2-main"),
    )
    args = parser.parse_args()
    root = args.module2_root.resolve()

    for path in root.rglob("*.csv"):
        normalize_csv(path)

    schema_dir = root / "_pipeline/schema"
    for path in schema_dir.glob("RebarTying_0*.json"):
        normalize_json(path)
    rename_pair(schema_dir, ".json")

    rename_specs = [
        (root / "02_joint_angle", "_angle.csv"),
        (root / "03_1_REBA_table/results", "_angle_SCORE.csv"),
        (root / "03_1_REBA_table/results_v2", "_angle_SCORE.csv"),
        (root / "03_2_Duration/duration", "_angle_integrated_analysis.csv"),
        (root / "03_2_Duration/_selffix_out/duration", "_angle_integrated_analysis.csv"),
        (root / "03_2_Duration/segments", "_angle_segments_wide.csv"),
        (root / "03_2_Duration/_selffix_out/segments", "_angle_segments_wide.csv"),
    ]
    for directory, suffix in rename_specs:
        if directory.exists():
            rename_pair(directory, suffix)

    print(f"[ok] normalized RebarTying IDs under {root}")


if __name__ == "__main__":
    main()
