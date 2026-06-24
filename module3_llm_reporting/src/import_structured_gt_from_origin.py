#!/usr/bin/env python3
"""Import structured key-factor GT labels from the origin reference files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from structured_common import KEY_FACTOR_FIELDS, project_root, write_json


VIDEO_ID_MAP = {
    "V1": "MansoryBrickLaying_00",
    "V2": "MansoryBrickLaying_01",
    "V3": "MansoryBrickLaying_02",
    "V4": "MansoryCement_02",
    "V5": "RebarPlacement_00",
    "V6": "RebarTying_01",
    "V7": "RebarTying_02",
    "V8": "WallPlacement_00",
}

PART_TO_FACTOR = {
    "Trunk": "trunk_overflexion",
    "Neck": "neck_overflexion_or_extension",
    "Upper_arm": "upper_arm_elevation",
    "Wrist": "wrist_deviation",
    "Knee": "knee_overflexion",
    "Static_posture": "prolonged_static_posture",
    "Repetitive_activity": "repetitive_work",
}

FACTOR_CRITERIA = {
    "trunk_overflexion": {
        "source_item": "Excessive trunk flexion",
        "criterion": "Trunk flexion >= 60 degrees",
    },
    "neck_overflexion_or_extension": {
        "source_item": "Neck flexion / extension abduction, twisting",
        "criterion": "Max(neck flexion, abs(neck bending), abs(neck twisting)) >= 20 degrees",
    },
    "upper_arm_elevation": {
        "source_item": "Elevated upper arm; extreme shoulder abduction",
        "criterion": "Upper arm flexion >= 90 degrees or upper arm abduction >= 90 degrees",
    },
    "wrist_deviation": {
        "source_item": "Wrist deviation",
        "criterion": "Abs(wrist flexion) >= 15 degrees or abs(wrist twisting) >= 45 degrees",
    },
    "knee_overflexion": {
        "source_item": "Deep knee flexion / squatting",
        "criterion": "Knee angle >= 60 degrees",
    },
    "prolonged_static_posture": {
        "source_item": "Prolonged static posture",
        "criterion": "Final REBA >= 8 maintained for at least 120 consecutive frames",
    },
    "repetitive_work": {
        "source_item": "High repetition motion",
        "criterion": "Detected repetitive activity based on the Module 2 repetition results table",
    },
}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--origin-key-csv",
        type=Path,
        default=Path("<private_workspace>/m3_origin/results_structured_key.csv"),
    )
    parser.add_argument(
        "--origin-criteria-xlsx",
        type=Path,
        default=Path("<private_workspace>/m3_origin/GT_key_factors_llm.xlsx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "data"
        / "structured_validation"
        / "reference"
        / "key_factor_reference.json",
    )
    return parser.parse_args()


def yes_no(value: str) -> str:
    normalized = str(value).strip()
    if normalized in {"1", "1.0", "Yes", "yes", "Y", "y"}:
        return "Yes"
    if normalized in {"0", "0.0", "No", "no", "N", "n"}:
        return "No"
    raise ValueError(f"Unsupported GT label: {value!r}")


def load_gt_rows(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {
        sample_id: {} for sample_id in VIDEO_ID_MAP.values()
    }
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            video_code = row["VIDEO"].strip()
            part = row["Part"].strip()
            if video_code not in VIDEO_ID_MAP or part not in PART_TO_FACTOR:
                continue
            sample_id = VIDEO_ID_MAP[video_code]
            factor = PART_TO_FACTOR[part]
            labels[sample_id][factor] = yes_no(row["GT"])

    for sample_id, sample_labels in labels.items():
        missing = [factor for factor in KEY_FACTOR_FIELDS if factor not in sample_labels]
        if missing:
            raise ValueError(f"Missing GT labels for {sample_id}: {missing}")
    return labels


def build_reference(labels: dict[str, dict[str, str]], args: argparse.Namespace) -> dict[str, Any]:
    samples = []
    for video_code, sample_id in VIDEO_ID_MAP.items():
        factor_labels = {}
        for factor in KEY_FACTOR_FIELDS:
            criteria = FACTOR_CRITERIA[factor]
            factor_labels[factor] = {
                "label": labels[sample_id][factor],
                "evidence": (
                    f"{criteria['criterion']} "
                    f"(source item: {criteria['source_item']}; CSV video code: {video_code})"
                ),
            }
        samples.append({"sample_id": sample_id, "video_code": video_code, "key_risk_factors": factor_labels})

    return {
        "reference_type": "origin_curated_key_factor_gt",
        "source": "GT labels imported from origin structured key-factor CSV",
        "source_files": {
            "gt_csv": str(args.origin_key_csv),
            "criteria_xlsx": str(args.origin_criteria_xlsx),
        },
        "video_id_map": VIDEO_ID_MAP,
        "factor_criteria": FACTOR_CRITERIA,
        "risk_summary_reference": None,
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    if not args.origin_key_csv.exists():
        raise SystemExit(f"Missing GT CSV: {args.origin_key_csv}")
    if not args.origin_criteria_xlsx.exists():
        raise SystemExit(f"Missing criteria workbook: {args.origin_criteria_xlsx}")

    labels = load_gt_rows(args.origin_key_csv)
    payload = build_reference(labels, args)
    write_json(args.output, payload)
    print(f"Imported structured key-factor GT for {len(payload['samples'])} samples")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
