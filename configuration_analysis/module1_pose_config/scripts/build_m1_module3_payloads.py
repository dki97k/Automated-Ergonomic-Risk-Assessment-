#!/usr/bin/env python3
"""Build Module 3 payloads for Module 1 configuration analysis."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CONDITIONS = ("alphapose_motionbert", "sam3db")
CASES = (
    "MansoryBrickLaying_00",
    "MansoryBrickLaying_01",
    "MansoryBrickLaying_02",
    "MansoryCement_02",
    "RebarPlacement_00",
    "RebarTying_01",
    "RebarTying_02",
    "WallPlacement_00",
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
M3_SRC = PROJECT_ROOT / "m3" / "src"
sys.path.insert(0, str(M3_SRC))

from structured_common import build_input_summary, read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--sam-full-module2-dir",
        type=Path,
        default=PROJECT_ROOT / "module3_llm_reporting" / "data" / "module2_processed" / "processed",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def rounded(value: Any, digits: int = 3) -> float:
    return round(as_float(value), digits)


def stats(values: list[float]) -> dict[str, float]:
    clean = [value for value in values if value == value]
    if not clean:
        return {"mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": round(float(pd.Series(clean).mean()), 3),
        "p90": round(float(pd.Series(clean).quantile(0.9)), 3),
        "max": round(float(pd.Series(clean).max()), 3),
    }


def freq_band(rpm: float) -> str:
    if rpm >= 10:
        return "very frequent"
    if rpm >= 5:
        return "frequent"
    if rpm >= 2:
        return "moderate"
    return "low"


def load_prepared_meta(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = root / "results" / "prepared_inputs" / "m1_configuration_input_summary.csv"
    out = {}
    for row in csv_rows(path):
        key = (row["condition"], row["sequence"])
        frames = as_int(row.get("prepared_rows"))
        out[key] = {
            "fps": 30,
            "total_frames": frames,
            "total_duration_sec": round(frames / 30.0, 3) if frames else 0.0,
            "missing_rows": as_int(row.get("missing_rows")),
        }
    return out


def load_shared_regions(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = root / "results" / "metrics" / "shared_region_case_metrics.csv"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in csv_rows(path):
        key = (row["condition"], row["clip"])
        out.setdefault(key, {})[row["region"]] = {
            "mean_score": rounded(row.get("mean_score")),
            "p90_score": rounded(row.get("p90_score")),
            "peak_score": as_int(row.get("peak_score")),
            "risk_bin": row.get("risk_bin", ""),
            "n_frames": as_int(row.get("n_frames")),
        }
    return out


def load_shared_angle_summaries(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for condition in CONDITIONS:
        angle_dir = root / "inputs" / "shared_angle_csv" / condition
        for path in sorted(angle_dir.glob("*_angle.csv")):
            if path.name.startswith("._"):
                continue
            clip = path.name.removesuffix("_angle.csv")
            df = pd.read_csv(path, header=[0, 1], index_col=0)
            out[(condition, clip)] = {
                "trunk_flexion_deg": stats(df[("trunk", "flexion")].tolist()),
                "trunk_bending_abs_deg": stats(df[("trunk", "bending")].abs().tolist()),
                "trunk_twisting_abs_deg": stats(df[("trunk", "twisting")].abs().tolist()),
                "upper_arm_flexion_deg": stats(
                    df[[("upperarm", "left_flexion"), ("upperarm", "right_flexion")]].max(axis=1).tolist()
                ),
                "upper_arm_abduction_deg": stats(
                    df[[("upperarm", "left_abduction"), ("upperarm", "right_abduction")]].max(axis=1).tolist()
                ),
                "lower_arm_flexion_deg": stats(
                    df[[("lower arm", "left_flexion"), ("lower arm", "right_flexion")]].max(axis=1).tolist()
                ),
                "knee_flexion_deg": stats(
                    df[[("knee", "left_flexion"), ("knee", "right_flexion")]].max(axis=1).tolist()
                ),
                "leg_support_ratio": stats(df[("leg_support", "ratio")].tolist()),
            }
    return out


def load_duration(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = root / "results" / "shared_duration" / "shared_duration_summary.csv"
    out = {}
    for row in csv_rows(path):
        clip = row["VIDEO_NM"].removesuffix("_angle")
        key = (row["condition"], clip)
        out[key] = {
            "scope": "shared trunk/arm/leg angle signals",
            "total_duration_sec": rounded(row.get("Total duration (s)")),
            "total_static_duration_sec": rounded(row.get("Total static duration (s)")),
            "static_posture_ratio_percent": rounded(row.get("Static posture ratio (%)")),
            "static_event_frequency_per_min": rounded(row.get("Frequency (events/min)")),
            "mean_holding_time_sec": rounded(row.get("Mean holding time (s)")),
        }
    return out


def load_repetition(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for condition in CONDITIONS:
        path = root / "results" / "repetition" / condition / "repetition_case_summary.csv"
        for row in csv_rows(path):
            rpm = as_float(row.get("rpm_mean"))
            out[(condition, row["clip"])] = {
                "method": "REP++ on normalized major-body-joint coordinates",
                "total_repetitions": as_int(row.get("repetitions_total_peaks")),
                "total_repetitions_from_peaks": as_int(row.get("repetitions_total_peaks")),
                "mean_period_sec": rounded(row.get("mean_period_sec")),
                "repetition_rate_cycle_per_min": round(rpm, 3),
                "frequency_band": freq_band(rpm),
                "quality_flag": row.get("quality_flag", ""),
                "interpretation_note": (
                    "Use as configuration-sensitive temporal evidence; quality_flag=poor "
                    "indicates that the repetition estimate should be interpreted cautiously."
                ),
            }
    return out


REBAR_FIELD_TO_MODULE2_CASE = {
    "RebarTying_01": "RebarTying_00",
    "RebarTying_02": "RebarTying_01",
}


def load_module2_summary(processed_dir: Path, source_case: str) -> dict[str, Any] | None:
    candidates = []
    mapped_case = REBAR_FIELD_TO_MODULE2_CASE.get(source_case)
    if mapped_case:
        candidates.append(mapped_case)
    candidates.append(source_case)

    for candidate in candidates:
        path = processed_dir / f"{candidate}.json"
        if not path.exists():
            continue
        payload = read_json(path)
        if "posture_summary" in payload and "joint_angle_summary" in payload:
            return payload
        return build_input_summary(payload)
    return None


def sam_additional_evidence(processed_dir: Path, source_case: str) -> dict[str, Any] | None:
    summary = load_module2_summary(processed_dir, source_case)
    if summary is None:
        return None
    joint = summary.get("joint_angle_summary", {})
    body = summary.get("posture_summary", {}).get("body_part_reba", {})
    return {
        "full_reba_final": summary.get("posture_summary", {}).get("final_reba", {}),
        "neck_reba": body.get("neck", {}),
        "wrist_reba": body.get("wrist", {}),
        "neck_joint_angles": {
            "flexion_deg": joint.get("neck_flexion_deg", {}),
            "bending_deg": joint.get("neck_bending_deg", {}),
            "twisting_deg": joint.get("neck_twisting_deg", {}),
        },
        "wrist_joint_angles": {
            "flexion_deg": joint.get("wrist_flexion_deg", {}),
            "twisting_deg": joint.get("wrist_twisting_deg", {}),
        },
    }


def build_payload(
    *,
    root: Path,
    condition: str,
    case_id: str,
    source_case: str,
    meta_map: dict[tuple[str, str], dict[str, Any]],
    region_map: dict[tuple[str, str], dict[str, Any]],
    angle_map: dict[tuple[str, str], dict[str, Any]],
    duration_map: dict[tuple[str, str], dict[str, Any]],
    repetition_map: dict[tuple[str, str], dict[str, Any]],
    sam_full_module2_dir: Path,
) -> dict[str, Any]:
    has_full = condition == "sam3db"
    additional = sam_additional_evidence(sam_full_module2_dir, source_case) if has_full else None
    if additional is None:
        additional = {
            "not_estimable": [
                "full final REBA score",
                "neck flexion/bending/twisting",
                "wrist flexion/twisting",
            ],
            "reason": (
                "AlphaPose-MotionBERT provides common major body joints but does not provide "
                "the head, ear, and hand landmarks required for Module 2 neck and wrist scoring."
            ),
        }

    evidence_limitation = (
        "SAM-3DB provides shared body posture, duration, repetition, and additional neck/wrist evidence. "
        "RGB visual context is not included in this Module 1 configuration payload."
        if has_full
        else (
            "AlphaPose-MotionBERT provides shared major-body-joint posture, duration, and repetition evidence. "
            "Full final REBA, neck, and wrist evidence are explicitly not estimable from the available joints."
        )
    )

    return {
        "case_id": case_id,
        "evidence_configuration": condition,
        "evidence_available": {
            "shared_body_posture": True,
            "full_reba_final": has_full,
            "neck_scoring": has_full,
            "wrist_scoring": has_full,
            "duration": (condition, source_case) in duration_map,
            "repetition": (condition, source_case) in repetition_map,
            "rgb": False,
        },
        "evidence_limitation": evidence_limitation,
        "meta": meta_map.get((condition, source_case), {"fps": 30}),
        "shared_posture_summary": {
            "scope": "trunk, upper arm, lower arm, and leg only",
            "risk_bin_basis": "p90 shared-region posture score",
            "regions": region_map.get((condition, source_case), {}),
        },
        "shared_joint_angle_summary": {
            "scope": "angles computable from major body joints common to both pose configurations",
            **angle_map.get((condition, source_case), {}),
        },
        "additional_pose_dependent_evidence": additional,
        "duration_summary": duration_map.get((condition, source_case), {}),
        "repetition_summary": repetition_map.get((condition, source_case), {}),
    }


def main() -> None:
    args = parse_args()
    root = args.root
    payload_root = root / "payloads"
    meta_map = load_prepared_meta(root)
    region_map = load_shared_regions(root)
    angle_map = load_shared_angle_summaries(root)
    duration_map = load_duration(root)
    repetition_map = load_repetition(root)
    manifest_cases = []

    for index, source_case in enumerate(CASES, start=1):
        case_id = f"case_{index:02d}"
        case_entry = {"case_id": case_id, "source_case": source_case, "payloads": {}}
        for condition in CONDITIONS:
            out_dir = payload_root / condition
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{case_id}.json"
            if out_path.exists() and not args.force:
                raise FileExistsError(f"Refusing to overwrite without --force: {out_path}")
            payload = build_payload(
                root=root,
                condition=condition,
                case_id=case_id,
                source_case=source_case,
                meta_map=meta_map,
                region_map=region_map,
                angle_map=angle_map,
                duration_map=duration_map,
                repetition_map=repetition_map,
                sam_full_module2_dir=args.sam_full_module2_dir,
            )
            write_json(out_path, payload)
            case_entry["payloads"][condition] = str(out_path)
        manifest_cases.append(case_entry)

    manifest = {
        "analysis": "module1_pose_configuration_contribution",
        "conditions": list(CONDITIONS),
        "case_count": len(manifest_cases),
        "case_mapping_note": (
            "Prompt-visible payloads use anonymized case IDs. Source case IDs are "
            "retained only in this manifest for auditability."
        ),
        "cases": manifest_cases,
    }
    write_json(payload_root / "manifest.json", manifest)
    print(f"[ok] wrote Module 3 payloads under {payload_root}")


if __name__ == "__main__":
    main()
