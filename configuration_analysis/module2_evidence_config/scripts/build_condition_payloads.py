#!/usr/bin/env python3
"""Build anonymized Module #2 configuration-analysis payloads.

This prepares three comparable evidence configurations for Module #3:

- rgb_only: anonymized peak-risk RGB image evidence only
- reba_only: REBA-derived posture summaries only
- full_module2: posture, joint-angle, duration, and repetition summaries

The prompt-visible payloads use neutral case IDs so task names such as masonry
or rebar do not become implicit evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M3_SRC = PROJECT_ROOT / "m3" / "src"
sys.path.insert(0, str(M3_SRC))

from structured_common import (  # noqa: E402
    REBA_PART_FIELDS,
    build_input_summary,
    read_json,
    safe_float,
    sorted_frame_items,
    stats,
    write_json,
)


def parse_args() -> argparse.Namespace:
    m2_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m2-root",
        type=Path,
        default=m2_root,
        help="configuration_analysis/m2 root directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing payload files",
    )
    return parser.parse_args()


def risk_bin_distribution(final_values: list[float]) -> dict[str, float]:
    count = len(final_values) or 1
    return {
        "low": round(sum(1 for value in final_values if value <= 3) / count, 3),
        "medium": round(sum(1 for value in final_values if 4 <= value <= 7) / count, 3),
        "high": round(sum(1 for value in final_values if value >= 8) / count, 3),
    }


def longest_high_run(final_values: list[float], fps: float) -> dict[str, float | int]:
    longest = 0
    current = 0
    for value in final_values:
        if value >= 8:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {
        "longest_high_risk_run_frames": longest,
        "longest_high_risk_run_sec": round(longest / fps, 3) if fps else 0.0,
    }


def collect_reba_values(reba_frames: dict[str, Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for key in (*REBA_PART_FIELDS, "final"):
        values[key] = []
    for _, frame in sorted_frame_items(reba_frames):
        if not isinstance(frame, dict):
            continue
        for key in values:
            value = safe_float(frame.get(key))
            if value is not None:
                values[key].append(value)
    return values


def build_reba_only_summary(processed: dict[str, Any], case_id: str) -> dict[str, Any]:
    if "posture_summary" in processed:
        meta = processed.get("meta", {})
        return {
            "case_id": case_id,
            "evidence_configuration": "reba_only",
            "evidence_available": {
                "rgb": False,
                "reba": True,
                "joint_angles": False,
                "duration": False,
                "repetition": False,
            },
            "evidence_limitation": (
                "Only REBA-derived posture score summaries are provided. Duration, "
                "repetition, joint-angle, and RGB evidence are absent."
            ),
            "meta": meta,
            "posture_summary": processed.get("posture_summary", {}),
        }

    meta = processed.get("meta", {})
    fps = safe_float(meta.get("fps")) or 30.0
    reba_frames = processed.get("reba", {})
    total_frames = int(safe_float(meta.get("total_frames")) or len(reba_frames))
    total_duration_sec = round(total_frames / fps, 3) if fps else 0.0

    values = collect_reba_values(reba_frames)
    final_values = values.get("final", [])
    high_run = longest_high_run(final_values, fps)

    return {
        "case_id": case_id,
        "evidence_configuration": "reba_only",
        "evidence_available": {
            "rgb": False,
            "reba": True,
            "joint_angles": False,
            "duration": False,
            "repetition": False,
        },
        "evidence_limitation": (
            "Only REBA-derived posture score summaries are provided. Duration, "
            "repetition, joint-angle, and RGB evidence are absent."
        ),
        "meta": {
            "fps": int(fps) if float(fps).is_integer() else fps,
            "total_frames": total_frames,
            "total_duration_sec": total_duration_sec,
        },
        "posture_summary": {
            "final_reba": {
                **stats(final_values),
                "risk_bin_distribution": risk_bin_distribution(final_values),
                **high_run,
            },
            "body_part_reba": {
                part: stats(values.get(part, [])) for part in REBA_PART_FIELDS
            },
        },
    }


def anonymize_full_summary(processed: dict[str, Any], case_id: str) -> dict[str, Any]:
    if "posture_summary" in processed and "joint_angle_summary" in processed:
        summary = dict(processed)
    else:
        summary = build_input_summary(processed)
    summary.pop("sample_id", None)
    return {
        "case_id": case_id,
        "evidence_configuration": "full_module2",
        "evidence_available": {
            "rgb": False,
            "reba": True,
            "joint_angles": True,
            "duration": True,
            "repetition": True,
        },
        "evidence_limitation": (
            "Full Module #2 numerical summaries are provided. RGB visual "
            "context is absent in this condition."
        ),
        **summary,
    }


def build_rgb_only_payload(case_id: str, asset_file: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "evidence_configuration": "rgb_only",
        "evidence_available": {
            "rgb": True,
            "reba": False,
            "joint_angles": False,
            "duration": False,
            "repetition": False,
        },
        "evidence_limitation": (
            "Only one representative RGB image is provided. Numerical posture, "
            "duration, repetition, and time-series evidence are absent."
        ),
        "rgb_evidence": {
            "asset_file": asset_file,
            "description": (
                "A representative peak-risk RGB frame is provided separately "
                "as image evidence for this anonymized case."
            ),
        },
    }


def find_rgb_image(rgb_dir: Path, sample_id: str) -> Path:
    matches = sorted(
        path
        for path in rgb_dir.glob(f"{sample_id}*")
        if path.is_file() and not path.name.startswith("._")
    )
    if not matches:
        raise FileNotFoundError(f"No RGB image found for {sample_id} in {rgb_dir}")
    return matches[0]


def write_payload(path: Path, payload: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file without --force: {path}")
    write_json(path, payload)


def main() -> None:
    args = parse_args()
    root = args.m2_root
    full_input_dir = root / "inputs" / "full_module2"
    reba_input_dir = root / "inputs" / "reba"
    rgb_input_dir = root / "inputs" / "rgb"
    payload_root = root / "payloads"
    rgb_payload_dir = payload_root / "rgb_only"
    reba_payload_dir = payload_root / "reba_only"
    full_payload_dir = payload_root / "full_module2"
    rgb_asset_dir = rgb_payload_dir / "assets"

    for directory in (rgb_payload_dir, reba_payload_dir, full_payload_dir, rgb_asset_dir):
        directory.mkdir(parents=True, exist_ok=True)

    full_files = sorted(
        path for path in full_input_dir.glob("*.json") if not path.name.startswith("._")
    )
    if not full_files:
        raise SystemExit(f"No full Module #2 JSON files found: {full_input_dir}")

    manifest_entries = []
    for index, full_path in enumerate(full_files, start=1):
        source_sample_id = full_path.stem
        case_id = f"case_{index:02d}"
        reba_path = reba_input_dir / full_path.name
        rgb_path = find_rgb_image(rgb_input_dir, source_sample_id)
        if not reba_path.exists():
            raise FileNotFoundError(f"Missing REBA-only file: {reba_path}")

        full_processed = read_json(full_path)
        reba_processed = read_json(reba_path)

        rgb_ext = rgb_path.suffix.lower() or ".jpg"
        rgb_asset_name = f"{case_id}{rgb_ext}"
        rgb_asset_path = rgb_asset_dir / rgb_asset_name
        if rgb_asset_path.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite existing RGB asset without --force: {rgb_asset_path}"
            )
        shutil.copy2(rgb_path, rgb_asset_path)

        write_payload(
            rgb_payload_dir / f"{case_id}.json",
            build_rgb_only_payload(case_id, f"assets/{rgb_asset_name}"),
            args.force,
        )
        write_payload(
            reba_payload_dir / f"{case_id}.json",
            build_reba_only_summary(reba_processed, case_id),
            args.force,
        )
        write_payload(
            full_payload_dir / f"{case_id}.json",
            anonymize_full_summary(full_processed, case_id),
            args.force,
        )

        manifest_entries.append(
            {
                "case_id": case_id,
                "source_sample_id": source_sample_id,
                "rgb_source_file": str(rgb_path),
                "rgb_payload_file": str(rgb_payload_dir / f"{case_id}.json"),
                "rgb_asset_file": str(rgb_asset_path),
                "reba_payload_file": str(reba_payload_dir / f"{case_id}.json"),
                "full_module2_payload_file": str(full_payload_dir / f"{case_id}.json"),
            }
        )

    manifest = {
        "analysis": "module2_information_configuration_contribution",
        "conditions": ["rgb_only", "reba_only", "full_module2"],
        "case_count": len(manifest_entries),
        "case_mapping_note": (
            "Prompt-visible payloads use anonymized case IDs. Source sample IDs "
            "are retained only in this manifest for auditability."
        ),
        "cases": manifest_entries,
    }
    write_json(payload_root / "manifest.json", manifest)

    print(f"Prepared {len(manifest_entries)} anonymized cases")
    print(f"Payload root: {payload_root}")
    print(f"Manifest: {payload_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
