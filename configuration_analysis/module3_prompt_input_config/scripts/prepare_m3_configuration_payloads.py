#!/usr/bin/env python3
"""Prepare Module #3 configuration-analysis payloads.

The Module #3 configuration uses a 2 x 3 design:

- input condition: Module #2 only, Module #2 + RGB
- prompt condition: neutral, original RGB-compatible, bounded context-augmented

This script builds the two input conditions from the latest Module #2
configuration payloads so Module #3 can be evaluated with identical cases.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


INPUT_CONDITIONS = ("module2_only", "module2_rgb")
PROMPT_CONDITIONS = (
    "neutral",
    "original_rgb_compatible",
    "bounded_context_augmented",
)


def parse_args() -> argparse.Namespace:
    default_m3_root = Path(__file__).resolve().parents[1]
    default_m2_root = default_m3_root.parent / "m2_config"
    if not default_m2_root.exists():
        default_m2_root = default_m3_root.parent / "m2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path, default=default_m3_root)
    parser.add_argument("--m2-root", type=Path, default=default_m2_root)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite without --force: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def source_manifest_cases(m2_root: Path) -> list[dict[str, Any]]:
    manifest_path = m2_root / "payloads" / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        cases = manifest.get("cases")
        if isinstance(cases, list) and cases:
            return cases

    cases = []
    for full_path in sorted((m2_root / "payloads" / "full_module2").glob("case_*.json")):
        if full_path.name.startswith("._"):
            continue
        case_id = full_path.stem
        rgb_candidates = sorted(
            path
            for path in (m2_root / "payloads" / "rgb_only" / "assets").glob(f"{case_id}.*")
            if path.is_file() and not path.name.startswith("._")
        )
        if not rgb_candidates:
            raise FileNotFoundError(f"Missing RGB asset for {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "source_sample_id": case_id,
                "full_module2_payload_file": str(full_path),
                "rgb_asset_file": str(rgb_candidates[0]),
            }
        )
    if not cases:
        raise FileNotFoundError(f"No Module #2 payload cases found under {m2_root}")
    return cases


def resolve_case_path(
    *,
    recorded_path: str | None,
    m2_root: Path,
    case_id: str,
    payload_relative: Path,
) -> Path:
    if recorded_path:
        candidate = Path(recorded_path)
        if candidate.exists():
            return candidate

    fallback = m2_root / payload_relative
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"Missing Module #2 source for {case_id}. "
        f"Recorded path: {recorded_path or 'none'}; fallback: {fallback}"
    )


def build_module2_only_payload(full_payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    payload = deepcopy(full_payload)
    payload["case_id"] = case_id
    payload["evidence_configuration"] = "module2_only"
    evidence = dict(payload.get("evidence_available", {}))
    evidence.update(
        {
            "rgb": False,
            "reba": True,
            "joint_angles": True,
            "duration": True,
            "repetition": True,
        }
    )
    payload["evidence_available"] = evidence
    payload.pop("rgb_evidence", None)
    payload["evidence_limitation"] = (
        "Full Module #2 numerical summaries are provided. RGB visual context is "
        "absent in this condition."
    )
    return payload


def build_module2_rgb_payload(
    full_payload: dict[str, Any],
    case_id: str,
    rgb_asset_name: str,
) -> dict[str, Any]:
    payload = deepcopy(full_payload)
    payload["case_id"] = case_id
    payload["evidence_configuration"] = "module2_rgb"
    evidence = dict(payload.get("evidence_available", {}))
    evidence.update(
        {
            "rgb": True,
            "reba": True,
            "joint_angles": True,
            "duration": True,
            "repetition": True,
        }
    )
    payload["evidence_available"] = evidence
    payload["rgb_evidence"] = {
        "asset_file": f"assets/{rgb_asset_name}",
        "description": (
            "A representative peak-risk RGB frame is provided as supplementary "
            "visual context for this anonymized case."
        ),
    }
    payload["evidence_limitation"] = (
        "Full Module #2 numerical summaries are provided. RGB visual context is "
        "also provided, but it should be used only for directly observable "
        "posture or scene context."
    )
    return payload


def main() -> None:
    args = parse_args()
    payload_root = args.m3_root / "payloads"
    module2_only_dir = payload_root / "module2_only"
    module2_rgb_dir = payload_root / "module2_rgb"
    rgb_asset_dir = module2_rgb_dir / "assets"
    for directory in (module2_only_dir, module2_rgb_dir, rgb_asset_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    for case in source_manifest_cases(args.m2_root):
        case_id = case["case_id"]
        full_path = resolve_case_path(
            recorded_path=case.get("full_module2_payload_file"),
            m2_root=args.m2_root,
            case_id=case_id,
            payload_relative=Path("payloads") / "full_module2" / f"{case_id}.json",
        )
        rgb_source_path = resolve_case_path(
            recorded_path=case.get("rgb_asset_file"),
            m2_root=args.m2_root,
            case_id=case_id,
            payload_relative=Path("payloads") / "rgb_only" / "assets" / f"{case_id}.jpg",
        )

        full_payload = read_json(full_path)
        rgb_asset_name = f"{case_id}{rgb_source_path.suffix.lower() or '.jpg'}"
        rgb_target_path = rgb_asset_dir / rgb_asset_name
        if rgb_target_path.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite without --force: {rgb_target_path}")
        shutil.copy2(rgb_source_path, rgb_target_path)

        module2_only_path = module2_only_dir / f"{case_id}.json"
        module2_rgb_path = module2_rgb_dir / f"{case_id}.json"
        write_json(
            module2_only_path,
            build_module2_only_payload(full_payload, case_id),
            args.force,
        )
        write_json(
            module2_rgb_path,
            build_module2_rgb_payload(full_payload, case_id, rgb_asset_name),
            args.force,
        )

        manifest_entries.append(
            {
                "case_id": case_id,
                "source_sample_id": case.get("source_sample_id", case_id),
                "module2_source_payload_file": str(full_path),
                "rgb_source_file": str(rgb_source_path),
                "module2_only_payload_file": str(module2_only_path),
                "module2_rgb_payload_file": str(module2_rgb_path),
                "rgb_asset_file": str(rgb_target_path),
            }
        )

    manifest = {
        "analysis": "module3_configuration_contribution",
        "input_conditions": list(INPUT_CONDITIONS),
        "prompt_conditions": list(PROMPT_CONDITIONS),
        "case_count": len(manifest_entries),
        "case_mapping_note": (
            "Prompt-visible payloads use anonymized case IDs. Source sample IDs "
            "are retained only in this manifest for auditability."
        ),
        "cases": manifest_entries,
    }
    write_json(payload_root / "manifest.json", manifest, args.force)
    print(f"Prepared {len(manifest_entries)} cases")
    print(f"Payload root: {payload_root}")
    print(f"Manifest: {payload_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
