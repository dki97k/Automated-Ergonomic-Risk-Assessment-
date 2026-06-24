#!/usr/bin/env python3
"""Normalize repetition JSONL coordinates for REP++ pipeline input.

The provided Newkeypoints files use snake_case joint names and list coordinates.
The REP++ pipeline expects display-style names, such as ``Wrist (L)``, and
coordinate dictionaries with ``x``, ``y``, and ``z`` keys.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NAME_MAP = {
    "nose": "Head",
    "neck": "Neck",
    "left_shoulder": "Shoulder (L)",
    "right_shoulder": "Shoulder (R)",
    "left_elbow": "Elbow (L)",
    "right_elbow": "Elbow (R)",
    "left_wrist": "Wrist (L)",
    "right_wrist": "Wrist (R)",
    "left_hip": "Hip (L)",
    "right_hip": "Hip (R)",
    "left_knee": "Knee (L)",
    "right_knee": "Knee (R)",
    "left_ankle": "Ankle (L)",
    "right_ankle": "Ankle (R)",
}


def vec(joints: dict, name: str) -> list[float] | None:
    value = joints.get(name)
    if value is None:
        return None
    if isinstance(value, dict):
        return [float(value["x"]), float(value["y"]), float(value["z"])]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def mean_vec(*values: list[float] | None) -> list[float] | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return [sum(v[i] for v in valid) / len(valid) for i in range(3)]


def coord(value: list[float]) -> dict[str, float]:
    return {"x": float(value[0]), "y": float(value[1]), "z": float(value[2])}


def convert_record(record: dict) -> dict:
    src = record.get("joints", {})
    out = {}
    for source_name, target_name in NAME_MAP.items():
        value = vec(src, source_name)
        if value is not None:
            out[target_name] = coord(value)

    left_hip = vec(src, "left_hip")
    right_hip = vec(src, "right_hip")
    pelvis = mean_vec(left_hip, right_hip)
    if pelvis is not None:
        out["Pelvis (Origin)"] = coord(pelvis)

    neck = vec(src, "neck")
    spine = mean_vec(pelvis, neck)
    if spine is not None:
        out["Spine"] = coord(spine)

    return {"frame": int(record.get("frame", 0)), "joints": out}


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            converted = convert_record(json.loads(line))
            if "Pelvis (Origin)" in converted["joints"] and "Neck" in converted["joints"]:
                n_out += 1
                fout.write(json.dumps(converted, separators=(",", ":")) + "\n")
    return n_in, n_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("<private_workspace>/m2/REP++/Newkeypoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("<private_workspace>/configuration_analysis/m1/inputs/repetition_jsonl/normalized"),
    )
    args = parser.parse_args()

    rows = []
    for src in sorted(args.input_dir.glob("*.jsonl")):
        if src.name.startswith("._"):
            continue
        dst = args.output_dir / src.name
        n_in, n_out = convert_file(src, dst)
        rows.append((src.name, n_in, n_out, str(dst)))

    for name, n_in, n_out, dst in rows:
        print(f"{name}: {n_out}/{n_in} -> {dst}")


if __name__ == "__main__":
    main()
