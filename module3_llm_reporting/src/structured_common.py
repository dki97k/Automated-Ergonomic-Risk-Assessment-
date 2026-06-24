#!/usr/bin/env python3
"""Shared utilities for Module 3 structured-report validation."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable


RISK_SUMMARY_FIELDS = (
    "posture_risk",
    "duration_risk",
    "repetition_risk",
)

KEY_FACTOR_FIELDS = (
    "trunk_overflexion",
    "neck_overflexion_or_extension",
    "upper_arm_elevation",
    "wrist_deviation",
    "knee_overflexion",
    "prolonged_static_posture",
    "repetitive_work",
)

REBA_PART_FIELDS = (
    "neck",
    "trunk",
    "leg",
    "upper_arm",
    "lower_arm",
    "wrist",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def stats(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if safe_float(v) is not None]
    if not clean:
        return {"mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": round(mean(clean), 3),
        "p90": round(percentile(clean, 0.9), 3),
        "max": round(max(clean), 3),
    }


def sorted_frame_items(frame_dict: dict[str, Any]) -> list[tuple[int, Any]]:
    items: list[tuple[int, Any]] = []
    for key, value in frame_dict.items():
        try:
            frame = int(key)
        except ValueError:
            continue
        items.append((frame, value))
    return sorted(items, key=lambda item: item[0])


def collect_angle_values(
    angle_frames: dict[str, Any],
    extractor: Callable[[dict[str, Any]], float | None],
) -> list[float]:
    values: list[float] = []
    for _, frame in sorted_frame_items(angle_frames):
        if not isinstance(frame, dict):
            continue
        value = extractor(frame)
        if value is not None:
            values.append(value)
    return values


def get_nested_float(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return safe_float(current)


def max_present(*values: float | None) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return max(clean)


def max_abs_present(*values: float | None) -> float | None:
    clean = [abs(v) for v in values if v is not None]
    if not clean:
        return None
    return max(clean)


def build_input_summary(processed: dict[str, Any]) -> dict[str, Any]:
    meta = processed.get("meta", {})
    fps = safe_float(meta.get("fps")) or 30.0
    total_frames = int(safe_float(meta.get("total_frames")) or len(processed.get("angles", {})))
    total_duration_sec = round(total_frames / fps, 3) if fps else 0.0
    sample_id = meta.get("video_id") or meta.get("sample_id") or "unknown"

    reba_frames = processed.get("reba", {})
    reba_values = {
        key: [
            safe_float(frame.get(key))
            for _, frame in sorted_frame_items(reba_frames)
            if isinstance(frame, dict)
        ]
        for key in (*REBA_PART_FIELDS, "final")
    }
    reba_values = {
        key: [value for value in values if value is not None]
        for key, values in reba_values.items()
    }

    final_values = reba_values.get("final", [])
    final_count = len(final_values) or 1
    risk_bin_distribution = {
        "low": round(sum(1 for value in final_values if value <= 3) / final_count, 3),
        "medium": round(sum(1 for value in final_values if 4 <= value <= 7) / final_count, 3),
        "high": round(sum(1 for value in final_values if value >= 8) / final_count, 3),
    }
    longest_high_run_frames = 0
    current_high_run_frames = 0
    for value in final_values:
        if value >= 8:
            current_high_run_frames += 1
            longest_high_run_frames = max(longest_high_run_frames, current_high_run_frames)
        else:
            current_high_run_frames = 0

    angle_frames = processed.get("angles", {})
    joint_angle_summary = {
        "trunk_flexion_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: max_present(get_nested_float(f, "trunk", "flexion"), 0.0),
            )
        ),
        "neck_flexion_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: (
                    abs(value)
                    if (value := get_nested_float(f, "neck", "flexion")) is not None
                    else None
                ),
            )
        ),
        "neck_bending_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: (
                    abs(value)
                    if (value := get_nested_float(f, "neck", "bending")) is not None
                    else None
                ),
            )
        ),
        "neck_twisting_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: (
                    abs(value)
                    if (value := get_nested_float(f, "neck", "twisting")) is not None
                    else None
                ),
            )
        ),
        "upper_arm_flexion_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: max_present(
                    get_nested_float(f, "upperarm", "left_flexion"),
                    get_nested_float(f, "upperarm", "right_flexion"),
                ),
            )
        ),
        "upper_arm_abduction_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: max_present(
                    get_nested_float(f, "upperarm", "left_abduction"),
                    get_nested_float(f, "upperarm", "right_abduction"),
                ),
            )
        ),
        "wrist_flexion_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: max_abs_present(
                    get_nested_float(f, "wrist", "left_flexion"),
                    get_nested_float(f, "wrist", "right_flexion"),
                ),
            )
        ),
        "wrist_twisting_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: max_abs_present(
                    get_nested_float(f, "wrist", "left_twisting"),
                    get_nested_float(f, "wrist", "right_twisting"),
                ),
            )
        ),
        "knee_flexion_deg": stats(
            collect_angle_values(
                angle_frames,
                lambda f: max_present(
                    get_nested_float(f, "knee", "left_flexion"),
                    get_nested_float(f, "knee", "right_flexion"),
                ),
            )
        ),
    }

    duration = processed.get("duration", {})
    whole_body_segments = duration.get("whole_body", []) if isinstance(duration, dict) else []
    segment_durations = [
        safe_float(segment.get("duration_sec")) or 0.0
        for segment in whole_body_segments
        if isinstance(segment, dict)
    ]
    total_static_duration_sec = sum(segment_durations)
    max_static_segment_sec = max(segment_durations) if segment_durations else 0.0
    max_moderate_or_high = 0.0
    max_high = 0.0
    final_by_frame = {
        frame: safe_float(data.get("final"))
        for frame, data in sorted_frame_items(reba_frames)
        if isinstance(data, dict)
    }
    for segment in whole_body_segments:
        if not isinstance(segment, dict):
            continue
        start = int(safe_float(segment.get("start_frame")) or 0)
        end = int(safe_float(segment.get("end_frame")) or start)
        duration_sec = safe_float(segment.get("duration_sec")) or 0.0
        segment_scores = [
            score for frame, score in final_by_frame.items() if start <= frame <= end and score is not None
        ]
        segment_max = max(segment_scores) if segment_scores else 0.0
        if segment_max >= 4:
            max_moderate_or_high = max(max_moderate_or_high, duration_sec)
        if segment_max >= 8:
            max_high = max(max_high, duration_sec)

    repetition = processed.get("repetition", {})
    input_summary = {
        "sample_id": sample_id,
        "meta": {
            "fps": int(fps) if fps.is_integer() else fps,
            "total_frames": total_frames,
            "total_duration_sec": total_duration_sec,
        },
        "posture_summary": {
            "final_reba": {
                **stats(final_values),
                "risk_bin_distribution": risk_bin_distribution,
                "longest_high_risk_run_frames": longest_high_run_frames,
                "longest_high_risk_run_sec": round(longest_high_run_frames / fps, 3)
                if fps
                else 0.0,
            },
            "body_part_reba": {
                part: stats(reba_values.get(part, [])) for part in REBA_PART_FIELDS
            },
        },
        "joint_angle_summary": joint_angle_summary,
        "duration_summary": {
            "static_posture_ratio": round(
                total_static_duration_sec / total_duration_sec, 3
            )
            if total_duration_sec
            else 0.0,
            "total_static_duration_sec": round(total_static_duration_sec, 3),
            "max_static_segment_sec": round(max_static_segment_sec, 3),
            "static_event_count": len(segment_durations),
            "max_moderate_or_high_risk_exposure_sec": round(max_moderate_or_high, 3),
            "max_high_risk_exposure_sec": round(max_high, 3),
        },
        "repetition_summary": {
            "total_repetitions": int(safe_float(repetition.get("total_repetitions")) or 0),
            "repetition_rate_cycle_per_min": round(
                safe_float(repetition.get("repetition_rate_cycle_per_min")) or 0.0, 3
            ),
            "mean_period_sec": round(safe_float(repetition.get("mean_period_sec")) or 0.0, 3),
            "std_period_sec": round(safe_float(repetition.get("std_period_sec")) or 0.0, 3),
        },
    }
    return input_summary


def label(value: bool) -> str:
    return "Yes" if value else "No"


def factor(label_value: bool, evidence: str) -> dict[str, str]:
    return {"label": label(label_value), "evidence": evidence}


def build_key_factor_reference(summary: dict[str, Any]) -> dict[str, Any]:
    joint = summary["joint_angle_summary"]
    posture = summary["posture_summary"]["final_reba"]
    repetition = summary["repetition_summary"]

    trunk = joint["trunk_flexion_deg"]
    neck = joint["neck_flexion_deg"]
    neck_bending = joint["neck_bending_deg"]
    neck_twisting = joint["neck_twisting_deg"]
    upper_flexion = joint["upper_arm_flexion_deg"]
    upper_abduction = joint["upper_arm_abduction_deg"]
    wrist_flexion = joint["wrist_flexion_deg"]
    wrist_twisting = joint["wrist_twisting_deg"]
    knee = joint["knee_flexion_deg"]

    return {
        "sample_id": summary["sample_id"],
        "key_risk_factors": {
            "trunk_overflexion": factor(
                trunk["max"] >= 60,
                f"trunk flexion max={trunk['max']}",
            ),
            "neck_overflexion_or_extension": factor(
                max(neck["max"], neck_bending["max"], neck_twisting["max"]) >= 20,
                (
                    f"neck flexion max={neck['max']}; "
                    f"bending max={neck_bending['max']}; "
                    f"twisting max={neck_twisting['max']}"
                ),
            ),
            "upper_arm_elevation": factor(
                max(upper_flexion["max"], upper_abduction["max"]) >= 90,
                (
                    f"upper arm flexion max={upper_flexion['max']}; "
                    f"abduction max={upper_abduction['max']}"
                ),
            ),
            "wrist_deviation": factor(
                wrist_flexion["max"] >= 15 or wrist_twisting["max"] >= 45,
                (
                    f"wrist flexion max={wrist_flexion['max']}; "
                    f"twisting max={wrist_twisting['max']}"
                ),
            ),
            "knee_overflexion": factor(
                knee["max"] >= 60,
                f"knee flexion max={knee['max']}",
            ),
            "prolonged_static_posture": factor(
                posture["longest_high_risk_run_frames"] >= 120,
                (
                    "longest final REBA >= 8 run="
                    f"{posture['longest_high_risk_run_frames']} frames"
                ),
            ),
            "repetitive_work": factor(
                repetition["total_repetitions"] > 0
                and repetition["repetition_rate_cycle_per_min"] > 0,
                (
                    f"total repetitions={repetition['total_repetitions']}, "
                    f"rate={repetition['repetition_rate_cycle_per_min']}"
                ),
            ),
        },
    }
