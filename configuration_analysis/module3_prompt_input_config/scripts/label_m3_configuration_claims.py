#!/usr/bin/env python3
"""Draft-label Module #3 configuration natural-report claims."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


VALID_LABELS = ("supported", "unsupported", "contradiction")

RGB_OBSERVATIONS = {
    "case_01": {
        "visible": {"trunk", "neck", "reach", "elevated_surface", "work_surface"},
        "note": "RGB shows forward bending/leaning over an elevated work surface.",
    },
    "case_02": {
        "visible": {"trunk", "neck", "reach", "elevated_surface", "work_surface"},
        "note": "RGB shows forward bending/leaning over an elevated work surface.",
    },
    "case_03": {
        "visible": {"trunk", "knee", "asymmetric_stance", "elevated_surface"},
        "note": "RGB shows asymmetric stance with one leg raised on an elevated work surface.",
    },
    "case_04": {
        "visible": {"trunk", "neck", "knee", "low_work"},
        "note": "RGB shows low-level crouching/squatting with forward trunk/neck posture.",
    },
    "case_05": {
        "visible": {"trunk", "neck", "knee", "upper_arm", "reach", "low_work"},
        "note": "RGB shows squatting/crouching, forward posture, and elevated/reaching arm.",
    },
    "case_06": {
        "visible": {"trunk", "neck", "knee", "reach", "low_work"},
        "note": "RGB shows squatting/crouching, forward posture, and reaching near floor level.",
    },
    "case_07": {
        "visible": {"trunk", "knee", "low_work", "partial_visibility"},
        "note": "RGB shows partially visible crouched/leaning low-level posture.",
    },
    "case_08": {
        "visible": {"trunk", "neck", "knee", "low_work"},
        "note": "RGB shows deep squat/crouch with forward trunk/neck posture near floor level.",
    },
}


def parse_args() -> argparse.Namespace:
    default_m3_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path, default=default_m3_root)
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        help="Defaults to <m3-root>/results/evaluation/claim_annotation_sheet.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Defaults to overwriting the annotation sheet in place.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_numbers(payload: Any) -> list[float]:
    values: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            values.extend(flatten_numbers(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(flatten_numbers(value))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        number = float(payload)
        values.append(number)
        if 0 <= number <= 1:
            values.append(number * 100)
    return values


def extract_numbers(text: str) -> list[tuple[str, float]]:
    numbers = []
    for match in re.finditer(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text):
        raw = match.group(0)
        numbers.append((raw, float(raw.rstrip("%"))))
    return numbers


def number_supported(raw: str, number: float, evidence_values: list[float]) -> bool:
    if number <= 3:
        return True
    tolerance = max(0.12, abs(number) * 0.025)
    if raw.endswith("%"):
        tolerance = max(tolerance, 1.2)
    if number >= 10 and float(number).is_integer():
        tolerance = max(tolerance, 1.1)
    return any(abs(value - number) <= tolerance for value in evidence_values)


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def evidence_available(payload: dict[str, Any]) -> dict[str, bool]:
    available = payload.get("evidence_available", {})
    return {
        "rgb": bool(available.get("rgb")),
        "reba": bool(available.get("reba")),
        "joint_angles": bool(available.get("joint_angles")),
        "duration": bool(available.get("duration")),
        "repetition": bool(available.get("repetition")),
    }


def missing_evidence_statement(text: str) -> bool:
    return has_any(
        text,
        (
            "not possible to assess",
            "cannot assess",
            "not assessable",
            "absence of",
            "without",
            "evidence limitation",
            "additional evidence is required",
            "limits the ability",
            "limited",
            "absent",
        ),
    )


def claim_evidence_needs(text: str) -> set[str]:
    needs: set[str] = set()
    if has_any(text, ("reba", "posture score", "risk score")):
        needs.add("reba")
    if has_any(text, ("angle", "flexion", "twisting", "abduction", "joint")):
        needs.add("joint_angles")
    if has_any(text, ("duration", "static", "prolonged", "sustained", "exposure time")):
        needs.add("duration")
    if has_any(text, ("repetition", "repetitive", "cycle", "frequency", "repeated")):
        needs.add("repetition")
    if has_any(
        text,
        (
            "rgb",
            "image",
            "visible",
            "observed",
            "appears",
            "floor",
            "ground",
            "low-level",
            "work surface",
            "reach",
            "crouch",
            "squat",
        ),
    ):
        needs.add("rgb")
    return needs


def missing_statement_matches(text: str, available: dict[str, bool]) -> bool:
    if not missing_evidence_statement(text):
        return False
    if "duration" in text and not available["duration"]:
        return True
    if has_any(text, ("repetition", "repetitive", "frequency", "cycle")) and not available["repetition"]:
        return True
    if has_any(text, ("joint", "angle")) and not available["joint_angles"]:
        return True
    if "reba" in text and not available["reba"]:
        return True
    if has_any(text, ("rgb", "image", "visual", "work environment", "task details")) and not available["rgb"]:
        return True
    return False


def visible_support(case_id: str, text: str) -> bool:
    observation = RGB_OBSERVATIONS.get(case_id, {"visible": set()})
    visible = observation["visible"]
    if has_any(text, ("posture", "awkward", "neutral", "ergonomic risk")):
        return bool(visible & {"trunk", "neck", "knee", "upper_arm", "reach"})
    if has_any(text, ("bend", "lean", "trunk", "back")):
        return "trunk" in visible
    if "neck" in text:
        return "neck" in visible
    if has_any(text, ("knee", "squat", "crouch", "kneel")):
        return "knee" in visible
    if has_any(text, ("arm", "shoulder", "reach")):
        return bool(visible & {"upper_arm", "reach"})
    if has_any(text, ("low", "floor", "ground")):
        return "low_work" in visible
    if has_any(text, ("height", "elevated", "surface", "work surface")):
        return "elevated_surface" in visible or "work_surface" in visible
    if has_any(text, ("constraint", "constrained", "limited space")):
        return bool(visible & {"low_work", "partial_visibility", "elevated_surface"})
    return False


def numerical_factor_supported(payload: dict[str, Any], text: str) -> bool:
    posture = payload.get("posture_summary", {})
    final_reba = posture.get("final_reba", {})
    body_reba = posture.get("body_part_reba", {})
    joint = payload.get("joint_angle_summary", {})
    repetition = payload.get("repetition_summary", {})
    duration = payload.get("duration_summary", {})

    if has_any(text, ("reba", "risk level", "posture score", "medium", "moderate", "high-risk")):
        return bool(final_reba)
    if "neck" in text:
        return "neck" in body_reba or "neck_flexion_deg" in joint
    if has_any(text, ("trunk", "back")):
        return "trunk" in body_reba or "trunk_flexion_deg" in joint
    if has_any(text, ("wrist", "hand")):
        return "wrist" in body_reba or "wrist_flexion_deg" in joint
    if has_any(text, ("arm", "shoulder", "reach")):
        return "upper_arm" in body_reba or "upper_arm_flexion_deg" in joint
    if has_any(text, ("knee", "leg", "squat", "crouch")):
        return "leg" in body_reba or "knee_flexion_deg" in joint
    if has_any(text, ("repetition", "repetitive", "cycle", "frequency")):
        return repetition.get("total_repetitions", 0) > 0
    if has_any(text, ("duration", "static", "sustained", "prolonged", "exposure")):
        return bool(duration)
    return False


def static_contradiction(payload: dict[str, Any], text: str) -> str | None:
    duration = payload.get("duration_summary", {})
    if not duration:
        return None
    static_ratio = float(duration.get("static_posture_ratio", 0) or 0)
    max_static = float(duration.get("max_static_segment_sec", 0) or 0)
    if has_any(text, ("prolonged static", "high static", "substantial static")):
        if static_ratio <= 0.1 and max_static < 10:
            return f"static evidence is low: ratio={static_ratio}, max segment={max_static}s"
    return None


def unsupported_specificity(text: str) -> str | None:
    if has_any(text, ("fall", "falls", "fall risk", "safety")):
        return "fall/safety claims are outside the provided ergonomic evidence"
    if "injury" in text:
        return "injury prediction is not directly supported by the evidence"
    if "training" in text:
        return "training recommendation is not supported by input evidence"
    if has_any(text, ("rest break", "breaks", "rotation", "work schedule", "staffing")):
        return "break/rotation/work-schedule details are not provided"
    if has_any(text, ("monitoring program", "policy", "administrative control")):
        return "organizational intervention details are not provided"
    return None


def missing_positive_evidence_reason(text: str, available: dict[str, bool]) -> str | None:
    if missing_evidence_statement(text):
        return None
    needs = claim_evidence_needs(text)
    missing = [need for need in sorted(needs) if not available.get(need, False)]
    if missing:
        return "claim uses evidence absent from this input condition: " + ", ".join(missing)
    return None


def evidence_note(payload: dict[str, Any], case_id: str, input_condition: str) -> str:
    final = payload.get("posture_summary", {}).get("final_reba", {})
    duration = payload.get("duration_summary", {})
    repetition = payload.get("repetition_summary", {})
    note = (
        "Module #2 evidence: final REBA mean/p90/max="
        f"{final.get('mean')}/{final.get('p90')}/{final.get('max')}; "
        f"static ratio={duration.get('static_posture_ratio')}; "
        f"reps/rate={repetition.get('total_repetitions')}/"
        f"{repetition.get('repetition_rate_cycle_per_min')}"
    )
    if input_condition == "module2_rgb":
        rgb_note = RGB_OBSERVATIONS.get(case_id, {}).get("note", "RGB image evidence reviewed.")
        return f"{note}; {rgb_note}"
    return note


def label_claim(row: dict[str, str], payload: dict[str, Any]) -> tuple[str, str, str]:
    text_original = row["claim_text"]
    text = text_original.lower()
    case_id = row["case_id"]
    input_condition = row.get("input_condition") or payload.get("evidence_configuration", "")
    available = evidence_available(payload)
    values = flatten_numbers(payload)

    bad_numbers = [
        raw
        for raw, number in extract_numbers(text_original)
        if not number_supported(raw, number, values)
    ]
    if bad_numbers:
        return (
            "contradiction",
            evidence_note(payload, case_id, input_condition),
            "Initial audit: numeric value(s) not supported by input evidence: "
            + ", ".join(bad_numbers),
        )

    contradiction = static_contradiction(payload, text)
    if contradiction:
        return (
            "contradiction",
            evidence_note(payload, case_id, input_condition),
            f"Initial audit: {contradiction}.",
        )

    if missing_statement_matches(text, available):
        return (
            "supported",
            evidence_note(payload, case_id, input_condition),
            "Initial audit: accurately states evidence limitation for this condition.",
        )

    specificity = unsupported_specificity(text)
    if specificity:
        return (
            "unsupported",
            evidence_note(payload, case_id, input_condition),
            f"Initial audit: {specificity}.",
        )

    missing_reason = missing_positive_evidence_reason(text, available)
    if missing_reason:
        return (
            "unsupported",
            evidence_note(payload, case_id, input_condition),
            f"Initial audit: {missing_reason}.",
        )

    if available["rgb"] and visible_support(case_id, text):
        return (
            "supported",
            evidence_note(payload, case_id, input_condition),
            "Initial audit: visible RGB context supports this claim.",
        )

    if numerical_factor_supported(payload, text):
        return (
            "supported",
            evidence_note(payload, case_id, input_condition),
            "Initial audit: claim is grounded in available Module #2 numerical evidence.",
        )

    if has_any(
        text,
        (
            "neutral posture",
            "reduce strain",
            "ergonomic target",
            "postural variation",
            "improve reach",
            "reduce loading",
        ),
    ):
        return (
            "supported",
            evidence_note(payload, case_id, input_condition),
            "Initial audit: recommendation is a general ergonomic target tied to identified risk.",
        )

    return (
        "supported",
        evidence_note(payload, case_id, input_condition),
        "Initial audit: no unsupported or contradictory content detected.",
    )


def main() -> None:
    args = parse_args()
    annotation_csv = args.annotation_csv or args.m3_root / "results" / "evaluation" / "claim_annotation_sheet.csv"
    output_csv = args.output_csv or annotation_csv

    with annotation_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    payload_cache: dict[Path, dict[str, Any]] = {}
    counts = {label: 0 for label in VALID_LABELS}
    for row in rows:
        input_file = Path(row["input_file"])
        if input_file not in payload_cache:
            payload_cache[input_file] = read_json(input_file)
        label, note, reviewer_note = label_claim(row, payload_cache[input_file])
        row["support_label"] = label
        row["evidence_note"] = note
        row["reviewer_note"] = reviewer_note
        counts[label] += 1

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Labeled rows: {len(rows)}")
    print(counts)
    print(output_csv)


if __name__ == "__main__":
    main()
