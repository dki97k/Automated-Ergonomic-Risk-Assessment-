#!/usr/bin/env python3
"""Draft-label natural-language validation claims against Module 2 evidence.

This produces a first-pass evidence audit for user review. It is intentionally
conservative for unsupported implementation or factor claims, but it does not
replace the user's final review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from natural_common import project_root


FACTOR_TERMS = {
    "trunk": ("trunk", "lumbar", "lower back", "spinal", "spine", "forward bend"),
    "neck": ("neck", "cervical"),
    "upper_arm": ("upper arm", "shoulder", "arm elevation", "elevated arm"),
    "wrist": ("wrist", "hand"),
    "knee": ("knee", "leg", "kneeling", "squat"),
    "static": (
        "static",
        "postural variation",
        "held posture",
    ),
    "repetition": (
        "repetition",
        "repetitive",
        "cycle",
        "cycles",
        "cumulative",
        "frequency",
        "period",
    ),
}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only"
        / "evaluation"
        / "claim_annotation_sheet.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only"
        / "evaluation"
        / "claim_annotation_sheet_labeled_draft.csv",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_numeric_values(payload: Any) -> list[float]:
    values: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            values.extend(flatten_numeric_values(value))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(flatten_numeric_values(value))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        number = float(payload)
        values.append(number)
        if 0 <= number <= 1:
            values.append(number * 100)
    return values


def extract_claim_numbers(text: str) -> list[tuple[str, float]]:
    numbers = []
    for match in re.finditer(r"(?<![A-Za-z])\d+(?:\.\d+)?%?", text):
        raw = match.group(0)
        numbers.append((raw, float(raw.rstrip("%"))))
    return numbers


def number_matches(raw: str, number: float, evidence_values: list[float]) -> bool:
    if raw == "90" and "90th" in raw:
        return True
    if number in (0, 1, 2, 3):
        return True
    tolerance = max(0.12, abs(number) * 0.02)
    if raw.endswith("%"):
        tolerance = max(tolerance, 1.2)
    if number >= 5 and number.is_integer():
        tolerance = max(tolerance, 0.35)
    if number >= 10 and number.is_integer():
        tolerance = max(tolerance, 1.1)
    return any(abs(value - number) <= tolerance for value in evidence_values)


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def mentioned_factors(text: str) -> set[str]:
    lower = text.lower()
    return {
        factor
        for factor, terms in FACTOR_TERMS.items()
        if any(term in lower for term in terms)
    }


def sample_evidence(summary: dict[str, Any]) -> dict[str, Any]:
    joint = summary["joint_angle_summary"]
    posture = summary["posture_summary"]["final_reba"]
    duration = summary["duration_summary"]
    repetition = summary["repetition_summary"]

    upper_flex = joint["upper_arm_flexion_deg"]
    upper_abd = joint["upper_arm_abduction_deg"]
    wrist_flex = joint["wrist_flexion_deg"]
    wrist_twist = joint["wrist_twisting_deg"]
    knee = joint["knee_flexion_deg"]
    neck_flex = joint["neck_flexion_deg"]
    neck_twist = joint["neck_twisting_deg"]
    neck_bend = joint["neck_bending_deg"]
    trunk = joint["trunk_flexion_deg"]

    factors = {
        "trunk": trunk["mean"] >= 25 or trunk["p90"] >= 60 or trunk["max"] >= 90,
        "neck": (
            neck_flex["mean"] >= 20
            or neck_flex["p90"] >= 30
            or neck_twist["mean"] >= 10
            or neck_twist["p90"] >= 20
            or neck_bend["p90"] >= 15
            or max(neck_flex["max"], neck_twist["max"], neck_bend["max"]) >= 40
        ),
        "upper_arm": (
            max(upper_flex["max"], upper_abd["max"]) >= 90
            or max(upper_flex["p90"], upper_abd["p90"]) >= 60
            or max(upper_flex["mean"], upper_abd["mean"]) >= 40
        ),
        "wrist": (
            wrist_flex["mean"] >= 15
            or wrist_flex["max"] >= 45
            or wrist_twist["max"] >= 45
        ),
        "knee": knee["mean"] >= 60 or knee["p90"] >= 90 or knee["max"] >= 90,
        "static": (
            duration["static_posture_ratio"] >= 0.3
            or duration["max_static_segment_sec"] >= 15
            or posture["longest_high_risk_run_sec"] >= 10
        ),
        "repetition": (
            repetition["total_repetitions"] > 0
            and repetition["repetition_rate_cycle_per_min"] > 0
        ),
    }

    return {
        "numeric_values": flatten_numeric_values(summary),
        "factors": factors,
        "posture": posture,
        "duration": duration,
        "repetition": repetition,
        "joint": joint,
    }


def evidence_note_for(factors: set[str], evidence: dict[str, Any]) -> str:
    notes = []
    joint = evidence["joint"]
    duration = evidence["duration"]
    repetition = evidence["repetition"]
    posture = evidence["posture"]
    if "trunk" in factors:
        t = joint["trunk_flexion_deg"]
        notes.append(f"trunk flexion mean/p90/max={t['mean']}/{t['p90']}/{t['max']}")
    if "neck" in factors:
        nf = joint["neck_flexion_deg"]
        nt = joint["neck_twisting_deg"]
        notes.append(
            "neck flexion/twisting mean="
            f"{nf['mean']}/{nt['mean']}, max={nf['max']}/{nt['max']}"
        )
    if "upper_arm" in factors:
        uf = joint["upper_arm_flexion_deg"]
        ua = joint["upper_arm_abduction_deg"]
        notes.append(
            "upper-arm flexion/abduction mean="
            f"{uf['mean']}/{ua['mean']}, max={uf['max']}/{ua['max']}"
        )
    if "wrist" in factors:
        wf = joint["wrist_flexion_deg"]
        wt = joint["wrist_twisting_deg"]
        notes.append(
            "wrist flexion/twisting mean="
            f"{wf['mean']}/{wt['mean']}, max={wf['max']}/{wt['max']}"
        )
    if "knee" in factors:
        k = joint["knee_flexion_deg"]
        notes.append(f"knee flexion mean/p90/max={k['mean']}/{k['p90']}/{k['max']}")
    if "static" in factors:
        notes.append(
            "static ratio/max segment/high-risk run="
            f"{duration['static_posture_ratio']}/{duration['max_static_segment_sec']}/"
            f"{posture['longest_high_risk_run_sec']}"
        )
    if "repetition" in factors:
        notes.append(
            "repetition total/rate/period="
            f"{repetition['total_repetitions']}/"
            f"{repetition['repetition_rate_cycle_per_min']}/"
            f"{repetition['mean_period_sec']}"
        )
    if not notes:
        notes.append(
            "overall REBA mean/p90/max="
            f"{posture['mean']}/{posture['p90']}/{posture['max']}; "
            f"risk bins={posture['risk_bin_distribution']}"
        )
    return "; ".join(notes)


def qualitative_mismatch(text: str, evidence: dict[str, Any]) -> str | None:
    lower = text.lower()
    posture = evidence["posture"]
    duration = evidence["duration"]
    mean_reba = posture["mean"]
    high_share = posture["risk_bin_distribution"]["high"]
    static_ratio = duration["static_posture_ratio"]
    max_static = duration["max_static_segment_sec"]
    high_run = posture["longest_high_risk_run_sec"]

    if "medium risk" in lower and not (4 <= mean_reba <= 7):
        return f"mean REBA {mean_reba} is not in the medium-risk range"
    if "moderate risk" in lower and not (4 <= mean_reba <= 7):
        return f"mean REBA {mean_reba} does not support moderate risk"
    if "significant ergonomic risk" in lower and not (mean_reba >= 6 or high_share >= 0.3):
        return f"mean REBA {mean_reba} and high-risk share {high_share} do not support significant risk"
    if "high-risk postures" in lower and high_share == 0:
        return "high-risk posture claim conflicts with zero high-risk share"
    if "prolonged high-risk" in lower and high_run < 10:
        return f"prolonged high-risk exposure is not supported by high-risk run={high_run}"
    return None


def unsupported_reason(text: str, factors: set[str], evidence: dict[str, Any]) -> str | None:
    lower = text.lower()
    supported_factors = evidence["factors"]

    absent = sorted(
        factor
        for factor in factors
        if factor in supported_factors and not supported_factors[factor]
    )
    if absent:
        if absent == ["static"] and "static" not in lower:
            absent = []
    if absent:
        if any(term in lower for term in ("primary", "main", "recommended", "reduce", "lower", "elevation", "stress")):
            return f"mentions factor(s) without sufficient mapped evidence: {', '.join(absent)}"

    if "static" in factors and not supported_factors["static"] and "static" in lower:
        if any(term in lower for term in ("significant", "prolonged", "warrant attention", "fatigue", "localized")):
            return "static exposure is present but not strong enough for this claim"

    if "frequency of repetitions" in lower and evidence["repetition"]["total_repetitions"] <= 10:
        return "repetition is present, but total repetitions are low for a frequency-focused factor claim"

    return None


def label_row(row: dict[str, str], evidence_by_sample: dict[str, dict[str, Any]]) -> dict[str, str]:
    sample_id = row["sample_id"]
    evidence = evidence_by_sample[sample_id]
    text = row["claim_text"]
    factors = mentioned_factors(text)

    bad_numbers = []
    for raw, number in extract_claim_numbers(text):
        if raw == "90" and "90th" in text:
            continue
        if not number_matches(raw, number, evidence["numeric_values"]):
            bad_numbers.append(raw)
    if bad_numbers:
        row["support_label"] = "contradiction"
        row["evidence_note"] = evidence_note_for(factors, evidence)
        row["reviewer_note"] = (
            "Audit note: numeric value(s) not found within tolerance: "
            + ", ".join(bad_numbers)
        )
        return row

    mismatch = qualitative_mismatch(text, evidence)
    if mismatch:
        row["support_label"] = "contradiction"
        row["evidence_note"] = evidence_note_for(factors, evidence)
        row["reviewer_note"] = f"Audit note: {mismatch}"
        return row

    unsupported = unsupported_reason(text, factors, evidence)
    if unsupported:
        row["support_label"] = "unsupported"
        row["evidence_note"] = evidence_note_for(factors, evidence)
        row["reviewer_note"] = f"Audit note: {unsupported}."
        return row

    row["support_label"] = "supported"
    row["evidence_note"] = evidence_note_for(factors, evidence)

    if any(term in text.lower() for term in ("could", "may", "potential", "expected", "beneficial")):
        row["reviewer_note"] = "Audit note: cautious biomechanical inference."
    else:
        row["reviewer_note"] = "Audit note: direct numerical evidence match."
    return row


def main() -> None:
    args = parse_args()
    input_dir = (
        project_root()
        / "data"
        / "structured_validation"
        / "inputs"
        / "numerical_only_m2_current_remapped"
    )
    evidence_by_sample = {
        path.stem: sample_evidence(load_json(path))
        for path in sorted(input_dir.glob("*.json"))
    }

    with args.annotation_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = [label_row(dict(row), evidence_by_sample) for row in reader]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["support_label"]] = counts.get(row["support_label"], 0) + 1
    print(f"Wrote labeled draft rows: {len(rows)}")
    print(counts)
    print(args.output_csv)


if __name__ == "__main__":
    main()
