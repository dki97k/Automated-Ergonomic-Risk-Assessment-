#!/usr/bin/env python3
"""Shared utilities for Module 3 natural-language validation."""

from __future__ import annotations

import re
from pathlib import Path

from structured_common import read_json, write_json


NATURAL_SECTIONS = (
    "Risk Interpretation",
    "Key Contributing Factors",
    "Recommendations",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_prompt(prompt_template: str, input_payload: dict) -> str:
    input_text = __import__("json").dumps(input_payload, ensure_ascii=False, indent=2)
    return prompt_template.replace("{MODULE3_INPUT_JSON}", input_text)


def normalize_heading(line: str) -> str:
    clean = line.strip()
    clean = re.sub(r"^#+\s*", "", clean)
    clean = re.sub(r"^\d+[\.\)]\s*", "", clean)
    clean = clean.strip("*: \t")
    return clean


def parse_sections(report_text: str) -> dict[str, str]:
    sections = {section: "" for section in NATURAL_SECTIONS}
    current: str | None = None
    buffers = {section: [] for section in NATURAL_SECTIONS}

    for raw_line in report_text.splitlines():
        heading = normalize_heading(raw_line)
        if heading in sections:
            current = heading
            continue
        if current is not None:
            buffers[current].append(raw_line.rstrip())

    return {
        section: "\n".join(lines).strip()
        for section, lines in buffers.items()
    }


def missing_sections(report_text: str) -> list[str]:
    sections = parse_sections(report_text)
    return [section for section, text in sections.items() if not text]


def compact_text(text: str) -> str:
    text = re.sub(r"^[-*]\s+", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_claims(section_text: str) -> list[str]:
    bullet_blocks = re.findall(
        r"(?ms)^\s*[-*]\s+(.+?)(?=^\s*[-*]\s+|\Z)",
        section_text.strip(),
    )
    if bullet_blocks:
        return [compact_text(block) for block in bullet_blocks if compact_text(block)]

    text = compact_text(section_text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9*])", text)
    claims = []
    for part in parts:
        clean = part.strip()
        if clean:
            claims.append(clean)
    return claims


def report_json_files(reports_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in reports_dir.glob("*/*.json")
        if not path.name.startswith("._")
    )
