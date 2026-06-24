#!/usr/bin/env python3
"""Run natural-language Module 3 validation reports."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from natural_common import build_prompt, missing_sections, parse_sections, project_root
from structured_common import read_json, write_json


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"


def validation_prompt_payload(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Keep validation input numerical by removing task-identifying sample text."""
    prompt_payload = json.loads(json.dumps(input_payload))
    prompt_payload.pop("sample_id", None)
    return prompt_payload


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root
        / "data"
        / "structured_validation"
        / "inputs"
        / "numerical_only_m2_current_remapped",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=root / "prompts" / "natural" / "p1_evidence_grounded_advanced.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root
        / "results"
        / "natural_validation"
        / "evidence_grounded_numerical_only_m2_current_remapped",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompt payloads without calling the OpenAI API.",
    )
    return parser.parse_args()


def call_openai(
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write only the requested evidence-grounded report text "
                    "using the required section headings."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_report_text(response_payload: dict[str, Any]) -> str:
    return response_payload["choices"][0]["message"]["content"].strip()


def main() -> None:
    args = parse_args()
    input_files = sorted(args.input_dir.glob("*.json"))
    if not input_files:
        raise SystemExit(f"No validation input files found: {args.input_dir}")

    prompt_template = args.prompt_file.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_payload_dir = args.output_dir / "prompt_payloads"
    reports_dir = args.output_dir / "reports"
    reports_txt_dir = args.output_dir / "reports_txt"
    prompt_payload_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reports_txt_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    run_manifest = {
        "condition": "evidence_grounded_numerical_only_m2_current_remapped",
        "report_type": "natural_language",
        "model": args.model,
        "temperature": args.temperature,
        "runs": args.runs,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "prompt_file": str(args.prompt_file),
        "sample_id_in_prompt": False,
        "dry_run": args.dry_run,
        "samples": [path.stem for path in input_files],
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)

    total = len(input_files) * args.runs
    completed = 0
    for input_path in input_files:
        input_payload = read_json(input_path)
        sample_id = input_payload["sample_id"]
        sample_dir = reports_dir / sample_id
        sample_txt_dir = reports_txt_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_txt_dir.mkdir(parents=True, exist_ok=True)
        prompt_text = build_prompt(prompt_template, validation_prompt_payload(input_payload))
        (prompt_payload_dir / f"{sample_id}.txt").write_text(prompt_text, encoding="utf-8")

        for run_idx in range(1, args.runs + 1):
            completed += 1
            output_path = sample_dir / f"run_{run_idx:02d}.json"
            output_txt_path = sample_txt_dir / f"run_{run_idx:02d}.txt"
            if output_path.exists() and not args.force:
                print(f"[{completed}/{total}] skip existing {sample_id} run {run_idx}")
                continue

            if args.dry_run:
                write_json(
                    output_path,
                    {
                        "sample_id": sample_id,
                        "run_index": run_idx,
                        "dry_run": True,
                        "report_text": "",
                    },
                )
                output_txt_path.write_text("", encoding="utf-8")
                print(f"[{completed}/{total}] wrote dry run {sample_id} run {run_idx}")
                continue

            print(f"[{completed}/{total}] calling {args.model}: {sample_id} run {run_idx}")
            last_error = None
            for attempt in range(1, 5):
                try:
                    response_payload = call_openai(
                        api_key=api_key or "",
                        model=args.model,
                        prompt=prompt_text,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    report_text = parse_report_text(response_payload)
                    missing = missing_sections(report_text)
                    output_txt_path.write_text(report_text + "\n", encoding="utf-8")
                    write_json(
                        output_path,
                        {
                            "sample_id": sample_id,
                            "run_index": run_idx,
                            "model": args.model,
                            "temperature": args.temperature,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "input_file": str(input_path),
                            "prompt_file": str(args.prompt_file),
                            "report_text": report_text,
                            "sections": parse_sections(report_text),
                            "missing_sections": missing,
                            "postprocessing": "none",
                            "usage": response_payload.get("usage", {}),
                            "response_id": response_payload.get("id"),
                        },
                    )
                    break
                except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as exc:
                    last_error = exc
                    if attempt == 4:
                        write_json(
                            output_path,
                            {
                                "sample_id": sample_id,
                                "run_index": run_idx,
                                "model": args.model,
                                "temperature": args.temperature,
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                                "input_file": str(input_path),
                                "prompt_file": str(args.prompt_file),
                                "report_text": "",
                                "error": str(last_error),
                            },
                        )
                        output_txt_path.write_text("", encoding="utf-8")
                        print(f"  failed after 4 attempts: {last_error}")
                    else:
                        time.sleep(2 * attempt)


if __name__ == "__main__":
    main()
