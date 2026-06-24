#!/usr/bin/env python3
"""Generate Module 3 reports for Module 1 configuration analysis."""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import http.client
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M3_SRC = PROJECT_ROOT / "m3" / "src"
sys.path.insert(0, str(M3_SRC))

from natural_common import missing_sections, parse_sections  # noqa: E402
from structured_common import read_json, write_json  # noqa: E402


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
CONDITIONS = ("alphapose_motionbert", "sam3db")
REPORT_TYPES = ("structured", "natural")


def verified_ssl_context() -> ssl.SSLContext | None:
    for module_name in ("certifi", "pip._vendor.certifi"):
        try:
            module = __import__(module_name, fromlist=["where"])
            cafile = module.where()
            if cafile and Path(cafile).exists():
                return ssl.create_default_context(cafile=cafile)
        except Exception:
            continue
    return None


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--report-type", choices=("both", *REPORT_TYPES), default="both")
    parser.add_argument("--condition", dest="conditions", action="append", choices=CONDITIONS)
    parser.add_argument("--case-id", dest="case_ids", action="append")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--structured-temperature", type=float, default=0.1)
    parser.add_argument("--natural-temperature", type=float, default=0.2)
    parser.add_argument("--structured-max-tokens", type=int, default=1800)
    parser.add_argument("--natural-max-tokens", type=int, default=2000)
    parser.add_argument("--api-delay-sec", type=float, default=0.25)
    parser.add_argument("--retry-base-sec", type=float, default=3.0)
    parser.add_argument("--rate-limit-retry-sec", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--payload-root", type=Path, default=root / "payloads")
    parser.add_argument("--output-root", type=Path, default=root / "results" / "generated_reports")
    parser.add_argument(
        "--structured-prompt-file",
        type=Path,
        default=root / "prompts" / "structured_pose_neutral_contribution.txt",
    )
    parser.add_argument(
        "--natural-prompt-file",
        type=Path,
        default=root / "prompts" / "natural_pose_neutral_contribution.txt",
    )
    return parser.parse_args()


def prompt_text(prompt_template: str, input_payload: dict[str, Any]) -> str:
    input_text = json.dumps(input_payload, ensure_ascii=False, indent=2)
    return prompt_template.replace("{MODULE3_INPUT_JSON}", input_text)


def call_openai(
    *,
    api_key: str,
    model: str,
    system_text: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    json_response: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_response:
        payload["response_format"] = {"type": "json_object"}

    request = urllib.request.Request(
        OPENAI_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60, context=verified_ssl_context()) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_structured_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(response_payload["choices"][0]["message"]["content"])


def parse_natural_response(response_payload: dict[str, Any]) -> str:
    return response_payload["choices"][0]["message"]["content"].strip()


def existing_output_is_success(path: Path, report_type: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    if payload.get("dry_run") is True or payload.get("error"):
        return False
    if report_type == "structured":
        return isinstance(payload.get("report"), dict)
    return isinstance(payload.get("report_text"), str) and bool(payload.get("report_text", "").strip())


def should_skip_output(path: Path, report_type: str, force: bool, retry_errors: bool) -> bool:
    if not path.exists():
        return False
    if force:
        return False
    if retry_errors:
        return existing_output_is_success(path, report_type)
    return True


def retry_sleep_seconds(exc: Exception, attempt: int, args: argparse.Namespace) -> float:
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        retry_after = exc.headers.get("Retry-After")
        try:
            if retry_after:
                return max(float(retry_after), args.rate_limit_retry_sec)
        except ValueError:
            pass
        return args.rate_limit_retry_sec * attempt
    return args.retry_base_sec * attempt


def selected_report_types(report_type: str) -> tuple[str, ...]:
    return REPORT_TYPES if report_type == "both" else (report_type,)


def input_files_for(payload_root: Path, condition: str, case_ids: set[str] | None) -> list[Path]:
    files = sorted(
        path
        for path in (payload_root / condition).glob("*.json")
        if path.is_file() and not path.name.startswith("._")
    )
    if case_ids is not None:
        files = [path for path in files if path.stem in case_ids]
    if not files:
        raise SystemExit(f"No input payloads found for {condition}: {payload_root / condition}")
    return files


def report_settings(args: argparse.Namespace, report_type: str) -> dict[str, Any]:
    if report_type == "structured":
        return {
            "prompt_file": args.structured_prompt_file,
            "temperature": args.structured_temperature,
            "max_tokens": args.structured_max_tokens,
            "json_response": True,
            "system_text": "Return only valid JSON that follows the requested schema.",
        }
    return {
        "prompt_file": args.natural_prompt_file,
        "temperature": args.natural_temperature,
        "max_tokens": args.natural_max_tokens,
        "json_response": False,
        "system_text": (
            "Write only the requested evidence-grounded report text using the "
            "required section headings."
        ),
    }


def dry_run_payload(
    *,
    case_id: str,
    condition: str,
    report_type: str,
    run_index: int,
    input_file: Path,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "sample_id": case_id,
        "condition": condition,
        "report_type": report_type,
        "run_index": run_index,
        "dry_run": True,
        "report": None,
        "report_text": None,
        "input_file": str(input_file),
        "image_file": "",
    }


def write_report_output(
    *,
    output_path: Path,
    report_type: str,
    case_id: str,
    condition: str,
    run_index: int,
    args: argparse.Namespace,
    input_file: Path,
    response_payload: dict[str, Any],
) -> None:
    if report_type == "structured":
        payload = {
            "case_id": case_id,
            "sample_id": case_id,
            "condition": condition,
            "report_type": report_type,
            "run_index": run_index,
            "model": args.model,
            "temperature": args.structured_temperature,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "report": parse_structured_response(response_payload),
            "usage": response_payload.get("usage", {}),
            "response_id": response_payload.get("id"),
            "input_file": str(input_file),
            "image_file": "",
        }
    else:
        report_text = parse_natural_response(response_payload)
        payload = {
            "case_id": case_id,
            "sample_id": case_id,
            "condition": condition,
            "report_type": report_type,
            "run_index": run_index,
            "model": args.model,
            "temperature": args.natural_temperature,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "report_text": report_text,
            "sections": parse_sections(report_text),
            "missing_sections": missing_sections(report_text),
            "usage": response_payload.get("usage", {}),
            "response_id": response_payload.get("id"),
            "input_file": str(input_file),
            "image_file": "",
        }
    write_json(output_path, payload)


def run_report_type(args: argparse.Namespace, report_type: str, condition: str) -> None:
    settings = report_settings(args, report_type)
    prompt_template = Path(settings["prompt_file"]).read_text(encoding="utf-8")
    case_ids = set(args.case_ids) if args.case_ids else None
    input_files = input_files_for(args.payload_root, condition, case_ids)

    condition_dir = args.output_root / report_type / condition
    prompt_dir = condition_dir / "prompt_payloads"
    reports_dir = condition_dir / "reports"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "analysis": "module1_pose_configuration_contribution",
        "report_type": report_type,
        "condition": condition,
        "model": args.model,
        "runs": args.runs,
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "prompt_file": str(settings["prompt_file"]),
        "payload_root": str(args.payload_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
        "case_ids": [path.stem for path in input_files],
    }
    write_json(condition_dir / "run_manifest.json", manifest)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    total = len(input_files) * args.runs
    completed = 0
    for input_file in input_files:
        input_payload = read_json(input_file)
        case_id = input_payload.get("case_id") or input_file.stem
        prompt = prompt_text(prompt_template, input_payload)
        (prompt_dir / f"{case_id}.txt").write_text(prompt, encoding="utf-8")

        case_report_dir = reports_dir / case_id
        case_report_dir.mkdir(parents=True, exist_ok=True)
        for run_index in range(1, args.runs + 1):
            completed += 1
            output_path = case_report_dir / f"run_{run_index:02d}.json"
            if should_skip_output(output_path, report_type, args.force, args.retry_errors):
                print(f"[{completed}/{total}] skip existing {report_type}/{condition}/{case_id} run {run_index}")
                continue
            if args.dry_run:
                write_json(
                    output_path,
                    dry_run_payload(
                        case_id=case_id,
                        condition=condition,
                        report_type=report_type,
                        run_index=run_index,
                        input_file=input_file,
                    ),
                )
                print(f"[{completed}/{total}] wrote dry run {report_type}/{condition}/{case_id} run {run_index}")
                continue

            print(f"[{completed}/{total}] calling {args.model}: {report_type}/{condition}/{case_id} run {run_index}")
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    response_payload = call_openai(
                        api_key=api_key or "",
                        model=args.model,
                        system_text=settings["system_text"],
                        prompt=prompt,
                        temperature=float(settings["temperature"]),
                        max_tokens=int(settings["max_tokens"]),
                        json_response=bool(settings["json_response"]),
                    )
                    write_report_output(
                        output_path=output_path,
                        report_type=report_type,
                        case_id=case_id,
                        condition=condition,
                        run_index=run_index,
                        args=args,
                        input_file=input_file,
                        response_payload=response_payload,
                    )
                    break
                except (
                    urllib.error.URLError,
                    urllib.error.HTTPError,
                    http.client.RemoteDisconnected,
                    TimeoutError,
                    ConnectionError,
                    json.JSONDecodeError,
                    KeyError,
                ) as exc:
                    last_error = exc
                    if attempt == 3:
                        write_json(
                            output_path,
                            {
                                "case_id": case_id,
                                "sample_id": case_id,
                                "condition": condition,
                                "report_type": report_type,
                                "run_index": run_index,
                                "model": args.model,
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                                "report": None,
                                "report_text": None,
                                "error": str(last_error),
                                "input_file": str(input_file),
                                "image_file": "",
                            },
                        )
                        print(f"  failed after 3 attempts: {last_error}")
                    else:
                        sleep_sec = retry_sleep_seconds(exc, attempt, args)
                        print(f"  retrying after {sleep_sec:.1f}s: {exc}")
                        time.sleep(sleep_sec)
            time.sleep(args.api_delay_sec)


def main() -> None:
    args = parse_args()
    conditions = tuple(args.conditions or CONDITIONS)
    for report_type in selected_report_types(args.report_type):
        for condition in conditions:
            run_report_type(args, report_type, condition)
    print("Module 1 configuration report generation complete.")
    print(args.output_root)


if __name__ == "__main__":
    main()
