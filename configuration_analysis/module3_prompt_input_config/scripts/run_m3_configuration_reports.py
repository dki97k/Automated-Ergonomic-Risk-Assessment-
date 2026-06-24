#!/usr/bin/env python3
"""Generate natural-language reports for Module #3 configuration analysis."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M3_SRC = PROJECT_ROOT / "m3" / "src"
if M3_SRC.exists():
    sys.path.insert(0, str(M3_SRC))
else:
    sys.path.insert(0, str(Path("<private_workspace>/m3/src")))

from natural_common import missing_sections, parse_sections  # noqa: E402
from structured_common import read_json, write_json  # noqa: E402


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
INPUT_CONDITIONS = ("module2_only", "module2_rgb")
PROMPT_CONDITIONS = (
    "neutral",
    "original_rgb_compatible",
    "bounded_context_augmented",
)
PROMPT_FILES = {
    "neutral": "natural_neutral_contribution.txt",
    "original_rgb_compatible": "natural_original_rgb_compatible.txt",
    "bounded_context_augmented": "natural_bounded_context_augmented.txt",
}


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
    default_m3_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m3-root", type=Path, default=default_m3_root)
    parser.add_argument(
        "--input-condition",
        dest="input_conditions",
        action="append",
        choices=INPUT_CONDITIONS,
        help="Input condition to run. Repeat for multiple. Default: all.",
    )
    parser.add_argument(
        "--prompt-condition",
        dest="prompt_conditions",
        action="append",
        choices=PROMPT_CONDITIONS,
        help="Prompt condition to run. Repeat for multiple. Default: all.",
    )
    parser.add_argument("--case-id", dest="case_ids", action="append")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--api-delay-sec", type=float, default=0.25)
    parser.add_argument("--retry-base-sec", type=float, default=3.0)
    parser.add_argument("--rate-limit-retry-sec", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Skip successful outputs, but overwrite dry-run/error outputs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--payload-root",
        type=Path,
        help="Defaults to <m3-root>/payloads.",
    )
    parser.add_argument(
        "--prompt-root",
        type=Path,
        help="Defaults to <m3-root>/prompts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to <m3-root>/results/generated_reports.",
    )
    return parser.parse_args()


def prompt_text(prompt_template: str, input_payload: dict[str, Any]) -> str:
    input_text = json.dumps(input_payload, ensure_ascii=False, indent=2)
    return prompt_template.replace("{MODULE3_INPUT_JSON}", input_text)


def image_path_for(input_file: Path, input_payload: dict[str, Any]) -> Path | None:
    evidence = input_payload.get("evidence_available", {})
    rgb_evidence = input_payload.get("rgb_evidence", {})
    if not evidence.get("rgb") or not isinstance(rgb_evidence, dict):
        return None
    asset_file = rgb_evidence.get("asset_file")
    if not asset_file:
        return None
    image_path = input_file.parent / str(asset_file)
    if not image_path.exists():
        raise FileNotFoundError(f"RGB asset not found: {image_path}")
    return image_path


def image_content_part(image_path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
            "detail": "high",
        },
    }


def user_message_content(prompt: str, image_path: Path | None) -> str | list[dict[str, Any]]:
    if image_path is None:
        return prompt
    return [
        {"type": "text", "text": prompt},
        image_content_part(image_path),
    ]


def call_openai(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_path: Path | None,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write only the requested evidence-grounded natural-language "
                    "report text using the required section headings."
                ),
            },
            {"role": "user", "content": user_message_content(prompt, image_path)},
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
    with urllib.request.urlopen(
        request,
        timeout=180,
        context=verified_ssl_context(),
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_report_text(response_payload: dict[str, Any]) -> str:
    return response_payload["choices"][0]["message"]["content"].strip()


def input_files_for(payload_root: Path, input_condition: str, case_ids: set[str] | None) -> list[Path]:
    files = sorted(
        path
        for path in (payload_root / input_condition).glob("*.json")
        if path.is_file() and not path.name.startswith("._")
    )
    if case_ids is not None:
        files = [path for path in files if path.stem in case_ids]
    if not files:
        raise SystemExit(f"No input payloads found for {input_condition}: {payload_root / input_condition}")
    return files


def existing_output_is_success(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    if payload.get("dry_run") is True or payload.get("error"):
        return False
    return isinstance(payload.get("report_text"), str) and bool(payload.get("report_text", "").strip())


def should_skip(path: Path, force: bool, retry_errors: bool) -> bool:
    if not path.exists():
        return False
    if force:
        return False
    if retry_errors:
        return existing_output_is_success(path)
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


def output_condition_dir(output_root: Path, input_condition: str, prompt_condition: str) -> Path:
    return output_root / "natural" / input_condition / prompt_condition


def write_dry_run_output(
    output_path: Path,
    *,
    case_id: str,
    input_condition: str,
    prompt_condition: str,
    run_index: int,
    input_file: Path,
    image_path: Path | None,
) -> None:
    write_json(
        output_path,
        {
            "case_id": case_id,
            "sample_id": case_id,
            "condition": f"{input_condition}__{prompt_condition}",
            "input_condition": input_condition,
            "prompt_condition": prompt_condition,
            "report_type": "natural",
            "run_index": run_index,
            "dry_run": True,
            "report_text": None,
            "sections": {},
            "input_file": str(input_file),
            "image_file": str(image_path) if image_path else "",
        },
    )


def write_report_output(
    output_path: Path,
    *,
    case_id: str,
    input_condition: str,
    prompt_condition: str,
    run_index: int,
    args: argparse.Namespace,
    input_file: Path,
    image_path: Path | None,
    response_payload: dict[str, Any],
) -> None:
    report_text = parse_report_text(response_payload)
    write_json(
        output_path,
        {
            "case_id": case_id,
            "sample_id": case_id,
            "condition": f"{input_condition}__{prompt_condition}",
            "input_condition": input_condition,
            "prompt_condition": prompt_condition,
            "report_type": "natural",
            "run_index": run_index,
            "model": args.model,
            "temperature": args.temperature,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "report_text": report_text,
            "sections": parse_sections(report_text),
            "missing_sections": missing_sections(report_text),
            "usage": response_payload.get("usage", {}),
            "response_id": response_payload.get("id"),
            "input_file": str(input_file),
            "image_file": str(image_path) if image_path else "",
        },
    )


def run_condition(
    args: argparse.Namespace,
    *,
    input_condition: str,
    prompt_condition: str,
    payload_root: Path,
    prompt_root: Path,
    output_root: Path,
) -> None:
    prompt_file = prompt_root / PROMPT_FILES[prompt_condition]
    prompt_template = prompt_file.read_text(encoding="utf-8")
    case_ids = set(args.case_ids) if args.case_ids else None
    input_files = input_files_for(payload_root, input_condition, case_ids)

    condition_dir = output_condition_dir(output_root, input_condition, prompt_condition)
    prompt_dir = condition_dir / "prompt_payloads"
    reports_dir = condition_dir / "reports"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "analysis": "module3_configuration_contribution",
        "report_type": "natural_language",
        "input_condition": input_condition,
        "prompt_condition": prompt_condition,
        "condition": f"{input_condition}__{prompt_condition}",
        "model": args.model,
        "runs": args.runs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_file": str(prompt_file),
        "payload_root": str(payload_root),
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
        image_path = image_path_for(input_file, input_payload)
        prompt = prompt_text(prompt_template, input_payload)
        (prompt_dir / f"{case_id}.txt").write_text(prompt, encoding="utf-8")

        case_report_dir = reports_dir / case_id
        case_report_dir.mkdir(parents=True, exist_ok=True)
        for run_index in range(1, args.runs + 1):
            completed += 1
            output_path = case_report_dir / f"run_{run_index:02d}.json"
            if should_skip(output_path, args.force, args.retry_errors):
                print(
                    f"[{completed}/{total}] skip existing "
                    f"{input_condition}/{prompt_condition}/{case_id} run {run_index}"
                )
                continue
            if args.dry_run:
                write_dry_run_output(
                    output_path,
                    case_id=case_id,
                    input_condition=input_condition,
                    prompt_condition=prompt_condition,
                    run_index=run_index,
                    input_file=input_file,
                    image_path=image_path,
                )
                print(
                    f"[{completed}/{total}] wrote dry run "
                    f"{input_condition}/{prompt_condition}/{case_id} run {run_index}"
                )
                continue

            print(
                f"[{completed}/{total}] calling {args.model}: "
                f"{input_condition}/{prompt_condition}/{case_id} run {run_index}"
            )
            last_error: Exception | None = None
            for attempt in range(1, 4):
                try:
                    response_payload = call_openai(
                        api_key=api_key or "",
                        model=args.model,
                        prompt=prompt,
                        image_path=image_path,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    write_report_output(
                        output_path,
                        case_id=case_id,
                        input_condition=input_condition,
                        prompt_condition=prompt_condition,
                        run_index=run_index,
                        args=args,
                        input_file=input_file,
                        image_path=image_path,
                        response_payload=response_payload,
                    )
                    break
                except (
                    urllib.error.URLError,
                    urllib.error.HTTPError,
                    json.JSONDecodeError,
                    KeyError,
                    FileNotFoundError,
                ) as exc:
                    last_error = exc
                    if attempt == 3:
                        write_json(
                            output_path,
                            {
                                "case_id": case_id,
                                "sample_id": case_id,
                                "condition": f"{input_condition}__{prompt_condition}",
                                "input_condition": input_condition,
                                "prompt_condition": prompt_condition,
                                "report_type": "natural",
                                "run_index": run_index,
                                "model": args.model,
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                                "report_text": None,
                                "sections": {},
                                "error": str(last_error),
                                "input_file": str(input_file),
                                "image_file": str(image_path) if image_path else "",
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
    payload_root = args.payload_root or args.m3_root / "payloads"
    prompt_root = args.prompt_root or args.m3_root / "prompts"
    output_root = args.output_root or args.m3_root / "results" / "generated_reports"
    input_conditions = tuple(args.input_conditions or INPUT_CONDITIONS)
    prompt_conditions = tuple(args.prompt_conditions or PROMPT_CONDITIONS)

    for input_condition in input_conditions:
        for prompt_condition in prompt_conditions:
            run_condition(
                args,
                input_condition=input_condition,
                prompt_condition=prompt_condition,
                payload_root=payload_root,
                prompt_root=prompt_root,
                output_root=output_root,
            )

    print("Module #3 configuration report generation complete.")
    print(output_root)


if __name__ == "__main__":
    main()
