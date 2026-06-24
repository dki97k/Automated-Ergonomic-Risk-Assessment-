#!/usr/bin/env python3
"""Generate structured reports for Module #3 configuration analysis."""

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

from structured_common import read_json, write_json  # noqa: E402


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
INPUT_CONDITIONS = ("module2_only", "module2_rgb")


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
    parser.add_argument("--case-id", dest="case_ids", action="append")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1600)
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
    parser.add_argument("--payload-root", type=Path, help="Defaults to <m3-root>/payloads.")
    parser.add_argument("--prompt-file", type=Path, help="Defaults to <m3-root>/prompts/structured_original_rgb_compatible.txt.")
    parser.add_argument("--output-root", type=Path, help="Defaults to <m3-root>/results/generated_reports.")
    return parser.parse_args()


def load_case_mapping(m3_root: Path) -> dict[str, str]:
    manifest_path = m3_root / "payloads" / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    mapping: dict[str, str] = {}
    for case in manifest.get("cases", []):
        case_id = case.get("case_id")
        source_sample_id = case.get("source_sample_id")
        if isinstance(case_id, str) and isinstance(source_sample_id, str):
            mapping[case_id] = source_sample_id
    return mapping


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
                "content": "Return only valid JSON that follows the requested schema.",
            },
            {"role": "user", "content": user_message_content(prompt, image_path)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
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


def parse_report(response_payload: dict[str, Any]) -> dict[str, Any]:
    content = response_payload["choices"][0]["message"]["content"]
    return json.loads(content)


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
    return isinstance(payload.get("report"), dict)


def should_skip(path: Path, force: bool, retry_errors: bool) -> bool:
    if force:
        return False
    if retry_errors:
        return existing_output_is_success(path)
    return path.exists()


def main() -> None:
    args = parse_args()
    payload_root = args.payload_root or args.m3_root / "payloads"
    prompt_file = args.prompt_file or args.m3_root / "prompts" / "structured_original_rgb_compatible.txt"
    output_root = args.output_root or args.m3_root / "results" / "generated_reports"
    input_conditions = tuple(args.input_conditions or INPUT_CONDITIONS)
    case_ids = set(args.case_ids) if args.case_ids else None

    prompt_template = prompt_file.read_text(encoding="utf-8")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    case_mapping = load_case_mapping(args.m3_root)
    run_manifest = {
        "analysis": "module3_structured_configuration",
        "model": args.model,
        "temperature": args.temperature,
        "runs": args.runs,
        "input_conditions": list(input_conditions),
        "prompt_file": str(prompt_file),
        "payload_root": str(payload_root),
        "output_root": str(output_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
    }
    write_json(output_root / "structured" / "run_manifest.json", run_manifest)

    all_inputs = [
        (input_condition, input_path)
        for input_condition in input_conditions
        for input_path in input_files_for(payload_root, input_condition, case_ids)
    ]
    total = len(all_inputs) * args.runs
    completed = 0

    for input_condition, input_path in all_inputs:
        input_payload = read_json(input_path)
        case_id = input_payload.get("case_id") or input_path.stem
        sample_id = case_mapping.get(case_id, case_id)
        image_path = image_path_for(input_path, input_payload)
        prompt = prompt_text(prompt_template, input_payload)
        condition_root = output_root / "structured" / input_condition
        prompt_payload_dir = condition_root / "prompt_payloads"
        reports_dir = condition_root / "reports" / case_id
        prompt_payload_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        (prompt_payload_dir / f"{case_id}.txt").write_text(prompt, encoding="utf-8")

        for run_idx in range(1, args.runs + 1):
            completed += 1
            output_path = reports_dir / f"run_{run_idx:02d}.json"
            if should_skip(output_path, args.force, args.retry_errors):
                print(f"[{completed}/{total}] skip existing {input_condition}/{case_id} run {run_idx}")
                continue

            if args.dry_run:
                write_json(
                    output_path,
                    {
                        "case_id": case_id,
                        "sample_id": sample_id,
                        "input_condition": input_condition,
                        "run_index": run_idx,
                        "dry_run": True,
                        "report": None,
                    },
                )
                print(f"[{completed}/{total}] wrote dry run {input_condition}/{case_id} run {run_idx}")
                continue

            print(f"[{completed}/{total}] calling {args.model}: {input_condition}/{case_id} run {run_idx}")
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
                    report = parse_report(response_payload)
                    write_json(
                        output_path,
                        {
                            "case_id": case_id,
                            "sample_id": sample_id,
                            "input_condition": input_condition,
                            "run_index": run_idx,
                            "model": args.model,
                            "temperature": args.temperature,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "report": report,
                            "usage": response_payload.get("usage", {}),
                            "response_id": response_payload.get("id"),
                        },
                    )
                    time.sleep(args.api_delay_sec)
                    break
                except urllib.error.HTTPError as exc:
                    last_error = exc
                    retry_after = exc.headers.get("retry-after")
                    if exc.code == 429:
                        wait_sec = float(retry_after or args.rate_limit_retry_sec)
                    else:
                        wait_sec = args.retry_base_sec * attempt
                    if attempt == 3:
                        write_json(
                            output_path,
                            {
                                "case_id": case_id,
                                "sample_id": sample_id,
                                "input_condition": input_condition,
                                "run_index": run_idx,
                                "model": args.model,
                                "temperature": args.temperature,
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                                "report": None,
                                "error": str(last_error),
                            },
                        )
                        print(f"  failed after 3 attempts: {last_error}")
                    else:
                        time.sleep(wait_sec)
                except (urllib.error.URLError, json.JSONDecodeError, KeyError, FileNotFoundError) as exc:
                    last_error = exc
                    if attempt == 3:
                        write_json(
                            output_path,
                            {
                                "case_id": case_id,
                                "sample_id": sample_id,
                                "input_condition": input_condition,
                                "run_index": run_idx,
                                "model": args.model,
                                "temperature": args.temperature,
                                "created_at": datetime.now().isoformat(timespec="seconds"),
                                "report": None,
                                "error": str(last_error),
                            },
                        )
                        print(f"  failed after 3 attempts: {last_error}")
                    else:
                        time.sleep(args.retry_base_sec * attempt)


if __name__ == "__main__":
    main()
