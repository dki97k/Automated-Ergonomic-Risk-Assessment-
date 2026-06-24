#!/usr/bin/env python3
"""Run LLM-as-judge pairwise comparisons for natural reports."""

from __future__ import annotations

import argparse
import json
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
sys.path.insert(0, str(M3_SRC))

from structured_common import read_json, write_json  # noqa: E402


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
CONDITION_PAIRS = (
    ("rgb_only", "reba_only"),
    ("reba_only", "full_module2"),
    ("rgb_only", "full_module2"),
)


def verified_ssl_context() -> ssl.SSLContext | None:
    """Use a bundled CA file when the Python.org default cert path is empty."""
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
    m2_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-root", type=Path, default=m2_root)
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=m2_root / "results" / "generated_reports",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=m2_root / "prompts" / "pairwise_judge_evidence_aware.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=m2_root / "results" / "pairwise_judge",
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o-mini"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--api-delay-sec", type=float, default=0.25)
    parser.add_argument("--retry-base-sec", type=float, default=3.0)
    parser.add_argument("--rate-limit-retry-sec", type=float, default=45.0)
    parser.add_argument(
        "--pairing",
        choices=("same-run", "all-runs"),
        default="same-run",
        help="How to pair reports for the same case and condition pair.",
    )
    parser.add_argument(
        "--include-reverse",
        action="store_true",
        help="Also judge the reversed A/B order to estimate order sensitivity.",
    )
    parser.add_argument("--case-id", dest="case_ids", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def report_files(generated_root: Path, condition: str) -> list[Path]:
    reports_dir = generated_root / "natural" / condition / "reports"
    return sorted(
        path
        for path in reports_dir.glob("*/*.json")
        if path.is_file() and not path.name.startswith("._")
    )


def load_reports(generated_root: Path, case_ids: set[str] | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    reports: dict[tuple[str, str, int], dict[str, Any]] = {}
    for condition in ("rgb_only", "reba_only", "full_module2"):
        for path in report_files(generated_root, condition):
            payload = read_json(path)
            report_text = payload.get("report_text")
            if not isinstance(report_text, str) or not report_text.strip():
                continue
            case_id = payload.get("case_id") or payload.get("sample_id") or path.parent.name
            if case_ids is not None and case_id not in case_ids:
                continue
            run_index = int(payload.get("run_index") or 0)
            reports[(condition, case_id, run_index)] = {
                "condition": condition,
                "case_id": case_id,
                "run_index": run_index,
                "report_text": report_text,
                "file": str(path),
            }
    return reports


def build_comparisons(
    reports: dict[tuple[str, str, int], dict[str, Any]],
    pairing: str,
    include_reverse: bool,
) -> list[dict[str, Any]]:
    by_condition_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for report in reports.values():
        by_condition_case.setdefault((report["condition"], report["case_id"]), []).append(report)

    case_ids = sorted({report["case_id"] for report in reports.values()})
    comparisons: list[dict[str, Any]] = []
    for left_condition, right_condition in CONDITION_PAIRS:
        for case_id in case_ids:
            left_reports = sorted(
                by_condition_case.get((left_condition, case_id), []),
                key=lambda item: item["run_index"],
            )
            right_reports = sorted(
                by_condition_case.get((right_condition, case_id), []),
                key=lambda item: item["run_index"],
            )
            if pairing == "same-run":
                right_by_run = {item["run_index"]: item for item in right_reports}
                pairs = [
                    (left, right_by_run[left["run_index"]])
                    for left in left_reports
                    if left["run_index"] in right_by_run
                ]
            else:
                pairs = [(left, right) for left in left_reports for right in right_reports]

            for left, right in pairs:
                comparisons.append(
                    {
                        "condition_pair": f"{left_condition}_vs_{right_condition}",
                        "case_id": case_id,
                        "run_index_A": left["run_index"],
                        "run_index_B": right["run_index"],
                        "report_A": left,
                        "report_B": right,
                        "order": "forward",
                    }
                )
                if include_reverse:
                    comparisons.append(
                        {
                            "condition_pair": f"{left_condition}_vs_{right_condition}",
                            "case_id": case_id,
                            "run_index_A": right["run_index"],
                            "run_index_B": left["run_index"],
                            "report_A": right,
                            "report_B": left,
                            "order": "reverse",
                        }
                    )
    return comparisons


def render_prompt(template: str, comparison: dict[str, Any]) -> str:
    report_a = comparison["report_A"]
    report_b = comparison["report_B"]
    return (
        template.replace("{REPORT_A_CONDITION}", report_a["condition"])
        .replace("{REPORT_A_TEXT}", report_a["report_text"])
        .replace("{REPORT_B_CONDITION}", report_b["condition"])
        .replace("{REPORT_B_TEXT}", report_b["report_text"])
    )


def call_openai(
    *,
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
                "content": "Return only valid JSON following the requested pairwise schema.",
            },
            {"role": "user", "content": prompt},
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


def parse_decision(response_payload: dict[str, Any]) -> dict[str, Any]:
    content = response_payload["choices"][0]["message"]["content"]
    return json.loads(content)


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


def jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def comparison_id(comparison: dict[str, Any], index: int) -> str:
    return (
        f"{index:04d}__{comparison['condition_pair']}__"
        f"{comparison['case_id']}__A{comparison['run_index_A']:02d}_B{comparison['run_index_B']:02d}__"
        f"{comparison['order']}"
    )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = args.output_dir / f"pairwise_results_{timestamp}.jsonl"
    prompt_dir = args.output_dir / f"prompts_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(
        args.generated_root,
        set(args.case_ids) if args.case_ids else None,
    )
    comparisons = build_comparisons(reports, args.pairing, args.include_reverse)
    if not comparisons:
        raise SystemExit("No pairwise comparisons available. Generate natural reports first.")

    run_manifest = {
        "analysis": "module2_configuration_pairwise_judge",
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "pairing": args.pairing,
        "include_reverse": args.include_reverse,
        "prompt_file": str(args.prompt_file),
        "generated_root": str(args.generated_root),
        "result_path": str(result_path),
        "comparison_count": len(comparisons),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
    }
    write_json(args.output_dir / f"run_manifest_{timestamp}.json", run_manifest)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    template = args.prompt_file.read_text(encoding="utf-8")
    for index, comparison in enumerate(comparisons, start=1):
        comp_id = comparison_id(comparison, index)
        prompt = render_prompt(template, comparison)
        (prompt_dir / f"{comp_id}.txt").write_text(prompt, encoding="utf-8")

        if args.dry_run:
            item = {
                "comparison_id": comp_id,
                "dry_run": True,
                "judge_model": args.model,
                "condition_pair": comparison["condition_pair"],
                "case_id": comparison["case_id"],
                "order": comparison["order"],
                "report_A": {
                    "condition": comparison["report_A"]["condition"],
                    "run_index": comparison["report_A"]["run_index"],
                    "file": comparison["report_A"]["file"],
                },
                "report_B": {
                    "condition": comparison["report_B"]["condition"],
                    "run_index": comparison["report_B"]["run_index"],
                    "file": comparison["report_B"]["file"],
                },
                "decision": None,
            }
            jsonl_append(result_path, item)
            print(f"[{index}/{len(comparisons)}] wrote dry run {comp_id}")
            continue

        print(f"[{index}/{len(comparisons)}] judging {comp_id}")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response_payload = call_openai(
                    api_key=api_key or "",
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                item = {
                    "comparison_id": comp_id,
                    "dry_run": False,
                    "judge_model": args.model,
                    "condition_pair": comparison["condition_pair"],
                    "case_id": comparison["case_id"],
                    "order": comparison["order"],
                    "report_A": {
                        "condition": comparison["report_A"]["condition"],
                        "run_index": comparison["report_A"]["run_index"],
                        "file": comparison["report_A"]["file"],
                    },
                    "report_B": {
                        "condition": comparison["report_B"]["condition"],
                        "run_index": comparison["report_B"]["run_index"],
                        "file": comparison["report_B"]["file"],
                    },
                    "decision": parse_decision(response_payload),
                    "usage": response_payload.get("usage", {}),
                    "response_id": response_payload.get("id"),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                jsonl_append(result_path, item)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc
                if attempt == 3:
                    item = {
                        "comparison_id": comp_id,
                        "dry_run": False,
                        "judge_model": args.model,
                        "condition_pair": comparison["condition_pair"],
                        "case_id": comparison["case_id"],
                        "order": comparison["order"],
                        "report_A": {
                            "condition": comparison["report_A"]["condition"],
                            "run_index": comparison["report_A"]["run_index"],
                            "file": comparison["report_A"]["file"],
                        },
                        "report_B": {
                            "condition": comparison["report_B"]["condition"],
                            "run_index": comparison["report_B"]["run_index"],
                            "file": comparison["report_B"]["file"],
                        },
                        "decision": None,
                        "error": str(last_error),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    }
                    jsonl_append(result_path, item)
                    print(f"  failed after 3 attempts: {last_error}")
                else:
                    sleep_sec = retry_sleep_seconds(exc, attempt, args)
                    print(f"  retrying after {sleep_sec:.1f}s: {exc}")
                    time.sleep(sleep_sec)
        time.sleep(args.api_delay_sec)

    print("Pairwise judge run complete.")
    print(result_path)


if __name__ == "__main__":
    main()
