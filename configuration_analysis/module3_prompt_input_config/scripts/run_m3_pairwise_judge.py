#!/usr/bin/env python3
"""Run LLM-as-Judge pairwise comparisons for Module #3 configuration reports."""

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
if M3_SRC.exists():
    sys.path.insert(0, str(M3_SRC))
else:
    sys.path.insert(0, str(Path("<private_workspace>/m3/src")))

from structured_common import read_json, write_json  # noqa: E402


OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
INPUT_CONDITIONS = ("module2_only", "module2_rgb")
PROMPT_CONDITIONS = (
    "neutral",
    "original_rgb_compatible",
    "bounded_context_augmented",
)
DEFAULT_PAIR_MODES = ("input_effect", "prompt_effect_adjacent")


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
        "--generated-root",
        type=Path,
        help="Defaults to <m3-root>/results/generated_reports.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Defaults to <m3-root>/prompts/pairwise_judge_evidence_aware.txt.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <m3-root>/results/pairwise_judge.",
    )
    parser.add_argument(
        "--pair-mode",
        action="append",
        choices=("input_effect", "prompt_effect_adjacent", "prompt_effect_all"),
        help="Pair family to judge. Repeat for multiple. Default: input_effect and prompt_effect_adjacent.",
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
    )
    parser.add_argument("--include-reverse", action="store_true")
    parser.add_argument("--case-id", dest="case_ids", action="append")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def condition_id(input_condition: str, prompt_condition: str) -> str:
    return f"{input_condition}__{prompt_condition}"


def condition_pairs(pair_modes: tuple[str, ...]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if "input_effect" in pair_modes:
        for prompt_condition in PROMPT_CONDITIONS:
            pairs.append(
                (
                    condition_id("module2_only", prompt_condition),
                    condition_id("module2_rgb", prompt_condition),
                )
            )
    prompt_pairs = []
    if "prompt_effect_adjacent" in pair_modes:
        prompt_pairs.extend(
            [
                ("neutral", "original_rgb_compatible"),
                ("original_rgb_compatible", "bounded_context_augmented"),
            ]
        )
    if "prompt_effect_all" in pair_modes:
        prompt_pairs.extend(
            [
                ("neutral", "original_rgb_compatible"),
                ("original_rgb_compatible", "bounded_context_augmented"),
                ("neutral", "bounded_context_augmented"),
            ]
        )
    seen_prompt_pairs: set[tuple[str, str]] = set()
    for left_prompt, right_prompt in prompt_pairs:
        if (left_prompt, right_prompt) in seen_prompt_pairs:
            continue
        seen_prompt_pairs.add((left_prompt, right_prompt))
        for input_condition in INPUT_CONDITIONS:
            pairs.append(
                (
                    condition_id(input_condition, left_prompt),
                    condition_id(input_condition, right_prompt),
                )
            )
    return pairs


def report_files(generated_root: Path, condition: str) -> list[Path]:
    input_condition, prompt_condition = condition.split("__", 1)
    reports_dir = generated_root / "natural" / input_condition / prompt_condition / "reports"
    return sorted(
        path
        for path in reports_dir.glob("*/*.json")
        if path.is_file() and not path.name.startswith("._")
    )


def load_reports(
    generated_root: Path,
    case_ids: set[str] | None,
    pairs: list[tuple[str, str]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    needed_conditions = sorted({condition for pair in pairs for condition in pair})
    reports: dict[tuple[str, str, int], dict[str, Any]] = {}
    for condition in needed_conditions:
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
                "input_condition": payload.get("input_condition", condition.split("__", 1)[0]),
                "prompt_condition": payload.get("prompt_condition", condition.split("__", 1)[1]),
                "case_id": case_id,
                "run_index": run_index,
                "report_text": report_text,
                "file": str(path),
            }
    return reports


def build_comparisons(
    reports: dict[tuple[str, str, int], dict[str, Any]],
    pairs: list[tuple[str, str]],
    pairing: str,
    include_reverse: bool,
) -> list[dict[str, Any]]:
    by_condition_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for report in reports.values():
        by_condition_case.setdefault((report["condition"], report["case_id"]), []).append(report)
    case_ids = sorted({report["case_id"] for report in reports.values()})
    comparisons: list[dict[str, Any]] = []
    for left_condition, right_condition in pairs:
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
                report_pairs = [
                    (left, right_by_run[left["run_index"]])
                    for left in left_reports
                    if left["run_index"] in right_by_run
                ]
            else:
                report_pairs = [(left, right) for left in left_reports for right in right_reports]
            for left, right in report_pairs:
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


def report_meta(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": report["condition"],
        "input_condition": report.get("input_condition"),
        "prompt_condition": report.get("prompt_condition"),
        "run_index": report["run_index"],
        "file": report["file"],
    }


def main() -> None:
    args = parse_args()
    generated_root = args.generated_root or args.m3_root / "results" / "generated_reports"
    prompt_file = args.prompt_file or args.m3_root / "prompts" / "pairwise_judge_evidence_aware.txt"
    output_dir = args.output_dir or args.m3_root / "results" / "pairwise_judge"
    pair_modes = tuple(args.pair_mode or DEFAULT_PAIR_MODES)
    pairs = condition_pairs(pair_modes)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"pairwise_results_{timestamp}.jsonl"
    prompt_dir = output_dir / f"prompts_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(
        generated_root,
        set(args.case_ids) if args.case_ids else None,
        pairs,
    )
    comparisons = build_comparisons(reports, pairs, args.pairing, args.include_reverse)
    if not comparisons:
        raise SystemExit("No pairwise comparisons available. Generate natural reports first.")

    run_manifest = {
        "analysis": "module3_configuration_pairwise_judge",
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "pair_modes": list(pair_modes),
        "condition_pairs": [f"{left}_vs_{right}" for left, right in pairs],
        "pairing": args.pairing,
        "include_reverse": args.include_reverse,
        "prompt_file": str(prompt_file),
        "generated_root": str(generated_root),
        "result_path": str(result_path),
        "comparison_count": len(comparisons),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dry_run": args.dry_run,
    }
    write_json(output_dir / f"run_manifest_{timestamp}.json", run_manifest)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not args.dry_run and not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    template = prompt_file.read_text(encoding="utf-8")
    for index, comparison in enumerate(comparisons, start=1):
        comp_id = comparison_id(comparison, index)
        prompt = render_prompt(template, comparison)
        (prompt_dir / f"{comp_id}.txt").write_text(prompt, encoding="utf-8")
        if args.dry_run:
            jsonl_append(
                result_path,
                {
                    "comparison_id": comp_id,
                    "dry_run": True,
                    "judge_model": args.model,
                    "condition_pair": comparison["condition_pair"],
                    "case_id": comparison["case_id"],
                    "order": comparison["order"],
                    "report_A": report_meta(comparison["report_A"]),
                    "report_B": report_meta(comparison["report_B"]),
                    "decision": None,
                },
            )
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
                jsonl_append(
                    result_path,
                    {
                        "comparison_id": comp_id,
                        "dry_run": False,
                        "judge_model": args.model,
                        "condition_pair": comparison["condition_pair"],
                        "case_id": comparison["case_id"],
                        "order": comparison["order"],
                        "report_A": report_meta(comparison["report_A"]),
                        "report_B": report_meta(comparison["report_B"]),
                        "decision": parse_decision(response_payload),
                        "usage": response_payload.get("usage", {}),
                        "response_id": response_payload.get("id"),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
                break
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc
                if attempt == 3:
                    jsonl_append(
                        result_path,
                        {
                            "comparison_id": comp_id,
                            "dry_run": False,
                            "judge_model": args.model,
                            "condition_pair": comparison["condition_pair"],
                            "case_id": comparison["case_id"],
                            "order": comparison["order"],
                            "report_A": report_meta(comparison["report_A"]),
                            "report_B": report_meta(comparison["report_B"]),
                            "decision": None,
                            "error": str(last_error),
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                        },
                    )
                    print(f"  failed after 3 attempts: {last_error}")
                else:
                    sleep_sec = retry_sleep_seconds(exc, attempt, args)
                    print(f"  retrying after {sleep_sec:.1f}s: {exc}")
                    time.sleep(sleep_sec)
        time.sleep(args.api_delay_sec)

    print("Module #3 pairwise judge run complete.")
    print(result_path)


if __name__ == "__main__":
    main()
