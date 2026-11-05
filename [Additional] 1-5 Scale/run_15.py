"""
run.py

Stage-1 prompt robustness experiment (5-point Likert, 1 .. 5; 3 = neutral).

Default design:
    N topics
    x 7 prompt framings (P1-P3, I1-I4)
    x 1 label order (normal)
    x 3 statement versions (original / paraphrase / negation)
    x runs
    per model-language pair

  Qwen/Qwen3.6-27B-FP8
  openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

from stimuli_15 import (
    build_conditions,
    parse_response,
    align_score,
    stance_score,
    SCALE_MID,
)


DEFAULT_RUNS = 20
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_SLEEP = 0.5
DEFAULT_SEED = 42
MAX_RETRIES = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env")


def make_client(provider: str):
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return Groq(api_key=api_key)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key)

    if provider == "mdx":
        api_key = os.getenv("MDX_API_KEY")
        base_url = os.getenv("MDX_BASE_URL")
        if not api_key:
            raise RuntimeError("MDX_API_KEY is not set.")
        if not base_url:
            raise RuntimeError(
                "MDX_BASE_URL is not set. "
                "Set it to the mdx endpoint URL in your .env file."
            )
        return OpenAI(base_url=base_url, api_key=api_key)

    if provider == "plamo":
        api_key = os.getenv("PLAMO_API_KEY")
        if not api_key:
            raise RuntimeError("PLAMO_API_KEY is not set.")
        return OpenAI(
            api_key=api_key,
            base_url="https://api.platform.preferredai.jp/v1",
        )

    raise ValueError(f"Unsupported provider: {provider}")


def extract_reasoning(message) -> str:
    """
    Reasoning / thinking text returned separately from the answer, if any.
    Field name differs by server: reasoning_content (vLLM etc.) or reasoning (Groq etc.).
    """
    for name in ("reasoning_content", "reasoning"):
        value = getattr(message, name, None)
        if value:
            return str(value)

    extra = getattr(message, "model_extra", None) or {}
    for name in ("reasoning_content", "reasoning"):
        if extra.get(name):
            return str(extra[name])

    return ""


def usage_to_dict(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    try:
        return usage.model_dump()
    except Exception:
        return {}


def call_model(client, model, prompt, temperature, max_tokens, system=None):
    """
    Returns (text, latency, meta).
    meta = reasoning / finish_reason / usage / response_id / response_model
    `system` (optional) is sent as the system message; `prompt` as the user message.
    """
    last_error = None

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            latency = time.perf_counter() - start

            choice = response.choices[0]
            text = (choice.message.content or "").strip()

            meta = {
                "reasoning": extract_reasoning(choice.message),
                "finish_reason": choice.finish_reason,
                "usage": usage_to_dict(response),
                "response_id": getattr(response, "id", None),
                "response_model": getattr(response, "model", None),
            }
            return text, latency, meta

        except Exception as exc:
            last_error = exc
            wait = min(2 ** (attempt - 1), 30)
            print(f"[WARN] API call failed ({attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(f"API call failed after retries: {last_error}")


def score_to_stance(canonical_score: Optional[int]) -> Optional[str]:
    """
    Map a canonical score (1..5) to a coarse stance label.
      > 3 -> support, < 3 -> oppose, 3 -> neutral
    """
    if canonical_score is None:
        return None
    if canonical_score > SCALE_MID:
        return "support"
    if canonical_score < SCALE_MID:
        return "oppose"
    return "neutral"


def make_scores(response_text: str, label_order: str, statement_polarity: int):
    """
    raw_score        : integer the model answered (1..5), as shown on its scale
    aligned_score    : raw_score put on the normal axis (reversed -> 6 - raw)
                       (5 = agree / beneficial / favorable toward the statement)
    canonical_score  : aligned_score flipped around the midpoint (3) by the
                       statement polarity
                       (> 3 = supports the ORIGINAL-direction claim, regardless
                        of whether the statement is original / paraphrase /
                        negation)
    """
    raw = parse_response(response_text)
    if raw is None:
        return None, None, None

    aligned = align_score(raw, label_order)
    canonical = stance_score(aligned, statement_polarity)
    return raw, aligned, canonical


CSV_FIELDS = [
    "timestamp_utc",
    "provider",
    "model",
    "language",
    "seed",
    "temperature",
    "max_tokens",
    "run",
    "condition_id",
    "topic_id",
    "topic",
    "topic_en",
    "topic_ja",
    "statement_version",
    "statement",
    "statement_polarity",
    "framing_id",
    "framing_type",
    "label_pair",
    "label_order",
    "positive_label",
    "negative_label",
    "shown_label_a",
    "shown_label_b",
    "scale_minus2_label",
    "scale_plus2_label",
    "prompt",
    "response_raw",
    "finish_reason",
    "raw_score",
    "aligned_score",
    "canonical_score",
    "canonical_stance",
    "parse_success",
    "latency_seconds",
    "error",
]


def sanitize_filename(value: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in value
    )


def make_output_paths(provider, model, language):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{stamp}_{sanitize_filename(provider)}_{sanitize_filename(model)}_{language}"
    return OUTPUT_DIR / f"{stem}.csv", OUTPUT_DIR / f"{stem}.jsonl"


def append_csv(path: Path, row: dict):
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_experiment(args):
    client = make_client(args.provider)

    conditions = build_conditions(
        language=args.language,
        topic_ids=args.topics,
        statement_versions=args.statement_versions,
    )

    jobs = []
    for condition in conditions:
        for run_no in range(1, args.runs + 1):
            jobs.append((condition, run_no))

    rng = random.Random(args.seed)
    rng.shuffle(jobs)

    csv_path, jsonl_path = make_output_paths(
        args.provider, args.model, args.language
    )

    print("=" * 70)
    print("PROMPT ROBUSTNESS PILOT (5-point Likert, 1..5)")
    print("=" * 70)
    print(f"Provider        : {args.provider}")
    print(f"Model           : {args.model}")
    print(f"Language        : {args.language}")
    print(f"Topics          : {len(set(c['topic_id'] for c in conditions))}")
    print(f"Framings        : {len(set(c['framing_id'] for c in conditions))}")
    print(f"Label orders    : {len(set(c['label_order'] for c in conditions))}")
    print(f"Statement vers. : {len(set(c['statement_version'] for c in conditions))}")
    print(f"Runs/condition  : {args.runs}")
    print(f"TOTAL API CALLS : {len(jobs)}")
    print(f"CSV             : {csv_path}")
    print(f"JSONL           : {jsonl_path}")
    print("=" * 70)

    for idx, (condition, run_no) in enumerate(jobs, start=1):
        print(
            f"[{idx:04d}/{len(jobs)}] "
            f"{condition['topic_id']} | "
            f"{condition['framing_id']} | "
            f"{condition['label_order']} | "
            f"{condition['statement_version']} | run={run_no}"
        )

        error = ""
        response_text = ""
        latency = None
        meta = {}
        raw_score = aligned_score = canonical_score = None
        stance = None

        try:
            response_text, latency, meta = call_model(
                client=client,
                model=args.model,
                prompt=condition["prompt"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                system=condition["system"],
            )

            raw_score, aligned_score, canonical_score = make_scores(
                response_text,
                condition["label_order"],
                condition["statement_polarity"],
            )
            stance = score_to_stance(canonical_score)

        except Exception as exc:
            error = repr(exc)

        def _val(x):
            return "" if x is None else x

        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "provider": args.provider,
            "model": args.model,
            "language": args.language,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "run": run_no,
            **condition,  # includes "system" and "scenario" (JSONL only)
            "response_raw": response_text,
            # JSONL-only extras (CSV ignores keys not in CSV_FIELDS, except finish_reason)
            "reasoning": meta.get("reasoning", ""),
            "finish_reason": meta.get("finish_reason", ""),
            "usage": meta.get("usage", {}),
            "response_id": meta.get("response_id", ""),
            "response_model": meta.get("response_model", ""),
            "raw_score": _val(raw_score),
            "aligned_score": _val(aligned_score),
            "canonical_score": _val(canonical_score),
            "canonical_stance": stance or "",
            "parse_success": int(raw_score is not None),
            "latency_seconds": round(latency, 4) if latency is not None else "",
            "error": error,
        }

        append_csv(csv_path, row)
        append_jsonl(jsonl_path, row)

        print(
            f"  response={response_text!r} | raw={raw_score} | "
            f"aligned={aligned_score} | canonical={canonical_score} | "
            f"parsed={raw_score is not None}"
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("\nFinished.")
    print(f"CSV   : {csv_path}")
    print(f"JSONL : {jsonl_path}")
    print(f'\nAnalyze with:\n  python analyze.py "{csv_path}"')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--provider",
        choices=["groq", "openai", "mdx", "plamo"],
        default="groq",
    )
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--language", choices=["en", "ja"], default="en")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--topics",
        nargs="*",
        default=None,
        help=(
            "Optional topic IDs defined in stimuli.TOPICS. "
            "If omitted, all active topics are used. "
            "Example: --topics sns_under16_regulation"
        ),
    )

    parser.add_argument(
        "--statement-versions",
        nargs="*",
        default=None,
        choices=["original", "paraphrase", "negation"],
        help="Optional subset; if omitted, the default three versions are used.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())