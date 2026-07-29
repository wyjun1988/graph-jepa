from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_contract import (
    LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY,
)
from stock_v2.news_structuring import (
    LABEL_SCHEMA,
    OUTPUT_SCHEMA_VERSION,
    PROMPT_VERSION,
    build_messages,
    compatible_resume_ids,
    deterministic_sample,
    materialize_event,
    parse_json_content,
    validate_labels,
)


_LOCAL = threading.local()


def _session() -> requests.Session:
    if not hasattr(_LOCAL, "session"):
        _LOCAL.session = requests.Session()
    return _LOCAL.session


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"non-object queue row at {path}:{line_number}")
            yield payload


def completed_ids(
    path: Path,
    args: argparse.Namespace,
    target_records: list[dict[str, Any]],
) -> set[str]:
    if not path.exists():
        return set()
    return compatible_resume_ids(
        list(iter_jsonl(path)),
        target_records,
        required_lineage={
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "prompt_version": PROMPT_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "inference_engine": "openai_compatible_http",
        },
    )


def request_labels(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    body = {
        "model": args.model,
        "messages": build_messages(record),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "kr_stock_news_event", "schema": LABEL_SCHEMA},
        },
    }
    error = ""
    for attempt in range(args.retries + 1):
        try:
            response = _session().post(args.endpoint, json=body, timeout=args.timeout_sec)
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            return validate_labels(parse_json_content(content)), ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
            if attempt < args.retries:
                time.sleep(0.5 * (2**attempt))
    return None, error


def process_one(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    labels, error = request_labels(record, args)
    generated_at = datetime.now(tz=timezone.utc).isoformat()
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "queue_id": record["queue_id"],
        "article_id": record["article_id"],
        "ticker": record["ticker"],
        "effective_session": record["effective_session"],
        "input_sha256": record["input_sha256"],
        "llm_used": labels is not None,
        "llm_error": error,
        "lineage": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "served_model": args.model,
            "prompt_version": PROMPT_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "input_hash_policy": str(
                record.get("input_hash_policy") or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
            ),
            "generated_at_utc": generated_at,
            "inference_engine": "openai_compatible_http",
            "decoding": {"temperature": 0.0, "top_p": 1.0, "enable_thinking": False},
        },
    }
    if labels is not None:
        result["labels"] = labels
        result["event"] = materialize_event(record, labels)
    return result


def chunks(rows: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structure a frozen stock-news queue through a vLLM endpoint.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", required=True, help="Served model name used in API requests.")
    parser.add_argument("--model-id", required=True, help="Immutable upstream model identifier.")
    parser.add_argument("--model-revision", required=True, help="Immutable upstream model commit.")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260712)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_records = list(iter_jsonl(input_path))
    selected_records = deterministic_sample(all_records, args.sample_size, args.sample_seed)
    done = completed_ids(output_path, args, selected_records) if args.resume else set()
    records = [
        row for row in selected_records if str(row.get("queue_id") or "") not in done
    ]
    mode = "a" if args.resume and output_path.exists() else "w"
    started = time.perf_counter()
    processed = ok = errors = 0
    with output_path.open(mode, encoding="utf-8") as output, ThreadPoolExecutor(
        max_workers=max(1, args.concurrency)
    ) as executor:
        for batch in chunks(records, max(1, args.concurrency) * 4):
            for result in executor.map(lambda row: process_one(row, args), batch):
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                processed += 1
                ok += int(bool(result["llm_used"]))
                errors += int(not result["llm_used"])
                if processed % max(1, args.print_every) == 0:
                    output.flush()
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    print(
                        json.dumps(
                            {
                                "processed": processed,
                                "ok": ok,
                                "errors": errors,
                                "remaining": len(records) - processed,
                                "rows_per_second": round(processed / elapsed, 3),
                            }
                        ),
                        flush=True,
                    )
        output.flush()
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "selected": len(records),
                "ok": ok,
                "errors": errors,
                "elapsed_sec": round(time.perf_counter() - started, 3),
            },
            indent=2,
        )
    )
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
