from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.qwen_events import QwenEventExtractor


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for record in iter_jsonl(path):
        if record.get("id"):
            done.add(str(record["id"]))
    return done


def parse_date(record: dict[str, Any]) -> datetime | None:
    values = [record.get("published"), record.get("ts")]
    article = record.get("article")
    if isinstance(article, dict):
        values.extend([article.get("published"), article.get("updated")])
    for value in values:
        if not value:
            continue
        text = str(value)
        try:
            if "," in text:
                return parsedate_to_datetime(text).replace(tzinfo=None)
        except Exception:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
    return None


def article_text(record: dict[str, Any]) -> tuple[str, str]:
    article = record.get("article") if isinstance(record.get("article"), dict) else {}
    title = str(article.get("title") or record.get("title") or "")
    summary = str(article.get("summary") or record.get("summary") or "")
    return title, summary


def score_event(event_payload: dict[str, Any], ticker: str) -> float:
    score = 0.0
    for delta in event_payload.get("node_deltas", []) or []:
        if not isinstance(delta, dict):
            continue
        node = str(delta.get("node", "")).replace("A", "")
        if node != ticker:
            continue
        if str(delta.get("field", "news_score")) != "news_score":
            continue
        try:
            score += float(delta.get("delta", 0.0)) * float(delta.get("confidence", event_payload.get("confidence", 0.0)))
        except Exception:
            continue
    return float(score)


def priority_value(record: dict[str, Any], mode: str) -> tuple[int, float]:
    date = parse_date(record)
    ts = 0.0 if date is None else date.timestamp()
    score = abs(float(record.get("score_contribution") or 0.0))
    if mode == "recent_first":
        return (0, -ts)
    if mode == "nonzero_first":
        return (0 if score > 1e-12 else 1, -score)
    if mode == "nonzero_recent_first":
        return (0 if score > 1e-12 else 1, -ts)
    return (0, 0.0)


def qwen_available(endpoint: str) -> bool:
    if "/v1/" in endpoint:
        probe = endpoint.split("/v1/", 1)[0].rstrip("/") + "/v1/models"
    else:
        probe = endpoint.replace("/api/generate", "/api/tags")
    try:
        response = requests.get(probe, timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescore historical stock-news JSONL events with Qwen.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no explicit cap")
    parser.add_argument("--priority", choices=["file", "recent_first", "nonzero_first", "nonzero_recent_first"], default="nonzero_recent_first")
    parser.add_argument("--min-abs-heuristic-score", type=float, default=0.0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-endpoint", action="store_true")
    parser.add_argument("--keep-original-event", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.require_endpoint and not qwen_available(args.endpoint):
        raise RuntimeError(f"Qwen endpoint is not available: {args.endpoint}")

    done_ids = load_done_ids(output_path) if args.resume else set()
    records = [record for record in iter_jsonl(input_path) if str(record.get("id", "")) not in done_ids]
    if args.min_abs_heuristic_score > 0:
        records = [
            record for record in records
            if abs(float(record.get("score_contribution") or 0.0)) >= args.min_abs_heuristic_score
        ]
    if args.priority != "file":
        records.sort(key=lambda record: priority_value(record, args.priority))
    if args.limit > 0:
        records = records[: args.limit]

    extractor = QwenEventExtractor(model=args.model, endpoint=args.endpoint, timeout=args.timeout_sec)
    mode = "a" if args.resume else "w"
    started = time.perf_counter()
    ok = 0
    errors = 0
    with output_path.open(mode, encoding="utf-8") as output:
        for idx, record in enumerate(records, start=1):
            ticker = str(record.get("ticker", "")).replace("A", "").strip()
            title, summary = article_text(record)
            new_record = dict(record)
            previous_event = record.get("event")
            try:
                event = extractor.extract_one(title=title, summary=summary, universe=[ticker])
                event_payload = event.raw or {
                    "event_type": event.event_type,
                    "summary": event.summary,
                    "polarity": event.polarity,
                    "magnitude": event.magnitude,
                    "confidence": event.confidence,
                    "horizon_days": event.horizon_days,
                    "affected_nodes": event.affected_nodes,
                    "node_deltas": [asdict(delta) for delta in event.node_deltas],
                    "edge_deltas": [asdict(delta) for delta in event.edge_deltas],
                }
                new_record["event"] = event_payload
                new_record["score_contribution"] = score_event(event_payload, ticker)
                new_record["llm_used"] = True
                new_record["llm_error"] = ""
                new_record["source"] = "qwen_rescore"
                ok += 1
            except Exception as exc:
                new_record["llm_used"] = False
                new_record["llm_error"] = str(exc)[:500]
                new_record["source"] = "qwen_rescore_error"
                errors += 1
            new_record["rescore"] = {
                "model": args.model,
                "endpoint": args.endpoint,
                "rescored_at": datetime.now().isoformat(timespec="seconds"),
                "input": str(input_path),
                "previous_source": record.get("source"),
                "previous_score_contribution": record.get("score_contribution"),
            }
            if args.keep_original_event:
                new_record["previous_event"] = previous_event
            output.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            output.flush()

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
            if idx % max(1, args.print_every) == 0:
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "processed": idx,
                    "ok": ok,
                    "errors": errors,
                    "remaining_in_batch": len(records) - idx,
                    "elapsed_sec": round(elapsed, 1),
                    "events_per_hour": round(idx / max(elapsed, 1e-6) * 3600, 1),
                    "last_ticker": ticker,
                }, ensure_ascii=False), flush=True)

    print(json.dumps({
        "input": str(input_path),
        "output": str(output_path),
        "selected": len(records),
        "ok": ok,
        "errors": errors,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
