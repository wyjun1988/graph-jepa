from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.qwen_events import QwenEventExtractor


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def select_samples(rows: list[dict[str, Any]], per_ticker: int, limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reversed(rows):
        ticker = str(row.get("ticker", ""))
        article = row.get("article", {})
        key = row.get("id") or f"{ticker}:{article.get('title')}:{article.get('link')}"
        if not ticker or key in seen:
            continue
        seen.add(str(key))
        if len(buckets[ticker]) < per_ticker:
            buckets[ticker].append(row)
    samples: list[dict[str, Any]] = []
    for ticker in sorted(buckets):
        samples.extend(reversed(buckets[ticker]))
    return samples[:limit]


def score_event(raw: dict[str, Any], ticker: str) -> float:
    total = 0.0
    for delta in raw.get("node_deltas", []):
        if str(delta.get("node", "")).replace("A", "") == ticker and delta.get("field", "news_score") == "news_score":
            total += float(delta.get("delta", 0.0)) * float(delta.get("confidence", raw.get("confidence", 0.0)))
    if not raw.get("node_deltas"):
        total = float(raw.get("polarity", 0.0)) * float(raw.get("magnitude", 0.0)) * float(raw.get("confidence", 0.0))
    return total


def sign_bucket(score: float, threshold: float) -> str:
    if score > threshold:
        return "positive"
    if score < -threshold:
        return "negative"
    return "neutral"


def eval_one(extractor: QwenEventExtractor, sample: dict[str, Any]) -> dict[str, Any]:
    ticker = str(sample["ticker"])
    article = sample.get("article", {})
    started = time.perf_counter()
    event = extractor.extract_one(str(article.get("title", "")), str(article.get("summary", "")), [ticker])
    latency = time.perf_counter() - started
    raw = event.raw
    return {
        "ok": True,
        "latency_sec": latency,
        "event_type": raw.get("event_type", event.event_type),
        "polarity": float(raw.get("polarity", event.polarity)),
        "magnitude": float(raw.get("magnitude", event.magnitude)),
        "confidence": float(raw.get("confidence", event.confidence)),
        "horizon_days": int(raw.get("horizon_days", event.horizon_days)),
        "score": score_event(raw, ticker),
        "themes": raw.get("raw_llm", {}).get("themes") or [d.get("src") for d in raw.get("edge_deltas", [])],
        "raw": raw,
    }


def safe_eval(extractor: QwenEventExtractor, sample: dict[str, Any]) -> dict[str, Any]:
    try:
        return eval_one(extractor, sample)
    except Exception as exc:
        return {"ok": False, "latency_sec": math.nan, "error": str(exc)[:500]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ops/sensing/news_events.jsonl")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--per-ticker", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--score-tolerance", type=float, default=0.20)
    parser.add_argument("--endpoint-9b", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--endpoint-4b", default="http://127.0.0.1:8002/v1/chat/completions")
    parser.add_argument("--endpoint-exaone", default="")
    parser.add_argument("--out-dir", default="reports/qwen_size_eval")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    samples = select_samples(rows, per_ticker=args.per_ticker, limit=args.limit)
    if len(samples) < args.limit:
        print(f"warning: only selected {len(samples)} samples", file=sys.stderr)

    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    extractors = {
        "qwen9b": QwenEventExtractor(endpoint=args.endpoint_9b, model="auto", timeout=180),
        "qwen4b": QwenEventExtractor(endpoint=args.endpoint_4b, model="auto", timeout=180),
    }
    if args.endpoint_exaone:
        extractors["exaone24"] = QwenEventExtractor(endpoint=args.endpoint_exaone, model="auto", timeout=180)

    results: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples, 1):
        ticker = str(sample.get("ticker", ""))
        article = sample.get("article", {})
        row: dict[str, Any] = {
            "idx": idx,
            "ticker": ticker,
            "name": sample.get("name", ""),
            "title": article.get("title", ""),
            "published": article.get("published", ""),
            "source": article.get("source", ""),
        }
        print(json.dumps({"idx": idx, "ticker": ticker, "title": row["title"][:80]}, ensure_ascii=False), flush=True)
        for label, extractor in extractors.items():
            res = safe_eval(extractor, sample)
            row[label] = res
            if res.get("ok"):
                row[f"{label}_sign"] = sign_bucket(float(res["score"]), args.threshold)
                row[f"{label}_score"] = float(res["score"])
                row[f"{label}_event_type"] = res.get("event_type", "")
                row[f"{label}_latency_sec"] = float(res["latency_sec"])
            else:
                row[f"{label}_sign"] = "error"
                row[f"{label}_score"] = math.nan
                row[f"{label}_event_type"] = "error"
                row[f"{label}_latency_sec"] = math.nan
        row["sign_agree"] = row["qwen9b_sign"] == row["qwen4b_sign"]
        row["event_type_agree"] = row["qwen9b_event_type"] == row["qwen4b_event_type"]
        if row["qwen9b_sign"] != "error" and row["qwen4b_sign"] != "error":
            row["score_abs_diff"] = abs(row["qwen9b_score"] - row["qwen4b_score"])
            row["trade_relevant_agree"] = row["sign_agree"] and row["score_abs_diff"] <= args.score_tolerance
        else:
            row["score_abs_diff"] = math.nan
            row["trade_relevant_agree"] = False
        if "exaone24" in extractors:
            row["exaone24_sign_agree_9b"] = row["qwen9b_sign"] == row["exaone24_sign"]
            row["exaone24_event_type_agree_9b"] = row["qwen9b_event_type"] == row["exaone24_event_type"]
            if row["qwen9b_sign"] != "error" and row["exaone24_sign"] != "error":
                row["exaone24_score_abs_diff_9b"] = abs(row["qwen9b_score"] - row["exaone24_score"])
                row["exaone24_trade_relevant_agree_9b"] = row["exaone24_sign_agree_9b"] and row["exaone24_score_abs_diff_9b"] <= args.score_tolerance
            else:
                row["exaone24_score_abs_diff_9b"] = math.nan
                row["exaone24_trade_relevant_agree_9b"] = False
        results.append(row)

    flat_fields = [
        "idx", "ticker", "name", "source", "published", "title",
        "qwen9b_event_type", "qwen4b_event_type", "event_type_agree",
        "qwen9b_sign", "qwen4b_sign", "sign_agree",
        "qwen9b_score", "qwen4b_score", "score_abs_diff", "trade_relevant_agree",
        "qwen9b_latency_sec", "qwen4b_latency_sec",
        "exaone24_event_type", "exaone24_sign", "exaone24_score", "exaone24_latency_sec",
        "exaone24_sign_agree_9b", "exaone24_event_type_agree_9b", "exaone24_score_abs_diff_9b", "exaone24_trade_relevant_agree_9b",
    ]
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in flat_fields})

    (out_dir / "comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ok_rows = [r for r in results if r["qwen9b_sign"] != "error" and r["qwen4b_sign"] != "error"]
    lat9 = [r["qwen9b_latency_sec"] for r in ok_rows]
    lat4 = [r["qwen4b_latency_sec"] for r in ok_rows]
    summary = {
        "sample_count": len(results),
        "ok_count": len(ok_rows),
        "qwen9b_success": sum(1 for r in results if r["qwen9b_sign"] != "error"),
        "qwen4b_success": sum(1 for r in results if r["qwen4b_sign"] != "error"),
        "sign_agreement_rate": sum(1 for r in ok_rows if r["sign_agree"]) / len(ok_rows) if ok_rows else 0.0,
        "event_type_agreement_rate": sum(1 for r in ok_rows if r["event_type_agree"]) / len(ok_rows) if ok_rows else 0.0,
        "trade_relevant_agreement_rate": sum(1 for r in ok_rows if r["trade_relevant_agree"]) / len(ok_rows) if ok_rows else 0.0,
        "mean_score_abs_diff": statistics.mean([r["score_abs_diff"] for r in ok_rows]) if ok_rows else None,
        "median_score_abs_diff": statistics.median([r["score_abs_diff"] for r in ok_rows]) if ok_rows else None,
        "qwen9b_mean_latency_sec": statistics.mean(lat9) if lat9 else None,
        "qwen4b_mean_latency_sec": statistics.mean(lat4) if lat4 else None,
        "qwen4b_speedup": (statistics.mean(lat9) / statistics.mean(lat4)) if lat9 and lat4 else None,
        "threshold": args.threshold,
        "score_tolerance": args.score_tolerance,
        "out_dir": str(out_dir),
    }
    if "exaone24" in extractors:
        ex_ok_rows = [r for r in results if r["qwen9b_sign"] != "error" and r["exaone24_sign"] != "error"]
        ex_lat = [r["exaone24_latency_sec"] for r in ex_ok_rows]
        summary.update({
            "exaone24_success": sum(1 for r in results if r.get("exaone24_sign") != "error"),
            "exaone24_sign_agreement_rate_vs_9b": sum(1 for r in ex_ok_rows if r.get("exaone24_sign_agree_9b")) / len(ex_ok_rows) if ex_ok_rows else 0.0,
            "exaone24_event_type_agreement_rate_vs_9b": sum(1 for r in ex_ok_rows if r.get("exaone24_event_type_agree_9b")) / len(ex_ok_rows) if ex_ok_rows else 0.0,
            "exaone24_trade_relevant_agreement_rate_vs_9b": sum(1 for r in ex_ok_rows if r.get("exaone24_trade_relevant_agree_9b")) / len(ex_ok_rows) if ex_ok_rows else 0.0,
            "exaone24_mean_score_abs_diff_vs_9b": statistics.mean([r["exaone24_score_abs_diff_9b"] for r in ex_ok_rows]) if ex_ok_rows else None,
            "exaone24_median_score_abs_diff_vs_9b": statistics.median([r["exaone24_score_abs_diff_9b"] for r in ex_ok_rows]) if ex_ok_rows else None,
            "exaone24_mean_latency_sec": statistics.mean(ex_lat) if ex_lat else None,
            "exaone24_speedup_vs_9b": (statistics.mean(lat9) / statistics.mean(ex_lat)) if lat9 and ex_lat else None,
        })
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
