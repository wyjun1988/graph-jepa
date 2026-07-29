from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any

import feedparser
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from stock_v2.ops.config import OpsConfig
from stock_v2.ops.engine import OpsEngine
from stock_v2.qwen_events import QwenEventExtractor
from stock_v2.news_sources import fetch_naver_search

POSITIVE_WORDS = [
    "호실적", "수주", "계약", "증설", "상향", "흑자", "개선", "성장", "최대", "회복",
    "승인", "인수", "투자", "배당", "자사주", "돌파", "강세", "상승", "도약",
]
NEGATIVE_WORDS = [
    "적자", "하락", "하향", "부진", "소송", "규제", "제재", "파업", "리콜", "손실",
    "압수수색", "감소", "철회", "취소", "약세", "급락", "우려", "위기", "부도",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def qwen_available(endpoint: str) -> bool:
    if "/v1/" in endpoint:
        base = endpoint.split("/v1/", 1)[0].rstrip("/")
        probe_endpoint = f"{base}/v1/models"
    else:
        probe_endpoint = endpoint.replace("/api/generate", "/api/tags")
    try:
        response = requests.get(probe_endpoint, timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def heuristic_event(ticker: str, name: str, title: str, summary: str, link: str) -> dict[str, Any]:
    text = f"{title} {summary}"
    pos = sum(1 for word in POSITIVE_WORDS if word in text)
    neg = sum(1 for word in NEGATIVE_WORDS if word in text)
    raw_score = pos - neg
    if raw_score > 0:
        polarity = min(1.0, 0.25 * raw_score)
    elif raw_score < 0:
        polarity = max(-1.0, 0.25 * raw_score)
    else:
        polarity = 0.0
    magnitude = min(1.0, 0.15 + 0.15 * abs(raw_score)) if raw_score else 0.1
    confidence = 0.35 if raw_score else 0.2
    return {
        "event_type": "news_heuristic",
        "summary": title[:180],
        "polarity": polarity,
        "magnitude": magnitude,
        "confidence": confidence,
        "horizon_days": 3,
        "affected_nodes": [ticker],
        "node_deltas": [
            {
                "node": ticker,
                "field": "news_score",
                "delta": polarity * magnitude,
                "confidence": confidence,
                "half_life_days": 3,
            }
        ],
        "edge_deltas": [],
        "raw": {"method": "keyword_fallback", "positive_hits": pos, "negative_hits": neg, "link": link},
    }


def fetch_articles(
    query: str,
    limit: int,
    source: str,
    lookback_days: int,
    timeout_sec: float,
) -> list[dict[str, str]]:
    if source == "naver_search":
        end = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize() + pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=max(1, int(lookback_days)))
        return fetch_naver_search(
            query,
            start=start,
            end=end,
            limit=limit,
            timeout_sec=timeout_sec,
            sort="1",
        )
    if source != "google_rss":
        raise ValueError(f"unknown news source: {source}")
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query) + "&hl=ko&gl=KR&ceid=KR:ko"
    response = requests.get(url, timeout=max(float(timeout_sec), 0.1), headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    articles: list[dict[str, str]] = []
    for entry in feed.entries[:limit]:
        source = entry.get("source", {})
        if isinstance(source, dict):
            source_name = str(source.get("title", ""))
        else:
            source_name = ""
        articles.append(
            {
                "title": str(entry.get("title", "")),
                "summary": str(entry.get("summary", "")),
                "link": str(entry.get("link", "")),
                "published": str(entry.get("published", "")),
                "source": source_name,
            }
        )
    return articles


def get_target_universe(config_path: str, limit_tickers: int) -> list[tuple[str, str]]:
    config = OpsConfig.load(config_path)
    config.mode = "paper"
    config.use_intraday_quotes = False
    engine = OpsEngine(config)
    try:
        return [(signal.ticker, signal.name) for signal in engine.generate_signals()[:limit_tickers]]
    finally:
        engine.close()


def run_cycle(args: argparse.Namespace, state: dict[str, Any], extractor: QwenEventExtractor | None) -> dict[str, Any]:
    targets = get_target_universe(args.config, args.limit_tickers)
    known_ids = set(state.get("seen_article_ids", []))
    events: list[dict[str, Any]] = []
    per_ticker: dict[str, dict[str, Any]] = {}

    llm_attempts = 0
    llm_used_count = 0
    llm_error_count = 0
    heuristic_count = 0
    unlimited_llm = args.max_llm_articles_per_cycle < 0

    for ticker, name in targets:
        query = str(name) if args.news_source == "naver_search" else f"{name} OR {ticker}"
        for article in fetch_articles(
            query,
            args.articles_per_ticker,
            source=args.news_source,
            lookback_days=args.news_lookback_days,
            timeout_sec=args.news_request_timeout_sec,
        ):
            article_id = stable_id(ticker, article["title"], article["link"])
            if article_id in known_ids:
                continue
            known_ids.add(article_id)
            llm_used = False
            llm_error = ""
            event_payload: dict[str, Any] | None = None
            can_use_llm = extractor is not None and (unlimited_llm or llm_attempts < args.max_llm_articles_per_cycle)
            if can_use_llm:
                llm_attempts += 1
                try:
                    event = extractor.extract_one(article["title"], article["summary"], [ticker])
                    event_payload = event.raw or {
                        "event_type": event.event_type,
                        "summary": event.summary,
                        "polarity": event.polarity,
                        "magnitude": event.magnitude,
                        "confidence": event.confidence,
                        "horizon_days": event.horizon_days,
                        "affected_nodes": event.affected_nodes,
                        "node_deltas": [delta.__dict__ for delta in event.node_deltas],
                        "edge_deltas": [delta.__dict__ for delta in event.edge_deltas],
                    }
                    llm_used = True
                    llm_used_count += 1
                except Exception as exc:
                    llm_error = str(exc)[:300]
                    llm_error_count += 1
            if event_payload is None:
                event_payload = heuristic_event(ticker, name, article["title"], article["summary"], article["link"])
                heuristic_count += 1

            score = 0.0
            for delta in event_payload.get("node_deltas", []):
                if str(delta.get("node", "")).replace("A", "") == ticker and delta.get("field", "news_score") == "news_score":
                    score += float(delta.get("delta", 0.0)) * float(delta.get("confidence", event_payload.get("confidence", 0.0)))
            rec = {
                "id": article_id,
                "ts": now_iso(),
                "source": f"{args.news_source}_live",
                "ticker": ticker,
                "name": name,
                "article": article,
                "llm_used": llm_used,
                "llm_error": llm_error,
                "event": event_payload,
                "score_contribution": score,
            }
            events.append(rec)
            bucket = per_ticker.setdefault(ticker, {"name": name, "new_articles": 0, "news_score_delta": 0.0})
            bucket["new_articles"] += 1
            bucket["news_score_delta"] += score

    if events:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as file:
            for event in events:
                file.write(json.dumps(event, ensure_ascii=False) + "\n")

    current_scores = state.get("ticker_scores", {})
    for ticker, bucket in per_ticker.items():
        prev = float(current_scores.get(ticker, {}).get("news_score", 0.0))
        score = prev * args.decay + float(bucket["news_score_delta"])
        current_scores[ticker] = {
            "name": bucket["name"],
            "news_score": score,
            "last_delta": bucket["news_score_delta"],
            "last_new_articles": bucket["new_articles"],
            "updated_at": now_iso(),
        }

    state.update(
        {
            "updated_at": now_iso(),
            "mode": "shadow_not_trading",
            "qwen_endpoint": args.qwen_endpoint,
            "qwen_enabled": extractor is not None,
            "target_count": len(targets),
            "articles_per_ticker": args.articles_per_ticker,
            "news_source": args.news_source,
            "max_llm_articles_per_cycle": args.max_llm_articles_per_cycle,
            "seen_article_ids": sorted(known_ids)[-args.max_seen_ids :],
            "ticker_scores": current_scores,
            "last_cycle": {
                "new_events": len(events),
                "llm_attempts": llm_attempts,
                "llm_used": llm_used_count,
                "llm_errors": llm_error_count,
                "heuristic_events": heuristic_count,
                "articles_per_ticker": args.articles_per_ticker,
                "news_source": args.news_source,
                "max_llm_articles_per_cycle": args.max_llm_articles_per_cycle,
                "per_ticker": per_ticker,
                "targets": [{"ticker": ticker, "name": name} for ticker, name in targets],
            },
        }
    )
    save_json(Path(args.state_output), state)
    return state["last_cycle"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="stock-v2 news sensing loop")
    parser.add_argument("--config", default="configs/ops.world-model-shadow.json")
    parser.add_argument("--cycles", type=int, default=1, help="0 means run until interrupted")
    parser.add_argument("--interval-sec", type=float, default=900.0)
    parser.add_argument("--limit-tickers", type=int, default=10)
    parser.add_argument("--articles-per-ticker", type=int, default=10)
    parser.add_argument("--news-source", choices=["naver_search", "google_rss"], default="naver_search")
    parser.add_argument("--news-lookback-days", type=int, default=7)
    parser.add_argument("--news-request-timeout-sec", type=float, default=15.0)
    parser.add_argument("--output", default="ops/sensing/news_events.jsonl")
    parser.add_argument("--state-output", default="ops/sensing/news_state.json")
    parser.add_argument("--qwen-endpoint", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--qwen-model", default="auto")
    parser.add_argument("--use-qwen", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-llm-articles-per-cycle", type=int, default=24, help="-1 means no LLM cap")
    parser.add_argument("--decay", type=float, default=0.95)
    parser.add_argument("--max-seen-ids", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_output)
    state = load_json(state_path, {"seen_article_ids": [], "ticker_scores": {}})
    qwen_ok = qwen_available(args.qwen_endpoint)
    if args.use_qwen == "always" and not qwen_ok:
        raise RuntimeError(f"Qwen endpoint is not available: {args.qwen_endpoint}")
    extractor = None
    if args.use_qwen == "always" or (args.use_qwen == "auto" and qwen_ok):
        extractor = QwenEventExtractor(model=args.qwen_model, endpoint=args.qwen_endpoint)

    cycle = 0
    while args.cycles <= 0 or cycle < args.cycles:
        cycle += 1
        started = time.perf_counter()
        last_cycle = run_cycle(args, state, extractor)
        latency_sec = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "cycle": cycle,
                    "latency_sec": round(latency_sec, 3),
                    "qwen_enabled": extractor is not None,
                    "new_events": last_cycle["new_events"],
                    "llm_used": last_cycle.get("llm_used", 0),
                    "llm_attempts": last_cycle.get("llm_attempts", 0),
                    "heuristic_events": last_cycle.get("heuristic_events", 0),
                    "sleep_sec": round(max(0.0, args.interval_sec - latency_sec), 3),
                    "per_ticker": last_cycle["per_ticker"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.cycles > 0 and cycle >= args.cycles:
            break
        time.sleep(max(0.0, args.interval_sec - latency_sec))


if __name__ == "__main__":
    main()
