from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import OpsConfig
from stock_v2.ops.engine import OpsEngine
from stock_v2.ops.store import OpsStore


def cmd_init_config(args: argparse.Namespace) -> None:
    config = OpsConfig()
    config.save(args.config)
    print(f"wrote {args.config}")


def cmd_init_state(args: argparse.Namespace) -> None:
    config = OpsConfig.load(args.config)
    store = OpsStore(config.state_db)
    store.init_cash(args.cash or config.paper_initial_cash, reset=args.reset)
    store.close()
    print(f"initialized {config.state_db}")


def cmd_signals(args: argparse.Namespace) -> None:
    config = OpsConfig.load(args.config)
    config.mode = "paper"
    engine = OpsEngine(config)
    try:
        signals = engine.current_signals()[: args.limit]
        print(json.dumps([signal.__dict__ for signal in signals], indent=2, ensure_ascii=False))
    finally:
        engine.close()


def cmd_quotes(args: argparse.Namespace) -> None:
    config = OpsConfig.load(args.config)
    broker = KiwoomRestBroker(config.kiwoom, dry_run=True)
    tickers = args.tickers
    if not tickers:
        config.mode = "paper"
        engine = OpsEngine(config)
        try:
            tickers = [signal.ticker for signal in engine.generate_signals()[: args.limit]]
        finally:
            engine.close()
    quotes = broker.get_quotes(tickers, sleep_sec=config.intraday_quote_sleep_sec)
    payload = {}
    for ticker, quote in quotes.items():
        data = asdict(quote)
        if not args.raw:
            data.pop("raw", None)
        payload[ticker] = data
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def cmd_status(args: argparse.Namespace) -> None:
    config = OpsConfig.load(args.config)
    engine = OpsEngine(config)
    try:
        print(json.dumps(engine.status(), indent=2, ensure_ascii=False))
    finally:
        engine.close()


def cmd_sense_loop(args: argparse.Namespace) -> None:
    config = OpsConfig.load(args.config)
    if args.mode:
        config.mode = args.mode
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    engine = OpsEngine(config)
    cycle = 0
    try:
        while args.cycles <= 0 or cycle < args.cycles:
            cycle += 1
            started = time.perf_counter()
            status = engine.status()
            latency_sec = time.perf_counter() - started
            interval_sec = max(args.min_interval_sec, min(args.max_interval_sec, latency_sec * args.latency_multiplier))
            payload = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "cycle": cycle,
                "mode": config.mode,
                "latency_sec": round(latency_sec, 3),
                "next_interval_sec": round(interval_sec, 3),
                "status": status,
            }
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            top = status["top_signals"][0] if status.get("top_signals") else {}
            print(
                json.dumps(
                    {
                        "cycle": cycle,
                        "latency_sec": round(latency_sec, 3),
                        "next_interval_sec": round(interval_sec, 3),
                        "intraday_quotes": status.get("intraday_quotes", 0),
                        "cash": status.get("cash"),
                        "equity": status.get("equity"),
                        "top": top,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.cycles > 0 and cycle >= args.cycles:
                break
            time.sleep(interval_sec)
    finally:
        engine.close()


def cmd_run_once(args: argparse.Namespace) -> None:
    config = OpsConfig.load(args.config)
    if args.mode:
        config.mode = args.mode
    engine = OpsEngine(config)
    try:
        result = engine.run_once()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        engine.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="stock-v2 operational runner")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("init-config")
    p.add_argument("--config", default="configs/ops.paper.json")
    p.set_defaults(func=cmd_init_config)

    p = sub.add_parser("init-state")
    p.add_argument("--config", default="configs/ops.paper.json")
    p.add_argument("--cash", type=int, default=None)
    p.add_argument("--reset", action="store_true")
    p.set_defaults(func=cmd_init_state)

    p = sub.add_parser("signals")
    p.add_argument("--config", default="configs/ops.paper.json")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_signals)

    p = sub.add_parser("quotes")
    p.add_argument("--config", default="configs/ops.paper.json")
    p.add_argument("--tickers", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--raw", action="store_true")
    p.set_defaults(func=cmd_quotes)

    p = sub.add_parser("status")
    p.add_argument("--config", default="configs/ops.paper.json")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("sense-loop")
    p.add_argument("--config", default="configs/ops.dry-live.json")
    p.add_argument("--mode", choices=["paper", "dry_live", "live"], default=None)
    p.add_argument("--cycles", type=int, default=0, help="0 means run until interrupted")
    p.add_argument("--min-interval-sec", type=float, default=30.0)
    p.add_argument("--max-interval-sec", type=float, default=180.0)
    p.add_argument("--latency-multiplier", type=float, default=3.0)
    p.add_argument("--output", default="ops/sensing/intraday_status.jsonl")
    p.set_defaults(func=cmd_sense_loop)

    p = sub.add_parser("run-once")
    p.add_argument("--config", default="configs/ops.paper.json")
    p.add_argument("--mode", choices=["paper", "dry_live", "live"], default=None)
    p.set_defaults(func=cmd_run_once)
    return parser.parse_args()


if __name__ == "__main__":
    ns = parse_args()
    ns.func(ns)
