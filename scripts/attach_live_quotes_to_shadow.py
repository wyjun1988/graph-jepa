from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import KiwoomConfig
from stock_v2.ops.types import Quote


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach read-only Kiwoom execution quotes to frozen shadow signals."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--sleep-sec", type=float, default=0.1)
    return parser.parse_args()


def attach_quotes(signals: list[dict], quotes: dict[str, Quote]) -> list[dict]:
    result: list[dict] = []
    for signal in signals:
        row = dict(signal)
        quote = quotes.get(str(signal.get("ticker") or ""))
        close = float(signal.get("price") or 0.0)
        if quote is None:
            row["live_quote"] = None
        else:
            current = quote.usable_price
            row["live_quote"] = {
                "last": quote.last_price,
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "exchange_time": quote.exchange_time,
                "received_at": quote.received_at,
                "source": quote.source,
                "return_since_model_state_close": (
                    None
                    if current is None or current <= 0.0 or close <= 0.0
                    else float(current / close - 1.0)
                ),
            }
        result.append(row)
    return result


def main() -> None:
    args = parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if source.get("approval_scope") != "read_only_shadow":
        raise ValueError("input is not an approved read-only shadow artifact")
    if source.get("live_orders_allowed") is not False or source.get("orders_submitted") != 0:
        raise ValueError("input does not enforce the zero-order contract")
    signals = list(source.get("signals") or [])
    broker = KiwoomRestBroker(
        KiwoomConfig(env_file=args.env_file, server="real"),
        dry_run=True,
    )
    if not broker.authenticate():
        raise RuntimeError("Kiwoom read-only authentication failed")
    tickers = [str(signal["ticker"]) for signal in signals]
    quotes = broker.get_quotes(tickers, sleep_sec=args.sleep_sec)
    payload = {
        **source,
        "status": "complete",
        "quote_snapshot_at": datetime.now(timezone.utc).isoformat(),
        "quote_overlay_mode": "execution_price_only",
        "model_state_updated": False,
        "orders_submitted": 0,
        "quoted_tickers": len(quotes),
        "signals": attach_quotes(signals, quotes),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": "complete",
                "quoted_tickers": len(quotes),
                "model_state_updated": False,
                "orders_submitted": 0,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
