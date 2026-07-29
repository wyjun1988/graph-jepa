from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, time
import fcntl
import json
from pathlib import Path
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KST = ZoneInfo("Asia/Seoul")

from stock_v2.ops.brokers import KiwoomRestBroker, PaperBroker
from stock_v2.ops.config import OpsConfig
from stock_v2.ops.risk import RiskManager, build_rebalance_orders
from stock_v2.ops.store import OpsStore
from stock_v2.ops.types import OrderResult, Signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one zero-live-order paper cycle from a frozen shadow signal artifact."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--output")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def is_market_session(now: datetime) -> bool:
    local = now.astimezone(KST)
    return local.weekday() < 5 and time(9, 0) <= local.time() <= time(15, 30)


def validate_paper_contract(config: OpsConfig, payload: dict) -> None:
    if config.mode != "paper":
        raise ValueError("paper shadow cycle requires mode=paper")
    if payload.get("approval_scope") != "read_only_shadow":
        raise ValueError("signal artifact is not approved for read-only shadow")
    if payload.get("live_orders_allowed") is not False:
        raise ValueError("signal artifact does not prohibit live orders")
    if config.risk.max_orders_per_day > 3:
        raise ValueError("paper shadow daily order limit must not exceed 3")
    if config.paper_initial_cash > 100_000:
        raise ValueError("paper shadow initial cash must not exceed 100,000 KRW")


def signal_with_quote(row: dict, quote) -> Signal | None:
    if quote is None or quote.usable_price is None or quote.usable_price <= 0.0:
        return None
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "daily_price": int(row["price"]),
            "intraday_price": int(round(quote.usable_price)),
            "intraday_bid": None if quote.bid_price is None else int(round(quote.bid_price)),
            "intraday_ask": None if quote.ask_price is None else int(round(quote.ask_price)),
            "intraday_order_price": int(
                round(quote.ask_price or quote.usable_price)
            ),
            "intraday_quote_time": quote.exchange_time,
            "intraday_received_at": quote.received_at,
            "price_source": quote.source,
            "model_state_updated_from_quote": False,
        }
    )
    return Signal(
        ticker=str(row["ticker"]),
        name=str(row["name"]),
        score=float(row["score"]),
        rank=int(row["rank"]),
        price=int(round(quote.usable_price)),
        model=str(row["model"]),
        asof=str(row["asof"]),
        metadata=metadata,
    )


def main() -> None:
    args = parse_args()
    config = OpsConfig.load(args.config)
    source = json.loads(Path(args.signals).read_text(encoding="utf-8"))
    validate_paper_contract(config, source)
    if not args.force and not is_market_session(datetime.now(KST)):
        print(json.dumps({"status": "skipped", "reason": "outside_krx_market_hours"}))
        return
    lock_path = Path(config.state_db).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("paper shadow cycle is already running") from exc

        store = OpsStore(config.state_db)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        store.start_run(run_id, "paper_shadow")
        try:
            paper = PaperBroker(
                store,
                config.paper_initial_cash,
                commission_bps=config.paper_commission_bps,
                sell_tax_bps=config.paper_sell_tax_bps,
            )
            quote_broker = KiwoomRestBroker(config.kiwoom, dry_run=True)
            if not quote_broker.authenticate():
                raise RuntimeError("Kiwoom read-only authentication failed")
            before = paper.get_account()
            signal_rows = list(source.get("signals") or [])
            tickers = list(
                dict.fromkeys(
                    [str(row["ticker"]) for row in signal_rows]
                    + [position.ticker for position in before.positions]
                )
            )
            quotes = quote_broker.get_quotes(
                tickers,
                sleep_sec=config.intraday_quote_sleep_sec,
            )
            signals = [
                signal
                for row in signal_rows
                if (signal := signal_with_quote(row, quotes.get(str(row["ticker"]))))
                is not None
            ]
            store.record_signals(run_id, signals)
            price_map = {
                ticker: float(quote.usable_price)
                for ticker, quote in quotes.items()
                if quote.usable_price is not None and quote.usable_price > 0.0
            }
            account = paper.get_account(price_map)
            risk = RiskManager(config.risk, store)
            intents = build_rebalance_orders(
                signals=signals[: max(config.top_k, config.risk.max_positions)],
                account=account,
                risk_config=config.risk,
                target_weight=config.target_weight,
            )
            results: list[OrderResult] = []
            dry_runs: list[dict] = []
            for intent in intents:
                latest_account = paper.get_account(price_map)
                decision = risk.validate(intent, latest_account)
                if not decision.allowed:
                    result = OrderResult(intent, "REJECTED", decision.reason)
                else:
                    preview = quote_broker.place_order(intent)
                    if preview.status != "DRY_RUN":
                        raise RuntimeError("Kiwoom order preview escaped dry-run mode")
                    dry_runs.append(preview.raw)
                    filled = paper.place_order(intent)
                    result = OrderResult(
                        intent=filled.intent,
                        status=filled.status,
                        message=filled.message,
                        raw={**filled.raw, "kiwoom_dry_run": preview.raw},
                    )
                store.record_order(run_id, "paper_shadow", result)
                results.append(result)

            final_account = paper.get_account(price_map)
            store.finish_run(
                run_id,
                "OK",
                f"signals={len(signals)} quotes={len(quotes)} orders={len(results)}",
            )
            payload = {
                "status": "complete",
                "mode": "paper_shadow",
                "run_id": run_id,
                "model_asof": signals[0].asof if signals else None,
                "model_state_updated_from_quote": False,
                "live_orders_allowed": False,
                "kiwoom_order_mode": "DRY_RUN",
                "paper_initial_cash": config.paper_initial_cash,
                "paper_commission_bps": config.paper_commission_bps,
                "paper_sell_tax_bps": config.paper_sell_tax_bps,
                "daily_order_limit": config.risk.max_orders_per_day,
                "orders_today": store.orders_today(),
                "quotes": len(quotes),
                "orders": [
                    {**asdict(result), "notional": result.intent.notional}
                    for result in results
                ],
                "kiwoom_dry_run_payloads": dry_runs,
                "account": {
                    "cash": final_account.cash,
                    "equity": final_account.equity,
                    "exposure": final_account.exposure,
                    "total_return": (
                        final_account.equity / config.paper_initial_cash - 1.0
                    ),
                    "positions": [asdict(position) for position in final_account.positions],
                },
                "top_signals": [asdict(signal) for signal in signals[: config.top_k]],
            }
            output = Path(args.output) if args.output else Path(config.reports_dir) / f"{run_id}.json"
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
                        "run_id": run_id,
                        "orders": len(results),
                        "orders_today": store.orders_today(),
                        "cash": final_account.cash,
                        "equity": final_account.equity,
                        "positions": len(final_account.positions),
                        "output": str(output),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            store.finish_run(run_id, "ERROR", str(exc))
            raise
        finally:
            store.close()


if __name__ == "__main__":
    main()
