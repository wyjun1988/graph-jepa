from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from stock_v2.ops.brokers import KiwoomRestBroker, PaperBroker
from stock_v2.ops.config import OpsConfig
from stock_v2.ops.risk import RiskManager, build_rebalance_orders
from stock_v2.ops.signals import SignalEngine
from stock_v2.ops.store import OpsStore
from stock_v2.ops.types import Account, OrderResult, Quote, Signal


class OpsEngine:
    def __init__(self, config: OpsConfig):
        self.config = config
        if config.mode == "live" and config.latent_path_head_path:
            raise ValueError("latent path head is approved for read-only shadow only")
        if (
            config.mode == "live"
            and config.use_intraday_quotes
            and not config.intraday_quote_include_orderbook
        ):
            raise ValueError("live mode requires intraday order-book quotes")
        self.store = OpsStore(config.state_db)
        self.signal_engine = SignalEngine(
            config.model_dir,
            config.signal_model,
            config.device,
            live_event_paths=config.live_event_paths,
            latent_path_head_path=config.latent_path_head_path,
        )
        self.risk = RiskManager(config.risk, self.store)

        if config.mode == "paper":
            self.broker = PaperBroker(
                self.store,
                config.paper_initial_cash,
                commission_bps=config.paper_commission_bps,
                sell_tax_bps=config.paper_sell_tax_bps,
            )
        elif config.mode == "live":
            self.broker = KiwoomRestBroker(config.kiwoom, dry_run=False)
        elif config.mode == "dry_live":
            self.broker = KiwoomRestBroker(config.kiwoom, dry_run=True)
        else:
            raise ValueError(f"unknown mode: {config.mode}")
        if config.use_intraday_quotes:
            self.quote_broker = self.broker if isinstance(self.broker, KiwoomRestBroker) else KiwoomRestBroker(config.kiwoom, dry_run=True)
        else:
            self.quote_broker = None

    def close(self) -> None:
        self.store.close()

    def generate_signals(
        self,
        intraday_quotes: dict[str, Quote] | None = None,
        intraday_session_date: str | None = None,
    ) -> list[Signal]:
        return self.signal_engine.latest_signals(
            start=self.config.data_start,
            end=self.config.data_end,
            cache_dir=self.config.cache_dir,
            top_n=max(
                self.config.top_k,
                self.config.risk.max_positions,
                self.config.intraday_quote_limit,
                20,
            ),
            intraday_quotes=intraday_quotes,
            intraday_session_date=intraday_session_date,
            max_intraday_missing_business_days=(
                self.config.max_intraday_missing_business_days
                if intraday_quotes
                else None
            ),
            min_coverage_ratio=self.config.min_latest_coverage_ratio,
            coverage_lookback=self.config.latest_coverage_lookback,
            min_price=self.config.risk.min_price,
            max_price=self.config.risk.max_price,
        )

    def get_intraday_quotes(self, tickers: list[str]) -> dict[str, Quote]:
        if self.quote_broker is None:
            return {}
        return self.quote_broker.get_quotes(
            tickers,
            sleep_sec=self.config.intraday_quote_sleep_sec,
            include_orderbook=self.config.intraday_quote_include_orderbook,
        )

    def current_signals(self) -> list[Signal]:
        daily_signals = self.generate_signals()
        signals, _ = self._refresh_intraday_state(daily_signals)
        return signals[: self.config.top_k]

    def _intraday_targets(
        self,
        signals: list[Signal],
        extra_tickers: list[str] | None = None,
    ) -> list[str]:
        targets: list[str] = []
        seen: set[str] = set()
        if self.config.intraday_quote_scope == "model_universe":
            candidates = list(self.signal_engine.tickers)
        else:
            candidates = [signal.ticker for signal in signals]
        quote_limit = max(0, self.config.intraday_quote_limit)
        if quote_limit:
            candidates = candidates[:quote_limit]
        for ticker in candidates:
            if ticker not in seen:
                targets.append(ticker)
                seen.add(ticker)
        for ticker in extra_tickers or []:
            normalized = ticker.replace("A", "").strip()
            if normalized and normalized not in seen:
                targets.append(normalized)
                seen.add(normalized)
        return targets

    def _collect_intraday_quotes(
        self,
        signals: list[Signal],
        extra_tickers: list[str] | None = None,
    ) -> dict[str, Quote]:
        if not self.config.use_intraday_quotes:
            return {}
        targets = self._intraday_targets(signals, extra_tickers)
        return self._collect_quote_targets(targets)

    def _collect_quote_targets(self, targets: list[str]) -> dict[str, Quote]:
        if not self.config.use_intraday_quotes or not targets:
            return {}
        normalized_targets: list[str] = []
        seen: set[str] = set()
        for ticker in targets:
            normalized = ticker.replace("A", "").strip()
            if normalized and normalized not in seen:
                normalized_targets.append(normalized)
                seen.add(normalized)
        quotes: dict[str, Quote] = {}
        remaining = list(normalized_targets)
        for _ in range(self.config.intraday_quote_retry_rounds + 1):
            try:
                quotes.update(self.get_intraday_quotes(remaining))
            except Exception:
                if self.config.require_intraday_quotes:
                    raise
            remaining = [
                ticker
                for ticker in normalized_targets
                if ticker not in quotes
                or quotes[ticker].usable_price is None
                or quotes[ticker].usable_price <= 0.0
            ]
            if not remaining:
                break
        if self.config.require_intraday_quotes:
            if remaining:
                raise RuntimeError(f"intraday quotes missing: {remaining[:10]}")
        return quotes

    def _missing_top_k_quote_tickers(
        self,
        signals: list[Signal],
        quotes: dict[str, Quote],
    ) -> list[str]:
        missing = []
        for signal in signals[: self.config.top_k]:
            quote = quotes.get(signal.ticker)
            price = None if quote is None else quote.usable_price
            if price is None or price <= 0.0:
                missing.append(signal.ticker)
        return missing

    @staticmethod
    def _intraday_session_date(quotes: dict[str, Quote]) -> str:
        dates = []
        for quote in quotes.values():
            if not quote.received_at:
                continue
            try:
                dates.append(datetime.fromisoformat(quote.received_at).date())
            except ValueError:
                continue
        return str(max(dates) if dates else datetime.now().date())

    def _refresh_intraday_state(
        self,
        daily_signals: list[Signal],
        extra_tickers: list[str] | None = None,
    ) -> tuple[list[Signal], dict[str, Quote]]:
        quotes = self._collect_intraday_quotes(daily_signals, extra_tickers)
        sensed_signals = (
            self.generate_signals(
                intraday_quotes=quotes,
                intraday_session_date=self._intraday_session_date(quotes),
            )
            if quotes
            else daily_signals
        )
        topup_rounds_used = 0
        for _ in range(self.config.intraday_quote_topup_rounds):
            missing = self._missing_top_k_quote_tickers(sensed_signals, quotes)
            if not missing:
                break
            previous_usable = {
                ticker
                for ticker, quote in quotes.items()
                if quote.usable_price is not None and quote.usable_price > 0.0
            }
            quotes.update(self._collect_quote_targets(missing))
            current_usable = {
                ticker
                for ticker, quote in quotes.items()
                if quote.usable_price is not None and quote.usable_price > 0.0
            }
            if current_usable == previous_usable:
                break
            topup_rounds_used += 1
            sensed_signals = self.generate_signals(
                intraday_quotes=quotes,
                intraday_session_date=self._intraday_session_date(quotes),
            )
        model_quote_count = (
            int(sensed_signals[0].metadata.get("model_input_quote_count", 0))
            if sensed_signals
            else 0
        )
        if model_quote_count < self.config.min_intraday_model_quote_count:
            raise RuntimeError(
                "intraday model quote coverage below configured minimum: "
                f"actual={model_quote_count} "
                f"required={self.config.min_intraday_model_quote_count}"
            )
        final_top_k_size = min(self.config.top_k, len(sensed_signals))
        final_top_k_quote_count = (
            final_top_k_size
            - len(self._missing_top_k_quote_tickers(sensed_signals, quotes))
        )
        if final_top_k_quote_count < self.config.min_top_k_intraday_quote_count:
            raise RuntimeError(
                "final top-k intraday quote coverage below configured minimum: "
                f"actual={final_top_k_quote_count} "
                f"required={self.config.min_top_k_intraday_quote_count} "
                f"top_k_size={final_top_k_size} "
                f"topup_rounds_used={topup_rounds_used}"
            )
        sensed_signals = [
            replace(
                signal,
                metadata={
                    **signal.metadata,
                    "intraday_quote_topup_rounds_used": topup_rounds_used,
                    "final_top_k_quote_count": final_top_k_quote_count,
                    "final_top_k_size": final_top_k_size,
                    "total_intraday_quote_count": len(quotes),
                },
            )
            for signal in sensed_signals
        ]
        return self._refresh_intraday_prices(
            sensed_signals,
            extra_tickers=extra_tickers,
            quotes=quotes,
        )

    def _refresh_intraday_prices(
        self,
        signals: list[Signal],
        extra_tickers: list[str] | None = None,
        quotes: dict[str, Quote] | None = None,
    ) -> tuple[list[Signal], dict[str, Quote]]:
        if not self.config.use_intraday_quotes:
            return signals, {}
        if quotes is None:
            quotes = self._collect_intraday_quotes(signals, extra_tickers)

        updated: list[Signal] = []
        for signal in signals:
            quote = quotes.get(signal.ticker)
            price = None if quote is None else quote.usable_price
            if price is None or price <= 0:
                updated.append(signal)
                continue
            order_price = quote.buy_reference_price or price
            metadata = dict(signal.metadata)
            metadata.update(
                {
                    "daily_price": signal.metadata.get(
                        "daily_close_before_quote_overlay",
                        signal.price,
                    ),
                    "intraday_price": int(round(price)),
                    "intraday_bid": None if quote.bid_price is None else int(round(quote.bid_price)),
                    "intraday_ask": None if quote.ask_price is None else int(round(quote.ask_price)),
                    "intraday_order_price": int(round(order_price)),
                    "intraday_quote_time": quote.exchange_time,
                    "intraday_received_at": quote.received_at,
                    "price_source": quote.source,
                    "model_state_updated_from_quote": (
                        signal.metadata.get("model_input_state")
                        == "partial_intraday_quote_overlay"
                    ),
                }
            )
            updated.append(replace(signal, price=int(round(price)), metadata=metadata))
        return updated, quotes

    @staticmethod
    def _price_map(signals: list[Signal], quotes: dict[str, Quote]) -> dict[str, float]:
        prices = {signal.ticker: float(signal.price) for signal in signals}
        for ticker, quote in quotes.items():
            price = quote.usable_price
            if price is not None and price > 0:
                prices[ticker] = float(price)
        return prices

    def run_once(self) -> dict:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
        self.store.start_run(run_id, self.config.mode)
        results: list[OrderResult] = []
        try:
            daily_signals = self.generate_signals()
            daily_price_map = {signal.ticker: signal.price for signal in daily_signals}
            preliminary_account = self.broker.get_account(daily_price_map)
            signals, quotes = self._refresh_intraday_state(
                daily_signals,
                extra_tickers=[position.ticker for position in preliminary_account.positions],
            )
            self.store.record_signals(run_id, signals)
            price_map = self._price_map(signals, quotes)
            account = self.broker.get_account(price_map)
            orders = build_rebalance_orders(
                signals=signals[: self.config.top_k],
                account=account,
                risk_config=self.config.risk,
                target_weight=self.config.target_weight,
            )

            for intent in orders:
                latest_account = self.broker.get_account(price_map)
                decision = self.risk.validate(intent, latest_account)
                if not decision.allowed:
                    result = OrderResult(intent, "REJECTED", decision.reason)
                else:
                    result = self.broker.place_order(intent)
                self.store.record_order(run_id, self.config.mode, result)
                results.append(result)

            self._write_report(run_id, signals, account, results, quotes)
            self.store.finish_run(run_id, "OK", f"signals={len(signals)} quotes={len(quotes)} orders={len(results)}")
            return {
                "run_id": run_id,
                "signals": len(signals),
                "quotes": len(quotes),
                "orders": len(results),
                "accepted": sum(result.status in {"FILLED", "ACCEPTED", "DRY_RUN"} for result in results),
            }
        except Exception as exc:
            self.store.finish_run(run_id, "ERROR", str(exc))
            raise

    def status(self) -> dict:
        daily_signals = self.generate_signals()
        daily_price_map = {signal.ticker: signal.price for signal in daily_signals}
        preliminary_account = self.broker.get_account(daily_price_map)
        signals, quotes = self._refresh_intraday_state(
            daily_signals,
            extra_tickers=[position.ticker for position in preliminary_account.positions],
        )
        account = self.broker.get_account(self._price_map(signals, quotes))
        return {
            "mode": self.config.mode,
            "cash": account.cash,
            "equity": account.equity,
            "exposure": account.exposure,
            "intraday_quotes": len(quotes),
            "model_quote_overlays": sum(
                bool(signal.metadata.get("model_state_updated_from_quote"))
                for signal in signals
            ),
            "positions": [
                {
                    "ticker": position.ticker,
                    "name": position.name,
                    "quantity": position.quantity,
                    "avg_price": position.avg_price,
                    "current_price": position.current_price,
                    "return_pct": position.return_pct,
                }
                for position in account.positions
            ],
            "top_signals": [
                {
                    "rank": signal.rank,
                    "ticker": signal.ticker,
                    "name": signal.name,
                    "score": signal.score,
                    "price": signal.price,
                    "daily_price": signal.metadata.get("daily_price"),
                    "bid": signal.metadata.get("intraday_bid"),
                    "ask": signal.metadata.get("intraday_ask"),
                    "price_source": signal.metadata.get("price_source", "daily"),
                    "quote_time": signal.metadata.get("intraday_quote_time"),
                    "model_input_state": signal.metadata.get("model_input_state", "daily_close"),
                    "model_input_quote_count": signal.metadata.get("model_input_quote_count", 0),
                    "asof": signal.asof,
                }
                for signal in signals[: self.config.top_k]
            ],
        }

    def _write_report(
        self,
        run_id: str,
        signals: list[Signal],
        account: Account,
        results: list[OrderResult],
        quotes: dict[str, Quote],
    ) -> None:
        root = Path(self.config.reports_dir) / run_id
        root.mkdir(parents=True, exist_ok=True)
        with (root / "signals.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "rank",
                    "ticker",
                    "name",
                    "score",
                    "price",
                    "daily_price",
                    "bid",
                    "ask",
                    "price_source",
                    "quote_time",
                    "model_input_state",
                    "model_input_quote_count",
                    "model",
                    "asof",
                ],
            )
            writer.writeheader()
            for signal in signals:
                writer.writerow(
                    {
                        "rank": signal.rank,
                        "ticker": signal.ticker,
                        "name": signal.name,
                        "score": signal.score,
                        "price": signal.price,
                        "daily_price": signal.metadata.get("daily_price", signal.price),
                        "bid": signal.metadata.get("intraday_bid", ""),
                        "ask": signal.metadata.get("intraday_ask", ""),
                        "price_source": signal.metadata.get("price_source", "daily"),
                        "quote_time": signal.metadata.get("intraday_quote_time", ""),
                        "model_input_state": signal.metadata.get("model_input_state", "daily_close"),
                        "model_input_quote_count": signal.metadata.get("model_input_quote_count", 0),
                        "model": signal.model,
                        "asof": signal.asof,
                    }
                )
        with (root / "orders.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["ticker", "name", "side", "quantity", "price", "notional", "status", "reason", "message"],
            )
            writer.writeheader()
            for result in results:
                intent = result.intent
                writer.writerow(
                    {
                        "ticker": intent.ticker,
                        "name": intent.name,
                        "side": intent.side,
                        "quantity": intent.quantity,
                        "price": intent.price,
                        "notional": intent.notional,
                        "status": result.status,
                        "reason": intent.reason,
                        "message": result.message,
                    }
                )
        (root / "summary.txt").write_text(
            "\n".join(
                [
                    f"run_id={run_id}",
                    f"mode={self.config.mode}",
                    f"cash={account.cash:.0f}",
                    f"equity={account.equity:.0f}",
                    f"exposure={account.exposure:.0f}",
                    f"signals={len(signals)}",
                    f"intraday_quotes={len(quotes)}",
                    f"orders={len(results)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
