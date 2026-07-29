from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from stock_v2.ops.config import RiskConfig
from stock_v2.ops.store import OpsStore
from stock_v2.ops.types import Account, OrderIntent, Position, Signal


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    def __init__(self, config: RiskConfig, store: OpsStore):
        self.config = config
        self.store = store

    def validate(self, intent: OrderIntent, account: Account) -> RiskDecision:
        if intent.quantity <= 0:
            return RiskDecision(False, "quantity must be positive")
        if intent.price < self.config.min_price or intent.price > self.config.max_price:
            return RiskDecision(False, "price outside allowed range")
        if self.config.allow_tickers and intent.ticker not in set(self.config.allow_tickers):
            return RiskDecision(False, "ticker is not in allow list")
        if intent.ticker in set(self.config.block_tickers):
            return RiskDecision(False, "ticker is blocked")
        if self.store.orders_today() >= self.config.max_orders_per_day:
            return RiskDecision(False, "daily order limit reached")

        if intent.side == "SELL":
            return RiskDecision(True, "sell risk accepted")

        if intent.signal_score is not None and intent.signal_score < self.config.min_score:
            return RiskDecision(False, "signal score below minimum")
        if intent.notional > self.config.max_cash_per_order:
            return RiskDecision(False, "order notional exceeds per-order cap")
        if account.equity <= 0:
            return RiskDecision(False, "account equity is zero")
        if intent.notional > account.equity * self.config.max_position_pct_equity:
            return RiskDecision(False, "order notional exceeds position pct cap")

        current_positions = {position.ticker: position for position in account.positions}
        if intent.ticker not in current_positions and len(current_positions) >= self.config.max_positions:
            return RiskDecision(False, "max positions reached")
        projected_exposure = account.exposure + intent.notional
        if projected_exposure > account.equity * self.config.max_total_exposure_pct:
            return RiskDecision(False, "projected exposure exceeds cap")
        if account.cash < intent.notional:
            return RiskDecision(False, "cash insufficient")
        return RiskDecision(True, "buy risk accepted")


def krx_tick_size(price: float) -> int:
    if price < 2_000:
        return 1
    if price < 5_000:
        return 5
    if price < 20_000:
        return 10
    if price < 50_000:
        return 50
    if price < 200_000:
        return 100
    if price < 500_000:
        return 500
    return 1_000


def round_up_to_krx_tick(price: float) -> int:
    tick = krx_tick_size(price)
    rounded = int(price)
    if rounded % tick:
        rounded = ((rounded // tick) + 1) * tick
    return max(tick, rounded)


def build_rebalance_orders(
    signals: list[Signal],
    account: Account,
    risk_config: RiskConfig,
    target_weight: float,
) -> list[OrderIntent]:
    top = signals[: risk_config.max_positions]
    target_tickers = {signal.ticker for signal in top}
    signal_map = {signal.ticker: signal for signal in signals}
    orders: list[OrderIntent] = []

    for position in account.positions:
        signal = signal_map.get(position.ticker)
        sell_reason = ""
        if position.return_pct >= risk_config.take_profit_pct:
            sell_reason = f"take_profit {position.return_pct:.2%}"
        elif position.return_pct <= risk_config.stop_loss_pct:
            sell_reason = f"stop_loss {position.return_pct:.2%}"
        elif position.ticker not in target_tickers:
            sell_reason = "no longer in target universe"

        if sell_reason:
            price = int(round(position.current_price))
            orders.append(
                OrderIntent(
                    ticker=position.ticker,
                    name=position.name,
                    side="SELL",
                    quantity=position.quantity,
                    price=max(1, price),
                    reason=sell_reason,
                    signal_score=None if signal is None else signal.score,
                )
            )

    existing = {position.ticker for position in account.positions}
    buys = 0
    budget = min(risk_config.max_cash_per_order, account.equity * target_weight)
    buffer = 1.0 + risk_config.limit_buffer_bps / 10_000.0
    for signal in top:
        if buys >= risk_config.max_new_buys_per_run:
            break
        if signal.ticker in existing:
            continue
        if signal.score < risk_config.min_score:
            continue
        base_price = float(signal.metadata.get("intraday_order_price") or signal.price)
        price = round_up_to_krx_tick(base_price * buffer)
        quantity = int(budget // price)
        if quantity <= 0:
            continue
        orders.append(
            OrderIntent(
                ticker=signal.ticker,
                name=signal.name,
                side="BUY",
                quantity=quantity,
                price=price,
                reason=f"top_rank_{signal.rank}",
                signal_score=signal.score,
            )
        )
        buys += 1
    return orders
