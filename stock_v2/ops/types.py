from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Literal, Optional


Side = Literal["BUY", "SELL"]
OrderStatus = Literal["PLANNED", "ACCEPTED", "REJECTED", "FILLED", "DRY_RUN", "ERROR"]


@dataclass(frozen=True)
class Signal:
    ticker: str
    name: str
    score: float
    rank: int
    price: int
    model: str
    asof: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Quote:
    ticker: str
    last_price: Optional[float]
    bid_price: Optional[float]
    ask_price: Optional[float]
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    volume: Optional[int] = None
    exchange_time: Optional[str] = None
    received_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    source: str = "unknown"
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def usable_price(self) -> Optional[float]:
        if self.last_price and self.last_price > 0:
            return self.last_price
        if self.bid_price and self.ask_price and self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2.0
        return self.ask_price or self.bid_price

    @property
    def buy_reference_price(self) -> Optional[float]:
        return self.ask_price or self.usable_price


@dataclass(frozen=True)
class Position:
    ticker: str
    name: str
    quantity: int
    avg_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def return_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return self.current_price / self.avg_price - 1.0


@dataclass(frozen=True)
class Account:
    cash: float
    positions: list[Position]

    @property
    def equity(self) -> float:
        return self.cash + sum(position.market_value for position in self.positions)

    @property
    def exposure(self) -> float:
        return sum(position.market_value for position in self.positions)


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    name: str
    side: Side
    quantity: int
    price: int
    reason: str
    signal_score: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def notional(self) -> int:
        return int(self.quantity * self.price)


@dataclass(frozen=True)
class OrderResult:
    intent: OrderIntent
    status: OrderStatus
    message: str
    broker_order_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)
