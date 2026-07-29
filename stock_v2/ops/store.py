from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional

from stock_v2.ops.types import OrderIntent, OrderResult, Position, Signal


class OpsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists meta (
                key text primary key,
                value text not null
            );
            create table if not exists runs (
                run_id text primary key,
                started_at text not null,
                mode text not null,
                status text not null,
                message text
            );
            create table if not exists signals (
                id integer primary key autoincrement,
                run_id text not null,
                ts text not null,
                ticker text not null,
                name text not null,
                score real not null,
                rank integer not null,
                price integer not null,
                model text not null,
                asof text not null,
                metadata_json text not null
            );
            create table if not exists orders (
                id integer primary key autoincrement,
                run_id text,
                ts text not null,
                mode text not null,
                ticker text not null,
                name text not null,
                side text not null,
                quantity integer not null,
                price integer not null,
                notional integer not null,
                status text not null,
                reason text not null,
                message text not null,
                raw_json text not null
            );
            create table if not exists positions (
                ticker text primary key,
                name text not null,
                quantity integer not null,
                avg_price real not null,
                updated_at text not null
            );
            """
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("select value from meta where key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "insert into meta(key, value) values(?, ?) on conflict(key) do update set value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def init_cash(self, cash: float, reset: bool = False) -> None:
        if reset:
            self.conn.execute("delete from positions")
            self.conn.execute("delete from orders")
            self.conn.execute("delete from signals")
        if reset or self.get_meta("paper_cash") is None:
            self.set_meta("paper_cash", str(float(cash)))
        self.conn.commit()

    def get_cash(self) -> float:
        value = self.get_meta("paper_cash")
        return 0.0 if value is None else float(value)

    def set_cash(self, cash: float) -> None:
        self.set_meta("paper_cash", str(float(cash)))

    def get_positions(self, prices: dict[str, float] | None = None) -> list[Position]:
        prices = prices or {}
        rows = self.conn.execute("select * from positions order by ticker").fetchall()
        positions = []
        for row in rows:
            avg_price = float(row["avg_price"])
            current_price = float(prices.get(row["ticker"], avg_price))
            positions.append(
                Position(
                    ticker=str(row["ticker"]),
                    name=str(row["name"]),
                    quantity=int(row["quantity"]),
                    avg_price=avg_price,
                    current_price=current_price,
                )
            )
        return positions

    def upsert_position(self, ticker: str, name: str, quantity: int, avg_price: float) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        if quantity <= 0:
            self.conn.execute("delete from positions where ticker = ?", (ticker,))
        else:
            self.conn.execute(
                """
                insert into positions(ticker, name, quantity, avg_price, updated_at)
                values(?, ?, ?, ?, ?)
                on conflict(ticker) do update set
                    name = excluded.name,
                    quantity = excluded.quantity,
                    avg_price = excluded.avg_price,
                    updated_at = excluded.updated_at
                """,
                (ticker, name, int(quantity), float(avg_price), now),
            )
        self.conn.commit()

    def start_run(self, run_id: str, mode: str) -> None:
        self.conn.execute(
            "insert or replace into runs(run_id, started_at, mode, status, message) values(?, ?, ?, ?, ?)",
            (run_id, datetime.now().isoformat(timespec="seconds"), mode, "RUNNING", ""),
        )
        self.conn.commit()

    def finish_run(self, run_id: str, status: str, message: str = "") -> None:
        self.conn.execute("update runs set status = ?, message = ? where run_id = ?", (status, message, run_id))
        self.conn.commit()

    def record_signals(self, run_id: str, signals: Iterable[Signal]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        self.conn.executemany(
            """
            insert into signals(run_id, ts, ticker, name, score, rank, price, model, asof, metadata_json)
            values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    now,
                    signal.ticker,
                    signal.name,
                    float(signal.score),
                    int(signal.rank),
                    int(signal.price),
                    signal.model,
                    signal.asof,
                    json.dumps(signal.metadata, ensure_ascii=False),
                )
                for signal in signals
            ],
        )
        self.conn.commit()

    def record_order(self, run_id: str, mode: str, result: OrderResult) -> None:
        intent = result.intent
        self.conn.execute(
            """
            insert into orders(run_id, ts, mode, ticker, name, side, quantity, price, notional, status, reason, message, raw_json)
            values(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                datetime.now().isoformat(timespec="seconds"),
                mode,
                intent.ticker,
                intent.name,
                intent.side,
                int(intent.quantity),
                int(intent.price),
                int(intent.notional),
                result.status,
                intent.reason,
                result.message,
                json.dumps(result.raw, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def orders_today(self) -> int:
        today = date.today().isoformat()
        row = self.conn.execute(
            """
            select count(*) as c
            from orders
            where substr(ts, 1, 10) = ?
              and status not in ('REJECTED', 'ERROR')
            """,
            (today,),
        ).fetchone()
        return int(row["c"])
