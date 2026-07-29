from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.ops.brokers import KiwoomRestBroker, PaperBroker
from stock_v2.ops.config import KiwoomConfig
from stock_v2.ops.store import OpsStore
from stock_v2.ops.types import OrderIntent


def main() -> None:
    ticker = "241840"
    state_db = ROOT / "ops/state/paper_roundtrip_smoke.sqlite3"
    store = OpsStore(state_db)
    store.init_cash(100_000, reset=True)
    paper = PaperBroker(store, 100_000, commission_bps=1.5, sell_tax_bps=20.0)
    kiwoom = KiwoomRestBroker(
        KiwoomConfig(env_file="../stock/.env", server="real", exchange="KRX"),
        dry_run=True,
    )
    if not kiwoom.authenticate():
        raise RuntimeError("Kiwoom read-only authentication failed")
    quote = kiwoom.get_quote(ticker)
    if quote is None or quote.bid_price is None or quote.ask_price is None:
        raise RuntimeError("round-trip smoke requires a complete bid/ask quote")

    buy = OrderIntent(
        ticker=ticker,
        name="에이스토리",
        side="BUY",
        quantity=1,
        price=int(round(quote.ask_price)),
        reason="isolated_roundtrip_smoke",
        signal_score=1.0,
    )
    buy_preview = kiwoom.place_order(buy)
    buy_result = paper.place_order(buy)
    sell = OrderIntent(
        ticker=ticker,
        name="에이스토리",
        side="SELL",
        quantity=1,
        price=int(round(quote.bid_price)),
        reason="isolated_roundtrip_smoke",
        signal_score=1.0,
    )
    sell_preview = kiwoom.place_order(sell)
    sell_result = paper.place_order(sell)
    account = paper.get_account({ticker: quote.usable_price or quote.bid_price})
    payload = {
        "status": "pass",
        "mode": "isolated_paper_roundtrip",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ticker": ticker,
        "live_orders_allowed": False,
        "buy": {"paper": asdict(buy_result), "kiwoom": asdict(buy_preview)},
        "sell": {"paper": asdict(sell_result), "kiwoom": asdict(sell_preview)},
        "account": asdict(account),
        "roundtrip_pnl": account.equity - 100_000,
    }
    if buy_preview.status != "DRY_RUN" or sell_preview.status != "DRY_RUN":
        raise RuntimeError("Kiwoom preview escaped dry-run mode")
    if buy_result.status != "FILLED" or sell_result.status != "FILLED":
        raise RuntimeError("paper round-trip did not fill")
    if account.positions:
        raise RuntimeError("paper round-trip left an unexpected position")
    output = ROOT / "ops/reports/latent_head_paper_100k/paper_roundtrip_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    store.close()
    print(
        json.dumps(
            {
                "status": "pass",
                "buy_api": buy_preview.raw.get("api_id"),
                "sell_api": sell_preview.raw.get("api_id"),
                "roundtrip_pnl": payload["roundtrip_pnl"],
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
