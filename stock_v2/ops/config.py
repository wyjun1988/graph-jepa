from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class RiskConfig:
    max_positions: int = 10
    max_new_buys_per_run: int = 3
    max_orders_per_day: int = 3
    max_cash_per_order: int = 500_000
    max_position_pct_equity: float = 0.08
    max_total_exposure_pct: float = 0.60
    min_score: float = 0.0
    min_price: int = 1_000
    max_price: int = 2_000_000
    take_profit_pct: float = 0.08
    stop_loss_pct: float = -0.05
    limit_buffer_bps: float = 20.0
    block_new_buys_when_market_score_below: Optional[float] = None
    allow_tickers: List[str] = field(default_factory=list)
    block_tickers: List[str] = field(default_factory=list)


@dataclass
class KiwoomConfig:
    env_file: str = "../stock/.env"
    server: str = "real"
    exchange: str = "KRX"
    timeout_sec: float = 10.0


@dataclass
class OpsConfig:
    mode: str = "paper"
    state_db: str = "ops/state/paper.sqlite3"
    paper_initial_cash: int = 10_000_000
    paper_commission_bps: float = 0.0
    paper_sell_tax_bps: float = 0.0
    model_dir: str = "models/krx100_mps"
    signal_model: str = "jepa_only_ridge"
    latent_path_head_path: Optional[str] = None
    data_start: str = "2020-01-01"
    data_end: Optional[str] = None
    top_k: int = 10
    target_weight: float = 0.08
    cache_dir: str = "data/cache"
    reports_dir: str = "ops/reports"
    live_event_paths: List[str] = field(default_factory=list)
    device: str = "cpu"
    use_intraday_quotes: bool = False
    intraday_quote_scope: str = "signal_candidates"
    intraday_quote_limit: int = 20
    intraday_quote_sleep_sec: float = 0.15
    intraday_quote_include_orderbook: bool = True
    require_intraday_quotes: bool = False
    intraday_quote_retry_rounds: int = 0
    intraday_quote_topup_rounds: int = 0
    max_intraday_missing_business_days: int = 0
    min_intraday_model_quote_count: int = 0
    min_top_k_intraday_quote_count: int = 0
    min_latest_coverage_ratio: float = 0.90
    latest_coverage_lookback: int = 20
    risk: RiskConfig = field(default_factory=RiskConfig)
    kiwoom: KiwoomConfig = field(default_factory=KiwoomConfig)

    def __post_init__(self) -> None:
        if self.intraday_quote_scope not in {"signal_candidates", "model_universe"}:
            raise ValueError(
                "intraday_quote_scope must be signal_candidates or model_universe"
            )
        if self.intraday_quote_limit < 0:
            raise ValueError("intraday_quote_limit must be non-negative")
        if self.intraday_quote_topup_rounds < 0:
            raise ValueError("intraday_quote_topup_rounds must be non-negative")
        if self.intraday_quote_retry_rounds < 0:
            raise ValueError("intraday_quote_retry_rounds must be non-negative")
        if self.max_intraday_missing_business_days < 0:
            raise ValueError("max_intraday_missing_business_days must be non-negative")
        if self.min_intraday_model_quote_count < 0:
            raise ValueError("min_intraday_model_quote_count must be non-negative")
        if self.min_top_k_intraday_quote_count < 0:
            raise ValueError("min_top_k_intraday_quote_count must be non-negative")
        if self.min_top_k_intraday_quote_count > self.top_k:
            raise ValueError("min_top_k_intraday_quote_count cannot exceed top_k")
        if self.min_top_k_intraday_quote_count and not self.use_intraday_quotes:
            raise ValueError(
                "min_top_k_intraday_quote_count requires use_intraday_quotes"
            )

    @classmethod
    def load(cls, path: str | Path) -> "OpsConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        risk = RiskConfig(**data.pop("risk", {}))
        kiwoom = KiwoomConfig(**data.pop("kiwoom", {}))
        return cls(**data, risk=risk, kiwoom=kiwoom)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
