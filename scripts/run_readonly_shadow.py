from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.ops.config import OpsConfig
from stock_v2.ops.engine import OpsEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate read-only shadow signals without entering the order path."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--data-end")
    parser.add_argument("--cache-dir")
    return parser.parse_args()


def resolve_release_end(cache_dir: str | Path) -> str:
    root = Path(cache_dir)
    for manifest_path in (root / "manifest.json", root.parent / "manifest.json"):
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        end = str(payload.get("end") or "").strip()
        if end:
            return end
    raise ValueError("data_end is unset and the OHLCV cache has no release manifest end")


def validate_readonly_config(config: OpsConfig) -> None:
    if config.mode == "live":
        raise ValueError("read-only shadow runner refuses mode=live")
    if config.signal_model != "world_model":
        raise ValueError("read-only world-model shadow requires signal_model=world_model")
    if config.target_weight != 0.0:
        raise ValueError("read-only shadow requires target_weight=0")
    limits = (
        config.risk.max_new_buys_per_run,
        config.risk.max_orders_per_day,
        config.risk.max_cash_per_order,
    )
    if any(value != 0 for value in limits):
        raise ValueError("read-only shadow requires all order limits to be zero")


def main() -> None:
    args = parse_args()
    config = OpsConfig.load(args.config)
    if args.device:
        config.device = args.device
    if args.cache_dir:
        config.cache_dir = args.cache_dir
    if args.data_end:
        config.data_end = args.data_end
    if config.data_end is None:
        config.data_end = resolve_release_end(config.cache_dir)
    validate_readonly_config(config)
    engine = OpsEngine(config)
    try:
        signals = engine.current_signals()
    finally:
        engine.close()
    payload = {
        "status": "complete",
        "approval_scope": "read_only_shadow",
        "live_orders_allowed": False,
        "orders_submitted": 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config)),
        "model_dir": config.model_dir,
        "latent_path_head_path": config.latent_path_head_path,
        "model_state_updated_from_quote": any(
            bool(signal.metadata.get("model_state_updated_from_quote"))
            for signal in signals
        ),
        "model_input_quote_count": max(
            (
                int(signal.metadata.get("model_input_quote_count", 0))
                for signal in signals
            ),
            default=0,
        ),
        "final_top_k_quote_count": max(
            (
                int(signal.metadata.get("final_top_k_quote_count", 0))
                for signal in signals
            ),
            default=0,
        ),
        "final_top_k_size": max(
            (
                int(signal.metadata.get("final_top_k_size", 0))
                for signal in signals
            ),
            default=0,
        ),
        "intraday_quote_topup_rounds_used": max(
            (
                int(signal.metadata.get("intraday_quote_topup_rounds_used", 0))
                for signal in signals
            ),
            default=0,
        ),
        "total_intraday_quote_count": max(
            (
                int(signal.metadata.get("total_intraday_quote_count", 0))
                for signal in signals
            ),
            default=0,
        ),
        "intraday_imputed_dynamic_node_count": max(
            (
                int(signal.metadata.get("intraday_imputed_dynamic_node_count", 0))
                for signal in signals
            ),
            default=0,
        ),
        "signals": [asdict(signal) for signal in signals],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "approval_scope": payload["approval_scope"],
                "signals": len(signals),
                "asof": signals[0].asof if signals else None,
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
