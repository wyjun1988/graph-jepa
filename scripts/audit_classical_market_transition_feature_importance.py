from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
)
from stock_v2.market_transition_head import MARKET_EVENT_TARGETS


def semantic_feature_group(name: str) -> str:
    parts = str(name).split(":")
    pool = parts[0]
    feature = parts[-1]
    if pool.startswith("transition_lag"):
        return "transition_history"
    if "available" in pool:
        return "availability"
    if feature.startswith("news_"):
        return "news"
    if feature.startswith("fund_"):
        return "fundamental"
    if feature.startswith("investor_"):
        return "investor_flow"
    if feature.startswith("ext_vix_"):
        return "external_volatility"
    if feature.startswith(("ext_kospi_", "ext_kosdaq_", "ext_sp500_", "ext_nasdaq_", "ext_dow_")):
        return "external_equity"
    if feature.startswith(("ext_usdkrw_", "ext_jpykrw_")):
        return "foreign_exchange"
    if feature.startswith(("ext_us10y_", "ext_bok_", "ext_fed_")):
        return "interest_rates"
    if feature.startswith(("ext_gold_", "ext_wti_")):
        return "commodities"
    if feature.startswith("ext_"):
        return "other_external"
    if feature.startswith(("volume_", "value_", "amihud_")):
        return "activity_liquidity"
    if feature.startswith(("volatility_", "downside_volatility_", "range_")):
        return "volatility_range"
    if feature.startswith(
        (
            "market_return_",
            "relative_return_",
            "cs_rank_",
            "market_beta_",
            "market_corr_",
        )
    ):
        return "market_structure"
    if feature.startswith(
        (
            "return_",
            "ma5_",
            "ma10_",
            "ma20_",
            "ma60_",
            "ma120_",
            "gap_open",
            "intraday_return",
            "drawdown_",
            "breakout_",
        )
    ):
        return "price_path"
    return "other_stock_state"


def _sorted_mapping(values: Mapping[str, float]) -> dict[str, float]:
    return dict(sorted(values.items(), key=lambda item: item[1], reverse=True))


def importance_payload(
    values: Sequence[float], feature_names: Sequence[str], *, top_k: int = 25
) -> dict[str, Any]:
    importance = np.asarray(values, dtype=np.float64)
    names = np.asarray([str(name) for name in feature_names], dtype=object)
    if importance.shape != (len(names),):
        raise ValueError("feature importance and feature names do not align")
    if not np.isfinite(importance).all() or (importance < 0.0).any():
        raise ValueError("feature importance must be finite and nonnegative")
    total = float(importance.sum())
    normalized = importance / total if total > 1e-12 else np.zeros_like(importance)
    order = np.argsort(-normalized, kind="mergesort")
    groups: dict[str, float] = {}
    pools: dict[str, float] = {}
    for name, value in zip(names.tolist(), normalized.tolist()):
        group = semantic_feature_group(name)
        groups[group] = groups.get(group, 0.0) + float(value)
        pool = name.partition(":")[0]
        pools[pool] = pools.get(pool, 0.0) + float(value)
    positive = normalized[normalized > 0.0]
    effective = (
        float(1.0 / np.square(positive).sum()) if positive.size else 0.0
    )
    count = max(0, min(int(top_k), len(order)))
    return {
        "features": len(names),
        "importance_sum": total,
        "top_feature_share": float(normalized[order[0]]) if len(order) else 0.0,
        "top_10_share": float(normalized[order[:10]].sum()),
        "effective_feature_count": effective,
        "top_features": [
            {"name": str(names[index]), "importance": float(normalized[index])}
            for index in order[:count]
        ],
        "semantic_group_importance": _sorted_mapping(groups),
        "pool_operator_importance": _sorted_mapping(pools),
    }


def classifier_importance(
    classifier,
    feature_names: Sequence[str],
    *,
    event_names: Sequence[str] = MARKET_EVENT_TARGETS,
    top_k: int = 25,
) -> dict[str, Any]:
    if isinstance(classifier, dict):
        if classifier.get("mode") != "by_event":
            raise ValueError("unknown classifier bundle")
        models = list(classifier.get("models", []))
        if len(models) != len(event_names):
            raise ValueError("event classifier bundle does not match event names")
        feature_indices = classifier.get("feature_indices")
        if feature_indices is None:
            feature_indices = [range(len(feature_names))] * len(models)
        if len(feature_indices) != len(models):
            raise ValueError("event feature contracts do not match event models")
        return {
            "mode": "by_event",
            "events": {
                str(name): importance_payload(
                    model.feature_importances_,
                    [feature_names[int(index)] for index in indices],
                    top_k=top_k,
                )
                for name, model, indices in zip(
                    event_names, models, feature_indices
                )
            },
        }
    return {
        "mode": "joint",
        "joint": importance_payload(
            classifier.feature_importances_, feature_names, top_k=top_k
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Classical Market-Transition Feature Importance",
        "",
        f"- Target: `{payload['target_version']}`",
        f"- Classifier mode: `{payload['classifier']['mode']}`",
        "- Live orders allowed: `false`",
        "",
    ]
    events = payload["classifier"].get("events")
    if events:
        lines.extend(
            [
                "| Event | Leading semantic group | Share | Leading feature | Share |",
                "|---|---|---:|---|---:|",
            ]
        )
        for event, item in events.items():
            group, group_share = next(iter(item["semantic_group_importance"].items()))
            feature = item["top_features"][0]
            lines.append(
                f"| {event} | {group} | {group_share:.3f} | "
                f"{feature['name']} | {feature['importance']:.3f} |"
            )
    else:
        item = payload["classifier"]["joint"]
        group, share = next(iter(item["semantic_group_importance"].items()))
        lines.append(f"Joint leading semantic group: `{group}` ({share:.3f}).")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit event-specific ExtraTrees market-transition sensors."
    )
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    model_root = Path(args.model_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = joblib.load(model_root / "classical_market_transition_head.joblib")
    feature_names = [str(name) for name in artifact["feature_names"]]
    if artifact.get("target_version") != MARKET_TRANSITION_TARGET_VERSION:
        raise ValueError("model target version does not match feature audit")
    if artifact.get("impact_metric_version") != MARKET_TRANSITION_IMPACT_METRIC_VERSION:
        raise ValueError("model impact metric version does not match feature audit")
    payload = {
        "status": "complete",
        "role": "event_specific_sensor_importance_audit",
        "model_root": str(model_root),
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "regressor": importance_payload(
            artifact["regressor"].feature_importances_,
            feature_names,
            top_k=int(args.top_k),
        ),
        "classifier": classifier_importance(
            artifact["classifier"],
            feature_names,
            top_k=int(args.top_k),
        ),
        "test_used_for_selection": False,
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "classifier_mode": payload["classifier"]["mode"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
