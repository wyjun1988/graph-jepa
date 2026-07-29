from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REQUIRED_HORIZONS = (1, 2, 3, 5, 10)
CRITICAL_RUNNER_PATHS = (
    "scripts/audit_rolling_validation_contract.py",
    "scripts/freeze_rolling_v6_preflight.py",
    "scripts/run_real_backtest.py",
    "scripts/run_walk_forward_node_eval.py",
    "scripts/evaluate_trained_market_transition_auxiliary.py",
    "scripts/run_v6_rolling5_preflight_rtx4000ada.sh",
    "scripts/run_v6_rolling5_train_rtx4000ada.sh",
    "scripts/run_v6_rolling5_preflight_lifecycle500_rtx4000ada.sh",
    "scripts/run_v6_rolling5_train_lifecycle500_rtx4000ada.sh",
    "scripts/run_v7_rolling5_preflight_global_context_lifecycle500_rtx4000ada.sh",
    "scripts/run_v7_rolling5_train_global_context_lifecycle500_rtx4000ada.sh",
    "scripts/run_v6_rolling_sensor_audit_m1max_v4.sh",
    "scripts/verify_rolling_v6_preflight.py",
)


def _iso_date(value: Any, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(source_root: Path) -> tuple[list[dict[str, str]], str]:
    candidates = set((source_root / "stock_v2").rglob("*.py"))
    candidates.update(
        path
        for relative in CRITICAL_RUNNER_PATHS
        if (path := source_root / relative).is_file()
    )
    rows = [
        {
            "path": path.relative_to(source_root).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in sorted(candidates)
        if path.is_file()
    ]
    if not rows:
        raise ValueError("no training source files were found to freeze")
    return rows, canonical_sha256({"files": rows})


def validate_rolling_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    contract = deepcopy(dict(payload))
    schema_version = int(contract.get("schema_version", 0) or 0)
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("rolling validation must explicitly prohibit live orders")
    if contract.get("promotion_scope") != "read_only_shadow_only":
        raise ValueError("promotion_scope must be read_only_shadow_only")
    selection = contract.get("selection") or {}
    if schema_version < 5:
        if selection.get("test_used_for_selection") is not False:
            raise ValueError("test outputs must not be used for model selection")
    else:
        if selection.get("test_used_for_selection") is not True:
            raise ValueError("v5 must disclose retrospective test-informed selection")
        if contract.get("known_development_period_reuse") is not True:
            raise ValueError("v5 requires known_development_period_reuse=true")
        if contract.get("forward_only_evidence_required_before_live_orders") is not True:
            raise ValueError("v5 requires forward-only evidence before live orders")
        if selection.get("retrospective_only") is not True:
            raise ValueError("v5 selection evidence must be marked retrospective_only")
        if selection.get("forward_shadow_required") is not True:
            raise ValueError("v5 requires forward_shadow_required=true")
    if selection.get("checkpoint") != "epoch24_fixed":
        raise ValueError("rolling validation must use the frozen epoch24 checkpoint")

    architecture = contract.get("architecture") or {}
    horizons = tuple(int(value) for value in architecture.get("horizons", []))
    if horizons != REQUIRED_HORIZONS:
        raise ValueError(f"horizons must be {REQUIRED_HORIZONS}")
    if int(architecture.get("hidden_dim", 0)) != 1024:
        raise ValueError("hidden_dim must remain frozen at 1024")
    if int(architecture.get("layers", 0)) != 10:
        raise ValueError("layers must remain frozen at 10")
    if int(architecture.get("epochs", 0)) != 24:
        raise ValueError("epochs must remain frozen at 24")
    if int(architecture.get("seed", -1)) != 17:
        raise ValueError("encoder seed must remain frozen at 17")

    cutoff = _iso_date(contract.get("data_cutoff"), "data_cutoff")
    folds = contract.get("folds") or []
    if len(folds) < 5:
        raise ValueError("at least five predeclared rolling folds are required")
    labels: set[str] = set()
    previous_eval_end: date | None = None
    for index, fold in enumerate(folds, start=1):
        label = str(fold.get("label") or "")
        if not label or label in labels:
            raise ValueError("fold labels must be non-empty and unique")
        labels.add(label)
        train_end = _iso_date(fold.get("train_end"), f"fold {label} train_end")
        eval_end = _iso_date(fold.get("eval_end"), f"fold {label} eval_end")
        if train_end >= eval_end:
            raise ValueError(f"fold {label} must have train_end < eval_end")
        if eval_end > cutoff:
            raise ValueError(f"fold {label} exceeds the frozen data cutoff")
        if previous_eval_end is not None and train_end != previous_eval_end:
            raise ValueError(
                f"fold {label} must start training at the prior evaluation boundary"
            )
        previous_eval_end = eval_end
        fold["ordinal"] = index

    gates = contract.get("gates") or {}
    per_fold = gates.get("per_fold") or {}
    aggregate = gates.get("aggregate") or {}
    if float(per_fold.get("direct_node_margin_min_exclusive", -1.0)) != 0.0:
        raise ValueError("direct node margin gate must remain strictly greater than zero")
    if float(per_fold.get("path_ic_min_exclusive", -1.0)) != 0.0:
        raise ValueError("every fold path IC must remain strictly positive")
    if float(aggregate.get("top300_h10_newey_west_t_min", 0.0)) < 2.58:
        raise ValueError("aggregate top300 h10 significance gate cannot be below 2.58")
    if float(aggregate.get("positive_fold_fraction_min", 0.0)) < 0.80:
        raise ValueError("positive-fold fraction gate cannot be below 0.80")
    if int(aggregate.get("significant_direct_losses_max", -1)) != 0:
        raise ValueError("significant direct challenger losses must be zero")

    if 2 <= schema_version < 4:
        geometry = contract.get("data_geometry") or {}
        sensors = contract.get("sensor_gates") or {}
        if int(geometry.get("feature_dates", 0)) != 1521:
            raise ValueError("v2 feature date count must remain frozen at 1521")
        if int(geometry.get("initial_training_feature_rows", 0)) < 500:
            raise ValueError("v2 initial training window must contain at least 500 rows")
        if int(geometry.get("raw_evaluation_rows_per_fold", 0)) != 204:
            raise ValueError("v2 raw evaluation windows must remain 204 rows")
        if int(per_fold.get("minimum_evaluation_steps", 0)) < 190:
            raise ValueError("v2 folds must contain at least 190 evaluable signal rows")
        if float(sensors.get("event_ticker_coverage_min", 0.0)) < 0.99:
            raise ValueError("event ticker coverage gate cannot be below 0.99")
        if float(sensors.get("fundamental_cell_coverage_min", 0.0)) < 0.79:
            raise ValueError("fundamental coverage gate cannot be below 0.79")
        if float(sensors.get("investor_cell_coverage_min", 0.0)) < 0.95:
            raise ValueError("investor coverage gate cannot be below 0.95")
        if (
            int(sensors.get("stock_nodes_required", 0)) != 453
            or int(sensors.get("external_nodes_required", 0)) != 13
            or int(sensors.get("features_required", 0)) != 149
        ):
            raise ValueError("v2 node and feature geometry must remain frozen")

    elif schema_version >= 4:
        geometry = contract.get("data_geometry") or {}
        sensors = contract.get("sensor_gates") or {}
        if int(geometry.get("raw_panel_dates", 0)) != 1601:
            raise ValueError("v4 raw panel date count must remain frozen at 1601")
        if int(geometry.get("feature_dates", 0)) != 1511:
            raise ValueError("v4 evaluable feature date count must remain frozen at 1511")
        if int(geometry.get("initial_training_feature_rows", 0)) < 500:
            raise ValueError("v4 initial training window must contain at least 500 rows")
        if int(geometry.get("raw_evaluation_rows_per_fold", 0)) != 204:
            raise ValueError("v4 raw evaluation windows must remain 204 rows")
        if int(per_fold.get("minimum_evaluation_steps", 0)) < 190:
            raise ValueError("v4 folds must contain at least 190 evaluable signal rows")
        if float(sensors.get("event_ticker_coverage_min", 0.0)) < 0.99:
            raise ValueError("event ticker coverage gate cannot be below 0.99")
        if float(sensors.get("fundamental_cell_coverage_min", 0.0)) < 0.79:
            raise ValueError("fundamental coverage gate cannot be below 0.79")
        if float(sensors.get("investor_cell_coverage_min", 0.0)) < 0.95:
            raise ValueError("investor coverage gate cannot be below 0.95")
        if sensors.get("fundamental_coverage_basis") != "price_observed":
            raise ValueError("v4 fundamental coverage must be conditioned on observed prices")
        if sensors.get("investor_coverage_basis") != "supported_traded_value":
            raise ValueError("v4 investor coverage must use supported traded-value cells")
        if float(sensors.get("target_finite_ratio_on_observed_state_min", 0.0)) < 0.98:
            raise ValueError("v4 observed-state target coverage gate cannot be below 0.98")
        if (
            int(sensors.get("stock_nodes_required", 0)) != 500
            or int(sensors.get("external_nodes_required", 0)) != 13
            or int(sensors.get("features_required", 0)) != 149
            or int(sensors.get("lifecycle_proxy_nodes_required", 0)) != 47
        ):
            raise ValueError("v4 node, feature, and lifecycle geometry must remain frozen")
        if sensors.get("lifecycle_release_required") is not True:
            raise ValueError("v4 must require a passing lifecycle release")
        if (
            int(architecture.get("stock_nodes", 0)) != 500
            or int(architecture.get("external_nodes", 0)) != 13
            or int(architecture.get("features", 0)) != 149
        ):
            raise ValueError("v4 architecture geometry must match the lifecycle panel")
        release = contract.get("data_release") or {}
        if release.get("kind") != "krx500_lifecycle_hybrid":
            raise ValueError("v4 must bind the lifecycle hybrid data release")
        if len(str(release.get("manifest_sha256") or "")) != 64:
            raise ValueError("v4 data release manifest hash must be frozen")
        if len(str(release.get("audit_sha256") or "")) != 64:
            raise ValueError("v4 data release audit hash must be frozen")

    if schema_version >= 3:
        if int(architecture.get("train_batch_size", 0)) != 16:
            raise ValueError("v3 train_batch_size must match the frozen base recipe")
        if architecture.get("amp_dtype") != "bfloat16":
            raise ValueError("v3 amp_dtype must match the frozen base recipe")
        if schema_version < 5:
            if selection.get("training_recipe_test_selected") is not False:
                raise ValueError("the training recipe cannot be selected from rolling tests")
            if selection.get("training_recipe_source") != "frozen_base_model_metadata":
                raise ValueError("v3 training recipe source must be frozen base metadata")
        else:
            if selection.get("training_recipe_test_selected") is not True:
                raise ValueError("v5 must disclose test-informed training recipe selection")
            if (
                selection.get("training_recipe_source")
                != "post_gate_methodological_amendment"
            ):
                raise ValueError("v5 training recipe source must name the gate amendment")

    contract["folds"] = folds
    contract["contract_sha256"] = canonical_sha256(
        {key: value for key, value in contract.items() if key != "contract_sha256"}
    )
    return contract


def freeze_preflight_manifests(
    contract: Mapping[str, Any],
    reports_root: Path,
    run_name: str,
    source_root: Path | None = None,
) -> dict[str, Any]:
    frozen = validate_rolling_contract(contract)
    source_files, source_tree_sha256 = _source_manifest(source_root or Path.cwd())
    rows = []
    for fold in frozen["folds"]:
        ordinal = int(fold["ordinal"])
        train_token = str(fold["train_end"]).replace("-", "")
        eval_token = str(fold["eval_end"]).replace("-", "")
        fold_name = f"{run_name}_fold{ordinal}_{train_token}_to_{eval_token}"
        fold_root = reports_root / fold_name
        data_path = fold_root / "training_data_manifest.json"
        edge_path = fold_root / "training_edge_manifest.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        edge = json.loads(edge_path.read_text(encoding="utf-8"))
        for role, payload in (("data", data), ("edge", edge)):
            declared = str(payload.get("sha256") or "")
            if len(declared) != 64:
                raise ValueError(f"{fold_name} {role} manifest has no SHA-256")
        rows.append(
            {
                "label": fold["label"],
                "fold_name": fold_name,
                "train_end": fold["train_end"],
                "eval_end": fold["eval_end"],
                "training_data_manifest_sha256": data["sha256"],
                "training_data_manifest_file_sha256": file_sha256(data_path),
                "training_edge_manifest_sha256": edge["sha256"],
                "training_edge_manifest_file_sha256": file_sha256(edge_path),
                "training_edge_steps": int(edge["step_count"]),
                "training_edge_count": int(edge["total_edges"]),
            }
        )
    return {
        "schema_version": 2,
        "role": "frozen_pretraining_rolling_manifest_contract",
        "base_contract": frozen,
        "base_contract_sha256": frozen["contract_sha256"],
        "run_name": run_name,
        "fold_manifests": rows,
        "source_files": source_files,
        "source_tree_sha256": source_tree_sha256,
        "preflight_generated_model_predictions": False,
        "test_used_for_selection": False,
        "model_recipe_test_used_for_selection": bool(
            (frozen.get("selection") or {}).get("test_used_for_selection")
        ),
        "live_orders_allowed": False,
    }


def verify_frozen_preflight(
    frozen: Mapping[str, Any],
    current_contract: Mapping[str, Any],
    reports_root: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if frozen.get("live_orders_allowed") is not False:
        raise ValueError("frozen preflight does not prohibit live orders")
    if frozen.get("preflight_generated_model_predictions") is not False:
        raise ValueError("preflight must not contain model predictions")
    if frozen.get("test_used_for_selection") is not False:
        raise ValueError("preflight test outputs cannot be used for selection")
    validated = validate_rolling_contract(current_contract)
    if frozen.get("base_contract_sha256") != validated["contract_sha256"]:
        raise ValueError("current rolling contract differs from the frozen preflight")
    current_sources, current_source_tree = _source_manifest(source_root or Path.cwd())
    if frozen.get("source_files") != current_sources:
        raise ValueError("training source files changed after preflight")
    if frozen.get("source_tree_sha256") != current_source_tree:
        raise ValueError("training source tree hash changed after preflight")
    run_name = str(frozen.get("run_name") or "")
    expected_folds = validated["folds"]
    rows = frozen.get("fold_manifests") or []
    if len(rows) != len(expected_folds):
        raise ValueError("frozen preflight fold count differs from the contract")
    verified = []
    for expected, row in zip(expected_folds, rows):
        if row.get("label") != expected.get("label"):
            raise ValueError("frozen preflight fold ordering differs")
        fold_root = reports_root / str(row["fold_name"])
        data_path = fold_root / "training_data_manifest.json"
        edge_path = fold_root / "training_edge_manifest.json"
        data = json.loads(data_path.read_text(encoding="utf-8"))
        edge = json.loads(edge_path.read_text(encoding="utf-8"))
        checks = {
            "training_data_manifest_sha256": data.get("sha256"),
            "training_data_manifest_file_sha256": file_sha256(data_path),
            "training_edge_manifest_sha256": edge.get("sha256"),
            "training_edge_manifest_file_sha256": file_sha256(edge_path),
        }
        for key, actual in checks.items():
            if actual != row.get(key):
                raise ValueError(f"{row['label']} changed after preflight: {key}")
        verified.append(
            {
                "label": row["label"],
                "fold_name": row["fold_name"],
                **checks,
            }
        )
    return {
        "status": "pass",
        "run_name": run_name,
        "verified_folds": verified,
        "frozen_contract_sha256": canonical_sha256(frozen),
        "source_tree_sha256": current_source_tree,
        "test_used_for_selection": False,
        "live_orders_allowed": False,
    }


def audit_rolling_sensor_reports(
    contract: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validated = validate_rolling_contract(contract)
    labels = [str(row["label"]) for row in validated["folds"]]
    if set(reports) != set(labels):
        raise ValueError("sensor report labels do not match the rolling contract")
    gates = validated["sensor_gates"]
    schema_version = int(validated.get("schema_version", 0) or 0)
    geometry = validated["data_geometry"]
    minimum_test_rows = int(
        validated["gates"]["per_fold"]["minimum_evaluation_steps"]
    )
    expected_test_rows = int(geometry["raw_evaluation_rows_per_fold"]) - max(
        REQUIRED_HORIZONS
    )
    event_freshness_floor = (
        _iso_date(validated["data_cutoff"], "data_cutoff") - timedelta(days=3)
    ).isoformat()
    rows = []
    blockers = []
    semantic_modes: set[str] = set()
    for ordinal, label in enumerate(labels, start=1):
        expected_fold = validated["folds"][ordinal - 1]
        report = reports[label]
        config = report.get("config") or {}
        training_manifest = report.get("training_data_manifest") or {}
        cache = report.get("cache") or {}
        features = report.get("features") or {}
        universe = report.get("universe") or {}
        ohlcv = report.get("ohlcv") or {}
        panel = report.get("panel") or {}
        event = report.get("event_ticker_coverage") or {}
        event_sources = list((report.get("events") or {}).values())
        event_features = report.get("event_features") or {}
        event_themes = report.get("event_theme_exposure") or {}
        fundamental = report.get("fundamental_sensors") or {}
        investor = report.get("investor_sensors") or {}
        external = report.get("external_factors") or {}
        lifecycle = report.get("lifecycle_release") or {}
        expected_train_rows = int(geometry["initial_training_feature_rows"]) + (
            ordinal - 1
        ) * int(geometry["raw_evaluation_rows_per_fold"])
        event_rows = sum(int(row.get("rows", 0)) for row in event_sources)
        event_invalid_json = sum(
            int(row.get("invalid_json", 0)) for row in event_sources
        )
        event_missing_dates = sum(
            int(row.get("missing_dates", 0)) for row in event_sources
        )
        event_invalid_tickers = sum(
            int(row.get("invalid_tickers", 0)) for row in event_sources
        )
        event_duplicate_keys = sum(
            int(row.get("duplicate_article_key_count", 0)) for row in event_sources
        )
        event_latest = max(
            (str(row.get("date_max") or "") for row in event_sources), default=""
        )
        scored_rows = sum(
            int(row.get("llm_used_rows", 0)) + int(row.get("qwen_source_rows", 0))
            for row in event_sources
        )
        score_abs_max = max(
            (
                float((row.get("abs_score") or {}).get("max") or 0.0)
                for row in event_sources
            ),
            default=0.0,
        )
        semantic_mode = (
            "scored" if scored_rows > 0 and score_abs_max > 0.0 else "neutral_count_theme_only"
        )
        semantic_modes.add(semantic_mode)
        if schema_version >= 4:
            target_coverage = float(
                features.get("target_finite_ratio_on_observed_state") or 0.0
            )
            fundamental_coverage = float(
                fundamental.get("coverage_on_observed_prices") or 0.0
            )
            investor_coverage = float(
                investor.get("coverage_on_observed_traded_value") or 0.0
            )
            panel_integrity = (
                int(panel.get("price_observed_cells", 0)) > 0
                and str(panel.get("date_max")) == str(expected_fold["eval_end"])
            )
            lifecycle_integrity = (
                lifecycle.get("status") == "pass"
                and int(lifecycle.get("verified_files", 0))
                == int(gates["stock_nodes_required"])
                and int(lifecycle.get("lifecycle_violations", -1)) == 0
                and int(
                    (lifecycle.get("provider_counts") or {}).get(
                        "finance_data_reader_adjusted_return_index_proxy",
                        0,
                    )
                )
                == int(gates["lifecycle_proxy_nodes_required"])
            )
        else:
            target_coverage = float(features.get("target_finite_ratio") or 0.0)
            fundamental_coverage = float(fundamental.get("coverage") or 0.0)
            investor_coverage = float(investor.get("coverage") or 0.0)
            panel_integrity = (
                int(panel.get("close_nonfinite_cells", -1)) == 0
                and int(panel.get("open_nonfinite_cells", -1)) == 0
                and str(panel.get("date_max")) == str(expected_fold["eval_end"])
            )
            lifecycle_integrity = True

        checks = {
            "issues_empty": not (report.get("issues") or []),
            "declared_start": str(config.get("start")) == str(validated["data_start"]),
            "declared_train_end": str(config.get("train_end"))
            == str(expected_fold["train_end"]),
            "declared_eval_end": str(config.get("end"))
            == str(expected_fold["eval_end"]),
            "declared_lags": all(
                int(config.get(name, -1)) == 1
                for name in (
                    "event_lag_days",
                    "fundamental_lag_days",
                    "investor_flow_lag_days",
                    "external_lag_days",
                )
            ),
            "manifest_train_end": str(training_manifest.get("train_end"))
            == str(expected_fold["train_end"]),
            "manifest_sha256": len(str(training_manifest.get("sha256") or ""))
            == 64,
            "stock_nodes": int(features.get("tickers", 0))
            == int(gates["stock_nodes_required"]),
            "universe_after_filter": int(
                universe.get("after_train_history_filter", 0)
            )
            == int(gates["stock_nodes_required"]),
            "features": int(features.get("feature_count", 0))
            == int(gates["features_required"]),
            "train_rows": int(features.get("train_rows", 0)) == expected_train_rows,
            "test_rows": int(features.get("test_rows", 0)) == expected_test_rows
            and expected_test_rows >= minimum_test_rows,
            "normalized_features_finite": int(
                features.get("normalized_nonfinite_cells", -1)
            )
            == 0,
            "target_finite_ratio": target_coverage
            >= float(gates.get("target_finite_ratio_on_observed_state_min", 0.95)),
            "cache_universe": int(cache.get("unique_valid_tickers", 0))
            == int(gates["stock_nodes_required"]),
            "ohlcv_integrity": not (ohlcv.get("issues") or {})
            and str(ohlcv.get("date_max")) == str(expected_fold["eval_end"]),
            "panel_integrity": panel_integrity,
            "lifecycle_integrity": lifecycle_integrity,
            "event_coverage": int(event.get("covered_tickers", 0))
            / max(1, int(gates["stock_nodes_required"]))
            >= float(gates["event_ticker_coverage_min"]),
            "event_inventory": event_rows >= 100_000
            and event_invalid_json == 0
            and event_missing_dates == 0
            and event_invalid_tickers == 0
            and event_duplicate_keys == 0
            and event_latest >= event_freshness_floor,
            "event_count_signal": int(
                (event_features.get("news_count_3d") or {}).get("nonzero_cells", 0)
            )
            > 0,
            "event_theme_signal": int(event_themes.get("themes", 0)) > 0
            and int(event_themes.get("nonzero_cells", 0)) > 0,
            "fundamental_coverage": fundamental_coverage
            >= float(gates["fundamental_cell_coverage_min"]),
            "investor_coverage": investor_coverage
            >= float(gates["investor_cell_coverage_min"]),
            "external_nodes": int(external.get("node_count", 0))
            == int(gates["external_nodes_required"]),
            "external_loaded": int(external.get("loaded", 0))
            == int(gates["external_nodes_required"]),
        }
        failed = [name for name, passed in checks.items() if not passed]
        blockers.extend(f"{label}:{name}" for name in failed)
        rows.append(
            {
                "label": label,
                "passed": not failed,
                "failed_checks": failed,
                "checks": checks,
                "train_rows": int(features.get("train_rows", 0)),
                "test_rows": int(features.get("test_rows", 0)),
                "stock_nodes": int(features.get("tickers", 0)),
                "external_nodes": int(external.get("node_count", 0)),
                "features": int(features.get("feature_count", 0)),
                "event_covered_tickers": int(event.get("covered_tickers", 0)),
                "target_gate_coverage": target_coverage,
                "fundamental_coverage": fundamental_coverage,
                "fundamental_raw_coverage": float(fundamental.get("coverage") or 0.0),
                "investor_coverage": investor_coverage,
                "investor_raw_coverage": float(investor.get("coverage") or 0.0),
                "event_rows": event_rows,
                "event_semantic_mode": semantic_mode,
                "event_scored_rows": scored_rows,
                "event_score_abs_max": score_abs_max,
                "source_warnings": report.get("warnings") or [],
            }
        )
    if len(semantic_modes) == 1:
        historical_news_semantics = next(iter(semantic_modes))
    else:
        historical_news_semantics = "mixed"
    return {
        "status": "pass" if not blockers else "blocked",
        "role": "rolling_v6_structured_sensor_audit",
        "folds": rows,
        "blockers": blockers,
        "historical_news_semantics": historical_news_semantics,
        "semantic_score_trained": historical_news_semantics == "scored",
        "capability_limitations": [
            "Historical news contributes event count and theme context, not LLM sentiment."
        ]
        if historical_news_semantics == "neutral_count_theme_only"
        else [],
        "live_orders_allowed": False,
    }
