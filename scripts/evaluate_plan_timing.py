"""Score the learned exit plan on returns from a DECISION, not a fixed horizon.

Every return number in this project so far answers "what did the stock do over
exactly d days". The user's objection is that this is not how the model would be
used: you might buy expecting a state change in 7 days, or intending to sell
tomorrow. The horizon is a choice the model should be making, and the return
should be scored against the choice it made.

So: for each date and stock the model predicts a path return at each horizon,
picks h* = argmax_h r_hat(h) -- a hard argmax, because at decision time you sell
on one day, not on a softmax over days -- and books r(h*). Training uses the
softmax so the loss has a gradient; evaluation uses the argmax because that is
the decision.

WHAT THIS IS MEASURED AGAINST, and why each control is here:

  hold10   the primary baseline. The plan must beat buying and holding to the
           longest horizon, which is what the model was previously scored on.
  fixed5   current practice. Beating hold10 by drifting to a shorter average
           holding period would be a fixed-horizon result wearing a plan's
           clothes; fixed5 catches that.
  placebo  h* labels permuted ACROSS STOCKS within a date. The holding-period
           distribution is preserved exactly; only the stock-to-plan pairing
           breaks. If the plan cannot beat this, the advantage is an artefact of
           which holding periods it likes, not of knowing which stock to hold.
           This is the control that matters.
  oracle   max_h r(h), perfect foresight. Not a baseline -- a ceiling, so the
           advantage can be read as a fraction of what is available.

PARITY IS ENFORCED, NOT ASSUMED. This script rebuilds the inference path from
evaluate_node_prediction's own functions (load_model, build_features_from_ckpt,
select_steps), but a rebuilt path that silently differs would make every number
here meaningless. So --parity-summary takes the main evaluator's summary for the
same checkpoint and this script recomputes pooled state skill through ITS OWN
path; a mismatch beyond tolerance is a hard failure, not a warning.

De-standardisation mirrors training exactly: the heads emit cross-sectionally
standardised values per (date, horizon), so comparing them across horizons
without undoing that would compare quantities on different scales and the argmax
would be meaningless. The raw path return and its per-date cross-sectional mean
and std are recomputed here from features.target_return_paths -- the same source
attach_downstream_targets reads -- rather than imported from the pod's patched
trainer, so this runs on any host.

Evidence class: research. Reads the fold's evaluation steps, which for a
qualification run are the same steps the gate scores. Nothing here promotes a
model or touches the prospective ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_node_prediction import (
    build_features_from_ckpt,
    future_state_metrics,
    graph_edge_kwargs,
    load_model,
    select_steps,
    state_target_feature_mask,
)
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.real_features import make_real_snapshot

PLACEBO_SEED = 20260717


def evaluator_contract_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fields build_features_from_ckpt reads that this script has no flag for.

    Same drift that broke benchmark_direct_baselines, the daily chain's cache
    build and measure_jepa_latency: the evaluator reads these unconditionally,
    so a field it gains later must land as a checkpoint fallback rather than an
    AttributeError. None/False/[] mean "use what the checkpoint recorded".
    """

    contract: dict[str, Any] = {
        "override_universe": False,
        "universe_manifest": None,
        "universe": None,
        "max_tickers": None,
        "start": None,
        "end": None,
        "train_end": None,
        "refresh": False,
        "allow_unverified_legacy": False,
        "allow_extrapolated_horizons": False,
        "min_train_rows": None,
        "edge_window": None,
        "edge_top_k": None,
        "min_abs_corr": None,
        "edge_correlation_mode": None,
        "partial_corr_top_k": None,
        "partial_corr_min_abs": None,
        "partial_corr_mode": None,
        "partial_corr_scale": None,
        "lead_lag_top_k": None,
        "lead_lag_days": None,
        "lead_lag_min_abs_corr": None,
        "lead_lag_mode": None,
        "lead_lag_scale": None,
        "policy_rate_edge_scale": None,
        "factor_sensitivity_top_k": None,
        "event_path": [],
        "event_half_life_days": None,
        "event_lag_days": None,
        "event_max_decay_days": None,
        "event_edge_top_k": None,
        "event_edge_min_weight": None,
        "event_edge_scale": None,
        "event_edge_max_themes": None,
        "event_edge_min_theme_count": None,
        "fundamental_path": [],
        "fundamental_lag_days": None,
        "investor_cache_dir": None,
        "investor_flow_lag_days": None,
        "external_symbol": [],
        "external_preset": None,
        "external_lag_days": None,
        "external_cache_dir": None,
        "external_etf_panel": None,
        "external_etf_symbols": None,
        "industry_profile_path": [],
        "industry_prefix_length": None,
        "industry_edge_scale": None,
    }
    for field, default in contract.items():
        if not hasattr(args, field):
            setattr(args, field, default)
    return args


def causal_path_scale(
    features, step: int, horizon: int, *, lookback: int
) -> tuple[float, float]:
    """Scale for (date, horizon) using only paths that FINISHED before `step`.

    The realized scale cannot be used at decision time: a path entered at date s
    for horizon h is only known at s + h, so at date t the most recent usable
    entry is t - h - 1. Estimating the mean and std over the `lookback` entries
    ending there keeps the transform a function of the past alone.

    This is the whole fix. Ranking horizons requires putting them on a common
    scale, and taking that scale from the current cross-section is what turned
    the argmax into hindsight. A trailing estimate is worse at describing today
    -- that is the point; today is not knowable yet.
    """

    stock_count = int(features.tradable_count)
    source = features.target_return_paths.get(int(horizon))
    if source is None:
        raise ValueError(f"missing target entry path horizon {horizon}")
    last_known = int(step) - int(horizon) - 1
    first = max(0, last_known - int(lookback) + 1)
    if last_known < first:
        return float("nan"), float("nan")
    window = np.asarray(source[first : last_known + 1, :stock_count], dtype=np.float64)
    finite = window[np.isfinite(window)]
    if finite.size < 30:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    std = float(finite.std())
    if not np.isfinite(std) or std < 1e-12:
        return float("nan"), float("nan")
    return mean, std


def raw_path_scale(features, step: int, horizon: int) -> tuple[np.ndarray, float, float]:
    """Realized raw path return at `horizon`, plus its cross-sectional mean/std.

    attach_downstream_targets standardises this exact quantity per (date,
    horizon) before the heads ever see it, and records the mean/std as the scale
    the plan loss multiplies back in. Recomputing it here from the same source
    keeps this script independent of the pod-side trainer patch, at the cost of
    having to keep the two in step -- which the plan_scale parity check below
    exists to catch.
    """

    stock_count = int(features.tradable_count)
    source = features.target_return_paths.get(int(horizon))
    if source is None:
        raise ValueError(f"missing target entry path horizon {horizon}")
    realized = np.asarray(source[int(step), :stock_count], dtype=np.float64)
    finite = np.isfinite(realized)
    if finite.sum() < 3:
        return realized, float("nan"), float("nan")
    mean = float(realized[finite].mean())
    std = float(realized[finite].std())
    if not np.isfinite(std) or std < 1e-12:
        return realized, float("nan"), float("nan")
    return realized, mean, std


def moving_block_bootstrap(
    daily: np.ndarray,
    *,
    block: int,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """95% interval for the mean of a session series, blocks preserving order.

    Daily plan-minus-baseline differences are autocorrelated -- the same names
    stay attractive for days -- so an iid bootstrap would understate the width.
    """

    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = len(values)
    if count < block * 2:
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    block_count = int(np.ceil(count / block))
    starts = generator.integers(0, count - block + 1, size=(samples, block_count))
    offsets = np.arange(block)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        drawn = values[(starts[index][:, None] + offsets[None, :]).ravel()[:count]]
        means[index] = drawn.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(daily: np.ndarray, *, block: int, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(daily, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"sessions": 0, "mean": float("nan")}
    lower, upper = moving_block_bootstrap(values, block=block, samples=samples, seed=seed)
    return {
        "sessions": int(len(values)),
        "mean": float(values.mean()),
        "boot_lower_95": lower,
        "boot_upper_95": upper,
        "positive_session_fraction": float((values > 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--max-steps", type=int, default=194)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--state-target-scope", choices=["all", "checkpoint_temporal"], default="all")
    parser.add_argument("--bootstrap-block", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--scale-lookback", type=int, default=60)
    parser.add_argument(
        "--scale-source",
        choices=["realized", "none", "causal"],
        default="causal",
        help=(
            "How to turn the head's cross-sectionally standardized path-return "
            "prediction back into a return before the argmax.\n"
            "  causal   (default) trailing scale from paths that finished before "
            "the decision date. The only honest option.\n"
            "  realized that date+horizon's REALIZED mean/std, which is what "
            "attach_downstream_targets records and what the training plan loss "
            "uses. LEAKY: the standardized prediction is zero-mean across the "
            "cross-section, so the realized mean is what separates the horizons, "
            "and the argmax becomes hindsight. Kept only to reproduce what "
            "training optimised.\n"
            "  none     rank the standardized predictions directly. Leak-free but "
            "compares horizons on different scales, so it understates the plan."
        ),
    )
    parser.add_argument(
        "--parity-summary",
        default=None,
        help=(
            "the main evaluator's summary.json for this same checkpoint. Its "
            "pooled state skill is recomputed through this script's inference "
            "path; a mismatch means the paths differ and every number here is void."
        ),
    )
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    args = evaluator_contract_defaults(parser.parse_args())

    horizons = [int(value) for value in args.horizons.split(",") if value.strip()]
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    model, ckpt = load_model(model_dir, device)
    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    namespace = argparse.Namespace(**ckpt_args)
    steps = select_steps(features, ckpt_args, args)
    stock_count = int(features.tradable_count)

    edge_window = int(ckpt_args.get("edge_window", 60))
    edge_top_k = int(ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(ckpt_args.get("min_abs_corr", 0.2))
    target_mask = state_target_feature_mask(
        features.feature_names,
        model.temporal_state_feature_weights,
        args.state_target_scope,
    )

    liquidity_index = (
        features.feature_names.index("value_ma20_log")
        if "value_ma20_log" in features.feature_names
        else None
    )

    rows: list[dict[str, Any]] = []
    parity_model_sse = 0.0
    parity_persistence_sse = 0.0
    generator = np.random.default_rng(PLACEBO_SEED)

    for step in steps:
        batch = make_real_snapshot(
            features,
            step=int(step),
            full_observation=True,
            edge_window=edge_window,
            top_k=edge_top_k,
            min_abs_corr=min_abs_corr,
            **graph_edge_kwargs(ckpt_args, args),
        ).to(device)
        with torch.no_grad():
            context = model.encode_temporal_context(batch)

        predicted_raw = np.full((stock_count, len(horizons)), np.nan)
        realized_raw = np.full((stock_count, len(horizons)), np.nan)

        for position, horizon in enumerate(horizons):
            steps_forward = max(1, int(rollout_steps_for_offset(namespace, int(horizon))))
            with torch.no_grad():
                z_pred = model.rollout_latent(context, steps=steps_forward)
                head = model.predict_downstream_auxiliary(
                    context, z_pred, rollout_steps=steps_forward
                )
                # index 0 is path_return in DOWNSTREAM_AUXILIARY_TASKS
                standardized = head[:stock_count, 0].detach().cpu().numpy().astype(np.float64)

                # Parity: recompute pooled state skill the way the main
                # evaluator does, through this script's own inference path.
                state_prediction = model.predict_temporal_state(
                    batch, z_pred, rollout_steps=steps_forward, z_context=context
                ).detach().cpu().numpy()[:stock_count]

            realized, mean, std = raw_path_scale(features, int(step), int(horizon))
            realized_raw[:, position] = realized
            if args.scale_source == "causal":
                mean, std = causal_path_scale(
                    features, int(step), int(horizon), lookback=args.scale_lookback
                )
            if args.scale_source == "none":
                predicted_raw[:, position] = standardized
            elif np.isfinite(mean) and np.isfinite(std):
                predicted_raw[:, position] = standardized * std + mean

            if args.parity_summary:
                # future_state_metrics is imported, not reimplemented. Its
                # validity rule -- target observed AND current observed AND all
                # three arrays finite -- is what makes the skill number mean what
                # it means, and a hand-rolled copy that drops one condition is
                # exactly how this check first failed (drift 2.6e-4 from omitting
                # current_available and the isfinite guards).
                target_available = (
                    features.available_mask[int(step) + int(horizon), :stock_count] > 0.5
                )
                current_available = features.available_mask[int(step), :stock_count] > 0.5
                metrics = future_state_metrics(
                    state_prediction,
                    features.features[int(step) + int(horizon), :stock_count],
                    features.features[int(step), :stock_count],
                    target_available & target_mask[None, :],
                    current_available & target_mask[None, :],
                )
                if metrics is not None:
                    parity_model_sse += float(metrics["model_sse"])
                    parity_persistence_sse += float(metrics["persistence_sse"])

        # A stock is only in scope if every horizon is both predictable and
        # realized -- otherwise the plan and the baselines would be selecting
        # from different menus and the comparison would not be like for like.
        usable = np.isfinite(predicted_raw).all(axis=1) & np.isfinite(realized_raw).all(axis=1)
        if usable.sum() < 3:
            continue

        predicted = predicted_raw[usable]
        realized = realized_raw[usable]
        chosen = predicted.argmax(axis=1)
        index = np.arange(len(chosen))
        hold_position = horizons.index(max(horizons))
        fixed5_position = horizons.index(5) if 5 in horizons else hold_position

        plan_return = realized[index, chosen]
        hold_return = realized[:, hold_position]
        fixed5_return = realized[:, fixed5_position]
        oracle_return = realized.max(axis=1)
        placebo_return = realized[index, generator.permutation(chosen)]

        row: dict[str, Any] = {
            "date": str(pd.Timestamp(features.dates[int(step)]).date()),
            "scored_stocks": int(usable.sum()),
            "plan_minus_hold10": float((plan_return - hold_return).mean()),
            "plan_minus_fixed5": float((plan_return - fixed5_return).mean()),
            "plan_minus_placebo": float((plan_return - placebo_return).mean()),
            "oracle_minus_hold10": float((oracle_return - hold_return).mean()),
            "placebo_minus_hold10": float((placebo_return - hold_return).mean()),
            "plan_return_mean": float(plan_return.mean()),
            "hold10_return_mean": float(hold_return.mean()),
        }
        # Every constant-horizon policy, as its own baseline. If "always h1"
        # already beats hold10 by what the plan beats hold10 by, then the plan
        # is not choosing anything -- the period simply rewarded short holds, and
        # a one-line rule with no model would score the same. The h*-permutation
        # placebo cannot separate that case from a real distribution effect,
        # because it preserves the very mix that would be doing the work.
        #
        # Per-date only. Which constant horizon is BEST must be settled across
        # the whole period in the summary: picking the winner date by date would
        # let the baseline see each day's realized returns before choosing, and
        # beating a look-ahead oracle is not the test.
        for position, horizon in enumerate(horizons):
            row[f"fixed{horizon}_minus_hold10"] = float(
                (realized[:, position] - hold_return).mean()
            )
        for position, horizon in enumerate(horizons):
            row[f"plan_argmax_h{horizon}"] = float((chosen == position).mean())

        # --- BUY + SELL timing (goal: 매수 매도 계획에 따른 수익률) ---
        # Entry menu: column 0 is "buy now" (open t+1, zero base), columns 1..H
        # are close(t+h). Exit menu: close(t+h). Rank pairs on predicted
        # close-to-close; realize with actual prices.
        n_scored = predicted.shape[0]
        entry_pred = np.concatenate([np.zeros((n_scored, 1)), predicted], axis=1)
        entry_real = np.concatenate([np.zeros((n_scored, 1)), realized], axis=1)
        best_gain = np.full(n_scored, -np.inf)
        best_entry = np.zeros(n_scored, dtype=int)
        best_exit = np.zeros(n_scored, dtype=int)
        n_cols = entry_pred.shape[1]
        for entry_col in range(n_cols):
            for exit_col in range(entry_col + 1, n_cols):
                gain = entry_pred[:, exit_col] - entry_pred[:, entry_col]
                improved = gain > best_gain
                best_gain = np.where(improved, gain, best_gain)
                best_entry = np.where(improved, entry_col, best_entry)
                best_exit = np.where(improved, exit_col, best_exit)
        bs_return = (1.0 + entry_real[index, best_exit]) / (1.0 + entry_real[index, best_entry]) - 1.0
        perm = generator.permutation(n_scored)
        bs_placebo = (
            (1.0 + entry_real[index, best_exit[perm]]) / (1.0 + entry_real[index, best_entry[perm]]) - 1.0
        )
        row["bs_plan_return_mean"] = float(bs_return.mean())
        row["bs_plan_minus_hold10"] = float((bs_return - hold_return).mean())
        row["bs_plan_minus_placebo"] = float((bs_return - bs_placebo).mean())
        row["bs_plan_minus_exit_only"] = float((bs_return - plan_return).mean())
        # entry: 0 = buy now, k = buy at close(t+h_k). exit: close(t+h_k).
        for entry_col in range(n_cols):
            label = "now" if entry_col == 0 else f"h{horizons[entry_col-1]}"
            row[f"bs_entry_{label}"] = float((best_entry == entry_col).mean())
        for exit_col in range(1, n_cols):
            row[f"bs_exit_h{horizons[exit_col-1]}"] = float((best_exit == exit_col).mean())

        if liquidity_index is not None:
            liquidity = features.raw_features[int(step), :stock_count, liquidity_index][usable]
            finite = np.isfinite(liquidity)
            if finite.sum() >= 300:
                order = np.argsort(-np.where(finite, liquidity, -np.inf))[:300]
                row["top300_plan_minus_hold10"] = float(
                    (plan_return[order] - hold_return[order]).mean()
                )
                row["top300_plan_minus_placebo"] = float(
                    (plan_return[order] - placebo_return[order]).mean()
                )
        rows.append(row)

    if not rows:
        raise SystemExit("no session produced a scorable plan")
    frame = pd.DataFrame(rows)

    parity: dict[str, Any] = {"checked": False}
    if args.parity_summary:
        reference = json.loads(Path(args.parity_summary).read_text(encoding="utf-8"))
        rebuilt = 1.0 - parity_model_sse / parity_persistence_sse
        recorded = reference.get("pooled_mse_skill_vs_persistence")
        if recorded is None:
            csv_path = Path(args.parity_summary).with_name("future_rollout.csv")
            table = pd.read_csv(csv_path)
            recorded = 1.0 - table["model_sse"].sum() / table["persistence_sse"].sum()
        drift = abs(rebuilt - float(recorded))
        parity = {
            "checked": True,
            "reference": str(args.parity_summary),
            "reference_pooled_state_skill": float(recorded),
            "rebuilt_pooled_state_skill": float(rebuilt),
            "absolute_drift": drift,
            "tolerance": args.parity_tolerance,
            "passed": drift <= args.parity_tolerance,
        }
        if not parity["passed"]:
            raise SystemExit(
                f"PARITY FAILED: this script's inference path gives pooled state skill "
                f"{rebuilt:.6f} against the evaluator's {float(recorded):.6f} "
                f"(drift {drift:.6f} > {args.parity_tolerance}). The paths differ; "
                f"every plan number would be measuring a different model. Refusing to write."
            )

    boot = dict(block=args.bootstrap_block, samples=args.bootstrap_samples, seed=args.seed)
    comparisons = {
        name: summarize(frame[name].to_numpy(), **boot)
        for name in (
            "plan_minus_hold10",
            "plan_minus_fixed5",
            "plan_minus_placebo",
            "oracle_minus_hold10",
            "placebo_minus_hold10",
        )
        if name in frame
    }
    if "top300_plan_minus_hold10" in frame:
        comparisons["top300_plan_minus_hold10"] = summarize(
            frame["top300_plan_minus_hold10"].to_numpy(), **boot
        )
        comparisons["top300_plan_minus_placebo"] = summarize(
            frame["top300_plan_minus_placebo"].to_numpy(), **boot
        )
    for horizon in horizons:
        column = f"fixed{horizon}_minus_hold10"
        if column in frame:
            comparisons[column] = summarize(frame[column].to_numpy(), **boot)

    # The constant-horizon policy that won over the WHOLE period, chosen with
    # full hindsight over the same 194 sessions. That makes it a generous
    # baseline -- generous on purpose. The plan has a model, five predictions per
    # stock per day, and a training loss; if it cannot beat the best one-line
    # rule that a person could have written after the fact, it is not earning its
    # complexity, and reporting only plan-minus-hold10 would hide that.
    constant_means = {
        horizon: comparisons[f"fixed{horizon}_minus_hold10"]["mean"]
        for horizon in horizons
        if f"fixed{horizon}_minus_hold10" in comparisons
        and np.isfinite(comparisons[f"fixed{horizon}_minus_hold10"].get("mean", np.nan))
    }
    best_constant: dict[str, Any] = {}
    if constant_means:
        winner = max(constant_means, key=lambda h: constant_means[h])
        daily_gap = (
            frame["plan_minus_hold10"].to_numpy()
            - frame[f"fixed{winner}_minus_hold10"].to_numpy()
        )
        best_constant = {
            "horizon": int(winner),
            "selected": "in-sample over the full evaluation period -- deliberately generous",
            "mean_minus_hold10": float(constant_means[winner]),
            "plan_minus_best_constant": summarize(daily_gap, **boot),
        }
        comparisons["plan_minus_best_constant_horizon"] = best_constant["plan_minus_best_constant"]

    oracle_mean = comparisons.get("oracle_minus_hold10", {}).get("mean")
    plan_mean = comparisons.get("plan_minus_hold10", {}).get("mean")
    payload = {
        "role": "research_only_plan_timing_evaluation",
        "live_orders_allowed": False,
        "test_used_for_selection": False,
        "promotion_eligible": False,
        "model_dir": str(model_dir),
        "checkpoint_sha256": hashlib.sha256(
            (model_dir / "graph_jepa_real.pt").read_bytes()
        ).hexdigest(),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest_sha256"),
        "device": str(device),
        "horizons": horizons,
        "sessions": int(len(frame)),
        "selection": "hard argmax over de-standardized per-horizon path-return predictions",
        "placebo": f"h* permuted across stocks within each date, seed {PLACEBO_SEED}",
        "parity": parity,
        "bootstrap": {
            "kind": "moving_block over sessions",
            "block": args.bootstrap_block,
            "samples": args.bootstrap_samples,
        },
        "comparisons": comparisons,
        "best_constant_horizon_baseline": best_constant,
        "plan_argmax_distribution": {
            f"h{horizon}": float(frame[f"plan_argmax_h{horizon}"].mean())
            for horizon in horizons
            if f"plan_argmax_h{horizon}" in frame
        },
        "fraction_of_oracle": (
            float(plan_mean / oracle_mean)
            if plan_mean is not None
            and oracle_mean not in (None, 0.0)
            and np.isfinite(oracle_mean)
            and oracle_mean != 0
            else None
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "daily_plan_metrics.csv", index=False)
    payload["daily_csv_sha256"] = hashlib.sha256(
        (output / "daily_plan_metrics.csv").read_bytes()
    ).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"sessions: {payload['sessions']}")
    if parity["checked"]:
        print(f"parity: drift {parity['absolute_drift']:.2e} <= {parity['tolerance']} PASS")
    print()
    for name, stats in comparisons.items():
        if not np.isfinite(stats.get("mean", float("nan"))):
            continue
        print(
            f"  {name:28} mean {stats['mean']:+.5f}  "
            f"95% [{stats['boot_lower_95']:+.5f}, {stats['boot_upper_95']:+.5f}]  "
            f"pos {stats['positive_session_fraction']:.2f}"
        )
    print()
    print("  h* distribution:", payload["plan_argmax_distribution"])
    if payload["fraction_of_oracle"] is not None:
        print(f"  plan captures {payload['fraction_of_oracle']:.1%} of the oracle timing advantage")
    print(f"\n-> {output}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
