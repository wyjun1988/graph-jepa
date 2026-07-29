"""Differentiable exit-plan loss: score the plan, not a fixed-horizon regression.

Design: docs/PLAN_TIMING_LOSS_DESIGN_20260717.md

Every return metric in this project fixes the holding period (h1/h2/h3/h5/h10 IC,
`--horizon 5` backtests). The user's principle is the opposite: a position is
opened with a *plan* -- "bought on a 7-day state change" or "bought to sell
tomorrow" -- and the return comes from executing that plan. A fixed-h regression
cannot express it.

Here the model's own path predictions become the plan. For horizons H and
predicted path returns r_hat(h):

    w(h) = softmax( r_hat(h) / tau )                # the plan, differentiable
    A    = sum_h w(h) * r(h)  -  r(h_max)           # advantage over hold-to-end
    loss = -mean(A)

Gradient flows through w back into r_hat, so the path head is tuned to produce
predictions that make *good decisions*, not merely accurate regressions.

Why advantage over hold-to-h_max rather than the raw plan return: in a drifting
market the raw plan return is maximised by always holding to the end, which
would reward drift rather than timing. Dividing by holding days instead would
bias toward short exits. The advantage cancels both the drift and the round-trip
cost (both arms trade once), leaving timing skill.

This module is pure tensor math with no project imports so it can be unit tested
without a GPU, a checkpoint, or a data release.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor

__all__ = ["plan_weights", "plan_timing_loss"]


def plan_weights(
    predicted_by_horizon: Mapping[int, Tensor],
    horizons: Sequence[int],
    *,
    temperature: float,
    uniform: bool = False,
) -> Tensor:
    """Softmax plan over horizons. Returns [nodes, len(horizons)].

    `uniform=True` returns constant weights. It is a DIAGNOSTIC, not the
    placebo arm: constant weights carry no gradient back to the predictions, so
    training with it is identical to training without the loss at all. The
    placebo arm is `permute_seed` in `plan_timing_loss`, which keeps the
    gradient path intact and destroys only the node-to-plan pairing.
    """

    if not horizons:
        raise ValueError("plan requires at least one horizon")
    if float(temperature) <= 0.0:
        raise ValueError("plan temperature must be positive")
    stacked = torch.stack([predicted_by_horizon[int(h)] for h in horizons], dim=1)
    if uniform:
        return torch.full_like(stacked, 1.0 / len(horizons))
    return torch.softmax(stacked / float(temperature), dim=1)


def plan_timing_loss(
    predicted_by_horizon: Mapping[int, Tensor],
    realized_by_horizon: Mapping[int, Tensor],
    valid_by_horizon: Mapping[int, Tensor],
    *,
    temperature: float,
    uniform: bool = False,
    permute_seed: int = 0,
    buy_sell: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Loss and diagnostics for the exit plan.

    All mappings are keyed by horizon and hold [nodes] tensors. `predicted` and
    `realized` must both be in RAW return units -- standardized values are not
    economics and their weighted sum means nothing.

    Validity is intersected across every horizon on purpose. Masking per horizon
    would let a node that only has short-horizon labels (a name that stops
    trading, say) contribute exclusively to short-exit plans, so missingness
    would masquerade as timing skill.

    `permute_seed > 0` is the placebo arm: each node's plan is scored against
    another node's realized path (a deterministic permutation of the surviving
    rows). The gradient path, the loss magnitude, and the weight distribution
    are all preserved; only the node-to-plan pairing is destroyed. An advantage
    that survives against this comes from knowing when to exit THIS name.
    `uniform=True` cannot serve that role because constant weights detach the
    gradient entirely, making the arm identical to no loss at all.

    Returns a zero loss with `nodes=0` when nothing survives the mask, so a
    thin batch degrades quietly instead of producing NaN.
    """

    horizons = sorted(int(h) for h in predicted_by_horizon)
    if not horizons:
        raise ValueError("plan timing loss requires at least one horizon")
    for name, mapping in (
        ("realized", realized_by_horizon),
        ("valid", valid_by_horizon),
    ):
        missing = [h for h in horizons if int(h) not in mapping]
        if missing:
            raise ValueError(f"{name} is missing horizons {missing}")

    reference = predicted_by_horizon[horizons[0]]
    valid = torch.ones_like(reference, dtype=torch.bool)
    for h in horizons:
        valid = (
            valid
            & valid_by_horizon[h].to(dtype=torch.bool, device=reference.device)
            & torch.isfinite(predicted_by_horizon[h])
            & torch.isfinite(realized_by_horizon[h])
        )

    node_count = int(valid.sum())
    if node_count == 0:
        zero = reference.sum() * 0.0
        return zero, {"nodes": 0}

    predicted = {h: predicted_by_horizon[h][valid] for h in horizons}
    realized = torch.stack([realized_by_horizon[h][valid] for h in horizons], dim=1)

    if int(permute_seed) > 0:
        generator = torch.Generator(device="cpu").manual_seed(
            int(permute_seed) * 1_000_003 + node_count
        )
        order = torch.randperm(node_count, generator=generator).to(realized.device)
        realized = realized[order]

    hold_return = realized[:, horizons.index(max(horizons))]
    if buy_sell:
        # Entry menu: column 0 = buy now (t+1 open, zero base), columns 1..H =
        # close(t+h_k). Exit menu: close(t+h_x). A pair is valid when the exit is
        # strictly after the entry in time (entry 0 precedes every close; entry k
        # precedes exit x iff x > k). Predicted close-to-close ranks the pairs.
        node_n = realized.shape[0]
        zeros = realized.new_zeros(node_n, 1)
        pred_close = torch.cat([zeros] + [predicted[h].unsqueeze(1) for h in horizons], dim=1)
        real_close = torch.cat([zeros, realized], dim=1)
        entry_idx, exit_idx = [], []
        span = len(horizons)
        for entry in range(span + 1):
            for exit_col in range(1, span + 1):
                if entry == 0 or exit_col > entry:
                    entry_idx.append(entry)
                    exit_idx.append(exit_col)
        entry_t = torch.tensor(entry_idx, device=realized.device)
        exit_t = torch.tensor(exit_idx, device=realized.device)
        pred_pair = pred_close[:, exit_t] - pred_close[:, entry_t]
        real_pair = (1.0 + real_close[:, exit_t]) / (1.0 + real_close[:, entry_t]) - 1.0
        weights = torch.softmax(pred_pair / float(temperature), dim=1)
        plan_return = (weights * real_pair).sum(dim=1)
    else:
        weights = plan_weights(
            predicted, horizons, temperature=temperature, uniform=uniform
        )
        plan_return = (weights * realized).sum(dim=1)
    advantage = plan_return - hold_return
    loss = -advantage.mean()

    with torch.no_grad():
        entropy = -(weights.clamp_min(1e-12).log() * weights).sum(dim=1).mean()
        if buy_sell:
            histogram = {"plan_buy_sell": 1.0, "plan_pairs": float(weights.shape[1])}
        else:
            chosen = weights.argmax(dim=1)
            histogram = {
                f"plan_argmax_h{h}": float((chosen == index).float().mean())
                for index, h in enumerate(horizons)
            }
        diagnostics: dict[str, float] = {
            "nodes": node_count,
            "plan_advantage_mean": float(advantage.mean()),
            "plan_return_mean": float(plan_return.mean()),
            "hold_return_mean": float(hold_return.mean()),
            "plan_weight_entropy": float(entropy),
            "plan_oracle_advantage_mean": float(
                (realized.max(dim=1).values - hold_return).mean()
            ),
        }
        diagnostics.update(histogram)
    return loss, diagnostics
