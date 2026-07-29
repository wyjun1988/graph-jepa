"""Fix the NaN-gradient poisoning in the plan-loss collector. Pod trees only.

Symptom: the H6-1 full arm produced loss=nan from epoch 1 while the aux-only
control trained normally, so the aux heads were fine and the plan wiring was not.

Cause: attach_downstream_targets skips a date when its cross-sectional std is
degenerate, leaving NaN in the per-date scale. The collector then computed
`standardized_prediction * deviation + mean`, which puts NaN INSIDE the
autograd graph. Masking afterwards hides it in the forward -- the loss prints
finite -- but the backward still returns a NaN gradient for the masked element,
which poisons every weight on the first optimizer step:

    x = tensor([1., 2.], requires_grad=True)
    y = x * tensor([nan, 3.])
    y[isfinite(y)].sum().backward()   # forward 6.0, x.grad == [nan, 3.]

Fix: sanitize the scale BEFORE it enters the graph. Nodes whose scale is
degenerate get a neutral (mean 0, deviation 1) scale so no NaN is ever
multiplied, and the validity mask -- computed from the ORIGINAL scale -- still
excludes them from the loss. The label is sanitized the same way; it carries no
gradient, but leaving NaN there would make the isfinite checks in
plan_timing_loss meaningless.

Idempotent.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

OLD = """        scale = target_batch.target_downstream_scale.to(
            device=z_pred.device, dtype=z_pred.dtype
        )
        mean = scale[:, 0]
        deviation = scale[:, 1]
        head = self.predict_downstream_auxiliary(z_context, z_pred, rollout_steps)
        standardized_prediction = head[:, PATH_RETURN_TASK_INDEX]
        standardized_target = target_batch.target_downstream.to(
            device=z_pred.device, dtype=z_pred.dtype
        )[:, PATH_RETURN_TASK_INDEX]
        supervised = self._supervision_node_mask(target_batch)
        predicted[int(rollout_steps)] = standardized_prediction * deviation + mean
        realized[int(rollout_steps)] = standardized_target * deviation + mean
        valid[int(rollout_steps)] = (
            supervised
            & torch.isfinite(standardized_target)
            & torch.isfinite(deviation)
            & (deviation > 0)
        )"""

NEW = """        scale = target_batch.target_downstream_scale.to(
            device=z_pred.device, dtype=z_pred.dtype
        )
        mean = scale[:, 0]
        deviation = scale[:, 1]
        head = self.predict_downstream_auxiliary(z_context, z_pred, rollout_steps)
        standardized_prediction = head[:, PATH_RETURN_TASK_INDEX]
        standardized_target = target_batch.target_downstream.to(
            device=z_pred.device, dtype=z_pred.dtype
        )[:, PATH_RETURN_TASK_INDEX]
        supervised = self._supervision_node_mask(target_batch)

        # attach_downstream_targets leaves NaN in the scale for any date whose
        # cross-sectional std was degenerate. Multiplying the head's output by a
        # NaN puts NaN INSIDE the autograd graph: masking it afterwards hides it
        # in the forward but the backward still returns a NaN gradient, which
        # poisons every weight on the first step. Sanitize before the multiply
        # and let the mask -- built from the ORIGINAL scale -- do the excluding.
        usable = (
            torch.isfinite(mean)
            & torch.isfinite(deviation)
            & (deviation > 0)
        )
        safe_mean = torch.where(usable, mean, torch.zeros_like(mean))
        safe_deviation = torch.where(
            usable, deviation, torch.ones_like(deviation)
        )
        safe_target = torch.where(
            torch.isfinite(standardized_target),
            standardized_target,
            torch.zeros_like(standardized_target),
        )
        predicted[int(rollout_steps)] = (
            standardized_prediction * safe_deviation + safe_mean
        )
        realized[int(rollout_steps)] = safe_target * safe_deviation + safe_mean
        valid[int(rollout_steps)] = (
            supervised & usable & torch.isfinite(standardized_target)
        )"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if (root / "ops/prospective_live").is_dir():
        raise SystemExit("refusing to patch the prospective-chain tree")
    target = root / "stock_v2/graph_jepa.py"
    text = target.read_text(encoding="utf-8")
    if "safe_deviation" in text:
        print("already fixed")
        return 0
    if OLD not in text:
        raise SystemExit("collector anchor not found; inspect _collect_plan_horizon")
    target.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print(f"fixed NaN-gradient poisoning -> {sha256(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
