"""Wire the exit-plan timing loss into the trainer. Isolated trees only.

Design: docs/PLAN_TIMING_LOSS_DESIGN_20260717.md
Pure math + unit tests already live in stock_v2/plan_timing.py (20 tests).

This patch only connects them:
  1. graph_jepa.py  -- accept the weight/temperature/permute-seed, collect the
     per-horizon path prediction and realized return inside temporal_multi_loss,
     and add the plan loss once after the horizon loop.
  2. run_real_backtest.py -- CLI flags, pass-through to the model, and carry the
     per-date standardization scale so raw returns can be recovered.

With --downstream-plan-loss-weight 0.0 (the default) every added branch is
skipped and behaviour is byte-identical; §0-2 of the design requires proving
that before any arm runs.

Refuses any tree carrying ops/prospective_live: graph_jepa.py and
run_real_backtest.py are on the daily chain's import path via
build_stale_jepa_rollout_cache.py, and the 06:05 KST cycle rebuilds the
prospective cache from the working tree.

Idempotent.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# --- graph_jepa.py ----------------------------------------------------------

GJ_IMPORT_OLD = "DOWNSTREAM_AUXILIARY_TASKS = ("
GJ_IMPORT_NEW = """from stock_v2.plan_timing import plan_timing_loss

DOWNSTREAM_AUXILIARY_TASKS = ("""

GJ_INIT_OLD = "        downstream_auxiliary_loss_weight: float = 0.0,"
GJ_INIT_NEW = """        downstream_auxiliary_loss_weight: float = 0.0,
        downstream_plan_loss_weight: float = 0.0,
        plan_temperature: float = 0.01,
        plan_permute_seed: int = 0,"""

GJ_STORE_OLD = """        if (
            not math.isfinite(downstream_auxiliary_loss_weight)
            or downstream_auxiliary_loss_weight < 0.0
        ):"""
GJ_STORE_NEW = """        self.downstream_plan_loss_weight = float(downstream_plan_loss_weight)
        self.plan_temperature = float(plan_temperature)
        self.plan_permute_seed = int(plan_permute_seed)
        if (
            not math.isfinite(self.downstream_plan_loss_weight)
            or self.downstream_plan_loss_weight < 0.0
        ):
            raise ValueError(
                "downstream_plan_loss_weight must be finite and non-negative"
            )
        if self.downstream_plan_loss_weight > 0.0 and self.plan_temperature <= 0.0:
            raise ValueError("plan_temperature must be positive")
        if (
            not math.isfinite(downstream_auxiliary_loss_weight)
            or downstream_auxiliary_loss_weight < 0.0
        ):"""

# Collect the path prediction and the realized path return for each horizon.
# Both are recovered to RAW return units by inverting the per-date
# standardization that attach_downstream_targets applied; the standardized
# values are not economics and their weighted sum would be meaningless.
GJ_COLLECT_OLD = """            downstream_auxiliary_loss = self._downstream_auxiliary_loss(
                z_context,
                z_pred_for_step,
                target_batch,
                rollout_steps=int(steps),
            )"""
GJ_COLLECT_NEW = """            downstream_auxiliary_loss = self._downstream_auxiliary_loss(
                z_context,
                z_pred_for_step,
                target_batch,
                rollout_steps=int(steps),
            )
            if self.downstream_plan_loss_weight > 0.0:
                self._collect_plan_horizon(
                    plan_predicted,
                    plan_realized,
                    plan_valid,
                    z_context=z_context,
                    z_pred=z_pred_for_step,
                    target_batch=target_batch,
                    rollout_steps=int(steps),
                )"""

GJ_INIT_COLLECT_OLD = """        losses = []"""
GJ_INIT_COLLECT_NEW = """        plan_predicted: Dict[int, Tensor] = {}
        plan_realized: Dict[int, Tensor] = {}
        plan_valid: Dict[int, Tensor] = {}
        losses = []"""

GJ_ADD_OLD = """        loss = (
            temporal_loss
            + self.current_imputation_loss_weight
            * self.state_loss_weight
            * current_imputation_loss
        )"""
GJ_ADD_NEW = """        loss = (
            temporal_loss
            + self.current_imputation_loss_weight
            * self.state_loss_weight
            * current_imputation_loss
        )
        plan_diagnostics: Dict[str, float] = {}
        if self.downstream_plan_loss_weight > 0.0 and plan_predicted:
            plan_loss, plan_diagnostics = plan_timing_loss(
                plan_predicted,
                plan_realized,
                plan_valid,
                temperature=self.plan_temperature,
                permute_seed=self.plan_permute_seed,
            )
            loss = loss + self.downstream_plan_loss_weight * plan_loss"""

GJ_METRICS_OLD = """        metrics = {
            "loss": float(loss.detach().cpu()),"""
GJ_METRICS_NEW = """        metrics = {
            "plan_loss_weight": self.downstream_plan_loss_weight,
            "loss": float(loss.detach().cpu()),"""

GJ_HELPER = '''

    def _collect_plan_horizon(
        self,
        predicted: Dict[int, Tensor],
        realized: Dict[int, Tensor],
        valid: Dict[int, Tensor],
        *,
        z_context: Tensor,
        z_pred: Tensor,
        target_batch: GraphBatch,
        rollout_steps: int,
    ) -> None:
        """Stash one horizon's raw path prediction and realized return.

        `attach_downstream_targets` standardizes each date's targets as
        (x - mean) / std, so both the stored target and the head's prediction
        live in standardized space. Multiplying by std and adding mean recovers
        the raw return exactly, and reusing the same scale for both keeps the
        prediction and the label on one scale by construction -- safer than
        recomputing raw returns from prices in a second place.
        """

        if target_batch.target_downstream is None:
            raise ValueError("plan loss requires target_downstream")
        if target_batch.target_downstream_scale is None:
            raise ValueError(
                "plan loss requires target_downstream_scale; run "
                "attach_downstream_targets with the plan flag enabled"
            )
        scale = target_batch.target_downstream_scale.to(
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
        )
'''

GJ_HELPER_ANCHOR = "    def _downstream_auxiliary_loss("

GJ_TASK_INDEX_OLD = '''DOWNSTREAM_AUXILIARY_TASKS = (
    "path_return",'''
GJ_TASK_INDEX_NEW = '''PATH_RETURN_TASK_INDEX = 0

DOWNSTREAM_AUXILIARY_TASKS = (
    "path_return",'''

GJ_BATCH_FIELD_OLD = "    target_downstream: Optional[Tensor] = None"
GJ_BATCH_FIELD_NEW = """    target_downstream: Optional[Tensor] = None
    target_downstream_scale: Optional[Tensor] = None"""

GJ_BATCH_TO_OLD = """            target_downstream=(
                None
                if self.target_downstream is None
                else self.target_downstream.to(device)
            ),"""
GJ_BATCH_TO_NEW = """            target_downstream=(
                None
                if self.target_downstream is None
                else self.target_downstream.to(device)
            ),
            target_downstream_scale=(
                None
                if self.target_downstream_scale is None
                else self.target_downstream_scale.to(device)
            ),"""

# --- run_real_backtest.py ---------------------------------------------------

RB_ARG_OLD = '''    parser.add_argument(
        "--downstream-auxiliary-loss-weight",'''
RB_ARG_NEW = '''    parser.add_argument("--downstream-plan-loss-weight", type=float, default=0.0)
    parser.add_argument("--plan-temperature", type=float, default=0.01)
    parser.add_argument("--plan-permute-seed", type=int, default=0)
    parser.add_argument(
        "--downstream-auxiliary-loss-weight",'''

RB_MODEL_OLD = "        downstream_auxiliary_loss_weight=("
RB_MODEL_NEW = """        downstream_plan_loss_weight=float(
            getattr(args, "downstream_plan_loss_weight", 0.0)
        ),
        plan_temperature=float(getattr(args, "plan_temperature", 0.01)),
        plan_permute_seed=int(getattr(args, "plan_permute_seed", 0) or 0),
        downstream_auxiliary_loss_weight=("""

# The plan loss needs the downstream targets attached, so widen the gate.
RB_ATTACH_OLD = """                        if args.downstream_auxiliary_loss_weight > 0.0:
                            target_batch = attach_downstream_targets("""
RB_ATTACH_NEW = """                        if (
                            args.downstream_auxiliary_loss_weight > 0.0
                            or args.downstream_plan_loss_weight > 0.0
                        ):
                            target_batch = attach_downstream_targets("""

RB_SCALE_OLD = """            mean = float(target[valid].mean())
            std = float(target[valid].std())
            if not np.isfinite(std) or std < 1e-12:
                continue
            values[position, :stock_count, task_index] = (
                (target - mean) / std
            ).astype(np.float32)"""
RB_SCALE_NEW = """            mean = float(target[valid].mean())
            std = float(target[valid].std())
            if not np.isfinite(std) or std < 1e-12:
                continue
            values[position, :stock_count, task_index] = (
                (target - mean) / std
            ).astype(np.float32)
            if task_index == 0:
                # Carry the path task's per-date scale so the plan loss can
                # invert the standardization back to raw return units.
                scales[position, :stock_count, 0] = np.float32(mean)
                scales[position, :stock_count, 1] = np.float32(std)"""

RB_SCALE_INIT_OLD = """    values = np.full((len(steps), node_count, task_count), np.nan, dtype=np.float32)"""
RB_SCALE_INIT_NEW = """    values = np.full((len(steps), node_count, task_count), np.nan, dtype=np.float32)
    scales = np.full((len(steps), node_count, 2), np.nan, dtype=np.float32)"""

RB_ATTACH_RETURN_OLD = """    batch.target_downstream = torch.from_numpy("""
RB_ATTACH_RETURN_NEW = """    batch.target_downstream_scale = torch.from_numpy(
        scales.reshape(-1, 2)
    )
    batch.target_downstream = torch.from_numpy("""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply(path: Path, pairs, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"already patched: {path.name}")
        return
    for name, old, _new in pairs:
        if old not in text:
            raise SystemExit(f"anchor not found in {path.name}: {name}")
    for _name, old, new in pairs:
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print(f"patched {path.name} -> {sha256(path)}")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if (root / "ops/prospective_live").is_dir():
        raise SystemExit(
            "refusing to patch a tree that carries the prospective chain: "
            "graph_jepa.py and run_real_backtest.py are on the 06:05 cycle's import path"
        )
    graph = root / "stock_v2/graph_jepa.py"
    backtest = root / "scripts/run_real_backtest.py"
    for path in (graph, backtest):
        if not path.is_file():
            raise SystemExit(f"not a stock-v2 tree: {root}")

    apply(
        graph,
        [
            ("task index", GJ_TASK_INDEX_OLD, GJ_TASK_INDEX_NEW),
            ("plan_timing import", GJ_IMPORT_OLD, GJ_IMPORT_NEW),
            ("GraphBatch field", GJ_BATCH_FIELD_OLD, GJ_BATCH_FIELD_NEW),
            ("GraphBatch.to", GJ_BATCH_TO_OLD, GJ_BATCH_TO_NEW),
            ("init signature", GJ_INIT_OLD, GJ_INIT_NEW),
            ("init validation", GJ_STORE_OLD, GJ_STORE_NEW),
            ("helper", GJ_HELPER_ANCHOR, GJ_HELPER.rstrip() + "\n\n" + GJ_HELPER_ANCHOR),
            ("collector init", GJ_INIT_COLLECT_OLD, GJ_INIT_COLLECT_NEW),
            ("per-horizon collect", GJ_COLLECT_OLD, GJ_COLLECT_NEW),
            ("loss assembly", GJ_ADD_OLD, GJ_ADD_NEW),
            ("metrics", GJ_METRICS_OLD, GJ_METRICS_NEW),
        ],
        marker="downstream_plan_loss_weight",
    )
    apply(
        backtest,
        [
            ("scale init", RB_SCALE_INIT_OLD, RB_SCALE_INIT_NEW),
            ("scale capture", RB_SCALE_OLD, RB_SCALE_NEW),
            ("attach return", RB_ATTACH_RETURN_OLD, RB_ATTACH_RETURN_NEW),
            ("attach gate", RB_ATTACH_OLD, RB_ATTACH_NEW),
            ("cli flags", RB_ARG_OLD, RB_ARG_NEW),
            ("model construction", RB_MODEL_OLD, RB_MODEL_NEW),
        ],
        marker="downstream_plan_loss_weight",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
