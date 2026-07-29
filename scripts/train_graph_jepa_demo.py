from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from stock_v2.graph_jepa import StockGraphJEPA
from stock_v2.synthetic_market import generate_synthetic_market, make_snapshot


def train_demo(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    market = generate_synthetic_market(
        num_steps=args.steps,
        num_nodes=args.nodes,
        num_features=args.features,
        seed=args.seed,
    )
    model = StockGraphJEPA(
        num_features=args.features,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        ema_decay=args.ema_decay,
        state_loss_weight=args.state_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_steps = range(20, args.steps - 5)
    last_metrics = {}
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for step in train_steps:
            batch = make_snapshot(market, step=step, hide_ratio=args.hide_ratio, seed=args.seed + epoch + step).to(device)
            loss, metrics = model.loss(batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

            total += metrics["loss"]
            count += 1
            last_metrics = metrics

        print(
            f"epoch={epoch:02d} "
            f"loss={total / max(count, 1):.4f} "
            f"latent={last_metrics['latent_loss']:.4f} "
            f"state={last_metrics['state_loss']:.4f} "
            f"masked_mae={last_metrics['masked_mae']:.4f}"
        )

    eval_batch = make_snapshot(market, step=args.steps - 2, hide_ratio=args.hide_ratio, seed=args.seed + 999).to(device)
    with torch.no_grad():
        prediction = model.infer_unobserved_state(eval_batch)
        hidden = eval_batch.feature_mask < 0.5
        mae = (prediction[hidden] - eval_batch.node_features[hidden]).abs().mean().item()

    print(f"eval_masked_mae={mae:.4f}")
    print("sample_hidden_predictions:")
    coords = hidden.nonzero(as_tuple=False)[:8]
    for node_idx, feature_idx in coords.tolist():
        name = market.feature_names[feature_idx]
        pred = prediction[node_idx, feature_idx].item()
        target = eval_batch.node_features[node_idx, feature_idx].item()
        print(f"  node={node_idx:02d} feature={name:<14} pred={pred:+.3f} target={target:+.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny Graph-JEPA stock-state demo.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hide-ratio", type=float, default=0.30)
    parser.add_argument("--ema-decay", type=float, default=0.98)
    parser.add_argument("--state-loss-weight", type=float, default=0.35)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train_demo(parse_args())
