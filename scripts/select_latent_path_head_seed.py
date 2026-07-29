from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("inputs must use LABEL=SUMMARY_JSON")
    return label.strip(), Path(raw_path.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fold(values: Sequence[tuple[str, Path]], fold: str) -> list[dict[str, Any]]:
    if len(values) < 2:
        raise ValueError(f"{fold} requires at least two seed candidates")
    if len({label for label, _path in values}) != len(values):
        raise ValueError(f"{fold} labels must be unique")
    rows = []
    for label, path in values:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("live_orders_allowed") is not False:
            raise ValueError(f"{fold} {label} is not a complete research-only head")
        if payload.get("fold2_used_for_selection") is not False:
            raise ValueError(f"{fold} {label} violates the selection contract")
        validation = float(payload["best_validation_path_ic"])
        test_ic = float(payload["weighted_path_ic"])
        if not math.isfinite(validation) or not math.isfinite(test_ic):
            raise ValueError(f"{fold} {label} contains non-finite scores")
        rows.append(
            {
                "label": label,
                "summary_path": str(path),
                "summary_sha256": sha256_file(path),
                "parent_model_sha256": payload["parent_model_sha256"],
                "train_data_manifest_sha256": payload["train_data_manifest_sha256"],
                "fit_dates": int(payload["fit_dates"]),
                "validation_dates": int(payload["validation_dates"]),
                "test_dates": int(payload["test_dates"]),
                "best_validation_path_ic": validation,
                "weighted_test_path_ic": test_ic,
            }
        )
    signatures = {
        (
            row["parent_model_sha256"],
            row["train_data_manifest_sha256"],
            row["fit_dates"],
            row["validation_dates"],
            row["test_dates"],
        )
        for row in rows
    }
    if len(signatures) != 1:
        raise ValueError(f"{fold} candidates do not share the same parent and windows")
    return rows


def select_seed(
    fold1_values: Sequence[tuple[str, Path]],
    fold2_values: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    fold1 = load_fold(fold1_values, "fold1")
    fold2 = load_fold(fold2_values, "fold2")
    labels1 = {row["label"] for row in fold1}
    labels2 = {row["label"] for row in fold2}
    if labels1 != labels2:
        raise ValueError("fold1 and fold2 seed labels do not align")
    selected = max(
        fold1,
        key=lambda row: (row["best_validation_path_ic"], row["label"]),
    )
    fold2_by_label = {row["label"]: row for row in fold2}
    return {
        "status": "complete",
        "role": "fold1_validation_only_latent_path_head_seed_selection",
        "selection_policy": {
            "selection_fold": "fold1_validation_only",
            "criterion": "maximum best_validation_path_ic",
            "fold1_test_used_for_selection": False,
            "fold2_used_for_selection": False,
        },
        "selected_label": selected["label"],
        "fold1_candidates": fold1,
        "fold2_candidates": fold2,
        "selected_fold1_confirmation": selected,
        "selected_fold2_confirmation": fold2_by_label[selected["label"]],
        "live_orders_allowed": False,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Latent Path Head Seed Selection",
        "",
        "Selection uses Fold 1 validation IC only. Both test windows are confirmation-only.",
        "",
        f"Selected seed: `{payload['selected_label']}`",
        "",
        "| Fold | Seed | Validation IC | Test IC |",
        "|---|---:|---:|---:|",
    ]
    for fold, rows in (
        ("Fold 1", payload["fold1_candidates"]),
        ("Fold 2", payload["fold2_candidates"]),
    ):
        for row in rows:
            lines.append(
                f"| {fold} | {row['label']} | {row['best_validation_path_ic']:.6f} | "
                f"{row['weighted_test_path_ic']:.6f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one latent path head seed using Fold 1 validation only."
    )
    parser.add_argument("--fold1", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--fold2", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    payload = select_seed(args.fold1, args.fold2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps({"selected_label": payload["selected_label"]}))


if __name__ == "__main__":
    main()
