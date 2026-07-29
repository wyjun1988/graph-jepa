from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


REPAIR_CONTRACT = "post_impact_clock_context_metadata_repair_v1"
AUDIT_FIELDS = ("context_map_audit", "latent_context_map_audit")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in payload.items() if key != "contract"}


def repair_payload(
    clock: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    original_clock_sha256: str,
    reference_summary_sha256: str,
) -> dict[str, Any]:
    if clock.get("live_orders_allowed") is not False or summary.get(
        "live_orders_allowed"
    ) is not False:
        raise ValueError("clock metadata repair requires research-only inputs")
    parity = clock.get("reference_inference_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError("clock metadata repair requires inference parity")
    if clock.get("variant") != summary.get("variant"):
        raise ValueError("clock and summary variants differ")
    if clock.get("daily_context_placebo_mode", "none") != summary.get(
        "daily_context_placebo_mode", "none"
    ):
        raise ValueError("clock and summary context modes differ")
    corrected = json.loads(json.dumps(clock))
    repaired_fields: list[str] = []
    for name in AUDIT_FIELDS:
        clock_audit = clock.get(name)
        summary_audit = summary.get(name)
        if not isinstance(clock_audit, Mapping) or not isinstance(
            summary_audit, Mapping
        ):
            raise ValueError(f"context audit is missing: {name}")
        expected_contract = summary_audit.get("contract")
        if not isinstance(expected_contract, str) or not expected_contract:
            raise ValueError(f"summary context contract is missing: {name}")
        existing_contract = clock_audit.get("contract")
        if existing_contract not in {None, expected_contract}:
            raise ValueError(f"clock context contract conflicts: {name}")
        if _without_contract(clock_audit) != _without_contract(summary_audit):
            raise ValueError(f"clock context audit values differ: {name}")
        corrected[name]["contract"] = expected_contract
        if existing_contract is None:
            repaired_fields.append(name)
    corrected["context_metadata_repair"] = {
        "contract": REPAIR_CONTRACT,
        "original_clock_sha256": str(original_clock_sha256),
        "reference_summary_sha256": str(reference_summary_sha256),
        "repaired_fields": repaired_fields,
        "prediction_metrics_changed": False,
    }
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add omitted context-contract labels to a post-impact clock report."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    summary_path = Path(args.reference_summary)
    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError(f"refusing to overwrite repaired clock: {output_path}")
    clock = json.loads(input_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    repaired = repair_payload(
        clock,
        summary,
        original_clock_sha256=sha256_file(input_path),
        reference_summary_sha256=sha256_file(summary_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(repaired, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": sha256_file(output_path),
                "prediction_metrics_changed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
