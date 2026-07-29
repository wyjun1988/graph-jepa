from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_post_impact_clock_bucket_increment import sha256_file
from scripts.audit_post_impact_rank_adapter_multifold import evaluate


AUDIT_ROLE = "post_impact_rank_adapter_seed_stability_audit"
EVIDENCE_CLASS = "post_selection_seed_diagnostic"


def classify(summary: dict[str, object]) -> dict[str, object]:
    selected = summary.get("selected_candidate")
    passed = selected == "aligned"
    summary.update(
        {
            "role": AUDIT_ROLE,
            "evidence_class": EVIDENCE_CLASS,
            "counts_as_primary_forward_evidence": False,
            "changes_frozen_prospective_candidate": False,
            "decision": (
                "seed_replication_pass" if passed else "seed_replication_fail"
            ),
            "next_gate": (
                "retain_existing_candidate_and_collect_forward_evidence"
                if passed
                else "retain_existing_candidate_but_flag_seed_instability"
            ),
            "live_orders_allowed": False,
            "promotion_eligible": False,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a post-selection rank-adapter seed replication without "
            "promoting it to prospective evidence."
        )
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, daily = evaluate(Path(args.contract), Path(args.artifact_root))
    summary = classify(summary)
    output_dir.mkdir(parents=True)
    daily_path = output_dir / "daily_paired_deltas.csv"
    daily.to_csv(daily_path, index=False)
    summary["daily_paired_deltas_sha256"] = sha256_file(daily_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "selected_candidate": summary["selected_candidate"],
                "evidence_class": EVIDENCE_CLASS,
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
