from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_structuring import (
    OUTPUT_SCHEMA_VERSION,
    PROMPT_VERSION,
    build_messages,
    build_repair_messages,
    compatible_resume_ids,
    deterministic_sample,
    materialize_event,
    parse_json_content,
    validate_labels,
)
from stock_v2.news_contract import (
    LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY,
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield payload


def compatible_completed_ids(
    path: Path,
    args: argparse.Namespace,
    target_records: list[dict[str, Any]],
) -> set[str]:
    if not path.exists():
        return set()
    return compatible_resume_ids(
        list(iter_jsonl(path)),
        target_records,
        required_lineage={
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "prompt_version": PROMPT_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "inference_engine": "transformers_direct",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structure a frozen news queue with deterministic Hugging Face GPU batches."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--repair-retries", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=20260712)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def build_result(
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    labels: dict[str, Any] | None,
    error: str,
    attempts: int,
    raw_output: str,
    runtime_versions: dict[str, str],
) -> dict[str, Any]:
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "queue_id": record["queue_id"],
        "article_id": record["article_id"],
        "ticker": record["ticker"],
        "effective_session": record["effective_session"],
        "input_sha256": record["input_sha256"],
        "llm_used": labels is not None,
        "llm_error": error,
        "lineage": {
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "served_model": str(Path(args.model_path).resolve()),
            "prompt_version": PROMPT_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "inference_engine": "transformers_direct",
            "input_hash_policy": str(
                record.get("input_hash_policy") or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
            ),
            "runtime_versions": runtime_versions,
            "decoding": {
                "do_sample": False,
                "enable_thinking": False,
                "max_new_tokens": args.max_new_tokens,
                "dtype": args.dtype,
                "configured_batch_size": args.batch_size,
            },
            "generation_attempts": attempts,
            "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        },
    }
    if labels is not None:
        result["labels"] = labels
        result["event"] = materialize_event(record, labels)
    else:
        result["raw_output_excerpt"] = raw_output[:1000]
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.repair_retries < 0:
        raise ValueError("batch-size/max-new-tokens must be positive and retries nonnegative")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available() or not str(args.device).startswith("cuda"):
        raise RuntimeError("this production structurer requires an explicit CUDA device")
    torch.manual_seed(0)
    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"model snapshot not found: {model_path}")
    if model_path.name != args.model_revision:
        raise ValueError("model-path directory must equal the immutable model revision")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_records = list(iter_jsonl(input_path))
    selected_records = deterministic_sample(all_records, args.sample_size, args.sample_seed)
    done = (
        compatible_completed_ids(output_path, args, selected_records)
        if args.resume
        else set()
    )
    records = [
        row for row in selected_records if str(row.get("queue_id") or "") not in done
    ]

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    if getattr(processor, "tokenizer", None) is None:
        raise RuntimeError("Qwen processor did not expose a tokenizer")
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=args.device,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).eval()
    runtime_versions = {
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda": str(torch.version.cuda),
    }

    def generate(conversations: list[list[dict[str, str]]]) -> list[str]:
        prompts = [
            processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for conversation in conversations
        ]
        encoded = processor(text=prompts, padding=True, return_tensors="pt")
        encoded = {
            key: value.to(args.device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )
        return processor.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)

    oom_fallbacks = 0
    minimum_effective_batch = args.batch_size

    def generate_resilient(conversations: list[list[dict[str, str]]]) -> list[str]:
        nonlocal oom_fallbacks, minimum_effective_batch
        minimum_effective_batch = min(minimum_effective_batch, len(conversations))
        out_of_memory = False
        try:
            return generate(conversations)
        except torch.OutOfMemoryError:
            if len(conversations) <= 1:
                raise
            oom_fallbacks += 1
            out_of_memory = True
        if out_of_memory:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        midpoint = len(conversations) // 2
        return generate_resilient(conversations[:midpoint]) + generate_resilient(
            conversations[midpoint:]
        )

    mode = "a" if args.resume and output_path.exists() else "w"
    started = time.perf_counter()
    processed = ok = errors = 0
    with output_path.open(mode, encoding="utf-8") as output:
        for offset in range(0, len(records), args.batch_size):
            batch = records[offset : offset + args.batch_size]
            pending = list(range(len(batch)))
            conversations = [build_messages(record) for record in batch]
            raw_outputs = [""] * len(batch)
            parsed: list[dict[str, Any] | None] = [None] * len(batch)
            validation_errors = [""] * len(batch)
            attempts = [0] * len(batch)
            for attempt in range(args.repair_retries + 1):
                if not pending:
                    break
                outputs = generate_resilient([conversations[index] for index in pending])
                next_pending: list[int] = []
                for index, raw_output in zip(pending, outputs):
                    attempts[index] += 1
                    raw_outputs[index] = raw_output
                    try:
                        parsed[index] = validate_labels(parse_json_content(raw_output))
                        validation_errors[index] = ""
                    except Exception as exc:
                        validation_errors[index] = f"{type(exc).__name__}: {str(exc)[:500]}"
                        if attempt < args.repair_retries:
                            conversations[index] = build_repair_messages(
                                batch[index],
                                raw_output,
                                validation_errors[index],
                                repair_attempt=attempt + 1,
                            )
                            next_pending.append(index)
                pending = next_pending

            for index, record in enumerate(batch):
                result = build_result(
                    record,
                    args,
                    labels=parsed[index],
                    error=validation_errors[index],
                    attempts=attempts[index],
                    raw_output=raw_outputs[index],
                    runtime_versions=runtime_versions,
                )
                output.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                processed += 1
                ok += int(parsed[index] is not None)
                errors += int(parsed[index] is None)
            output.flush()
            if processed % max(1, args.print_every) < len(batch) or processed == len(records):
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "ok": ok,
                            "errors": errors,
                            "remaining": len(records) - processed,
                            "rows_per_second": round(processed / elapsed, 3),
                            "gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                        }
                    ),
                    flush=True,
                )

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "selected": len(records),
        "ok": ok,
        "errors": errors,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "oom_fallbacks": oom_fallbacks,
        "minimum_effective_batch": minimum_effective_batch,
        "runtime_versions": runtime_versions,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
