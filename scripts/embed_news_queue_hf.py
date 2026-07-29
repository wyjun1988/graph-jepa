from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("queue_id"):
                raise ValueError(f"invalid queue row at {path}:{line_number}")
            yield row


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed a frozen Korean news queue on CUDA.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--embedding-dimension", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--print-every", type=int, default=5000)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("news embedding requires an explicit CUDA device")
    model_path = Path(args.model_path).resolve()
    if not model_path.exists() or model_path.name != args.model_revision:
        raise ValueError("model-path must be an existing immutable revision directory")
    input_path = Path(args.input)
    records = list(iter_jsonl(input_path))
    if not records:
        raise ValueError("news embedding queue is empty")
    texts = [
        f"회사: {row.get('company_name', '')}\n제목: {row.get('title', '')}\n요약: {row.get('summary', '')}"
        for row in records
    ]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    hidden_size = int(getattr(model.config, "hidden_size"))
    if not 1 <= args.embedding_dimension <= hidden_size:
        raise ValueError("embedding-dimension exceeds model hidden size")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    embeddings = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), args.embedding_dimension),
    )

    def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
            return hidden_states[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), lengths]

    started = time.perf_counter()
    for offset in range(0, len(texts), args.batch_size):
        batch = tokenizer(
            texts[offset : offset + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(args.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)
            pooled = last_token_pool(output.last_hidden_state, batch["attention_mask"])
            pooled = functional.normalize(pooled[:, : args.embedding_dimension].float(), p=2, dim=1)
        embeddings[offset : offset + len(pooled)] = pooled.cpu().numpy().astype(np.float16)
        processed = offset + len(pooled)
        if processed % max(1, args.print_every) < args.batch_size or processed == len(texts):
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "remaining": len(texts) - processed,
                        "rows_per_second": round(processed / elapsed, 2),
                        "gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                    }
                ),
                flush=True,
            )
    embeddings.flush()
    del embeddings
    temporary.replace(output_path)
    manifest = {
        "schema_version": 1,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "rows": len(records),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_path": str(model_path),
        "embedding_dimension": args.embedding_dimension,
        "max_length": args.max_length,
        "dtype": "float16",
        "normalized": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda": str(torch.version.cuda),
        },
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

