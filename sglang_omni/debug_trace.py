from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def _trace_dir() -> str | None:
    return os.environ.get("QWEN3_TRACE_DIR")


def _trace_request_filter() -> str:
    return os.environ.get("QWEN3_TRACE_REQUEST_FILTER", "")


def _enabled_for(request_id: str) -> bool:
    trace_dir = _trace_dir()
    if not trace_dir:
        return False
    request_filter = _trace_request_filter()
    return (not request_filter) or (request_filter in request_id)


def _ensure_trace_dir() -> Path | None:
    trace_dir = _trace_dir()
    if not trace_dir:
        return None
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _round_list(values: torch.Tensor, count: int = 8) -> list[float]:
    flat = values.detach().float().reshape(-1).cpu()
    return [round(float(v), 6) for v in flat[:count].tolist()]


def summarize_tensor(value: Any, *, head: int = 8) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach().cpu()
    as_float = tensor.float()
    raw_bytes = as_float.numpy().tobytes()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "sum": round(float(as_float.sum().item()), 6),
        "norm": round(float(as_float.norm().item()), 6),
        "head": _round_list(tensor, count=head),
    }


def summarize_logits_row(logits: Any, *, topk: int = 8) -> dict[str, Any] | None:
    if not isinstance(logits, torch.Tensor):
        return None
    row = logits.detach().float().reshape(-1).cpu()
    if row.numel() == 0:
        return None
    k = min(int(topk), int(row.numel()))
    top_vals, top_ids = torch.topk(row, k=k)
    return {
        "sha256": hashlib.sha256(row.numpy().tobytes()).hexdigest(),
        "argmax_id": int(top_ids[0].item()),
        "argmax_logit": round(float(top_vals[0].item()), 6),
        "top_ids": [int(v) for v in top_ids.tolist()],
        "top_vals": [round(float(v), 6) for v in top_vals.tolist()],
    }


def append_trace(stage: str, request_id: str, event: str, **payload: Any) -> None:
    if not _enabled_for(request_id):
        return
    trace_dir = _ensure_trace_dir()
    if trace_dir is None:
        return
    record = {
        "ts": round(time.time(), 6),
        "stage": stage,
        "request_id": request_id,
        "event": event,
    }
    record.update(payload)
    with (trace_dir / f"{stage}.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_ar_trace(
    stage: str,
    requests: list[Any],
    logits_output: Any,
    next_token_ids: Any,
) -> None:
    if not isinstance(next_token_ids, torch.Tensor):
        return
    logits = getattr(logits_output, "next_token_logits", None)
    hidden = getattr(logits_output, "hidden_states", None)
    for row_idx, request in enumerate(requests):
        request_id = getattr(request, "request_id", None)
        if not isinstance(request_id, str):
            continue
        row_logits = None
        if (
            isinstance(logits, torch.Tensor)
            and logits.ndim >= 2
            and row_idx < logits.shape[0]
        ):
            row_logits = logits[row_idx]
        row_hidden = None
        if (
            isinstance(hidden, torch.Tensor)
            and hidden.ndim >= 2
            and row_idx < hidden.shape[0]
        ):
            row_hidden = hidden[row_idx]
        generation_steps = getattr(
            getattr(request, "data", None), "generation_steps", None
        )
        append_trace(
            stage,
            request_id,
            "ar_step",
            row=row_idx,
            generation_steps=(
                int(generation_steps) if generation_steps is not None else None
            ),
            next_token_id=int(next_token_ids[row_idx].item()),
            logits=summarize_logits_row(row_logits),
            hidden=summarize_tensor(row_hidden),
        )
