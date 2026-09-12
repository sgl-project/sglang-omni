# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 probes: request-event JSONL + optional NVTX ranges.

Hot-path ``print`` / ``logger.info`` is banned here. Flushing stderr
creates the GPU gaps we are trying to measure, multi-process serve
interleaves lines with no join key, and nothing overlays onto nsys.

``log_cosy_profile`` writes through ``event_recorder.emit`` and is a
no-op unless ``/start_request_profile`` (or ``/start_profile``) opened
the recorder. NVTX ranges are independent: they show up on an nsys
timeline even when the recorder is off.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from sglang_omni.profiler.event_recorder import emit, get_recorder


def request_ids_of(requests: Sequence[Any]) -> list[str]:
    return [str(getattr(req, "request_id", "unknown")) for req in requests]


def log_cosy_profile(
    event: str,
    *,
    request_ids: Sequence[str] | None = None,
    **fields: object,
) -> None:
    """Emit one batch-shaped event. No-op when the recorder is inactive."""
    if not get_recorder().is_active():
        return
    ids = [str(rid) for rid in request_ids] if request_ids else []
    emit(
        request_id=ids[0] if ids else "batch",
        stage=None,
        event_name=f"cosy_{event}",
        metadata={
            "clock": "CLOCK_MONOTONIC",
            "monotonic_ns": time.monotonic_ns(),
            "request_ids": ids,
            **fields,
        },
    )


def _nvtx_push(label: str) -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    torch.cuda.nvtx.range_push(label)
    return True


def _nvtx_pop() -> None:
    import torch

    torch.cuda.nvtx.range_pop()


@contextmanager
def cosy_profile_range(
    event: str,
    *,
    request_ids: Sequence[str] | None = None,
    **fields: object,
) -> Iterator[None]:
    """NVTX range plus start/end events around a GPU section we own."""
    batch = fields.get("batch")
    label = f"cosy_{event}" if batch is None else f"cosy_{event} batch={batch}"
    pushed = _nvtx_push(label)
    log_cosy_profile(f"{event}_start", request_ids=request_ids, **fields)
    try:
        yield
    finally:
        log_cosy_profile(f"{event}_end", request_ids=request_ids, **fields)
        if pushed:
            _nvtx_pop()
