# SPDX-License-Identifier: Apache-2.0
"""Stage factory for Nemotron 3.5 ASR."""

from __future__ import annotations

import math

from sglang_omni.models.nemotron3_5_asr.batch_engine import NemotronBatchEngine
from sglang_omni.models.nemotron3_5_asr.model_runner import Nemotron3_5ASRModelRunner
from sglang_omni.models.nemotron3_5_asr.session import NemotronSessionScheduler
from sglang_omni.utils.device import resolve_device_spec


def create_nemotron3_5_asr_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    num_lookahead_tokens: int = 3,
    max_batch_size: int = 8,
    max_batch_wait_ms: float = 2.0,
    session_max_concurrency: int | None = None,
    max_open_sessions: int = 64,
    max_state_bytes: int = 8 * 1024 * 1024 * 1024,
    max_pcm_bytes: int = 2 * 1024 * 1024,
    max_history_tokens: int = 16384,
    max_text_bytes: int = 2 * 1024 * 1024,
) -> NemotronSessionScheduler:
    concurrency = max(4, max_batch_size) if session_max_concurrency is None else session_max_concurrency
    if min(max_batch_size, concurrency, max_open_sessions, max_state_bytes,
           max_pcm_bytes, max_history_tokens, max_text_bytes) < 1:
        raise ValueError("Nemotron batch, concurrency and resource budgets must be positive")
    elif not math.isfinite(max_batch_wait_ms) or max_batch_wait_ms < 0:
        raise ValueError("max_batch_wait_ms must be finite and non-negative")
    else:
        runner = Nemotron3_5ASRModelRunner(
            model_path, device=resolve_device_spec(device, gpu_id), dtype=dtype,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        engine = NemotronBatchEngine(
            runner, max_batch_size=max_batch_size, max_batch_wait_ms=max_batch_wait_ms,
            max_pending_tasks=2 * concurrency + max_open_sessions,
            max_open_sessions=max_open_sessions, max_state_bytes=max_state_bytes,
            max_pcm_bytes=max_pcm_bytes, max_history_tokens=max_history_tokens,
            max_text_bytes=max_text_bytes,
        )
        return NemotronSessionScheduler(
            engine, max_concurrency=concurrency, max_open_sessions=max_open_sessions,
            max_state_bytes=max_state_bytes,
        )


__all__ = ["create_nemotron3_5_asr_executor"]
