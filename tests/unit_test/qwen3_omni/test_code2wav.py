# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from tests.unit_test.fixtures.qwen_fakes import FakeCode2WavModel, make_qwen_payload


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    """Preserves incremental waveform emission and request-state cleanup on abort."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    scheduler._payloads["req-1"] = make_qwen_payload(request_id="req-1")
    scheduler._ensure_request_state("req-1")

    chunk_meta = {"stream": False}  # non-streaming: final result carries full PCM
    scheduler._on_chunk(
        "req-1", SimpleNamespace(data=torch.tensor([1, 10]), metadata=chunk_meta)
    )
    scheduler._on_chunk(
        "req-1", SimpleNamespace(data=torch.tensor([2, 20]), metadata=chunk_meta)
    )
    scheduler._on_chunk(
        "req-1", SimpleNamespace(data=torch.tensor([3, 30]), metadata=chunk_meta)
    )
    scheduler._on_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler._payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler._ensure_request_state("req-2")
    scheduler._pending_done.add("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler._code_chunks
    assert "req-2" not in scheduler._payloads
    assert "req-2" not in scheduler._pending_done
