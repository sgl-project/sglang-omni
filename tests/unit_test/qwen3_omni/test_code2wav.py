# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
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
        "req-1",
        StreamItem(0, torch.tensor([1, 10]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(1, torch.tensor([2, 20]), "talker", metadata=chunk_meta),
    )
    scheduler._on_chunk(
        "req-1",
        StreamItem(2, torch.tensor([3, 30]), "talker", metadata=chunk_meta),
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


def _feed_chunks(scheduler: Code2WavScheduler, request_id: str, count: int) -> None:
    chunk_meta = {"stream": False}
    for i in range(count):
        scheduler._on_chunk(
            request_id,
            StreamItem(i, torch.tensor([i, i * 10]), "talker", metadata=chunk_meta),
        )


def test_qwen_code2wav_initial_chunk_override_shrinks_first_flush() -> None:
    """A request-level initial_codec_chunk_frames flushes chunk 0 early, and
    chunk 1's decode window uses the whole first chunk as left context."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        sample_rate=24000,
    )
    payload = make_qwen_payload(
        request_id="req-1", params={"initial_codec_chunk_frames": 4}
    )
    scheduler._on_streaming_new_request("req-1", payload)

    # First 4 chunks flush at the initial size, not the steady size of 10.
    _feed_chunks(scheduler, "req-1", 4)
    assert model.calls == [(1, 2, 4)]

    # Next 10 chunks (steady size, counted from the initial offset) flush the
    # second chunk. Its window covers all 14 frames seen so far, i.e. context
    # == 4 == the entire first (initial) chunk, so the seam is continuous.
    for i in range(4, 14):
        scheduler._on_chunk(
            "req-1",
            StreamItem(
                i, torch.tensor([i, i * 10]), "talker", metadata={"stream": False}
            ),
        )
    assert model.calls == [(1, 2, 4), (1, 2, 14)]


def test_qwen_code2wav_no_override_keeps_steady_size_for_first_flush() -> None:
    """Without a request-level override, the first flush still uses the
    configured steady stream_chunk_size (unchanged prior behavior)."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        sample_rate=24000,
    )
    payload = make_qwen_payload(request_id="req-1", params={})
    scheduler._on_streaming_new_request("req-1", payload)

    _feed_chunks(scheduler, "req-1", 9)
    assert model.calls == []  # not enough for a steady-size flush yet

    scheduler._on_chunk(
        "req-1",
        StreamItem(9, torch.tensor([9, 90]), "talker", metadata={"stream": False}),
    )
    assert model.calls == [(1, 2, 10)]


def test_qwen_code2wav_stage_default_applies_before_payload_arrives() -> None:
    """code2wav has can_accept_stream_before_payload=True, so chunks can reach
    on_stream_chunk before on_streaming_new_request resolves the per-request
    override. The stage-configured default must still apply in that race,
    not silently fall back to the steady size."""
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=25,
        initial_codec_chunk_frames=4,
        sample_rate=24000,
    )

    # Chunks arrive first; on_streaming_new_request has not run yet.
    _feed_chunks(scheduler, "req-1", 4)
    assert model.calls == [(1, 2, 4)]

    # The payload arrives afterward with no per-request override; the
    # already-applied stage default must not be disturbed.
    payload = make_qwen_payload(request_id="req-1", params={})
    scheduler._on_streaming_new_request("req-1", payload)
    assert scheduler._initial_chunk_frames["req-1"] == 4
