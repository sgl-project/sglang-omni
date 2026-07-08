# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage
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


def _chunk(chunk_id: int, code_a: int, code_b: int, *, stream: bool = False):
    return StreamItem(
        chunk_id,
        torch.tensor([code_a, code_b], dtype=torch.long),
        "talker",
        metadata={"stream": stream},
    )


def _seed_payloads(scheduler: Code2WavScheduler, request_ids: list[str]) -> None:
    for request_id in request_ids:
        scheduler._payloads[request_id] = make_qwen_payload(request_id=request_id)
        scheduler._ensure_request_state(request_id)


def _drain_final_audio(scheduler: Code2WavScheduler) -> dict[str, np.ndarray]:
    audio_by_request: dict[str, np.ndarray] = {}
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        if message.type != "result":
            continue
        audio_by_request[message.request_id] = np.frombuffer(
            message.data.data["audio_waveform"], dtype=np.float32
        )
    return audio_by_request


def test_qwen_code2wav_batches_ready_stream_windows_without_audio_changes() -> None:
    batched_model = FakeCode2WavModel(total_upsample=2)
    batched = Code2WavScheduler(
        batched_model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(batched, ["req-1", "req-2"])

    batched._on_chunk("req-1", _chunk(0, 1, 10))
    batched._on_chunk("req-2", _chunk(0, 2, 20))
    batched.inbox.put(IncomingMessage("req-2", "stream_chunk", _chunk(1, 4, 40)))
    batched._on_chunk("req-1", _chunk(1, 3, 30))
    batched._on_done("req-1")
    batched._on_done("req-2")
    batched_audio = _drain_final_audio(batched)

    single_model = FakeCode2WavModel(total_upsample=2)
    single = Code2WavScheduler(
        single_model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        enable_batched_decode=False,
    )
    _seed_payloads(single, ["req-1", "req-2"])
    for request_id, chunks in {
        "req-1": [_chunk(0, 1, 10), _chunk(1, 3, 30)],
        "req-2": [_chunk(0, 2, 20), _chunk(1, 4, 40)],
    }.items():
        for item in chunks:
            single._on_chunk(request_id, item)
        single._on_done(request_id)
    single_audio = _drain_final_audio(single)

    assert batched_model.calls == [(2, 2, 2)]
    assert single_model.calls == [(1, 2, 2), (1, 2, 2)]
    assert set(batched_audio) == {"req-1", "req-2"}
    for request_id in batched_audio:
        np.testing.assert_array_equal(
            batched_audio[request_id], single_audio[request_id]
        )


def test_qwen_code2wav_batches_same_length_final_partials() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=10,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1", "req-2", "req-3"])

    scheduler._on_chunk("req-1", _chunk(0, 1, 10))
    scheduler._on_chunk("req-1", _chunk(1, 2, 20))
    scheduler._on_chunk("req-2", _chunk(0, 3, 30))
    scheduler._on_chunk("req-2", _chunk(1, 4, 40))
    scheduler._on_chunk("req-3", _chunk(0, 5, 50))
    scheduler.inbox.put(IncomingMessage("req-2", "stream_done"))
    scheduler.inbox.put(IncomingMessage("req-3", "stream_done"))

    scheduler._on_done("req-1")
    _drain_final_audio(scheduler)

    assert model.calls == [(2, 2, 2), (1, 2, 1)]


def test_qwen_code2wav_skips_eos_chunks_before_batching() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
        sample_rate=24000,
        codec_eos_token_id=2150,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["req-1"])

    scheduler._on_chunk(
        "req-1",
        StreamItem(
            0,
            torch.tensor([2150, 2150], dtype=torch.long),
            "talker",
            metadata={"stream": False},
        ),
    )

    assert model.calls == []
    assert scheduler._code_chunks["req-1"] == []


def test_qwen_code2wav_batch_wait_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_QWEN3_CODE2WAV_MAX_BATCH_WAIT_MS", "7")
    scheduler = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=1,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )

    assert scheduler._decode_max_batch_wait_s == 0.007


def test_qwen_code2wav_left_context_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_QWEN3_CODE2WAV_LEFT_CONTEXT_SIZE", "5")
    scheduler = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=1,
        left_context_size=25,
        sample_rate=24000,
    )

    assert scheduler._left_context_size == 5


def test_qwen_code2wav_non_stream_chunk_size_env_keeps_streaming_threshold(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SGLANG_OMNI_QWEN3_CODE2WAV_NON_STREAM_CHUNK_SIZE", "3")
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=0,
        sample_rate=24000,
        max_batch_wait_ms=0,
    )
    _seed_payloads(scheduler, ["non-stream", "stream"])

    scheduler._on_chunk("non-stream", _chunk(0, 1, 10, stream=False))
    scheduler._on_chunk("non-stream", _chunk(1, 2, 20, stream=False))
    assert model.calls == []

    scheduler._on_chunk("stream", _chunk(0, 3, 30, stream=True))
    scheduler._on_chunk("stream", _chunk(1, 4, 40, stream=True))
    assert model.calls == [(1, 2, 2)]

    scheduler._on_chunk("non-stream", _chunk(2, 5, 50, stream=False))
    assert model.calls == [(1, 2, 2), (1, 2, 3)]


def test_qwen_code2wav_non_stream_chunk_size_arg_and_env_override(monkeypatch) -> None:
    scheduler = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        non_stream_chunk_size=4,
        left_context_size=0,
        sample_rate=24000,
    )
    assert scheduler._non_stream_chunk_size == 4

    monkeypatch.setenv("SGLANG_OMNI_QWEN3_CODE2WAV_NON_STREAM_CHUNK_SIZE", "6")
    scheduler = Code2WavScheduler(
        FakeCode2WavModel(total_upsample=2),
        device="cpu",
        stream_chunk_size=2,
        non_stream_chunk_size=4,
        left_context_size=0,
        sample_rate=24000,
    )
    assert scheduler._non_stream_chunk_size == 6
