# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.models.fishaudio_s2_pro.request_builders import apply_tts_result
from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import (
    S2ProVocoderScheduler,
    _apply_stream_crossfade,
    _StreamVocoderState,
    build_stream_vocoder_chunk,
    flush_stream_vocoder_chunk,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


class _FakeCodec:
    sample_rate = 44100
    frame_length = 4
    delay = 0

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(indices.shape))
        batch = indices.shape[0]
        tokens = indices.shape[-1]
        rows = []
        for row in range(batch):
            values = torch.arange(tokens * self.frame_length, dtype=torch.float32)
            rows.append(values.unsqueeze(0) + float(row + 1))
        return torch.stack(rows, dim=0)


class _ContextCodec:
    sample_rate = 44100
    frame_length = 3
    delay = 3

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        weights = torch.arange(
            1,
            indices.shape[1] + 1,
            dtype=torch.float32,
            device=indices.device,
        ).view(1, -1, 1)
        token_values = (indices.float() * weights).sum(dim=1)
        prev_values = torch.nn.functional.pad(token_values[:, :-1], (1, 0))
        frames = token_values + 0.25 * prev_values
        frame_offsets = torch.arange(
            self.frame_length,
            dtype=torch.float32,
            device=indices.device,
        ).view(1, 1, -1)
        audio = frames.unsqueeze(-1) + frame_offsets
        return audio.reshape(indices.shape[0], 1, -1)


def _payload(
    request_id: str,
    *,
    stream: bool = True,
    code_len: int = 4,
    usage: dict | None = None,
) -> StagePayload:
    output_codes = torch.arange(11 * code_len, dtype=torch.long).reshape(11, code_len)
    data = {
        "output_codes": output_codes.tolist(),
        "sample_rate": 123,
    }
    if usage is not None:
        data["usage"] = usage
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=data,
    )


def _empty_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": False}),
        data={},
    )


def _zero_length_payload(request_id: str) -> StagePayload:
    output_codes = torch.empty((11, 0), dtype=torch.long)
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": False}),
        data={"output_codes": output_codes.tolist()},
    )


def _code(value: int = 1) -> torch.Tensor:
    return torch.full((11, 1), value, dtype=torch.long)


def _chunk(value: int = 1) -> StreamItem:
    return StreamItem(chunk_id=value, data=_code(value), from_stage="tts_engine")


def _audio_tensor(payload: dict) -> torch.Tensor:
    audio = np.frombuffer(payload["audio_waveform"], dtype=np.float32)
    return torch.from_numpy(audio.copy()).reshape(payload["audio_waveform_shape"])


def _audio_len(payload: dict) -> int:
    return int(np.prod(payload["audio_waveform_shape"]))


def _start_scheduler(
    *,
    stream_stride: int = 3,
    stream_followup_stride: int = 5,
    stream_crossfade_samples: int = 0,
) -> tuple[S2ProVocoderScheduler, threading.Thread]:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=1,
        stream_crossfade_samples=stream_crossfade_samples,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    return scheduler, thread


def _stop_scheduler(scheduler: S2ProVocoderScheduler, thread: threading.Thread) -> None:
    scheduler.stop()
    thread.join(timeout=2.0)


def test_streaming_vocoder_chunk_cadence() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.1)

        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(3)))
        first = scheduler.outbox.get(timeout=2.0)
        assert first.type == "stream"
        assert first.data["modality"] == "audio"
        assert _audio_len(first.data) == 12

        for value in range(4, 8):
            scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(value)))
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.1)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_sample_level_matches_contextual_full_decode() -> None:
    codec = _ContextCodec()
    state = _StreamVocoderState()
    full_codes = torch.arange(11 * 7, dtype=torch.long).reshape(11, 7)
    chunks = []

    for idx in range(full_codes.shape[1]):
        output = build_stream_vocoder_chunk(
            state,
            full_codes[:, idx : idx + 1],
            codec=codec,
            device=torch.device("cpu"),
            stream_stride=3,
            stream_followup_stride=2,
            stream_overlap_tokens=1,
            stream_crossfade_samples=0,
        )
        if output is not None:
            chunks.append(_audio_tensor(output))

    output = flush_stream_vocoder_chunk(
        state,
        codec=codec,
        device=torch.device("cpu"),
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    if output is not None:
        chunks.append(_audio_tensor(output))

    streaming_audio = torch.cat(chunks)
    full_audio = codec.from_indices(full_codes[1:][None])[0, 0]

    torch.testing.assert_close(streaming_audio, full_audio)


def test_streaming_vocoder_crossfade_blends_tail_and_retains_next_tail() -> None:
    state = _StreamVocoderState(
        pending_tail=torch.tensor([10.0, 20.0, 30.0]),
    )
    delta_audio = torch.tensor([100.0, 200.0, 300.0, 400.0])

    output = _apply_stream_crossfade(
        state,
        delta_audio,
        stream_crossfade_samples=2,
        is_final=False,
    )

    assert output is not None
    torch.testing.assert_close(output, torch.tensor([10.0, 20.0, 200.0]))
    torch.testing.assert_close(state.pending_tail, torch.tensor([300.0, 400.0]))


def test_streaming_vocoder_zero_overlap_final_flush_emits_retained_tail() -> None:
    codec = _FakeCodec()
    state = _StreamVocoderState()

    first = build_stream_vocoder_chunk(
        state,
        torch.arange(22, dtype=torch.long).reshape(11, 2),
        codec=codec,
        device=torch.device("cpu"),
        stream_stride=2,
        stream_followup_stride=10,
        stream_overlap_tokens=0,
        stream_crossfade_samples=3,
    )
    assert first is not None
    assert state.codes == []
    assert state.pending_tail is not None

    flush = flush_stream_vocoder_chunk(
        state,
        codec=codec,
        device=torch.device("cpu"),
        stream_overlap_tokens=0,
        stream_crossfade_samples=3,
    )

    assert flush is not None
    assert flush["modality"] == "audio"
    assert _audio_len(flush) == 3
    assert state.pending_tail is None


def test_streaming_vocoder_final_flush_clears_tail_when_codes_remain() -> None:
    codec = _FakeCodec()
    state = _StreamVocoderState(
        codes=[torch.arange(11, dtype=torch.long).reshape(11, 1)],
        last_vocode_tokens=1,
        total_tokens=1,
        pending_tail=torch.tensor([1.0, 2.0, 3.0]),
    )

    flush = flush_stream_vocoder_chunk(
        state,
        codec=codec,
        device=torch.device("cpu"),
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )

    assert flush is not None
    assert _audio_tensor(flush).tolist() == [1.0, 2.0, 3.0]
    assert state.pending_tail is None


def test_streaming_vocoder_final_flush_emits_tail_before_result() -> None:
    scheduler, thread = _start_scheduler(stream_stride=2, stream_followup_stride=10)
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        first = scheduler.outbox.get(timeout=2.0)
        assert first.type == "stream"

        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(3)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        flush = scheduler.outbox.get(timeout=2.0)
        final = scheduler.outbox.get(timeout=2.0)
        assert flush.type == "stream"
        assert final.type == "result"
        assert final.data.data["modality"] == "audio"
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_done_before_payload_finalizes_after_new_request() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        stream = scheduler.outbox.get(timeout=2.0)
        final = scheduler.outbox.get(timeout=2.0)
        assert stream.type == "stream"
        assert final.type == "result"
        assert final.request_id == "req"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_zero_chunk_done_before_payload_finalizes() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        assert final.request_id == "req"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_final_payload_preserves_usage_and_authoritative_audio() -> (
    None
):
    scheduler, thread = _start_scheduler()
    usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
    try:
        scheduler.inbox.put(
            IncomingMessage(
                "req",
                "new_request",
                _payload("req", stream=True, code_len=4, usage=usage),
            )
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        data = final.data.data
        assert data["usage"] == usage
        assert data["sample_rate"] == 44100
        assert data["modality"] == "audio"
        assert len(data["audio_data"]) == 16
    finally:
        _stop_scheduler(scheduler, thread)


def test_non_streaming_vocoder_clears_static_stream_done_signal() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=False))
        )
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=True))
        )
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(3)))
        stream = scheduler.outbox.get(timeout=2.0)
        assert stream.type == "stream"
    finally:
        _stop_scheduler(scheduler, thread)


def test_non_streaming_vocoder_clears_prefetched_stream_done_signal() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(
            IncomingMessage("req", "new_request", _payload("req", stream=False))
        )
        final = scheduler.outbox.get(timeout=2.0)
        assert final.type == "result"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_abort_cleans_state_and_suppresses_final() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_stride=3,
        stream_followup_stride=5,
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
        max_batch_wait_ms=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    try:
        scheduler._payloads["req"] = _payload("req")
        scheduler._pending_done.add("req")
        scheduler._on_chunk("req", _chunk(1))
        scheduler._pending_messages.append(IncomingMessage("req", "stream_done"))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(2)))
        scheduler.inbox.put(IncomingMessage("req", "stream_done"))

        scheduler.abort("req")
        thread.start()

        assert "req" not in scheduler._payloads
        assert "req" not in scheduler._stream_states
        assert "req" not in scheduler._pending_done
        assert "req" in scheduler._aborted_request_ids
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_abort_does_not_block_other_request() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(
            IncomingMessage("aborted", "new_request", _payload("aborted"))
        )
        scheduler.inbox.put(IncomingMessage("aborted", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("other", "new_request", _payload("other")))
        scheduler.inbox.put(IncomingMessage("other", "stream_chunk", _chunk(1)))
        scheduler.inbox.put(IncomingMessage("other", "stream_chunk", _chunk(2)))
        scheduler.inbox.put(IncomingMessage("other", "stream_chunk", _chunk(3)))

        scheduler.abort("aborted")

        out = scheduler.outbox.get(timeout=2.0)
        assert out.request_id == "other"
        assert out.type == "stream"
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_chunk_failure_emits_one_error_and_no_success() -> None:
    scheduler, thread = _start_scheduler()

    def _raise_on_chunk(request_id, chunk) -> None:
        del request_id, chunk
        raise RuntimeError("chunk failed")

    scheduler._on_chunk = _raise_on_chunk
    try:
        scheduler.inbox.put(IncomingMessage("req", "new_request", _payload("req")))
        scheduler.inbox.put(IncomingMessage("req", "stream_chunk", _chunk(1)))
        error = scheduler.outbox.get(timeout=2.0)

        assert error.request_id == "req"
        assert error.type == "error"
        assert isinstance(error.data, RuntimeError)
        assert "req" in scheduler._aborted_request_ids

        scheduler.inbox.put(IncomingMessage("req", "stream_done"))
        with pytest.raises(queue.Empty):
            scheduler.outbox.get(timeout=0.2)
    finally:
        _stop_scheduler(scheduler, thread)


def test_streaming_vocoder_abort_during_final_vocode_suppresses_result() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    scheduler._on_streaming_new_request("req", _payload("req"))

    def _abort_during_vocode(payload):
        scheduler.abort(payload.request_id)
        return payload

    scheduler._vocode_payload = _abort_during_vocode

    scheduler._on_done("req")

    assert "req" in scheduler._aborted_request_ids
    assert scheduler.outbox.empty()


def test_non_streaming_vocoder_rejects_missing_output_codes() -> None:
    scheduler, thread = _start_scheduler()
    try:
        scheduler.inbox.put(
            IncomingMessage("req-empty", "new_request", _empty_payload("req-empty"))
        )
        output = scheduler.outbox.get(timeout=2.0)
    finally:
        _stop_scheduler(scheduler, thread)

    assert output.request_id == "req-empty"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert "req-empty" in str(output.data)
    assert scheduler._codec.calls == []


def test_non_streaming_vocoder_batch_rejects_zero_length_before_decode() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    with pytest.raises(ValueError, match="req-zero"):
        scheduler._vocode_payloads(
            [
                _payload("req-good", stream=False),
                _zero_length_payload("req-zero"),
            ]
        )
    assert scheduler._codec.calls == []


def test_non_streaming_vocoder_batch_isolates_invalid_payload() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    messages = [
        IncomingMessage("req-good", "new_request", _payload("req-good", stream=False)),
        IncomingMessage(
            "req-zero",
            "new_request",
            _zero_length_payload("req-zero"),
        ),
    ]

    scheduler._handle_new_request_batch(messages)

    outputs = [scheduler.outbox.get_nowait(), scheduler.outbox.get_nowait()]
    by_request = {out.request_id: out for out in outputs}
    assert by_request["req-zero"].type == "error"
    assert isinstance(by_request["req-zero"].data, ValueError)
    assert by_request["req-good"].type == "result"
    assert scheduler._codec.calls == [(1, 10, 4)]


def test_vocoder_preserves_finish_reason_from_tts_payload() -> None:
    req_data = SimpleNamespace(
        output_codes=[torch.arange(11, dtype=torch.long).reshape(11, 1)],
        input_ids=[1, 2, 3],
        finish_reason="length",
    )
    state = S2ProState(sample_rate=44100)
    apply_tts_result(state, req_data)
    payload = StagePayload(
        request_id="req-length",
        request=OmniRequest(inputs="hello", params={"stream": False}),
        data=state.to_dict(),
    )
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )

    result = scheduler._vocode_payload(payload)

    assert result.data["finish_reason"] == "length"


def test_non_streaming_vocoder_batch_skips_aborted_request() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
        max_batch_wait_ms=0,
    )
    scheduler.abort("aborted")
    first = IncomingMessage("other", "new_request", _payload("other", stream=False))
    scheduler.inbox.put(
        IncomingMessage("aborted", "new_request", _payload("aborted", stream=False))
    )

    batch = scheduler._collect_new_request_batch(first)
    scheduler._handle_new_request_batch(batch)

    out = scheduler.outbox.get_nowait()
    assert out.request_id == "other"
    assert out.type == "result"
    assert scheduler.outbox.empty()


def test_non_streaming_vocoder_batch_stops_before_streaming_request() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
        max_batch_wait_ms=0,
    )
    first = IncomingMessage(
        "nonstream-a",
        "new_request",
        _payload("nonstream-a", stream=False),
    )
    scheduler.inbox.put(
        IncomingMessage("stream-b", "new_request", _payload("stream-b", stream=True))
    )
    scheduler.inbox.put(
        IncomingMessage(
            "nonstream-c",
            "new_request",
            _payload("nonstream-c", stream=False),
        )
    )

    batch = scheduler._collect_new_request_batch(first)

    assert [msg.request_id for msg in batch] == ["nonstream-a"]
    assert scheduler._next_message().request_id == "stream-b"
    assert scheduler._next_message().request_id == "nonstream-c"


def test_non_streaming_vocoder_abort_during_batch_decode_suppresses_result() -> None:
    scheduler = S2ProVocoderScheduler(
        _FakeCodec(),
        device="cpu",
        stream_overlap_tokens=1,
        stream_crossfade_samples=0,
    )
    messages = [
        IncomingMessage("other", "new_request", _payload("other", stream=False)),
        IncomingMessage("aborted", "new_request", _payload("aborted", stream=False)),
    ]

    def _abort_during_decode(payloads):
        scheduler.abort("aborted")
        return payloads

    scheduler._batch_fn = _abort_during_decode

    scheduler._handle_new_request_batch(messages)

    out = scheduler.outbox.get_nowait()
    assert out.request_id == "other"
    assert out.type == "result"
    assert scheduler.outbox.empty()
