# SPDX-License-Identifier: Apache-2.0
"""CPU product tests for the MOSS-TTS-Realtime streaming vocoder."""

from __future__ import annotations

import queue
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts_realtime import (
    streaming_vocoder as streaming_vocoder_module,
)
from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.streaming_vocoder import (
    MossTTSRealtimeStreamingVocoderScheduler,
    _CodecStreamSession,
    _LegacyCodecStreamingStateAdapter,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.moss_tts_realtime.runtime_config import MODEL_CONFIG

N_VQ = int(MODEL_CONFIG.rvq)
SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920


class _FakeStreamingState:
    def __init__(self, batch_size: int) -> None:
        self.device = torch.device("cpu")
        self.offsets = torch.zeros(batch_size, dtype=torch.long)
        self.exec_mask = torch.ones(batch_size, dtype=torch.bool)

    def set_exec_mask(self, exec_mask: torch.Tensor) -> None:
        self.exec_mask.copy_(exec_mask.to(dtype=torch.bool))

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.offsets[reset_mask] = 0
        self.exec_mask[reset_mask] = True


class _FakeStateModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._streaming_state: _FakeStreamingState | None = None


class FakeLegacyCodec(nn.Module):
    """Legacy surface: module states exist, but no top-level exec-mask setter."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.state_modules = nn.ModuleList([_FakeStateModule(), _FakeStateModule()])
        self.config = SimpleNamespace(
            sampling_rate=SAMPLE_RATE,
            downsample_rate=SAMPLES_PER_FRAME,
            quantizer_kwargs={"num_quantizers": 32, "codebook_size": 1024},
        )
        self.frame_calls: list[tuple[tuple[int, ...], int]] = []
        self.batch_decode_calls = 0
        self.streaming_batch_sizes: list[int] = []
        self.fail_next_decode = False

    @contextmanager
    def streaming(self, batch_size: int):
        if any(module._streaming_state is not None for module in self.state_modules):
            raise RuntimeError("already streaming")
        self.streaming_batch_sizes.append(batch_size)
        for module in self.state_modules:
            module._streaming_state = _FakeStreamingState(batch_size)
        try:
            yield
        finally:
            for module in self.state_modules:
                module._streaming_state = None

    def _decode_frame(self, codes: torch.Tensor, codes_lengths: torch.Tensor):
        if self.fail_next_decode:
            self.fail_next_decode = False
            raise RuntimeError("injected codec failure")
        _, batch_size, step_frames = codes.shape
        states = [module._streaming_state for module in self.state_modules]
        active = tuple(
            index
            for index in range(batch_size)
            if states[0] is None or bool(states[0].exec_mask[index])
        )
        self.frame_calls.append((active, step_frames))
        audio = torch.zeros(batch_size, 1, step_frames * SAMPLES_PER_FRAME)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        state_count = len(states)
        for batch_index in range(batch_size):
            length = int(codes_lengths[batch_index])
            if length == 0:
                continue
            if states[0] is not None and not bool(states[0].exec_mask[batch_index]):
                continue
            if states[0] is None:
                offset_units = 0
            else:
                offset_units = sum(
                    int(state.offsets[batch_index]) for state in states if state
                )
            for frame_index in range(length):
                value = float(codes[:, batch_index, frame_index].sum())
                value += 1000.0 * (offset_units + state_count * frame_index)
                start = frame_index * SAMPLES_PER_FRAME
                audio[
                    batch_index,
                    0,
                    start : start + SAMPLES_PER_FRAME,
                ] = value
            audio_lengths[batch_index] = length * SAMPLES_PER_FRAME
            for state in states:
                if state is not None and bool(state.exec_mask[batch_index]):
                    state.offsets[batch_index] += length
        return SimpleNamespace(audio=audio, audio_lengths=audio_lengths)

    def batch_decode(
        self,
        codes_list: list[torch.Tensor],
        *,
        num_quantizers: int | None = None,
    ):
        self.batch_decode_calls += 1
        if num_quantizers is None:
            num_quantizers = int(codes_list[0].shape[0])
        max_frames = max(int(codes.shape[1]) for codes in codes_list)
        batch = torch.zeros(
            num_quantizers,
            len(codes_list),
            max_frames,
            dtype=torch.long,
        )
        lengths = torch.zeros(len(codes_list), dtype=torch.long)
        for index, codes in enumerate(codes_list):
            frames = int(codes.shape[1])
            batch[:, index, :frames] = codes[:num_quantizers]
            lengths[index] = frames
        return self._decode_frame(batch, lengths)


class _FakeCudaGraphRunner:
    def __init__(
        self,
        frames: list[int],
        *,
        fail: bool = False,
        length_delta: int = 0,
    ) -> None:
        self._frames = sorted(set(frames))
        self._fail = fail
        self._length_delta = int(length_delta)
        self.decode_calls: list[int] = []

    def captured_frames(self) -> list[int]:
        return list(self._frames)

    def decode_step(
        self,
        codes: torch.Tensor,
        exec_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        frame_count = int(codes.shape[2])
        self.decode_calls.append(frame_count)
        if self._fail:
            raise RuntimeError("injected graph replay failure")
        if frame_count not in self._frames:
            return None
        batch_size = int(codes.shape[1])
        audio = torch.zeros(
            batch_size,
            1,
            frame_count * SAMPLES_PER_FRAME,
        )
        audio_lengths = exec_mask.to(dtype=torch.long) * (
            frame_count * SAMPLES_PER_FRAME + self._length_delta
        )
        return audio, audio_lengths


def _rows(frames: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 100, (frames, N_VQ), generator=generator)


def _reference(rows: torch.Tensor) -> np.ndarray:
    waveform = np.empty(rows.shape[0] * SAMPLES_PER_FRAME, dtype=np.float32)
    state_count = 2
    for frame_index, row in enumerate(rows):
        value = float(row.sum()) + 1000.0 * state_count * frame_index
        start = frame_index * SAMPLES_PER_FRAME
        waveform[start : start + SAMPLES_PER_FRAME] = value
    return waveform


def _metadata(**extra: Any) -> dict[str, Any]:
    return {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": N_VQ,
        **extra,
    }


def _metadata(**extra: Any) -> dict[str, Any]:
    return {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": N_VQ,
        # Session identity rides every chunk, mirroring the engine's
        # stream_metadata stamping; the vocoder keys its codec slot by it.
        "session_id": "session-1",
        **extra,
    }


def _stream_item(row: torch.Tensor, chunk_id: int, **metadata: Any) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=row.clone(),
        from_stage="tts_engine",
        metadata=_metadata(**metadata),
    )


def _payload(
    rows: torch.Tensor,
    *,
    request_id: str,
    stream: bool,
) -> StagePayload:
    state = MossTTSRealtimeState(
        session_id=f"session-{request_id}",
        turn_id=request_id,
        audio_codes=rows.clone(),
    )
    state.prompt_tokens = 3
    state.completion_tokens = int(rows.shape[0])
    state.engine_time_s = 0.25
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={"stream": stream}),
        data=state.to_dict(),
    )


def _scheduler(
    *,
    stream_slots: int = 2,
    max_batch_size: int = 4,
) -> tuple[MossTTSRealtimeStreamingVocoderScheduler, FakeLegacyCodec]:
    codec = FakeLegacyCodec()
    scheduler = MossTTSRealtimeStreamingVocoderScheduler(
        codec,
        n_vq=N_VQ,
        stream_slots=stream_slots,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=0,
    )
    return scheduler, codec


def _active_fake_states(codec: FakeLegacyCodec) -> list[_FakeStreamingState]:
    states = [module._streaming_state for module in codec.state_modules]
    assert all(state is not None for state in states)
    return [state for state in states if state is not None]


def _drain(scheduler: MossTTSRealtimeStreamingVocoderScheduler) -> list[Any]:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _decode_audio(data: dict[str, Any]) -> np.ndarray:
    assert data["audio_waveform_dtype"] == "float32"
    waveform = np.frombuffer(data["audio_waveform"], dtype=np.float32)
    return waveform.reshape(data["audio_waveform_shape"])


def _stream_audio(messages: list[Any], request_id: str) -> np.ndarray:
    chunks = [
        _decode_audio(message.data)
        for message in messages
        if message.request_id == request_id and message.type == "stream"
    ]
    assert chunks
    assert all(chunk.ndim == 1 for chunk in chunks)
    return np.concatenate(chunks)


def _finish(
    scheduler: MossTTSRealtimeStreamingVocoderScheduler,
    rows: torch.Tensor,
    *,
    request_id: str,
) -> None:
    scheduler._on_done(request_id)
    scheduler._on_streaming_new_request(
        request_id,
        _payload(rows, request_id=request_id, stream=True),
    )


def test_ramp_final_flush_and_terminal_order() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows = _rows(8, seed=1)

    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, index))
    _finish(scheduler, rows, request_id="req")

    messages = _drain(scheduler)
    assert [message.type for message in messages] == [
        "stream",
        "stream",
        "stream",
        "stream",
        "result",
    ]
    assert [
        _decode_audio(message.data).shape[0]
        for message in messages
        if message.type == "stream"
    ] == [
        SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
        3 * SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
    ]
    assert [step for _, step in codec.frame_calls] == [1, 2, 3, 2]
    np.testing.assert_array_equal(_stream_audio(messages, "req"), _reference(rows))
    assert messages[-1].data.data["sample_rate"] == SAMPLE_RATE
    assert scheduler._session is not None
    # The successful turn keeps the session's slot: no reset, lease stays.
    assert scheduler._session.active_leases == 1
    assert list(scheduler._codec_sessions) == ["session-1"]

    # Session close releases the slot and resets the causal state.
    result = scheduler.admin("close_realtime_session", {"session_id": "session-1"})
    assert result["success"] is True
    assert result["data"]["released"] is True
    assert scheduler._session.active_leases == 0
    assert not scheduler._codec_sessions
    for state in codec.state_modules:
        assert (state._streaming_state.offsets == 0).all()


def test_equal_ramp_steps_coalesce_without_cross_slot_drift() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(6, seed=2)
    rows_b = _rows(6, seed=3)
    items = []
    for index in range(6):
        items.extend(
            [
                ("a", _stream_item(rows_a[index], index, session_id="session-a")),
                ("b", _stream_item(rows_b[index], index, session_id="session-b")),
            ]
        )

    scheduler.on_stream_chunk_batch(items)
    _finish(scheduler, rows_a, request_id="a")
    _finish(scheduler, rows_b, request_id="b")

    messages = _drain(scheduler)
    assert codec.frame_calls == [((0, 1), 1), ((0, 1), 2), ((0, 1), 3)]
    np.testing.assert_array_equal(_stream_audio(messages, "a"), _reference(rows_a))
    np.testing.assert_array_equal(_stream_audio(messages, "b"), _reference(rows_b))


def test_first_vocoder_step_events_cover_each_coalesced_participant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(
        streaming_vocoder_module,
        "realtime_events_active",
        lambda: True,
    )
    monkeypatch.setattr(
        streaming_vocoder_module,
        "_emit_event",
        lambda **kwargs: events.append(kwargs),
    )
    scheduler, _ = _scheduler(stream_slots=2)
    rows_a = _rows(3, seed=21)
    rows_b = _rows(3, seed=22)

    scheduler.on_stream_chunk_batch(
        [
            (
                "a",
                _stream_item(
                    rows_a[0],
                    0,
                    session_id="session-a",
                    turn_id="turn-a",
                    turn_index=0,
                ),
            ),
            (
                "b",
                _stream_item(
                    rows_b[0],
                    0,
                    session_id="session-b",
                    turn_id="turn-b",
                    turn_index=1,
                ),
            ),
        ]
    )
    scheduler.on_stream_chunk_batch(
        [
            ("a", _stream_item(rows_a[1], 1)),
            ("a", _stream_item(rows_a[2], 2)),
            ("b", _stream_item(rows_b[1], 1)),
            ("b", _stream_item(rows_b[2], 2)),
        ]
    )

    critical = [
        event
        for event in events
        if event["event_name"] in {"vocoder_step_start", "vocoder_step_end"}
    ]
    assert [event["event_name"] for event in critical] == [
        "vocoder_step_start",
        "vocoder_step_start",
        "vocoder_step_end",
        "vocoder_step_end",
    ]
    assert [event["request_id"] for event in critical] == ["a", "b", "a", "b"]
    for event in critical:
        assert event["metadata"]["step_frames"] == 1
        assert event["metadata"]["participant_count"] == 2
        assert event["metadata"]["codec_slot_width"] == 2
        assert event["metadata"]["codec_active_slots"] == 2
        assert event["metadata"]["execution_mode"] == "eager"
        assert event["metadata"]["decode_step_index"] == 0
    assert critical[0]["metadata"]["session_id"] == "session-a"
    assert critical[1]["metadata"]["turn_index"] == 1
    assert critical[2]["metadata"]["output_samples"] == SAMPLES_PER_FRAME
    scheduler.abort("a")
    scheduler.abort("b")


def test_observability_identity_is_best_effort_not_a_stream_contract() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    rows = _rows(3, seed=23)

    scheduler.on_stream_chunk_batch(
        [
            (
                "request",
                _stream_item(
                    rows[0],
                    0,
                    session_id="session-a",
                    turn_id="turn-a",
                    turn_index=0,
                ),
            )
        ]
    )
    scheduler.on_stream_chunk_batch(
        [
            (
                "request",
                _stream_item(
                    rows[1],
                    1,
                    session_id="session-b",
                    turn_id="turn-b",
                    turn_index=-1,
                ),
            ),
            ("request", _stream_item(rows[2], 2)),
        ]
    )

    state = scheduler._stream_states["request"]
    assert state.session_id == "session-a"
    assert state.turn_id == "turn-a"
    assert state.turn_index == 0
    assert all(message.type == "stream" for message in _drain(scheduler))
    scheduler.abort("request")


def test_staggered_requests_keep_their_exact_next_ramp_size() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(3, seed=31)
    rows_b = _rows(3, seed=32)
    scheduler._on_chunk("a", _stream_item(rows_a[0], 0, session_id="session-a"))
    scheduler.on_stream_chunk_batch(
        [
            ("a", _stream_item(rows_a[1], 1)),
            ("a", _stream_item(rows_a[2], 2)),
            ("b", _stream_item(rows_b[0], 0, session_id="session-b")),
        ]
    )
    scheduler.on_stream_chunk_batch(
        [
            ("b", _stream_item(rows_b[1], 1)),
            ("b", _stream_item(rows_b[2], 2)),
        ]
    )
    _finish(scheduler, rows_a, request_id="a")
    _finish(scheduler, rows_b, request_id="b")

    assert [step for _, step in codec.frame_calls] == [1, 1, 2, 2]
    messages = _drain(scheduler)
    np.testing.assert_array_equal(_stream_audio(messages, "a"), _reference(rows_a))
    np.testing.assert_array_equal(_stream_audio(messages, "b"), _reference(rows_b))


def test_codec_session_shares_one_exec_mask_and_resets_every_state() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=2,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    states = _active_fake_states(codec)
    shared_exec_mask = states[0].exec_mask

    assert all(state.exec_mask is shared_exec_mask for state in states)
    session._state_adapter.set_exec_mask(torch.tensor([False, False]))
    for index, state in enumerate(states, start=1):
        state.offsets.fill_(index)

    session._state_adapter.reset_slots([1], batch_size=2)

    assert torch.equal(shared_exec_mask, torch.tensor([False, True]))
    for index, state in enumerate(states, start=1):
        assert torch.equal(state.offsets, torch.tensor([index, 0]))
        assert state.exec_mask is shared_exec_mask
    session.close()


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device"])
def test_codec_state_adapter_rejects_incompatible_exec_masks(mismatch: str) -> None:
    codec = FakeLegacyCodec()
    with codec.streaming(2):
        states = _active_fake_states(codec)
        original_exec_masks = [state.exec_mask for state in states]
        if mismatch == "shape":
            states[1].exec_mask = torch.ones(3, dtype=torch.bool)
        elif mismatch == "dtype":
            states[1].exec_mask = torch.ones(2, dtype=torch.long)
        else:
            states[1].device = torch.device("meta")
            states[1].exec_mask = torch.ones(2, dtype=torch.bool, device="meta")

        with pytest.raises(RuntimeError, match="cannot share exec_mask"):
            _LegacyCodecStreamingStateAdapter(codec, device=torch.device("cpu"))

        assert states[0].exec_mask is original_exec_masks[0]


@pytest.mark.parametrize("stream_slots", [1, 16])
def test_shared_exec_mask_only_advances_active_eager_slot(stream_slots: int) -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=stream_slots,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()

    decoded = session.step({slot: _rows(1, seed=33).transpose(0, 1)})[slot]

    assert decoded.shape == (1, SAMPLES_PER_FRAME)
    assert codec.frame_calls == [((slot,), 1)]
    for state in _active_fake_states(codec):
        expected_offsets = torch.zeros(stream_slots, dtype=torch.long)
        expected_offsets[slot] = 1
        assert torch.equal(state.offsets, expected_offsets)
    session.release(slot)
    session.close()


def test_masked_release_preserves_peer_and_reused_slot_starts_fresh() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(3, seed=4)
    rows_b = _rows(6, seed=5)
    scheduler.on_stream_chunk_batch(
        [
            *(
                ("a", _stream_item(row, index, session_id="session-a"))
                for index, row in enumerate(rows_a)
            ),
            *(
                ("b", _stream_item(row, index, session_id="session-b"))
                for index, row in enumerate(rows_b[:3])
            ),
        ]
    )
    a_slot = scheduler._stream_states["a"].slot
    states = _active_fake_states(codec)

    # A successful turn drains PCM but keeps the session's slot warm: the peer
    # is untouched and nothing returns to the pool.
    _finish(scheduler, rows_a, request_id="a")
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 2
    assert all(int(state.offsets[a_slot]) == 3 for state in states)

    # Session close releases-and-resets that slot; the peer stream is untouched.
    result = scheduler.admin("close_realtime_session", {"session_id": "session-a"})
    assert result["success"] is True
    assert result["data"]["released"] is True
    assert all(int(state.offsets[a_slot]) == 0 for state in states)

    for index, row in enumerate(rows_b[3:], start=3):
        scheduler._on_chunk("b", _stream_item(row, index))

    # A new session reuses the reset slot and starts from a fresh causal state.
    rows_c = _rows(1, seed=6)
    scheduler._on_chunk("c", _stream_item(rows_c[0], 0, session_id="session-c"))
    assert scheduler._stream_states["c"].slot == a_slot
    _finish(scheduler, rows_c, request_id="c")
    _finish(scheduler, rows_b, request_id="b")

    messages = _drain(scheduler)
    np.testing.assert_array_equal(_stream_audio(messages, "b"), _reference(rows_b))
    np.testing.assert_array_equal(_stream_audio(messages, "c"), _reference(rows_c))


def test_slot_exhaustion_errors_without_displacing_live_request() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=7)[0]
    scheduler._on_chunk("live", _stream_item(row, 0, session_id="session-live"))
    scheduler._on_chunk("overflow", _stream_item(row, 0, session_id="session-overflow"))

    messages = _drain(scheduler)
    error = next(message for message in messages if message.request_id == "overflow")
    assert error.type == "error"
    assert "slots are exhausted" in str(error.data)
    assert "live" in scheduler._stream_states
    assert "overflow" not in scheduler._stream_states
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 1


def test_codec_model_info_tracks_acquire_release_reuse_and_exhaustion() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=71)[0]
    initial = scheduler.admin("model_info")["data"]
    assert initial["codec_slot_capacity"] == 1
    assert initial["codec_active_slots"] == 0
    assert initial["codec_free_slots"] == 1
    assert initial["codec_decoder_dtype"] == "float32"

    scheduler._on_chunk("live", _stream_item(row, 0, session_id="session-live"))
    leased_slot = scheduler._stream_states["live"].slot
    scheduler._on_chunk("overflow", _stream_item(row, 0, session_id="session-overflow"))
    active = scheduler.admin("model_info")["data"]
    assert active["codec_active_slots"] == 1
    assert active["codec_free_slots"] == 0
    assert active["codec_live_stream_states"] == 1
    assert active["codec_active_slots_high_water"] == 1
    assert active["codec_pending_frames_high_water"] == 1
    assert active["codec_slot_acquire_total"] == 1
    assert active["codec_slot_exhaustion_total"] == 1

    scheduler.abort("live")
    scheduler._on_chunk("reused", _stream_item(row, 0))
    assert scheduler._stream_states["reused"].slot == leased_slot
    scheduler.abort("reused")
    released = scheduler.admin("model_info")["data"]
    assert released["codec_active_slots"] == 0
    assert released["codec_free_slots"] == 1
    assert released["codec_live_stream_states"] == 0
    assert released["codec_slot_acquire_total"] == 2
    assert released["codec_slot_release_total"] == 2
    assert released["codec_slot_exhaustion_total"] == 1
    assert released["codec_slot_reset_error_total"] == 0


def test_codec_reset_failure_quarantines_slot_and_reports_error() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    row = _rows(1, seed=72)[0]
    scheduler._on_chunk("live", _stream_item(row, 0))
    streaming_state = codec.state_modules[0]._streaming_state
    assert streaming_state is not None

    def fail_reset(reset_mask: torch.Tensor) -> None:
        del reset_mask
        raise RuntimeError("injected codec reset failure")

    streaming_state.reset = fail_reset
    # Aborts release the session's slot; the failing reset quarantines it.
    with pytest.raises(RuntimeError, match="injected codec reset failure"):
        scheduler.abort("live")

    snapshot = scheduler.admin("model_info")["data"]
    assert snapshot["codec_active_slots"] == 0
    assert snapshot["codec_free_slots"] == 0
    assert snapshot["codec_quarantined_slots"] == 1
    assert snapshot["codec_live_stream_states"] == 0
    assert snapshot["codec_slot_acquire_total"] == 1
    assert snapshot["codec_slot_release_total"] == 0
    assert snapshot["codec_slot_reset_error_total"] == 1


def test_shared_codec_step_failure_aborts_every_participant() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(1, seed=33)
    rows_b = _rows(1, seed=34)
    codec.fail_next_decode = True

    scheduler.on_stream_chunk_batch(
        [
            ("a", _stream_item(rows_a[0], 0, session_id="session-a")),
            ("b", _stream_item(rows_b[0], 0, session_id="session-b")),
        ]
    )

    messages = _drain(scheduler)
    assert [(message.request_id, message.type) for message in messages] == [
        ("a", "error"),
        ("b", "error"),
    ]
    assert all("injected codec failure" in str(message.data) for message in messages)
    assert not scheduler._stream_states
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0


@pytest.mark.parametrize(
    ("codes", "metadata", "match"),
    [
        (torch.zeros(15, dtype=torch.long), {}, "shape"),
        (torch.zeros(16, dtype=torch.float32), {}, "integer dtype"),
        (torch.full((16,), 1024, dtype=torch.long), {}, "must be in"),
        (torch.zeros(16, dtype=torch.long), {"n_vq": 8}, "n_vq"),
        (
            torch.zeros(16, dtype=torch.long),
            {"sample_rate": 48000},
            "sample_rate",
        ),
    ],
)
def test_invalid_stream_chunk_aborts_and_releases_slot(
    codes: torch.Tensor,
    metadata: dict[str, Any],
    match: str,
) -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    item = StreamItem(
        chunk_id=0,
        data=codes,
        from_stage="tts_engine",
        metadata=_metadata(**metadata),
    )
    scheduler._on_chunk("bad", item)

    messages = _drain(scheduler)
    assert len(messages) == 1
    assert messages[0].type == "error"
    assert match in str(messages[0].data)
    assert "bad" not in scheduler._stream_states
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0


def test_abort_is_idempotent_and_late_chunks_do_not_reacquire() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=8)[0]
    scheduler._on_chunk("req", _stream_item(row, 0))
    scheduler.abort("req")
    scheduler.abort("req")
    scheduler._on_chunk("req", _stream_item(row, 1))

    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0
    assert "req" not in scheduler._stream_states


def test_stop_releases_all_live_slots_and_closes_context() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows = _rows(1, seed=9)
    scheduler._on_chunk("a", _stream_item(rows[0], 0))
    scheduler._on_chunk("b", _stream_item(rows[0], 0))

    scheduler.stop()

    assert scheduler._session is None
    assert not scheduler._stream_states
    assert all(module._streaming_state is None for module in codec.state_modules)


def test_serving_start_opens_fixed_slot_codec_session() -> None:
    scheduler, codec = _scheduler(stream_slots=3)

    scheduler.on_serving_start()

    assert scheduler._session is not None
    assert scheduler._session.free_slots == 3
    assert codec.streaming_batch_sizes == [3]
    scheduler.on_serving_stop()


def test_codec_session_routes_captured_shapes_and_falls_back_to_eager() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=1,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    graph = _FakeCudaGraphRunner([1])
    session._cg_runner = graph

    graphed = session.step({slot: _rows(1, seed=91).transpose(0, 1)})[slot]
    eager = session.step({slot: _rows(2, seed=92).transpose(0, 1)})[slot]

    assert graphed.shape == (1, SAMPLES_PER_FRAME)
    assert eager.shape == (1, 2 * SAMPLES_PER_FRAME)
    assert graph.decode_calls == [1, 2]
    assert codec.frame_calls == [((slot,), 2)]
    assert session._cg_graph_frames == {1: 1}
    assert session._cg_eager_frames == {2: 1}
    session.release(slot)
    session.close()


def test_codec_session_disables_graph_after_replay_failure() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=1,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    session._cg_runner = _FakeCudaGraphRunner([1], fail=True)
    codes = _rows(1, seed=93).transpose(0, 1)

    with pytest.raises(RuntimeError, match="graph replay failure"):
        session.step({slot: codes})

    assert session._cg_runner is None
    decoded = session.step({slot: codes})[slot]
    assert decoded.shape == (1, SAMPLES_PER_FRAME)
    assert codec.frame_calls == [((slot,), 1)]
    session.release(slot)
    session.close()


def test_codec_session_disables_graph_after_invalid_replay_output() -> None:
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=1,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    session._cg_runner = _FakeCudaGraphRunner([1], length_delta=1)

    with pytest.raises(RuntimeError, match="unexpected active length"):
        session.step({slot: _rows(1, seed=95).transpose(0, 1)})

    assert session._cg_runner is None
    session.release(slot)
    session.close()


def test_codec_session_attempts_cuda_graph_warmup_once(monkeypatch) -> None:
    from sglang_omni.models.moss_tts_realtime import vocoder_cuda_graph

    calls: list[tuple[list[int], float]] = []

    class FakeCaptureRunner:
        def __init__(self, *args: Any, min_free_gb: float, **kwargs: Any) -> None:
            del args, kwargs
            self._min_free_gb = min_free_gb
            self._frames: list[int] = []

        def warmup(self, frames: list[int]) -> None:
            calls.append((list(frames), self._min_free_gb))
            self._frames = list(frames)

        def captured_frames(self) -> list[int]:
            return list(self._frames)

    monkeypatch.setattr(
        vocoder_cuda_graph,
        "MossTTSRealtimeVocoderCudaGraphRunner",
        FakeCaptureRunner,
    )
    codec = FakeLegacyCodec()
    session = _CodecStreamSession(
        codec,
        stream_slots=2,
        n_vq=N_VQ,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    slot = session.acquire()
    session.step({slot: _rows(1, seed=94).transpose(0, 1)})

    assert session.warmup_cuda_graph([1, 2], min_free_gb=4.5) == [1, 2]
    assert session.warmup_cuda_graph([3], min_free_gb=8.0) == [1, 2]
    assert calls == [([1, 2], 4.5)]
    for module in codec.state_modules:
        state = module._streaming_state
        assert state is not None
        assert torch.equal(state.offsets, torch.zeros(2, dtype=torch.long))
    session.release(slot)
    session.close()


def test_scheduler_default_cuda_graph_frames_cover_dense_catchup_range(
    monkeypatch,
) -> None:
    calls: list[tuple[list[int], float]] = []
    monkeypatch.setattr(
        MossTTSRealtimeStreamingVocoderScheduler,
        "_codec_on_cuda",
        lambda self: True,
    )

    def fake_warmup(
        self: _CodecStreamSession,
        frames: list[int],
        *,
        min_free_gb: float = 3.0,
    ) -> list[int]:
        self.warmup_attempted = True
        calls.append((list(frames), min_free_gb))
        return []

    monkeypatch.setattr(_CodecStreamSession, "warmup_cuda_graph", fake_warmup)
    scheduler, _ = _scheduler(stream_slots=1)

    scheduler.warmup_now()
    scheduler.warmup_now()

    assert calls == [(list(range(1, 13)), 3.0)]
    assert scheduler._session is not None
    assert scheduler._session.warmup_attempted
    scheduler.on_serving_stop()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cuda_graph_frames": []}, "must not be empty"),
        ({"cuda_graph_frames": [0]}, "step range"),
        ({"cuda_graph_frames": [26]}, "step range"),
        ({"cuda_graph_min_free_gb": -1.0}, "non-negative"),
    ],
)
def test_scheduler_rejects_invalid_cuda_graph_settings(
    kwargs: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MossTTSRealtimeStreamingVocoderScheduler(
            FakeLegacyCodec(),
            n_vq=N_VQ,
            **kwargs,
        )


def test_scheduler_accepts_explicit_capture_through_25_frames() -> None:
    scheduler = MossTTSRealtimeStreamingVocoderScheduler(
        FakeLegacyCodec(),
        n_vq=N_VQ,
        cuda_graph_frames=[25, 1, 25],
    )

    assert scheduler._cuda_graph_capture_frames() == [1, 25]


def test_dense_cuda_graph_range_drains_existing_backlog_without_waiting() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    session = scheduler._ensure_session()
    graph = _FakeCudaGraphRunner(list(range(1, 13)))
    session._cg_runner = graph
    rows = _rows(20, seed=96)

    scheduler.on_stream_chunk_batch(
        [("req", _stream_item(row, index)) for index, row in enumerate(rows)]
    )

    assert graph.decode_calls == [1, 2, 12, 5]
    messages = _drain(scheduler)
    assert [
        _decode_audio(message.data).shape[0]
        for message in messages
        if message.type == "stream"
    ] == [
        1 * SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
        12 * SAMPLES_PER_FRAME,
        5 * SAMPLES_PER_FRAME,
    ]
    snapshot = scheduler.admin("model_info")["data"]
    assert snapshot["codec_resource_totals"]["codec_catchup_step_total"] == 2
    assert snapshot["codec_resource_totals"]["codec_catchup_frame_total"] == 17
    scheduler.abort("req")


def test_dense_cuda_graph_range_never_waits_to_fill_a_larger_shape() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    session = scheduler._ensure_session()
    graph = _FakeCudaGraphRunner(list(range(1, 13)))
    session._cg_runner = graph
    rows = _rows(6, seed=97)

    for index, row in enumerate(rows):
        scheduler._on_chunk("req", _stream_item(row, index))

    assert graph.decode_calls == [1, 2, 3]
    snapshot = scheduler.admin("model_info")["data"]
    assert snapshot["codec_cuda_graph_default_max_frames"] == 12
    assert snapshot["codec_resource_totals"].get("codec_catchup_step_total", 0) == 0
    scheduler.abort("req")


def test_idle_offline_batch_uses_full_batch_decode() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    rows_a = _rows(4, seed=10)
    rows_b = _rows(2, seed=11)
    payloads = [
        _payload(rows_a, request_id="a", stream=False),
        _payload(rows_b, request_id="b", stream=False),
    ]

    results = scheduler._vocode_batch(payloads)

    assert codec.batch_decode_calls == 1
    np.testing.assert_array_equal(
        _decode_audio(results[0].data),
        _reference(rows_a),
    )
    np.testing.assert_array_equal(
        _decode_audio(results[1].data),
        _reference(rows_b),
    )


def test_done_before_payload_falls_back_to_terminal_audio_codes_once() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows = _rows(4, seed=35)

    scheduler._on_done("req")
    scheduler._on_streaming_new_request(
        "req",
        _payload(rows, request_id="req", stream=True),
    )

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["stream", "result"]
    assert codec.batch_decode_calls == 1
    np.testing.assert_array_equal(_stream_audio(messages, "req"), _reference(rows))


def test_audio_eos_without_real_frames_emits_only_terminal_result() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows = torch.empty((0, N_VQ), dtype=torch.long)

    scheduler._on_done("req")
    scheduler._on_streaming_new_request(
        "req",
        _payload(rows, request_id="req", stream=True),
    )

    messages = _drain(scheduler)
    assert [message.type for message in messages] == ["result"]
    assert codec.batch_decode_calls == 0
    # A turn that never streamed a codec frame never binds a slot.
    assert scheduler._session is None
    assert not scheduler._codec_sessions


def test_offline_decode_borrows_free_slot_without_advancing_live_stream() -> None:
    scheduler, codec = _scheduler(stream_slots=2)
    live_rows = _rows(6, seed=12)
    for index, row in enumerate(live_rows[:3]):
        scheduler._on_chunk("live", _stream_item(row, index))
    offline_rows = _rows(4, seed=13)

    result = scheduler._vocode(
        _payload(offline_rows, request_id="offline", stream=False)
    )
    for index, row in enumerate(live_rows[3:], start=3):
        scheduler._on_chunk("live", _stream_item(row, index))
    _finish(scheduler, live_rows, request_id="live")

    messages = _drain(scheduler)
    assert codec.batch_decode_calls == 0
    np.testing.assert_array_equal(_decode_audio(result.data), _reference(offline_rows))
    np.testing.assert_array_equal(
        _stream_audio(messages, "live"),
        _reference(live_rows),
    )


def test_offline_decode_fails_when_all_fixed_slots_are_leased() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    row = _rows(1, seed=14)[0]
    scheduler._on_chunk("live", _stream_item(row, 0))

    with pytest.raises(RuntimeError, match="no free slot"):
        scheduler._vocode(
            _payload(_rows(2, seed=15), request_id="offline", stream=False)
        )


def test_turns_of_one_session_share_one_slot_and_continue_codec_state() -> None:
    """The issue-1812 contract: turn boundaries inside a session do not reset
    the causal codec state."""
    scheduler, codec = _scheduler(stream_slots=2)
    rows_t1 = _rows(8, seed=40)
    rows_t2 = _rows(4, seed=41)

    for index, row in enumerate(rows_t1):
        scheduler._on_chunk("turn-1", _stream_item(row, index))
    _finish(scheduler, rows_t1, request_id="turn-1")

    entry = scheduler._codec_sessions["session-1"]
    slot = entry.slot
    states = _active_fake_states(codec)
    assert all(int(state.offsets[slot]) == 8 for state in states)

    # Turn 2 of the same session reuses the slot and continues the context:
    # all four frames stream in, and ramp emissions continue per contract.
    for index, row in enumerate(rows_t2):
        scheduler._on_chunk("turn-2", _stream_item(row, index, session_id="session-1"))

    assert scheduler._codec_sessions["session-1"].slot == slot
    assert scheduler._stream_states["turn-2"].slot == slot
    # The first frame of turn 2 decodes against the 8-frame history: the fake
    # codec prices it with offset_units = 2 states * 8 committed frames.
    row0_value = float(rows_t2[0].sum()) + 1000.0 * (2 * 8)
    messages = _drain(scheduler)
    audio = _stream_audio(messages, "turn-2")
    assert audio[0] == pytest.approx(row0_value)

    _finish(scheduler, rows_t2, request_id="turn-2")
    assert all(int(state.offsets[slot]) == 12 for state in states)
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 1


def test_new_turn_after_session_close_restarts_from_fresh_codec_state() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows_t1 = _rows(3, seed=50)
    rows_t2 = _rows(3, seed=51)

    for index, row in enumerate(rows_t1):
        scheduler._on_chunk("turn-1", _stream_item(row, index))
    _finish(scheduler, rows_t1, request_id="turn-1")
    slot = scheduler._codec_sessions["session-1"].slot
    states = _active_fake_states(codec)
    assert all(int(state.offsets[slot]) == 3 for state in states)

    result = scheduler.admin("close_realtime_session", {"session_id": "session-1"})
    assert result["data"]["released"] is True
    assert not scheduler._codec_sessions
    assert all(int(state.offsets[slot]) == 0 for state in states)

    scheduler._on_chunk("turn-2", _stream_item(rows_t2[0], 0, session_id="session-1"))
    assert scheduler._stream_states["turn-2"].slot == slot
    row0_value = float(rows_t2[0].sum())  # fresh context: zero history offset
    messages = _drain(scheduler)
    audio = _stream_audio(messages, "turn-2")
    assert audio[0] == pytest.approx(row0_value)


def test_sessions_are_isolated_and_never_share_a_slot() -> None:
    scheduler, _ = _scheduler(stream_slots=2)
    row = _rows(1, seed=60)[0]

    scheduler._on_chunk("a", _stream_item(row, 0, session_id="session-a"))
    scheduler._on_chunk("b", _stream_item(row, 0, session_id="session-b"))

    entry_a = scheduler._codec_sessions["session-a"]
    entry_b = scheduler._codec_sessions["session-b"]
    assert entry_a.slot != entry_b.slot

    # Closing a session with a live turn defers until that turn's final flush;
    # the peer session is untouched throughout.
    scheduler.admin("close_realtime_session", {"session_id": "session-a"})
    assert scheduler._codec_sessions["session-a"].closing is True
    assert "session-b" in scheduler._codec_sessions
    assert scheduler._stream_states["b"].slot == entry_b.slot

    _finish(scheduler, _rows(1, seed=60), request_id="a")
    assert "session-a" not in scheduler._codec_sessions
    assert "session-b" in scheduler._codec_sessions

    scheduler.abort("b")
    assert not scheduler._codec_sessions


def test_close_marker_defers_release_until_final_pcm_is_drained() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    rows = _rows(5, seed=70)
    for index, row in enumerate(rows[:4]):
        scheduler._on_chunk("turn-1", _stream_item(row, index))
    early_messages = _drain(scheduler)

    # Engine-side ephemeral close rides the turn's stream edge ahead of the
    # terminal result; the slot release must wait for the final flush.
    marker = StreamItem(
        chunk_id=4,
        data=torch.empty(0, dtype=torch.long),
        from_stage="tts_engine",
        metadata=_metadata(
            session_control="close",
            session_id="session-1",
        ),
    )
    scheduler.on_stream_chunk_batch([("turn-1", marker)])

    entry = scheduler._codec_sessions["session-1"]
    assert entry.closing is True
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 1

    # The buffered fifth frame still decodes before the release lands.
    scheduler._on_chunk("turn-1", _stream_item(rows[4], 4))
    _finish(scheduler, rows, request_id="turn-1")

    assert not scheduler._codec_sessions
    assert scheduler._session.active_leases == 0
    late_messages = _drain(scheduler)
    messages = [
        *early_messages,
        *[message for message in late_messages if message.type != "result"],
    ]
    np.testing.assert_array_equal(_stream_audio(messages, "turn-1"), _reference(rows))


def test_close_marker_for_unknown_session_is_a_no_op() -> None:
    scheduler, _ = _scheduler(stream_slots=1)
    marker = StreamItem(
        chunk_id=0,
        data=torch.empty(0, dtype=torch.long),
        from_stage="tts_engine",
        metadata=_metadata(
            session_control="close",
            session_id="session-elsewhere",
        ),
    )
    scheduler.on_stream_chunk_batch([("some-request", marker)])
    assert not scheduler._stream_states
    assert not scheduler._codec_sessions
    assert not _drain(scheduler)


def test_idle_sweep_releases_slots_after_double_engine_ttl() -> None:
    scheduler, codec = _scheduler(stream_slots=1)
    scheduler._session_idle_ttl_s = 0.05
    row = _rows(1, seed=80)[0]
    scheduler._on_chunk("turn-1", _stream_item(row, 0))
    _finish(scheduler, _rows(1, seed=80), request_id="turn-1")

    states = _active_fake_states(codec)
    slot = scheduler._codec_sessions["session-1"].slot
    assert all(int(state.offsets[slot]) == 1 for state in states)

    # Not yet beyond 2 * TTL: a new turn still sees the warm context.
    scheduler._on_chunk("turn-2", _stream_item(row, 0, session_id="session-1"))
    assert scheduler._stream_states["turn-2"].slot == slot
    scheduler.abort("turn-2")  # releases (abort semantics)
    assert not scheduler._codec_sessions

    # Back to fresh demand after the entry aged past the sweep threshold.
    scheduler._on_chunk("turn-3", _stream_item(row, 0, session_id="session-1"))
    _finish(scheduler, _rows(1, seed=80), request_id="turn-3")
    with scheduler._state_lock:
        scheduler._sweep_idle_codec_sessions_locked(
            time.monotonic() + 10 * scheduler._session_idle_ttl_s
        )
    assert not scheduler._codec_sessions
    assert scheduler._session is not None
    assert scheduler._session.active_leases == 0
    assert all(int(state.offsets[slot]) == 0 for state in states)
