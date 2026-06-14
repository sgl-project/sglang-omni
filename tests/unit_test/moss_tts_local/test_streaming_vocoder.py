# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local streaming vocoder tests.

All tests are CPU-only and drive the scheduler hooks synchronously in the real
pipeline order (chunks -> stream_done -> terminal payload replay). The fake
codec implements the v2 streaming surface (persistent ``streaming()`` session,
per-slot ``exec_mask``/offsets, per-slot ``reset``) with a decode whose output
depends on each slot's cumulative frame offset, so any state-advance error,
cross-slot leak, or missed reset changes the waveform. The headline assertion
is that streamed PCM concatenates to exactly the offline decode of the same
codes — the property the v2 codec provides by construction.
"""

from __future__ import annotations

import queue
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts_local import stages
from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
from sglang_omni.models.moss_tts_local.request_builders import (
    build_moss_tts_local_stream_metadata,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.models.tts_streaming import INITIAL_CODEC_CHUNK_FRAMES_PARAM
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload

N_VQ = 4
SAMPLES_PER_FRAME = 4
SAMPLE_RATE = 48000


class _FakeStreamingState:
    def __init__(self, batch_size: int) -> None:
        self.device = torch.device("cpu")
        self.offsets = torch.zeros(batch_size, dtype=torch.long)
        self.exec_mask = torch.ones(batch_size, dtype=torch.bool)

    def set_exec_mask(self, exec_mask: torch.Tensor) -> None:
        self.exec_mask = exec_mask.clone().to(torch.bool)

    def reset(self, reset_mask: torch.Tensor) -> None:
        self.offsets[reset_mask] = 0
        self.exec_mask[reset_mask] = True


class FakeCodec(nn.Module):
    """Stateful fake of the MOSS-Audio-Tokenizer-v2 decode surface.

    Frame ``t`` of slot ``b`` decodes to ``sum(codes[:, b, t]) + 1000 * o``
    where ``o`` is the slot's cumulative frame offset, replicated over
    SAMPLES_PER_FRAME samples (negated on the second channel). Offsets only
    advance for exec-masked slots, mirroring the real codec.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self._streaming_state: _FakeStreamingState | None = None
        self.config = SimpleNamespace(sampling_rate=SAMPLE_RATE)
        self.frame_calls = 0

    @contextmanager
    def streaming(self, batch_size: int):
        if self._streaming_state is not None:
            raise RuntimeError("already streaming!")
        self._streaming_state = _FakeStreamingState(batch_size)
        try:
            yield
        finally:
            self._streaming_state = None

    def _set_streaming_exec_mask(self, exec_mask: torch.Tensor) -> None:
        assert self._streaming_state is not None
        self._streaming_state.set_exec_mask(exec_mask)

    def _decode_frame(self, codes: torch.Tensor, codes_lengths: torch.Tensor):
        self.frame_calls += 1
        _, batch_size, step_t = codes.shape
        state = self._streaming_state
        audio = torch.zeros(batch_size, 2, step_t * SAMPLES_PER_FRAME)
        audio_lengths = torch.zeros(batch_size, dtype=torch.long)
        for b in range(batch_size):
            if state is not None and not bool(state.exec_mask[b]):
                continue
            t_len = int(codes_lengths[b])
            if t_len == 0:
                continue
            base = int(state.offsets[b]) if state is not None else 0
            for t in range(t_len):
                value = float(codes[:, b, t].sum()) + 1000.0 * (base + t)
                start = t * SAMPLES_PER_FRAME
                audio[b, 0, start : start + SAMPLES_PER_FRAME] = value
                audio[b, 1, start : start + SAMPLES_PER_FRAME] = -value
            audio_lengths[b] = t_len * SAMPLES_PER_FRAME
            if state is not None:
                state.offsets[b] += t_len
        return SimpleNamespace(audio=audio, audio_lengths=audio_lengths)


class FakeProcessor:
    def __init__(self) -> None:
        self.audio_tokenizer = FakeCodec()
        self.model_config = SimpleNamespace(n_vq=N_VQ, sampling_rate=SAMPLE_RATE)
        self.decode_calls = 0

    def decode_audio_codes(self, codes_list, *, return_stereo: bool = True):
        # Mirrors the real processor: chunked decode inside its own streaming
        # context, which the codec forbids while a session is live.
        self.decode_calls += 1
        codec = self.audio_tokenizer
        wavs = []
        with codec.streaming(len(codes_list)):
            for index, rows in enumerate(codes_list):
                codes = rows[:, :N_VQ].T.unsqueeze(1)  # [n_vq, 1, T]
                exec_mask = torch.zeros(len(codes_list), dtype=torch.bool)
                exec_mask[index] = True
                codec._set_streaming_exec_mask(exec_mask)
                full = torch.zeros(
                    N_VQ, len(codes_list), codes.shape[2], dtype=torch.long
                )
                full[:, index, :] = codes[:, 0, :]
                lengths = torch.zeros(len(codes_list), dtype=torch.long)
                lengths[index] = codes.shape[2]
                result = codec._decode_frame(full, lengths)
                n = int(result.audio_lengths[index])
                wavs.append(result.audio[index, :, :n].to(torch.float32))
        return wavs


def reference_waveform(rows: torch.Tensor) -> torch.Tensor:
    """Stateless offline decode of [T, n_vq] codes rows."""
    codes = rows[:, :N_VQ]
    frames = int(codes.shape[0])
    audio = torch.zeros(2, frames * SAMPLES_PER_FRAME)
    for t in range(frames):
        value = float(codes[t].sum()) + 1000.0 * t
        start = t * SAMPLES_PER_FRAME
        audio[0, start : start + SAMPLES_PER_FRAME] = value
        audio[1, start : start + SAMPLES_PER_FRAME] = -value
    return audio


def _make_scheduler(
    monkeypatch: pytest.MonkeyPatch, processor: FakeProcessor, **kwargs: int
) -> MossTTSLocalStreamingVocoderScheduler:
    monkeypatch.setattr(
        stages,
        "_load_moss_tts_local_processor",
        lambda model_path, *, device: processor,
    )
    scheduler = stages.create_vocoder_executor("fake-model", device="cpu", **kwargs)
    assert isinstance(scheduler, MossTTSLocalStreamingVocoderScheduler)
    return scheduler


def _rows(frames: int, *, seed: int) -> torch.Tensor:
    """Full AR rows [frames, 1 + n_vq]: text token + codes."""
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 100, (frames, N_VQ), generator=generator)
    text = torch.full((frames, 1), 7, dtype=torch.long)
    return torch.cat([text, codes], dim=1)


def _metadata(**extra: Any) -> dict[str, Any]:
    return {"stream": True, "modality": "audio_codes", "n_vq": N_VQ, **extra}


def _stream_item(row: torch.Tensor, metadata: dict[str, Any], chunk_id: int = 0):
    return StreamItem(
        chunk_id=chunk_id,
        data=row.clone(),
        from_stage="tts_engine",
        metadata=metadata,
    )


def _terminal_payload(
    rows: torch.Tensor | None,
    *,
    request_id: str = "req",
    params: dict[str, Any] | None = None,
) -> StagePayload:
    state = MossTTSLocalState(
        text="hello",
        audio_codes=rows[:, 1:].clone() if rows is not None else None,
        prompt_tokens=3,
        completion_tokens=int(rows.shape[0]) if rows is not None else 0,
        engine_time_s=0.5,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="", params={"stream": True, **(params or {})}),
        data=state.to_dict(),
    )


def _drain(scheduler) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def _run_stream(
    scheduler,
    rows: torch.Tensor,
    *,
    request_id: str = "req",
    metadata: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> list:
    metadata = metadata if metadata is not None else _metadata()
    for index, row in enumerate(rows):
        scheduler._on_chunk(request_id, _stream_item(row, metadata, index))
    # Real pipeline order: chunks -> stream_done -> terminal payload replay.
    scheduler._on_done(request_id)
    scheduler._on_streaming_new_request(
        request_id, _terminal_payload(rows, request_id=request_id, params=params)
    )
    return _drain(scheduler)


def _decode_audio(data: dict[str, Any]) -> np.ndarray:
    assert data["audio_waveform_dtype"] == "float32"
    array = np.frombuffer(data["audio_waveform"], dtype=np.float32)
    return array.reshape(data["audio_waveform_shape"])


def _concat_stream_audio(messages: list, request_id: str) -> np.ndarray:
    chunks = [
        _decode_audio(msg.data)
        for msg in messages
        if msg.type == "stream" and msg.request_id == request_id
    ]
    assert chunks, "no stream chunks emitted"
    for chunk in chunks:
        assert chunk.ndim == 2 and chunk.shape[0] == 2  # stereo kept end to end
    return np.concatenate(chunks, axis=1)


def test_stream_metadata_builder() -> None:
    def payload(params: dict[str, Any]) -> StagePayload:
        return StagePayload(
            request_id="req",
            request=OmniRequest(inputs="", params=params),
            data={},
        )

    assert build_moss_tts_local_stream_metadata(payload({}), n_vq=12) is None
    metadata = build_moss_tts_local_stream_metadata(
        payload({"stream": True, INITIAL_CODEC_CHUNK_FRAMES_PARAM: 3}), n_vq=12
    )
    assert metadata == {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": 12,
        INITIAL_CODEC_CHUNK_FRAMES_PARAM: 3,
    }


def test_stream_concatenates_to_offline_decode(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    rows = _rows(23, seed=1)
    messages = _run_stream(scheduler, rows)

    stream_msgs = [m for m in messages if m.type == "stream"]
    # 23 frames at initial=5/steady=10: chunks of 5, 10, and the 8-frame tail.
    assert [
        _decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME for m in stream_msgs
    ] == [5, 10, 8]
    for msg in stream_msgs:
        assert msg.data["sample_rate"] == SAMPLE_RATE
        assert msg.data["modality"] == "audio"
        assert msg.metadata == {"modality": "audio"}

    audio = _concat_stream_audio(messages, "req")
    np.testing.assert_array_equal(audio, reference_waveform(rows[:, 1:]).numpy())

    results = [m for m in messages if m.type == "result"]
    assert len(results) == 1
    final = results[0].data
    assert isinstance(final, StagePayload)
    assert final.data["modality"] == "audio"
    assert final.data["sample_rate"] == SAMPLE_RATE
    assert final.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 23,
        "total_tokens": 26,
        "engine_time_s": 0.5,
    }


def test_initial_chunk_frames_request_override(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    rows = _rows(14, seed=2)
    metadata = _metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 2})
    messages = _run_stream(scheduler, rows, metadata=metadata)
    sizes = [
        _decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream"
    ]
    assert sizes == [2, 10, 2]
    audio = _concat_stream_audio(messages, "req")
    np.testing.assert_array_equal(audio, reference_waveform(rows[:, 1:]).numpy())


def test_explicit_zero_initial_chunk_means_steady_only(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=5,
    )
    rows = _rows(12, seed=3)
    metadata = _metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 0})
    messages = _run_stream(scheduler, rows, metadata=metadata)
    sizes = [
        _decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream"
    ]
    assert sizes == [10, 2]


def test_interleaved_streams_are_isolated(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=3,
    )
    rows_a = _rows(17, seed=10)
    rows_b = _rows(9, seed=11)
    metadata = _metadata()
    chunk_id = 0
    for index in range(max(len(rows_a), len(rows_b))):
        if index < len(rows_a):
            scheduler._on_chunk("a", _stream_item(rows_a[index], metadata, chunk_id))
            chunk_id += 1
        if index < len(rows_b):
            scheduler._on_chunk("b", _stream_item(rows_b[index], metadata, chunk_id))
            chunk_id += 1
    scheduler._on_done("b")
    scheduler._on_streaming_new_request("b", _terminal_payload(rows_b, request_id="b"))
    scheduler._on_done("a")
    scheduler._on_streaming_new_request("a", _terminal_payload(rows_a, request_id="a"))
    messages = _drain(scheduler)

    audio_a = _concat_stream_audio(messages, "a")
    audio_b = _concat_stream_audio(messages, "b")
    np.testing.assert_array_equal(audio_a, reference_waveform(rows_a[:, 1:]).numpy())
    np.testing.assert_array_equal(audio_b, reference_waveform(rows_b[:, 1:]).numpy())


def test_near_due_streams_coalesce_into_one_step(monkeypatch) -> None:
    """A due stream must not step alone past near-due peers.

    A decode step costs one forward over the full slot width regardless of
    how many slots are active, so when A (6 buffered, due) steps while B sits
    at 5, B's own step a moment later doubles the GPU work. The pump must
    instead step both at T=5 in a single _decode_frame call.
    """
    processor = FakeProcessor()
    codec = processor.audio_tokenizer
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=3,
    )
    rows_a = _rows(9, seed=40)
    rows_b = _rows(8, seed=41)
    metadata = _metadata()
    messages: list = []
    chunk_id = 0
    # Warm both streams past their initial chunk so both sit at the steady
    # threshold (6) with empty buffers.
    for index in range(3):
        scheduler._on_chunk("a", _stream_item(rows_a[index], metadata, chunk_id))
        chunk_id += 1
    for index in range(3):
        scheduler._on_chunk("b", _stream_item(rows_b[index], metadata, chunk_id))
        chunk_id += 1
    messages += _drain(scheduler)
    # B buffers 5 frames (one short of due); then A crosses its threshold.
    for index in range(3, 8):
        scheduler._on_chunk("b", _stream_item(rows_b[index], metadata, chunk_id))
        chunk_id += 1
    calls_before = codec.frame_calls
    for index in range(3, 9):
        scheduler._on_chunk("a", _stream_item(rows_a[index], metadata, chunk_id))
        chunk_id += 1
    assert codec.frame_calls - calls_before == 1
    coalesced = _drain(scheduler)
    sizes = {
        msg.request_id: _decode_audio(msg.data).shape[1]
        for msg in coalesced
        if msg.type == "stream"
    }
    assert sizes == {
        "a": 5 * SAMPLES_PER_FRAME,
        "b": 5 * SAMPLES_PER_FRAME,
    }
    messages += coalesced
    # Finishing both streams must still produce exactly the offline waveform,
    # proving the rider step advanced B's slot state correctly.
    scheduler._on_done("a")
    scheduler._on_streaming_new_request("a", _terminal_payload(rows_a, request_id="a"))
    scheduler._on_done("b")
    scheduler._on_streaming_new_request("b", _terminal_payload(rows_b, request_id="b"))
    messages += _drain(scheduler)
    np.testing.assert_array_equal(
        _concat_stream_audio(messages, "a"),
        reference_waveform(rows_a[:, 1:]).numpy(),
    )
    np.testing.assert_array_equal(
        _concat_stream_audio(messages, "b"),
        reference_waveform(rows_b[:, 1:]).numpy(),
    )


def test_explicit_zero_initial_chunk_is_not_pulled_below_steady(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=2,
    )
    rows_a = _rows(2, seed=42)
    rows_b = _rows(6, seed=43)
    metadata_a = _metadata()
    metadata_b = _metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 0})
    chunk_id = 0

    # B explicitly opts out of a smaller first chunk, so five buffered frames
    # must not ride along when A crosses its own first-chunk threshold.
    for index in range(5):
        scheduler._on_chunk("b", _stream_item(rows_b[index], metadata_b, chunk_id))
        chunk_id += 1
    for index in range(2):
        scheduler._on_chunk("a", _stream_item(rows_a[index], metadata_a, chunk_id))
        chunk_id += 1

    messages = _drain(scheduler)
    assert [m.request_id for m in messages if m.type == "stream"] == ["a"]

    scheduler._on_chunk("b", _stream_item(rows_b[5], metadata_b, chunk_id))
    messages += _drain(scheduler)
    b_chunks = [
        _decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream" and m.request_id == "b"
    ]
    assert b_chunks == [6]


def test_positive_initial_chunk_is_not_pulled_below_threshold(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=6,
        initial_chunk_frames=2,
    )
    rows_a = _rows(1, seed=44)
    rows_b = _rows(5, seed=45)
    metadata_a = _metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 1})
    metadata_b = _metadata(**{INITIAL_CODEC_CHUNK_FRAMES_PARAM: 5})
    chunk_id = 0

    # B asked for a 5-frame first chunk; four buffered frames must not ride
    # along when A becomes due with a 1-frame floor.
    for index in range(4):
        scheduler._on_chunk("b", _stream_item(rows_b[index], metadata_b, chunk_id))
        chunk_id += 1
    scheduler._on_chunk("a", _stream_item(rows_a[0], metadata_a, chunk_id))
    chunk_id += 1

    messages = _drain(scheduler)
    assert [m.request_id for m in messages if m.type == "stream"] == ["a"]

    scheduler._on_chunk("b", _stream_item(rows_b[4], metadata_b, chunk_id))
    messages += _drain(scheduler)
    b_chunks = [
        _decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages
        if m.type == "stream" and m.request_id == "b"
    ]
    assert b_chunks == [5]


def test_slot_reuse_after_release(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=4,
        initial_chunk_frames=2,
    )
    rows_a = _rows(7, seed=20)
    messages_a = _run_stream(scheduler, rows_a, request_id="a")
    np.testing.assert_array_equal(
        _concat_stream_audio(messages_a, "a"),
        reference_waveform(rows_a[:, 1:]).numpy(),
    )
    # The single slot was released and reset; a second stream must start from
    # a fresh offset, not continue where "a" left off.
    rows_c = _rows(6, seed=21)
    messages_c = _run_stream(scheduler, rows_c, request_id="c")
    np.testing.assert_array_equal(
        _concat_stream_audio(messages_c, "c"),
        reference_waveform(rows_c[:, 1:]).numpy(),
    )


def test_slot_exhaustion_falls_back_to_offline_decode(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=4,
        initial_chunk_frames=2,
    )
    metadata = _metadata()
    rows_a = _rows(9, seed=30)
    rows_b = _rows(8, seed=31)
    messages: list = []
    for index, row in enumerate(rows_a[:5]):
        scheduler._on_chunk("a", _stream_item(row, metadata, index))
    # "b" cannot get a slot while "a" holds the only one: nothing may stream.
    for index, row in enumerate(rows_b):
        scheduler._on_chunk("b", _stream_item(row, metadata, index))
    messages += _drain(scheduler)
    assert all(m.request_id != "b" for m in messages if m.type == "stream")
    scheduler._on_done("b")
    scheduler._on_streaming_new_request("b", _terminal_payload(rows_b, request_id="b"))
    messages_b = _drain(scheduler)
    sizes_b = [
        _decode_audio(m.data).shape[1] // SAMPLES_PER_FRAME
        for m in messages_b
        if m.type == "stream"
    ]
    assert sizes_b == [8]  # one catch-up chunk decoded through the offline lane
    np.testing.assert_array_equal(
        _concat_stream_audio(messages_b, "b"),
        reference_waveform(rows_b[:, 1:]).numpy(),
    )
    # "a" is unaffected by b's offline-lane decode.
    for index, row in enumerate(rows_a[5:], start=5):
        scheduler._on_chunk("a", _stream_item(row, metadata, index))
    scheduler._on_done("a")
    scheduler._on_streaming_new_request("a", _terminal_payload(rows_a, request_id="a"))
    messages += _drain(scheduler)
    np.testing.assert_array_equal(
        _concat_stream_audio(messages, "a"),
        reference_waveform(rows_a[:, 1:]).numpy(),
    )


def test_done_without_chunks_decodes_payload_codes(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(monkeypatch, processor)
    rows = _rows(5, seed=40)
    scheduler._on_done("req")
    scheduler._on_streaming_new_request("req", _terminal_payload(rows))
    messages = _drain(scheduler)
    np.testing.assert_array_equal(
        _concat_stream_audio(messages, "req"),
        reference_waveform(rows[:, 1:]).numpy(),
    )
    assert [m.type for m in messages] == ["stream", "result"]


def test_abort_releases_slot(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_slots=1,
        stream_chunk_frames=4,
        initial_chunk_frames=2,
    )
    metadata = _metadata()
    rows_a = _rows(3, seed=50)
    for index, row in enumerate(rows_a):
        scheduler._on_chunk("a", _stream_item(row, metadata, index))
    scheduler.abort("a")
    _drain(scheduler)
    rows_b = _rows(6, seed=51)
    messages_b = _run_stream(scheduler, rows_b, request_id="b")
    np.testing.assert_array_equal(
        _concat_stream_audio(messages_b, "b"),
        reference_waveform(rows_b[:, 1:]).numpy(),
    )


def test_non_streaming_path_with_and_without_live_session(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(monkeypatch, processor)

    def offline_payload(rows: torch.Tensor, request_id: str) -> StagePayload:
        state = MossTTSLocalState(
            text="x",
            audio_codes=rows[:, 1:].clone(),
            prompt_tokens=2,
            completion_tokens=int(rows.shape[0]),
            engine_time_s=0.25,
        )
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="", params={}),
            data=state.to_dict(),
        )

    rows_1 = _rows(11, seed=60)
    rows_2 = _rows(4, seed=61)

    # Before any stream: the pre-existing processor path is used.
    results = scheduler._vocode_batch(
        [offline_payload(rows_1, "r1"), offline_payload(rows_2, "r2")]
    )
    assert processor.decode_calls == 1
    waves_before = [_decode_audio(result.data) for result in results]
    for result in results:
        assert result.data["sample_rate"] == SAMPLE_RATE
        assert result.data["modality"] == "audio"
        assert result.data["usage"]["prompt_tokens"] == 2

    # A streaming request opens the persistent session...
    _run_stream(scheduler, _rows(6, seed=62))
    assert scheduler._session is not None

    # ...after which the processor path would raise ("already streaming"), so
    # offline decodes must go through the session's offline lane and still
    # produce identical audio.
    results = scheduler._vocode_batch(
        [offline_payload(rows_1, "r3"), offline_payload(rows_2, "r4")]
    )
    assert processor.decode_calls == 1
    waves_after = [_decode_audio(result.data) for result in results]
    for before, after in zip(waves_before, waves_after):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(
        waves_after[0], reference_waveform(rows_1[:, 1:]).numpy()
    )


def test_offline_lane_waves_split_across_slots(monkeypatch) -> None:
    del monkeypatch
    processor = FakeProcessor()
    # Constructed directly: max_step_frames is not a factory knob.
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        processor,
        max_batch_size=2,
        max_step_frames=3,
        stream_chunk_frames=3,
    )
    _run_stream(scheduler, _rows(5, seed=70))  # open the session
    rows_list = [_rows(7, seed=71), _rows(2, seed=72), _rows(5, seed=73)]
    payloads = []
    for index, rows in enumerate(rows_list):
        state = MossTTSLocalState(text="x", audio_codes=rows[:, 1:].clone())
        payloads.append(
            StagePayload(
                request_id=f"r{index}",
                request=OmniRequest(inputs="", params={}),
                data=state.to_dict(),
            )
        )
    results = scheduler._vocode_batch(payloads)
    for rows, result in zip(rows_list, results):
        np.testing.assert_array_equal(
            _decode_audio(result.data), reference_waveform(rows[:, 1:]).numpy()
        )


def test_stop_closes_persistent_streaming_session(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(monkeypatch, processor)
    scheduler._on_chunk("req", _stream_item(_rows(1, seed=74)[0], _metadata()))
    assert scheduler._session is not None
    assert processor.audio_tokenizer._streaming_state is not None

    scheduler.stop()

    assert scheduler._session is None
    assert scheduler._stream_states == {}
    assert processor.audio_tokenizer._streaming_state is None

    # Reusing the same codec instance after stop must be able to open a fresh
    # streaming context instead of tripping the codec's nested-session guard.
    restarted = _make_scheduler(monkeypatch, processor)
    restarted._on_chunk("req2", _stream_item(_rows(1, seed=75)[0], _metadata()))
    assert restarted._session is not None
    assert processor.audio_tokenizer._streaming_state is not None
    restarted.stop()


class _FailingCodec(FakeCodec):
    """FakeCodec whose Nth ``_decode_frame`` call raises."""

    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call

    def _decode_frame(self, codes: torch.Tensor, codes_lengths: torch.Tensor):
        if self.frame_calls + 1 == self._fail_on_call:
            self.frame_calls += 1
            raise RuntimeError("codec decode exploded")
        return super()._decode_frame(codes, codes_lengths)


def test_decode_step_failure_fails_participants_only(monkeypatch) -> None:
    """A failed decode step errors every participant, releases their slots,
    leaves non-participants and already-emitted audio untouched, and keeps
    the scheduler usable for new streams.
    """
    processor = FakeProcessor()
    processor.audio_tokenizer = _FailingCodec(fail_on_call=3)
    scheduler = _make_scheduler(
        monkeypatch,
        processor,
        stream_chunk_frames=10,
        initial_chunk_frames=2,
    )
    metadata = _metadata()

    # Decode #1 succeeds: "c" emits its initial 2-frame chunk.
    rows_c = _rows(2, seed=100)
    for index, row in enumerate(rows_c):
        scheduler._on_chunk("c", _stream_item(row, metadata, index))
    early = _drain(scheduler)
    assert [m.type for m in early] == ["stream"]
    assert early[0].request_id == "c"

    # "b" first emits its configured initial chunk, then buffers below its
    # steady threshold. When "a" crosses its 2-frame initial threshold, "b" can
    # ride along because it did not opt out of smaller initial chunks.
    rows_b = _rows(5, seed=101)
    for index, row in enumerate(rows_b):
        scheduler._on_chunk("b", _stream_item(row, metadata, index))
    early_b = _drain(scheduler)
    assert [m.type for m in early_b] == ["stream"]
    assert early_b[0].request_id == "b"
    rows_a = _rows(2, seed=102)
    for index, row in enumerate(rows_a):
        scheduler._on_chunk("a", _stream_item(row, metadata, index))

    messages = _drain(scheduler)
    errors = [m for m in messages if m.type == "error"]
    assert {m.request_id for m in errors} == {"a", "b"}
    assert all(m.request_id != "c" for m in messages if m.type == "stream")

    # Both participants' state is gone and their slots are back in the pool.
    assert "a" not in scheduler._stream_states
    assert "b" not in scheduler._stream_states
    assert len(scheduler._session._free_stream_slots) == scheduler._stream_slots - 1

    # The scheduler keeps serving: a fresh stream decodes normally.
    rows_d = _rows(6, seed=103)
    messages_d = _run_stream(scheduler, rows_d, request_id="d")
    np.testing.assert_array_equal(
        _concat_stream_audio(messages_d, "d"),
        reference_waveform(rows_d[:, 1:]).numpy(),
    )


def test_stream_chunk_requires_metadata_contract(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(monkeypatch, processor)
    row = _rows(1, seed=80)[0]
    with pytest.raises(RuntimeError, match="missing metadata"):
        scheduler.on_stream_chunk("req", _stream_item(row, None))
    with pytest.raises(RuntimeError, match="stream"):
        scheduler.on_stream_chunk(
            "req2", _stream_item(row, {"modality": "audio_codes"})
        )
    with pytest.raises(ValueError, match="modality"):
        scheduler.on_stream_chunk(
            "req3", _stream_item(row, {"stream": True, "modality": "text"})
        )
    with pytest.raises(ValueError, match="channels"):
        scheduler.on_stream_chunk(
            "req4", _stream_item(torch.zeros(2, dtype=torch.long), _metadata())
        )


def test_is_streaming_payload(monkeypatch) -> None:
    processor = FakeProcessor()
    scheduler = _make_scheduler(monkeypatch, processor)
    assert scheduler.is_streaming_payload(_terminal_payload(_rows(2, seed=90)))
    non_stream = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="", params={}),
        data={},
    )
    assert not scheduler.is_streaming_payload(non_stream)
