# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    MossTTSSGLangRequestData,
    make_moss_tts_stream_output_builder,
)
from sglang_omni.models.moss_tts.streaming_vocoder import MossStreamingVocoderScheduler
from sglang_omni.models.moss_tts.vocoder import MossTTSVocoder
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput


class _FakeAudioTokenizer:
    sample_rate = 24000

    class _Model:
        class config:
            hop_length = 4

    def __init__(self) -> None:
        self.model = self._Model()
        self.decode_inputs: list[torch.Tensor] = []

    def decode_codes(self, segments: list[torch.Tensor]) -> list[torch.Tensor]:
        decoded = []
        offsets = torch.arange(4, dtype=torch.float32) / 10.0
        for segment in segments:
            self.decode_inputs.append(segment.detach().clone())
            frames = [row.sum().float().repeat(4) + offsets for row in segment]
            body = torch.cat(frames) if frames else torch.empty(0)
            tail = torch.tensor([10000.0, 10001.0])
            decoded.append(torch.cat([body, tail]))
        return decoded


def _make_scheduler(
    *,
    stream_stride: int = 3,
    stream_followup_stride: int = 2,
    stream_overlap_tokens: int = 1,
    stream_holdback_tokens: int = 0,
) -> tuple[MossStreamingVocoderScheduler, _FakeAudioTokenizer, MossTTSVocoder]:
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            n_vq=3,
            audio_pad_code=99,
            sampling_rate=24000,
        )
    )
    tokenizer = _FakeAudioTokenizer()
    vocoder = MossTTSVocoder(processor, tokenizer, "cpu")
    scheduler = MossStreamingVocoderScheduler(
        vocoder,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
    )
    return scheduler, tokenizer, vocoder


def _apply_delay_pattern(raw_codes: torch.Tensor, pad_code: int = 99) -> torch.Tensor:
    frames, n_vq = raw_codes.shape
    delayed = torch.full(
        (frames + n_vq - 1, n_vq),
        int(pad_code),
        dtype=torch.long,
    )
    for channel in range(n_vq):
        delayed[channel : channel + frames, channel] = raw_codes[:, channel]
    return delayed


def _payload(request_id: str, delayed: torch.Tensor) -> StagePayload:
    state = MossTTSState(
        delayed_audio_codes=delayed,
        prompt_tokens=2,
        completion_tokens=int(delayed.shape[0]),
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data=state.to_dict(),
    )


def _item(data: torch.Tensor, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=data,
        from_stage="tts_engine",
        metadata={
            "modality": "audio_codes",
            "stream": True,
            "n_vq": 3,
            "audio_pad_code": 99,
            "sample_rate": 24000,
        },
    )


def _drain(scheduler: MossStreamingVocoderScheduler) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def test_zero_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="stream overlap must be > 0"):
        _make_scheduler(stream_overlap_tokens=0)


def test_stream_builder_emits_prefix_and_only_new_audio_rows() -> None:
    config = SimpleNamespace(
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_pad_code=99,
        audio_vocab_size=99,
        sampling_rate=24000,
    )
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs="hello", params={"stream": True}),
        data={},
    )
    data = MossTTSSGLangRequestData(
        req=SimpleNamespace(inflight_middle_chunks=0),
        stage_payload=payload,
        state=MossTTSState(),
        model_config=config,
        assistant_prefix_rows=torch.tensor(
            [[1, 99, 99], [10, 99, 99], [12, 1, 99]], dtype=torch.long
        ),
        output_rows=[torch.tensor([12, 2, 3])],
    )
    builder = make_moss_tts_stream_output_builder()

    messages = builder("req", data, RequestOutput("req", data=12))

    assert len(messages) == 1
    assert messages[0].data.tolist() == [[1, 99], [2, 3]]
    assert messages[0].metadata["row_index"] == 0
    assert messages[0].metadata["modality"] == "audio_codes"

    assert builder("req", data, RequestOutput("req", data=12)) == []
    data.output_rows.append(torch.tensor([12, 4, 5]))
    messages = builder("req", data, RequestOutput("req", data=12))
    assert messages[0].data.tolist() == [4, 5]
    assert messages[0].metadata["row_index"] == 2

    data.output_rows.append(torch.tensor([11, 99, 99]))
    assert builder("req", data, RequestOutput("req", data=11)) == []


def test_streaming_matches_full_decode_across_segments() -> None:
    raw_codes = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [99, 99, 99],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
        ],
        dtype=torch.long,
    )
    delayed = _apply_delay_pattern(raw_codes)
    scheduler, tokenizer, vocoder = _make_scheduler()
    payload = _payload("req", delayed)
    full_state, full_delayed = vocoder.prepare_item(payload)
    full, _ = vocoder._decode_audio(full_state, full_delayed)

    scheduler._on_streaming_new_request("req", payload)
    scheduler._on_chunk("req", _item(delayed[:5], 0))
    scheduler._on_chunk("req", _item(delayed[5:], 1))
    scheduler._on_done("req")

    messages = _drain(scheduler)
    chunks = [
        np.frombuffer(message.data["audio_waveform"], dtype=np.float32).copy()
        for message in messages
        if message.type == "stream"
    ]
    np.testing.assert_array_equal(np.concatenate(chunks), full.numpy())
    assert max(int(codes.shape[0]) for codes in tokenizer.decode_inputs[2:]) <= 3
    result = next(message for message in messages if message.type == "result")
    assert "delayed_audio_codes" not in result.data.data
    assert result.data.data["usage"]["completion_tokens"] == int(delayed.shape[0])


def test_streaming_path_does_not_call_nonstream_batch_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delayed = _apply_delay_pattern(
        torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.long)
    )
    scheduler, tokenizer, vocoder = _make_scheduler()

    async def fail_decode_batch(*_args, **_kwargs):
        pytest.fail("streaming requests must not use the non-streaming batch decoder")

    monkeypatch.setattr(vocoder, "decode_batch", fail_decode_batch)
    scheduler._on_streaming_new_request("req", _payload("req", delayed))
    scheduler._on_chunk("req", _item(delayed))
    scheduler._on_done("req")

    messages = _drain(scheduler)

    assert tokenizer.decode_inputs
    assert any(message.type == "stream" for message in messages)
    assert any(message.type == "result" for message in messages)


def test_chunks_and_done_before_payload_preserve_final_tail() -> None:
    raw_codes = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=torch.long,
    )
    delayed = _apply_delay_pattern(raw_codes)
    scheduler, _, vocoder = _make_scheduler(stream_holdback_tokens=1)
    payload = _payload("req", delayed)
    full_state, full_delayed = vocoder.prepare_item(payload)
    full, _ = vocoder._decode_audio(full_state, full_delayed)

    scheduler._on_chunk("req", _item(delayed))
    scheduler._on_done("req")
    assert "req" in scheduler._pending_done
    scheduler._on_streaming_new_request("req", payload)

    messages = _drain(scheduler)
    chunks = [
        np.frombuffer(message.data["audio_waveform"], dtype=np.float32).copy()
        for message in messages
        if message.type == "stream"
    ]
    np.testing.assert_array_equal(np.concatenate(chunks), full.numpy())
    assert "req" not in scheduler._pending_done
    assert "req" not in scheduler._stream_states


def test_abort_drops_state_and_late_chunks() -> None:
    delayed = _apply_delay_pattern(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    )
    scheduler, _, _ = _make_scheduler()
    scheduler._on_chunk("req", _item(delayed[:2]))
    assert "req" in scheduler._stream_states

    scheduler.abort("req")
    scheduler._on_chunk("req", _item(delayed[2:]))

    assert "req" not in scheduler._stream_states
    assert _drain(scheduler) == []


class _RealtimeFakeAudioTokenizer(_FakeAudioTokenizer):
    """Decoder whose output length matches a 10 Hz codec, so slack is measurable in seconds."""

    SAMPLES_PER_FRAME = 2400

    class _Model:
        class config:
            hop_length = 2400

    def __init__(self) -> None:
        self.model = self._Model()
        self.decode_inputs: list[torch.Tensor] = []

    def decode_codes(self, segments: list[torch.Tensor]) -> list[torch.Tensor]:
        decoded = []
        for segment in segments:
            self.decode_inputs.append(segment.detach().clone())
            decoded.append(torch.zeros(len(segment) * self.SAMPLES_PER_FRAME))
        return decoded


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _make_realtime_scheduler(
    *,
    stream_slack_ladder=None,
    stream_slack_margin_s: float = 1.0,
    clock=None,
    stream_followup_stride: int = 2,
):
    processor = SimpleNamespace(
        model_config=SimpleNamespace(n_vq=3, audio_pad_code=99, sampling_rate=24000)
    )
    tokenizer = _RealtimeFakeAudioTokenizer()
    vocoder = MossTTSVocoder(processor, tokenizer, "cpu")
    scheduler = MossStreamingVocoderScheduler(
        vocoder,
        stream_stride=3,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=1,
        stream_holdback_tokens=0,
        stream_slack_ladder=stream_slack_ladder,
        stream_slack_margin_s=stream_slack_margin_s,
        clock=clock,
    )
    return scheduler, tokenizer


def test_moss_slack_ladder_rejects_bad_config() -> None:
    with pytest.raises(TypeError, match="must be a tuple or list"):
        _make_realtime_scheduler(stream_slack_ladder=8)
    with pytest.raises(ValueError, match="at least one entry"):
        _make_realtime_scheduler(stream_slack_ladder=())
    for non_int in ((8, 16.0), (True, 16), (8, "16")):
        with pytest.raises(TypeError, match="entries must be ints"):
            _make_realtime_scheduler(stream_slack_ladder=non_int)
    for bad in ((0, 16), (-8, 16)):
        with pytest.raises(ValueError, match="entries must be > 0"):
            _make_realtime_scheduler(stream_slack_ladder=bad)
    for not_ascending in ((16, 8), (8, 8, 16), (8, 32, 16)):
        with pytest.raises(ValueError, match="strictly ascending"):
            _make_realtime_scheduler(stream_slack_ladder=not_ascending)
    for bad_margin in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="stream_slack_margin_s"):
            _make_realtime_scheduler(
                stream_slack_ladder=(2, 4), stream_slack_margin_s=bad_margin
            )
    with pytest.raises(TypeError, match="stream_slack_margin_s"):
        _make_realtime_scheduler(stream_slack_ladder=(2, 4), stream_slack_margin_s=True)


def test_moss_slack_stride_fit_rule_boundaries() -> None:
    clock = _FakeClock(100.0)
    scheduler, _ = _make_realtime_scheduler(
        stream_slack_ladder=(2, 4, 8), stream_slack_margin_s=1.0, clock=clock
    )
    state = scheduler.create_stream_state("req")
    frame_s = _RealtimeFakeAudioTokenizer.SAMPLES_PER_FRAME / 24000.0

    def stride_for(slack_s: float) -> int:
        state.playback_deadline_s = clock.now + 1.0 + slack_s
        return scheduler._steady_followup_stride(state, now=clock.now)

    state.playback_deadline_s = 0.0
    assert scheduler._steady_followup_stride(state, now=clock.now) == 2
    assert stride_for(-5.0) == 2
    assert stride_for(4 * frame_s - 1e-6) == 2
    assert stride_for(4 * frame_s + 1e-6) == 4
    assert stride_for(8 * frame_s - 1e-6) == 4
    assert stride_for(8 * frame_s + 1e-6) == 8
    assert stride_for(60.0) == 8


def test_moss_slack_ladder_default_off_keeps_configured_stride() -> None:
    clock = _FakeClock(0.0)
    scheduler, _ = _make_realtime_scheduler(clock=clock, stream_followup_stride=2)
    assert scheduler._slack_ladder is None
    state = scheduler.create_stream_state("req")
    state.playback_deadline_s = 1e6
    assert scheduler._steady_followup_stride(state, now=clock.now) == 2
    clock.now = 1e9
    assert scheduler._steady_followup_stride(state, now=clock.now) == 2


def test_moss_slack_ladder_climbs_then_steps_down_with_buffer() -> None:
    clock = _FakeClock(0.0)
    scheduler, _ = _make_realtime_scheduler(
        stream_slack_ladder=(2, 4, 8), stream_slack_margin_s=0.0, clock=clock
    )
    raw = torch.arange(120, dtype=torch.long).reshape(40, 3) % 90
    delayed = _apply_delay_pattern(raw)
    scheduler._on_streaming_new_request("req", _payload("req", delayed))

    strides: list[int] = []
    emitted = 0
    for chunk_id, row in enumerate(delayed):
        scheduler._on_chunk("req", _item(row.unsqueeze(0), chunk_id))
        messages = _drain(scheduler)
        if not messages:
            continue
        emitted += len(messages)
        state = scheduler._stream_states["req"]
        strides.append(state.next_decode_rows - state.delayed_count)

    assert emitted > 0, "no audio was emitted"
    # The clock is frozen, so the client buffer only grows: the ladder must climb and
    # never step back down, and it must reach a rung above the first one.
    assert strides == sorted(strides), strides
    assert strides[0] == 2 and max(strides) > 2, strides

    state = scheduler._stream_states["req"]
    grown = scheduler._steady_followup_stride(state, now=clock.now)
    clock.now = state.playback_deadline_s
    assert scheduler._steady_followup_stride(state, now=clock.now) == 2
    assert grown > 2
    scheduler.abort("req")


def test_moss_vocoder_factory_forwards_slack_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages

    processor = SimpleNamespace(
        model_config=SimpleNamespace(n_vq=3, audio_pad_code=99, sampling_rate=24000)
    )
    monkeypatch.setattr(stages, "_load_moss_processor", lambda path: processor)
    monkeypatch.setattr(
        stages, "_resolve_audio_tokenizer_model_path", lambda *a, **k: "fake"
    )
    monkeypatch.setattr(
        stages, "load_moss_audio_vocoder", lambda *a, **k: _RealtimeFakeAudioTokenizer()
    )
    scheduler = stages.create_vocoder_executor(
        "fake-model",
        device="cpu",
        stream_slack_ladder=[2, 4, 8],
        stream_slack_margin_s=0.5,
    )
    assert scheduler._slack_ladder == (2, 4, 8)
    assert scheduler._slack_margin_s == 0.5
    assert scheduler._stream_followup_stride == 8
