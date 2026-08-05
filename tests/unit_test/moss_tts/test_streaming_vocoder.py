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
from sglang_omni.models.moss_tts.stages import _MossTTSVocoder
from sglang_omni.models.moss_tts.streaming_vocoder import MossStreamingVocoderScheduler
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
) -> tuple[MossStreamingVocoderScheduler, _FakeAudioTokenizer, _MossTTSVocoder]:
    processor = SimpleNamespace(
        model_config=SimpleNamespace(
            n_vq=3,
            audio_pad_code=99,
            sampling_rate=24000,
        )
    )
    tokenizer = _FakeAudioTokenizer()
    vocoder = _MossTTSVocoder(processor, tokenizer, "cpu")
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
