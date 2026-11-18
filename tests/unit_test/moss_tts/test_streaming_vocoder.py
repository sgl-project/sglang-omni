# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from sglang_omni.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerVocoder,
    MossAudioVocoder,
)
from sglang_omni.models.moss_tts.codec_session import CodecStreamSession
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
from tests.unit_test.moss_tts.test_encoder import tiny_moss_audio_tokenizer_v1_config

N_VQ = 3
PAD = 99
SAMPLES_PER_FRAME = 4


class FakeQuantizer:
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.to(torch.float32).sum(dim=0).unsqueeze(1)


class FakeDecoderStage(nn.Module):
    module_type = "PatchedPretransform"
    patch_size = SAMPLES_PER_FRAME
    is_downsample = False

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor):
        return (
            x.repeat_interleave(SAMPLES_PER_FRAME, dim=-1),
            input_lengths * SAMPLES_PER_FRAME,
        )


class FakeCodec(nn.Module):
    """Stateful mono codec: a frame's PCM depends on its slot's frame offset."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.decoder = nn.ModuleList([FakeDecoderStage()])
        self.quantizer = FakeQuantizer()
        self.offsets: torch.Tensor | None = None
        self.batch_shapes: list[tuple[int, ...]] = []

    def initialize_decoder_state_pool(self, state_capacity, scratch_capacity=0):
        self.offsets = torch.zeros(state_capacity + scratch_capacity, dtype=torch.long)

    def reset_decoder_state_slots(self, slot_ids: torch.Tensor) -> None:
        self.offsets[slot_ids] = 0

    def close_decoder_state_pool(self) -> None:
        self.offsets = None

    def decode_streaming_tensors(self, codes, codes_lengths, slot_ids, valid_rows):
        self.batch_shapes.append(tuple(codes.shape))
        _, batch_size, step_t = codes.shape
        audio = torch.zeros(batch_size, 1, step_t * SAMPLES_PER_FRAME)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        for row, slot in enumerate(slot_ids.tolist()):
            if not bool(valid_rows[row]):
                continue
            base = int(self.offsets[slot])
            for frame in range(int(codes_lengths[row])):
                value = float(codes[:, row, frame].sum()) + 1000.0 * (base + frame)
                start = frame * SAMPLES_PER_FRAME
                audio[row, 0, start : start + SAMPLES_PER_FRAME] = value
            lengths[row] = int(codes_lengths[row]) * SAMPLES_PER_FRAME
            self.offsets[slot] += int(codes_lengths[row])
        return audio, lengths


class FakeAudioVocoder:
    sample_rate = 24000

    def __init__(self) -> None:
        self.model = FakeCodec()

    def decode_codes(self, segments: list[torch.Tensor]) -> list[torch.Tensor]:
        return [reference_waveform(segment) for segment in segments]


def reference_waveform(codes: torch.Tensor) -> torch.Tensor:
    """Offline decode of one [T, n_vq] segment from a fresh codec state."""
    values = [float(row.sum()) + 1000.0 * t for t, row in enumerate(codes)]
    return torch.tensor(values, dtype=torch.float32).repeat_interleave(
        SAMPLES_PER_FRAME
    )


def make_scheduler(**kwargs) -> tuple[MossStreamingVocoderScheduler, FakeCodec]:
    processor = SimpleNamespace(
        model_config=SimpleNamespace(n_vq=N_VQ, audio_pad_code=PAD, sampling_rate=24000)
    )
    audio_vocoder = FakeAudioVocoder()
    vocoder = MossTTSVocoder(processor, audio_vocoder, "cpu")
    kwargs.setdefault("stream_chunk_frames", 2)
    kwargs.setdefault("coalesce_floor_frames", 1)
    kwargs.setdefault("vocoder_cuda_graph", False)
    scheduler = MossStreamingVocoderScheduler(vocoder, **kwargs)
    scheduler.session = CodecStreamSession(
        audio_vocoder.model, stream_slots=scheduler.stream_slots, n_vq=N_VQ
    )
    return scheduler, audio_vocoder.model


def apply_delay_pattern(raw_codes: torch.Tensor) -> torch.Tensor:
    frames, n_vq = raw_codes.shape
    delayed = torch.full((frames + n_vq - 1, n_vq), PAD, dtype=torch.long)
    for channel in range(n_vq):
        delayed[channel : channel + frames, channel] = raw_codes[:, channel]
    return delayed


def make_payload(request_id: str, delayed: torch.Tensor) -> StagePayload:
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


def item(data: torch.Tensor, chunk_id: int = 0) -> StreamItem:
    return StreamItem(
        chunk_id=chunk_id,
        data=data,
        from_stage="tts_engine",
        metadata={
            "modality": "audio_codes",
            "stream": True,
            "n_vq": N_VQ,
            "audio_pad_code": PAD,
            "sample_rate": 24000,
        },
    )


def drain(scheduler: MossStreamingVocoderScheduler) -> list:
    messages = []
    while True:
        try:
            messages.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return messages


def streamed_audio(messages: list, request_id: str = "req") -> np.ndarray:
    return np.concatenate(
        [
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            for message in messages
            if message.type == "stream" and message.request_id == request_id
        ]
    )


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


def test_streaming_matches_per_segment_offline_decode() -> None:
    raw_codes = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [PAD, PAD, PAD],
            [13, 14, 15],
            [16, 17, 18],
            [19, 20, 21],
        ],
        dtype=torch.long,
    )
    delayed = apply_delay_pattern(raw_codes)
    scheduler, _ = make_scheduler()
    expected = torch.cat(
        [reference_waveform(raw_codes[:4]), reference_waveform(raw_codes[5:])]
    )

    scheduler.handle_streaming_new_request("req", make_payload("req", delayed))
    for index in range(int(delayed.shape[0])):
        scheduler.handle_stream_chunk("req", item(delayed[index], index))
    scheduler.handle_stream_done("req")

    messages = drain(scheduler)
    assert sum(message.type == "stream" for message in messages) > 2
    np.testing.assert_array_equal(streamed_audio(messages), expected.numpy())
    result = next(message for message in messages if message.type == "result")
    assert "delayed_audio_codes" not in result.data.data
    assert result.data.data["usage"]["completion_tokens"] == int(delayed.shape[0])
    assert "req" not in scheduler.stream_states


def test_concurrent_streams_share_one_batched_decode_step() -> None:
    scheduler, codec = make_scheduler(initial_chunk_frames=2)
    raw = {
        rid: torch.randint(
            0, PAD, (6, N_VQ), generator=torch.Generator().manual_seed(i)
        )
        for i, rid in enumerate(("a", "b"))
    }
    delayed = {rid: apply_delay_pattern(codes) for rid, codes in raw.items()}
    for rid in raw:
        scheduler.handle_streaming_new_request(rid, make_payload(rid, delayed[rid]))
    for index in range(int(delayed["a"].shape[0])):
        scheduler.on_stream_chunk_batch(
            [(rid, item(delayed[rid][index], index)) for rid in raw]
        )
    for rid in raw:
        scheduler.handle_stream_done(rid)

    messages = drain(scheduler)
    assert any(shape[1] >= 2 for shape in codec.batch_shapes)
    for rid, codes in raw.items():
        np.testing.assert_array_equal(
            streamed_audio(messages, rid), reference_waveform(codes).numpy()
        )


def test_chunks_and_done_before_payload_preserve_final_tail() -> None:
    raw_codes = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=torch.long,
    )
    delayed = apply_delay_pattern(raw_codes)
    scheduler, _ = make_scheduler()

    scheduler.handle_stream_chunk("req", item(delayed))
    scheduler.handle_stream_done("req")
    assert "req" in scheduler.pending_done
    scheduler.handle_streaming_new_request("req", make_payload("req", delayed))

    messages = drain(scheduler)
    np.testing.assert_array_equal(
        streamed_audio(messages), reference_waveform(raw_codes).numpy()
    )
    assert "req" not in scheduler.pending_done
    assert "req" not in scheduler.stream_states


def test_abort_drops_state_and_late_chunks() -> None:
    delayed = apply_delay_pattern(
        torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    )
    scheduler, _ = make_scheduler()
    scheduler.handle_stream_chunk("req", item(delayed[:2]))
    assert "req" in scheduler.stream_states

    scheduler.abort("req")
    scheduler.handle_stream_chunk("req", item(delayed[2:]))

    assert "req" not in scheduler.stream_states
    assert drain(scheduler) == []


def test_abort_releases_advanced_slot_for_next_stream() -> None:
    scheduler, _ = make_scheduler(stream_slots=1)
    first = apply_delay_pattern(torch.randint(0, PAD, (3, N_VQ)))
    scheduler.handle_streaming_new_request("a", make_payload("a", first))
    for index in range(N_VQ + 1):
        scheduler.handle_stream_chunk("a", item(first[index], index))
    assert scheduler.session.stream_slots_in_use == {0}
    assert len(drain(scheduler)) == 1

    scheduler.abort("a")

    assert scheduler.session.stream_slots_in_use == set()
    raw = torch.randint(0, PAD, (4, N_VQ))
    delayed = apply_delay_pattern(raw)
    scheduler.handle_streaming_new_request("b", make_payload("b", delayed))
    for index in range(int(delayed.shape[0])):
        scheduler.handle_stream_chunk("b", item(delayed[index], index))
    scheduler.handle_stream_done("b")
    np.testing.assert_array_equal(
        streamed_audio(drain(scheduler), "b"), reference_waveform(raw).numpy()
    )


def test_slot_starved_stream_uses_standalone_segment_decode(monkeypatch) -> None:
    scheduler, _ = make_scheduler(stream_slots=1)
    calls = []
    standalone = scheduler.vocoder.decode_segments
    monkeypatch.setattr(
        scheduler.vocoder,
        "decode_segments",
        lambda segments: calls.append(len(segments)) or standalone(segments),
    )
    raw = {rid: torch.randint(0, PAD, (4, N_VQ)) for rid in ("a", "b")}
    delayed = {rid: apply_delay_pattern(codes) for rid, codes in raw.items()}
    for rid in raw:
        scheduler.handle_streaming_new_request(rid, make_payload(rid, delayed[rid]))
    for index in range(int(delayed["a"].shape[0])):
        scheduler.on_stream_chunk_batch(
            [(rid, item(delayed[rid][index], index)) for rid in raw]
        )
    for rid in raw:
        scheduler.handle_stream_done(rid)

    messages = drain(scheduler)
    assert calls == [1]
    for rid, codes in raw.items():
        np.testing.assert_array_equal(
            streamed_audio(messages, rid), reference_waveform(codes).numpy()
        )


def test_pad_closed_short_segment_streams_before_done() -> None:
    raw_codes = torch.tensor(
        [[1, 2, 3], [PAD, PAD, PAD], [PAD, PAD, PAD]], dtype=torch.long
    )
    delayed = apply_delay_pattern(raw_codes)
    scheduler, _ = make_scheduler()

    scheduler.handle_streaming_new_request("req", make_payload("req", delayed))
    for index in range(int(delayed.shape[0])):
        scheduler.handle_stream_chunk("req", item(delayed[index], index))

    np.testing.assert_array_equal(
        streamed_audio(drain(scheduler)), reference_waveform(raw_codes[:1]).numpy()
    )


def test_streaming_codec_view_keeps_native_decoder_stateless() -> None:
    """Offline dispatch must preserve both its PCM and the live v1 continuation."""
    torch.manual_seed(2805)
    codec = MossAudioTokenizerVocoder(
        tiny_moss_audio_tokenizer_v1_config(),
        parameter_device="cpu",
        decoder_dtype=torch.float32,
        compute_dtype=torch.float32,
        attention_backend="sdpa",
    )
    codec.eval()
    processor = SimpleNamespace(model_config=SimpleNamespace(n_vq=2, audio_pad_code=4))
    vocoder = MossTTSVocoder(processor, MossAudioVocoder(codec, device="cpu"), "cpu")
    scheduler = MossStreamingVocoderScheduler(
        vocoder, stream_slots=2, vocoder_cuda_graph=False
    )
    offline_codes = torch.tensor([[0, 1], [1, 2], [3, 0], [2, 3]])

    def decode_offline() -> bytes:
        payload = StagePayload(
            request_id="offline",
            request=OmniRequest(inputs="hello", params={"stream": False}),
            data=MossTTSState(
                delayed_audio_codes=apply_delay_pattern(offline_codes)
            ).to_dict(),
        )
        return asyncio.run(scheduler.vocode_payload(payload)).data["audio_waveform"]

    before = decode_offline()
    session = scheduler.ensure_session()
    slot = session.acquire()
    first_codes = torch.tensor([[1, 3, 0], [2, 0, 1]])
    next_codes = torch.tensor([[2, 1], [3, 2]])
    try:
        session.step({slot: first_codes})
        expected_continuation = session.step({slot: next_codes})[slot]
        session.reset_slots([slot])
        session.step({slot: first_codes})
        during = decode_offline()
        actual_continuation = session.step({slot: next_codes})[slot]
        assert during == before
        torch.testing.assert_close(
            actual_continuation, expected_continuation, rtol=0, atol=0
        )
    finally:
        scheduler.on_serving_stop()
    after = decode_offline()
    assert after == before
