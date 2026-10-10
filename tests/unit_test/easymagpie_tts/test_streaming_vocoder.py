# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue

import numpy as np
import pytest
import torch

from sglang_omni.models.easymagpie_tts import streaming_vocoder
from sglang_omni.models.easymagpie_tts.payload_types import EasyMagpieTTSState
from sglang_omni.models.easymagpie_tts.streaming_vocoder import (
    EasyMagpieStreamingVocoder,
    EasyMagpieStreamState,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem, StreamItemBatch
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage


@pytest.fixture
def vocoder(codec) -> EasyMagpieStreamingVocoder:
    return EasyMagpieStreamingVocoder(
        codec, startup_chunk_frames=(2, 6), steady_chunk_frames=8
    )


def make_payload(request_id: str, *, stream: bool, codes=None) -> StagePayload:
    state = EasyMagpieTTSState(audio_codes=codes, prompt_tokens=9, completion_tokens=4)
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi", params={"stream": stream}),
        data=state.to_dict(),
    )


def chunk(request_id: str, codes: torch.Tensor, index: int) -> IncomingMessage:
    return IncomingMessage(
        request_id=request_id,
        type="stream_chunk",
        data=StreamItem(
            chunk_id=index,
            data=codes,
            from_stage="tts_engine",
            metadata={"modality": "audio_codes", "stream": True},
        ),
    )


def drain(vocoder: EasyMagpieStreamingVocoder) -> list[OutgoingMessage]:
    messages = []
    while True:
        try:
            messages.append(vocoder.outbox.get_nowait())
        except queue.Empty:
            return messages


def waveform(message: OutgoingMessage) -> np.ndarray:
    return np.frombuffer(message.data["audio_waveform"], dtype=np.float32)


def test_stream_follows_chunk_schedule_and_matches_offline_decode(
    vocoder, codec
) -> None:
    codes = torch.randint(0, 16, (13, 4))
    vocoder.handle_streaming_new_request("a", make_payload("a", stream=True))
    for index, row in enumerate(codes):
        vocoder.handle_stream_chunk_batch([chunk("a", row, index)])
    vocoder.handle_stream_done("a")

    messages = drain(vocoder)
    assert [m.type for m in messages] == ["stream", "stream", "stream", "result"]
    sizes = [waveform(m).size // 12 for m in messages[:-1]]
    # 2 and 6 frame startup chunks, then the 5 leftover frames at stream end.
    assert sizes == [2, 6, 5]
    streamed = np.concatenate([waveform(m) for m in messages[:-1]])
    np.testing.assert_allclose(
        streamed, codec.decode_batch([codes])[0].numpy(), atol=1e-5, rtol=0
    )
    result = messages[-1].data.data
    assert "audio_waveform" not in result
    assert result["sample_rate"] == 22050
    assert result["usage"]["completion_tokens"] == 4


def test_streams_waiting_on_the_same_chunk_size_decode_together(
    vocoder, codec, monkeypatch
) -> None:
    batch_sizes = []
    stream = codec.stream

    def counting_stream(codes, state):
        batch_sizes.append(int(codes.shape[0]))
        return stream(codes, state)

    monkeypatch.setattr(codec, "stream", counting_stream)
    for request_id in ("a", "b"):
        vocoder.handle_streaming_new_request(
            request_id, make_payload(request_id, stream=True)
        )
    rows = torch.randint(0, 16, (2, 4))
    vocoder.handle_stream_chunk_batch(
        [chunk(rid, rows[i : i + 1], i) for i in range(2) for rid in ("a", "b")]
    )

    assert batch_sizes == [2]
    assert {m.request_id for m in drain(vocoder)} == {"a", "b"}


def test_a_row_batched_message_decodes_its_streams_together(
    vocoder, codec, monkeypatch
) -> None:
    batch_sizes = []
    stream = codec.stream

    def counting_stream(codes, state):
        batch_sizes.append(int(codes.shape[0]))
        return stream(codes, state)

    monkeypatch.setattr(codec, "stream", counting_stream)
    for request_id in ("a", "b"):
        vocoder.handle_streaming_new_request(
            request_id, make_payload(request_id, stream=True)
        )
    for step in range(2):
        vocoder.handle_message(
            IncomingMessage(
                request_id="a",
                type="stream_chunk_batch",
                data=StreamItemBatch(
                    request_ids=("a", "b"),
                    rows=(0, 1),
                    chunk_ids=(step, step),
                    data=torch.randint(0, 16, (2, 1, 4)),
                    from_stage="tts_engine",
                    metadata={"modality": "audio_codes", "stream": True},
                ),
            ),
            loop=None,
        )

    assert vocoder.accepts_stream_chunk_batch is True
    assert batch_sizes == [2]
    assert {m.request_id for m in drain(vocoder)} == {"a", "b"}


@pytest.mark.parametrize("freeze_gc", [True, False])
def test_serving_start_freezes_the_gc_only_when_asked(
    codec, monkeypatch, freeze_gc
) -> None:
    frozen = []
    monkeypatch.setattr(streaming_vocoder, "freeze_gc_after_warmup", frozen.append)
    vocoder = EasyMagpieStreamingVocoder(codec, cuda_graph=False, freeze_gc=freeze_gc)

    vocoder.on_serving_start()

    assert frozen == (["vocoder"] if freeze_gc else [])


def test_finished_streams_return_their_codec_slots(vocoder) -> None:
    free = len(vocoder.runner.free)
    for request_id in ("a", "b"):
        vocoder.handle_streaming_new_request(
            request_id, make_payload(request_id, stream=True)
        )
        vocoder.handle_stream_chunk_batch(
            [chunk(request_id, torch.randint(0, 16, (3, 4)), 0)]
        )
    assert len(vocoder.runner.free) == free - 2

    for request_id in ("a", "b"):
        vocoder.handle_stream_done(request_id)
    assert len(vocoder.runner.free) == free


def test_first_chunks_take_priority_over_steady_chunks(vocoder) -> None:
    vocoder.stream_states["warm"] = EasyMagpieStreamState(
        pending=[torch.zeros(8, 4, dtype=torch.long)] * 2,
        pending_frames=16,
        decoded_chunks=2,
    )
    vocoder.stream_states["new"] = EasyMagpieStreamState(
        pending=[torch.zeros(2, 4, dtype=torch.long)], pending_frames=2
    )

    assert [rid for rid, _ in vocoder.select_step_participants()] == ["new"]


def test_rejects_rows_with_the_wrong_codebook_width(vocoder) -> None:
    with pytest.raises(ValueError, match=r"\[frames, 4\]"):
        vocoder.validate_chunk("a", EasyMagpieStreamState(), torch.zeros(1, 3))


def test_offline_requests_decode_whole_utterances(vocoder, codec) -> None:
    codes = torch.randint(0, 16, (3, 4))
    (result,) = vocoder.decode_payloads([make_payload("a", stream=False, codes=codes)])

    np.testing.assert_allclose(
        np.frombuffer(result.data["audio_waveform"], dtype=np.float32),
        codec.decode_batch([codes])[0].numpy(),
    )
    assert result.data["usage"]["prompt_tokens"] == 9
    with pytest.raises(ValueError, match="no audio frames"):
        vocoder.decode_payload(make_payload("b", stream=False))
