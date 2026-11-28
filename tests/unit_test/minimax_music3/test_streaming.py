# SPDX-License-Identifier: Apache-2.0
"""Music3 streaming resampling, audio delivery, and cancellation."""

from __future__ import annotations

import pytest
import torch
from torchaudio import functional as AF

from sglang_omni.models.minimax_music3.acoustic import (
    MiniMaxMusic3AcousticScheduler,
    StreamingResampler,
)
from sglang_omni.models.minimax_music3.chunking import chunk_windows
from sglang_omni.models.minimax_music3.constants import (
    DAV_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
)
from sglang_omni.models.minimax_music3.payload_types import MiniMaxMusic3State
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.mark.parametrize(
    "piece_samples",
    [
        [220_160, 220_672],
        [220_160, 176_128, 176_128, 176_128, 575_744],
        [1, 440, 441, 442, 882, 7, 50_000],
        [123_457],
    ],
)
def test_streaming_resampler_pieces_join_into_one_pass(
    piece_samples: list[int],
) -> None:
    """Stream and full responses both join these pieces, so they must equal one
    pass over the whole song, for decoder-sized and arbitrary pieces alike."""
    generator = torch.Generator().manual_seed(len(piece_samples))
    song = torch.rand((2, sum(piece_samples)), generator=generator) * 2 - 1
    resampler = StreamingResampler()
    pieces, start = [], 0
    for index, length in enumerate(piece_samples):
        piece = resampler.push(
            song[:, start : start + length], final=index == len(piece_samples) - 1
        )
        start += length
        if piece is not None:
            pieces.append(piece)
        else:
            pass

    expected = AF.resample(song, DAV_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
    assert torch.equal(torch.cat(pieces, dim=-1), expected)
    assert resampler.output_samples == expected.shape[-1]


class SeededAcousticDecoder:
    """Deterministic stereo audio per chunk, 1764 samples per frame like 44.1 kHz at 25 fps."""

    serial_offload = False

    def decode_with_state(
        self,
        hidden: torch.Tensor,
        *,
        chunk_idx: int,
        should_abort,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if should_abort():
            raise InterruptedError("aborted")
        else:
            pass
        generator = torch.Generator().manual_seed(chunk_idx)
        wave = torch.rand((2, hidden.shape[0] * 1764), generator=generator) * 2 - 1
        return wave, torch.zeros((1, 128, 1)), torch.zeros((1, 2048, 1))


def music3_hidden_chunks(frames: int, *, stream_audio: bool) -> list[StreamItem]:
    return [
        StreamItem(
            chunk_id=window.index,
            data=torch.zeros((1, window.length, 32768)),
            from_stage="minimax_music3_ar",
            metadata={
                "chunk_idx": window.index,
                "start_frame": window.start,
                "end_frame": window.end,
                "seed": 7,
                "is_final": window.is_last,
                "stream_audio": stream_audio,
            },
        )
        for window in chunk_windows(frames)
    ]


def acoustic_scheduler_with_payload(
    request_id: str, frames: int
) -> MiniMaxMusic3AcousticScheduler:
    scheduler = MiniMaxMusic3AcousticScheduler(decoder=SeededAcousticDecoder())  # type: ignore[arg-type]
    state = MiniMaxMusic3State(
        seed=7, generated_frames=frames, finish_reason="stop", max_audio_frames=frames
    )
    state.prompt_tokens = 115
    state.completion_tokens = frames
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="lyrics"),
        data=state.to_dict(),
    )
    scheduler.stream_payloads[request_id] = payload
    scheduler.on_streaming_new_request(request_id, payload)
    return scheduler


def waveform_from(data: dict) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(data["audio_waveform"]), dtype=torch.float32
    ).view(data["audio_waveform_shape"])


@pytest.mark.parametrize("frames", [150, 350, 1306])
def test_acoustic_stream_joins_into_the_full_response(frames: int) -> None:
    """A streaming request's pieces join into the exact audio a full request
    returns; its result closes the stream with finish reason and usage only."""
    full = acoustic_scheduler_with_payload("full", frames)
    for item in music3_hidden_chunks(frames, stream_audio=False):
        assert full.on_stream_chunk("full", item) == []
    [full_result] = full.on_stream_done("full")

    streamed = acoustic_scheduler_with_payload("streamed", frames)
    messages = []
    for item in music3_hidden_chunks(frames, stream_audio=True):
        messages.extend(streamed.on_stream_chunk("streamed", item))
    [stream_result] = streamed.on_stream_done("streamed")

    assert [message.type for message in messages] == ["stream"] * len(messages)
    assert len(messages) == len(chunk_windows(frames))
    joined = torch.cat([waveform_from(message.data) for message in messages], dim=-1)
    assert torch.equal(joined, waveform_from(full_result.data.data))
    assert messages[0].data["sample_rate"] == OUTPUT_SAMPLE_RATE
    assert all(message.data["keep_audio_channels"] for message in messages)
    assert waveform_from(messages[0].data).shape[0] == 2
    closing = stream_result.data.data
    assert "audio_waveform" not in closing
    assert closing["finish_reason"] == full_result.data.data["finish_reason"] == "stop"
    assert closing["usage"] == full_result.data.data["usage"]
    assert closing["usage"]["completion_tokens"] == frames


def test_acoustic_stream_abort_stops_pieces_and_frees_the_request() -> None:
    """A cancel during a chunk's decode emits nothing more; abort frees the request."""
    scheduler = acoustic_scheduler_with_payload("cancelled", 250)
    first, second = music3_hidden_chunks(250, stream_audio=True)
    assert len(scheduler.on_stream_chunk("cancelled", first)) == 1
    state = scheduler.stream_states["cancelled"]

    state.abort_event.set()

    assert scheduler.on_stream_chunk("cancelled", second) == []
    assert len(state.audio_pieces) == 1
    scheduler.abort("cancelled")
    assert scheduler.is_aborted("cancelled")
    assert "cancelled" not in scheduler.stream_states
    assert not state.audio_pieces


def test_acoustic_stream_rejects_a_non_boolean_stream_flag() -> None:
    scheduler = acoustic_scheduler_with_payload("bad", 150)
    [item] = music3_hidden_chunks(150, stream_audio=True)
    item.metadata["stream_audio"] = "yes"
    with pytest.raises(ValueError, match="stream_audio must be a bool"):
        scheduler.on_stream_chunk("bad", item)
