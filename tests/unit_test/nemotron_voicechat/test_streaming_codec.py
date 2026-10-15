# SPDX-License-Identifier: Apache-2.0
"""Incremental code2wav decoding must reproduce whole-utterance decoding."""

import pytest
import torch

from sglang_omni.models.nemotron_voicechat.code2wav_stream import (
    DECODE_WINDOW_FRAMES,
    TAIL_HOLDBACK_SAMPLES,
    NemotronCode2WavScheduler,
    StreamingCodec,
)
from sglang_omni.models.nemotron_voicechat.codec import RVQVAEDecoder
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload

NUM_QUANTIZERS = 4
SAMPLES_PER_FRAME = 512


class FrameLocalDecoder:
    """Decoder without cross-frame context: sample = frame id + intra-frame ramp.

    Frame-causal on the left, no lookahead, so the streaming path must match
    the whole-utterance render exactly once the holdback is accounted for.
    """

    samples_per_frame = SAMPLES_PER_FRAME

    def __init__(self) -> None:
        self.calls: list[int] = []

    def required_context_frames(self, holdback_samples: int) -> int:
        return (holdback_samples + self.samples_per_frame - 1) // self.samples_per_frame

    def __call__(self, codes_TQ: torch.Tensor) -> torch.Tensor:
        self.calls.append(int(codes_TQ.shape[0]))
        frame_id = codes_TQ[:, 0].to(torch.float32)  # quantizer 0 carries the id
        ramp = torch.arange(SAMPLES_PER_FRAME, dtype=torch.float32) / SAMPLES_PER_FRAME
        return (frame_id[:, None] * 1000.0 + ramp[None, :]).reshape(-1)


def make_codes(num_frames: int) -> torch.Tensor:
    codes = torch.zeros(num_frames, NUM_QUANTIZERS, dtype=torch.long)
    codes[:, 0] = torch.arange(num_frames)
    return codes


def test_streaming_matches_whole_utterance_decode() -> None:
    num_frames = DECODE_WINDOW_FRAMES * 2 + 5
    codes = make_codes(num_frames)
    full = FrameLocalDecoder()(codes)

    decoder = FrameLocalDecoder()
    codec = StreamingCodec(decoder, "cpu", chunk_frames=(1,))
    parts = [audio for row in codes for audio in codec.push(row[None, :])]
    parts.append(codec.flush())
    streamed = torch.cat(parts)

    torch.testing.assert_close(streamed, full, rtol=0, atol=0)
    assert streamed.numel() == num_frames * SAMPLES_PER_FRAME
    assert max(decoder.calls) <= DECODE_WINDOW_FRAMES


def test_each_push_holds_back_the_tail_until_the_next_frame() -> None:
    codec = StreamingCodec(FrameLocalDecoder(), "cpu", chunk_frames=(1,))
    (first,) = codec.push(make_codes(1))
    assert first.numel() == SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    (second,) = codec.push(make_codes(2)[1:])
    # The second push releases the first frame's holdback plus its own share.
    assert second.numel() == SAMPLES_PER_FRAME
    assert codec.flush().numel() == TAIL_HOLDBACK_SAMPLES


def test_multi_row_push_and_empty_flush() -> None:
    codec = StreamingCodec(FrameLocalDecoder(), "cpu", chunk_frames=(1,))
    assert codec.flush().numel() == 0
    out = torch.cat(codec.push(make_codes(3)))
    assert out.numel() == 3 * SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    assert codec.emitted_samples == out.numel()


@pytest.mark.parametrize("chunk_frames", [(1,), (1, 2, 4), (1, 1, 1, 1, 2, 2, 4)])
@pytest.mark.parametrize("frame_count", [0, 1, 2, 3, 8, 15, 16, 17, 37, 65])
@pytest.mark.parametrize("input_frames", [1, 3, 20])
def test_ramp_preserves_all_samples(
    chunk_frames: tuple[int, ...], frame_count: int, input_frames: int
) -> None:
    decoder = FrameLocalDecoder()
    codec = StreamingCodec(decoder, "cpu", chunk_frames=chunk_frames)
    codes = make_codes(frame_count)
    audio_parts = [
        audio
        for start in range(0, frame_count, input_frames)
        for audio in codec.push(codes[start : start + input_frames])
    ]
    audio_parts.append(codec.flush())
    torch.testing.assert_close(
        torch.cat(audio_parts), FrameLocalDecoder()(codes), rtol=0, atol=0
    )
    assert all(frames <= DECODE_WINDOW_FRAMES for frames in decoder.calls)
    assert codec.flush().numel() == 0


def test_ramp_chunk_boundaries_and_partial_flush() -> None:
    codec = StreamingCodec(
        FrameLocalDecoder(), "cpu", chunk_frames=(1, 1, 1, 1, 2, 2, 4)
    )
    audio_parts = codec.push(make_codes(14))
    assert [audio.numel() for audio in audio_parts] == [
        SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES,
        SAMPLES_PER_FRAME,
        SAMPLES_PER_FRAME,
        SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
        2 * SAMPLES_PER_FRAME,
        4 * SAMPLES_PER_FRAME,
    ]
    assert codec.flush().numel() == 2 * SAMPLES_PER_FRAME + TAIL_HOLDBACK_SAMPLES


@pytest.mark.parametrize("chunk_frames", [(), (0,), (1, -2), (1, 16)])
def test_reject_invalid_or_unsafe_chunks(chunk_frames: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        NemotronCode2WavScheduler(
            FrameLocalDecoder(), "cpu", chunk_frames=chunk_frames, compute_fn=None
        )


@pytest.mark.parametrize("chunk_frames", [(1, 2, 4), (1, 1, 1, 1, 2, 2, 4), (1, 13)])
@torch.inference_mode()
def test_ramp_preserves_causal_decoder_waveform(chunk_frames: tuple[int, ...]) -> None:
    torch.manual_seed(2459)
    decoder = RVQVAEDecoder(
        {
            "base_hidden_size": 8,
            "channel_mult": [1, 2, 4],
            "codebook_size": 32,
            "groups": 1,
            "hop_length": 4,
            "kernel_size": 7,
            "latent_size": 8,
            "n_fft": 16,
            "num_blocks": 3,
            "num_quantizers": 4,
            "rates": [7, 7, 9],
            "wav_to_token_ratio": 1764,
        }
    ).eval()
    for codebook in decoder.prvq.mus_list:
        codebook.normal_(std=0.1)
    decoder.control_codes.fill_(1000)
    decoder.silence_codes.zero_()
    codes = torch.randint(32, (37, 4))
    baseline = StreamingCodec(decoder, "cpu", chunk_frames=(1,))
    candidate = StreamingCodec(decoder, "cpu", chunk_frames=chunk_frames)
    expected_parts = [audio for row in codes for audio in baseline.push(row[None])]
    expected_parts.append(baseline.flush())
    actual_parts = candidate.push(codes)
    actual_parts.append(candidate.flush())
    torch.testing.assert_close(
        torch.cat(actual_parts), torch.cat(expected_parts), atol=1e-5, rtol=1e-5
    )
    with pytest.raises(ValueError, match="context"):
        StreamingCodec(decoder, "cpu", chunk_frames=(1, 14))


def test_scheduler_preserves_chunks_and_request_isolation() -> None:
    scheduler = NemotronCode2WavScheduler(
        FrameLocalDecoder(), "cpu", chunk_frames=(1, 2, 4), compute_fn=None
    )
    payload = StagePayload("first", OmniRequest(None), {})
    scheduler.handle_streaming_new_request("first", payload)
    first = scheduler.on_stream_chunk("first", StreamItem(0, make_codes(1), "talker"))
    assert len(first) == 1
    assert (
        scheduler.on_stream_chunk("first", StreamItem(1, make_codes(2)[1:], "talker"))
        == []
    )
    second = scheduler.on_stream_chunk("second", StreamItem(0, make_codes(3), "talker"))
    assert len(second) == 2
    scheduler.abort("first")
    assert "first" not in scheduler.states
    assert "second" in scheduler.states
    scheduler.handle_streaming_new_request("first", payload)
    restarted = scheduler.on_stream_chunk(
        "first", StreamItem(0, make_codes(2), "talker")
    )
    assert len(restarted) == 1
    scheduler.handle_stream_done("first")
    tail = scheduler.outbox.get_nowait()
    result = scheduler.outbox.get_nowait()
    assert tail.type == "stream"
    assert tail.data["audio_waveform_shape"] == [
        SAMPLES_PER_FRAME + TAIL_HOLDBACK_SAMPLES
    ]
    assert result.type == "result"
    assert result.data.data["audio_waveform_shape"] == [2 * SAMPLES_PER_FRAME]
    assert "first" not in scheduler.states
    assert "second" in scheduler.states


@pytest.mark.parametrize("frame_count", [0, 2])
def test_end_of_stream_before_payload(frame_count: int) -> None:
    scheduler = NemotronCode2WavScheduler(
        FrameLocalDecoder(), "cpu", chunk_frames=(1, 2, 4), compute_fn=None
    )
    scheduler.handle_stream_chunk(
        "request", StreamItem(0, make_codes(frame_count), "talker")
    )
    scheduler.handle_stream_done("request")
    scheduler.handle_streaming_new_request(
        "request", StagePayload("request", OmniRequest(None), {})
    )
    messages = []
    while not scheduler.outbox.empty():
        messages.append(scheduler.outbox.get_nowait())
    assert messages[-1].type == "result"
    assert messages[-1].data.data["audio_waveform_shape"] == [
        frame_count * SAMPLES_PER_FRAME
    ]
    assert "request" not in scheduler.states
