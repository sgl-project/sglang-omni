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
    num_quantizers = NUM_QUANTIZERS

    def __init__(self):
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


def test_streaming_matches_whole_utterance_decode():
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


def test_each_push_holds_back_the_tail_until_the_next_frame():
    codec = StreamingCodec(FrameLocalDecoder(), "cpu", chunk_frames=(1,))
    (first,) = codec.push(make_codes(1))
    assert first.numel() == SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    (second,) = codec.push(make_codes(2)[1:])
    # The second push releases the first frame's holdback plus its own share.
    assert second.numel() == SAMPLES_PER_FRAME
    assert codec.flush().numel() == TAIL_HOLDBACK_SAMPLES


def test_multi_row_push_and_empty_flush():
    codec = StreamingCodec(FrameLocalDecoder(), "cpu", chunk_frames=(1,))
    assert codec.flush().numel() == 0
    out = torch.cat(codec.push(make_codes(3)))
    assert out.numel() == 3 * SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    assert codec.emitted_samples == out.numel()


@pytest.mark.parametrize("rows_per_push", [1, DECODE_WINDOW_FRAMES + 1])
def test_retained_codes_stay_within_the_decode_window(rows_per_push):
    codec = StreamingCodec(FrameLocalDecoder(), "cpu", chunk_frames=(1,))
    codes = make_codes(DECODE_WINDOW_FRAMES * 3)
    for start in range(0, codes.shape[0], rows_per_push):
        codec.push(codes[start : start + rows_per_push])
    assert codec.window_codes_TQ.shape[0] == DECODE_WINDOW_FRAMES


@pytest.mark.parametrize("chunk_frames", [(1,), (1, 2, 4), (1, 1, 1, 1, 2, 2, 4)])
@pytest.mark.parametrize("frame_count", [0, 1, 15, 16, 17, 37])
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


def test_two_frame_push_crosses_chunk_boundary() -> None:
    """A push crossing the target must emit a chunk before the final flush."""
    codec = StreamingCodec(FrameLocalDecoder(), "cpu", chunk_frames=(4,))
    codes = make_codes(5)
    assert codec.push(codes[:3]) == []
    (audio,) = codec.push(codes[3:])
    assert audio.numel() == 4 * SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    tail = codec.flush()
    assert tail.numel() == SAMPLES_PER_FRAME + TAIL_HOLDBACK_SAMPLES
    torch.testing.assert_close(
        torch.cat((audio, tail)), FrameLocalDecoder()(codes), rtol=0, atol=0
    )


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


@pytest.mark.parametrize("chunk_frames", [(16, 1), (1, 16)])
def test_reject_unsafe_chunks(chunk_frames: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="context"):
        NemotronCode2WavScheduler(
            FrameLocalDecoder(), "cpu", chunk_frames=chunk_frames, compute_fn=None
        )


@torch.inference_mode()
def make_causal_decoder(samples_per_frame: int) -> RVQVAEDecoder:
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
            "wav_to_token_ratio": samples_per_frame,
        }
    ).eval()
    for codebook in decoder.prvq.mus_list:
        codebook.normal_(std=0.1)
    decoder.control_codes.fill_(1000)
    decoder.silence_codes.zero_()
    return decoder


def test_invalid_codec_ratio_reports_config_values() -> None:
    """A checkpoint with inconsistent frame sizes must report its config values."""
    decoder = make_causal_decoder(1765)
    with pytest.raises(ValueError) as error:
        decoder.required_context_frames(TAIL_HOLDBACK_SAMPLES)
    for field in ("wav_to_token_ratio=1765", "rates product=441", "hop_length=4"):
        assert field in str(error.value)


@pytest.mark.parametrize("chunk_frames", [(1, 2, 4), (1, 1, 1, 1, 2, 2, 4), (1, 13)])
@torch.inference_mode()
def test_ramp_preserves_causal_decoder_waveform(chunk_frames: tuple[int, ...]) -> None:
    decoder = make_causal_decoder(1764)
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
    NemotronCode2WavScheduler(
        decoder, "cpu", chunk_frames=chunk_frames, compute_fn=None
    )
    with pytest.raises(ValueError, match="context"):
        NemotronCode2WavScheduler(decoder, "cpu", chunk_frames=(1, 14), compute_fn=None)


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
    scheduler.handle_streaming_new_request(
        "third", StagePayload("third", OmniRequest(None), {})
    )
    third = scheduler.on_stream_chunk("third", StreamItem(0, make_codes(2), "talker"))
    assert len(third) == 1
    scheduler.handle_stream_done("third")
    tail = scheduler.outbox.get_nowait()
    result = scheduler.outbox.get_nowait()
    assert tail.type == "stream"
    assert tail.data["audio_waveform_shape"] == [
        SAMPLES_PER_FRAME + TAIL_HOLDBACK_SAMPLES
    ]
    assert result.type == "result"
    assert result.data.data["audio_waveform_shape"] == [2 * SAMPLES_PER_FRAME]
    assert "third" not in scheduler.states
    assert "second" in scheduler.states


@pytest.mark.parametrize(
    ("frame_count", "cancel_after_eos"), [(0, False), (2, False), (5, False), (2, True)]
)
def test_end_of_stream_before_payload(frame_count: int, cancel_after_eos: bool) -> None:
    """EOS must deliver the complete waveform before a delayed terminal payload."""
    decoder = FrameLocalDecoder()
    scheduler = NemotronCode2WavScheduler(
        decoder, "cpu", chunk_frames=(1, 2, 4), compute_fn=None
    )
    codes = make_codes(frame_count)
    scheduler.handle_stream_chunk("request", StreamItem(0, codes, "talker"))
    scheduler.handle_stream_done("request")
    audio_parts: list[torch.Tensor] = []
    while not scheduler.outbox.empty():
        message = scheduler.outbox.get_nowait()
        assert message.type == "stream"
        audio_parts.append(
            torch.frombuffer(
                bytearray(message.data["audio_waveform"]), dtype=torch.float32
            )
        )
    waveform = torch.cat(audio_parts) if audio_parts else torch.zeros(0)
    torch.testing.assert_close(waveform, FrameLocalDecoder()(codes), rtol=0, atol=0)
    decode_calls = len(decoder.calls)
    if cancel_after_eos:
        scheduler.abort("request")
        assert "request" not in scheduler.pending_done
        assert scheduler.outbox.empty()
    else:
        scheduler.handle_streaming_new_request(
            "request", StagePayload("request", OmniRequest(None), {})
        )
        result = scheduler.outbox.get_nowait()
        assert result.type == "result"
        assert result.data.data["audio_waveform"] == waveform.numpy().tobytes()
        assert result.data.data["audio_waveform_shape"] == [
            frame_count * SAMPLES_PER_FRAME
        ]
        assert len(decoder.calls) == decode_calls
        assert scheduler.outbox.empty()
        assert "request" not in scheduler.stream_payloads
        assert "request" not in scheduler.pending_done
    assert "request" not in scheduler.states
