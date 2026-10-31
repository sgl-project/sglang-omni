# SPDX-License-Identifier: Apache-2.0
"""Incremental code2wav decoding must reproduce whole-utterance decoding."""

import pytest
import torch
from torch import nn

from sglang_omni.models.nemotron_voicechat.code2wav_stream import (
    DECODE_WINDOW_FRAMES,
    TAIL_HOLDBACK_SAMPLES,
    StreamingCodec,
)
from sglang_omni.models.nemotron_voicechat.codec import RVQVAEDecoder

NUM_QUANTIZERS = 4
SAMPLES_PER_FRAME = 512
CODEBOOK_SIZE = 32
# note (Lushali): the checkpoint codec's rates, kernel and STFT sizes with
# narrow channels, so its context reach matches at a fraction of the cost.
SMALL_CODEC_CONFIG = {
    "num_quantizers": NUM_QUANTIZERS,
    "codebook_size": CODEBOOK_SIZE,
    "latent_size": 8,
    "n_fft": 16,
    "hop_length": 4,
    "base_hidden_size": 8,
    "channel_mult": [1, 2, 4],
    "rates": [7, 7, 9],
    "num_blocks": 3,
    "kernel_size": 7,
    "groups": 1,
    "wav_to_token_ratio": 1764,
}


class FrameLocalDecoder:
    """Decoder without cross-frame context: sample = frame id + intra-frame ramp.

    Frame-causal on the left, no lookahead, so the streaming path must match
    the whole-utterance render exactly once the holdback is accounted for.
    """

    samples_per_frame = SAMPLES_PER_FRAME
    num_quantizers = NUM_QUANTIZERS

    def __init__(self):
        self.calls: list[int] = []

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
    codec = StreamingCodec(decoder, "cpu")
    parts = [codec.push(row[None, :]) for row in codes]
    parts.append(codec.flush())
    streamed = torch.cat(parts)

    torch.testing.assert_close(streamed, full, rtol=0, atol=0)
    assert streamed.numel() == num_frames * SAMPLES_PER_FRAME
    assert max(decoder.calls) <= DECODE_WINDOW_FRAMES


def test_each_push_holds_back_the_tail_until_the_next_frame():
    codec = StreamingCodec(FrameLocalDecoder(), "cpu")
    first = codec.push(make_codes(1))
    assert first.numel() == SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    second = codec.push(make_codes(2)[1:])
    # The second push releases the first frame's holdback plus its own share.
    assert second.numel() == SAMPLES_PER_FRAME
    assert codec.flush().numel() == TAIL_HOLDBACK_SAMPLES


def test_multi_row_push_and_empty_flush():
    codec = StreamingCodec(FrameLocalDecoder(), "cpu")
    assert codec.flush().numel() == 0
    out = codec.push(make_codes(3))
    assert out.numel() == 3 * SAMPLES_PER_FRAME - TAIL_HOLDBACK_SAMPLES
    assert codec.emitted_samples == out.numel()


@pytest.mark.parametrize("rows_per_push", [1, DECODE_WINDOW_FRAMES + 1])
def test_retained_codes_stay_within_the_decode_window(rows_per_push):
    codec = StreamingCodec(FrameLocalDecoder(), "cpu")
    codes = make_codes(DECODE_WINDOW_FRAMES * 3)
    for start in range(0, codes.shape[0], rows_per_push):
        codec.push(codes[start : start + rows_per_push])
    assert codec.window_codes_TQ.shape[0] == DECODE_WINDOW_FRAMES


@pytest.mark.parametrize(
    "push_frame_counts",
    [[1, 15], [1, 16], [1, DECODE_WINDOW_FRAMES * 2 + 5], [16], [17], [3, 0, 3]],
)
def test_pushes_longer_than_the_window_match_whole_utterance_decode(
    push_frame_counts: list[int],
) -> None:
    """A long push must not lose audio that is no longer in the decode window."""
    codes = make_codes(sum(push_frame_counts))
    full = FrameLocalDecoder()(codes)

    decoder = FrameLocalDecoder()
    codec = StreamingCodec(decoder, "cpu")
    parts = [codec.push(piece) for piece in codes.split(push_frame_counts)]
    parts.append(codec.flush())

    torch.testing.assert_close(torch.cat(parts), full, rtol=0, atol=0)
    assert max(decoder.calls) <= DECODE_WINDOW_FRAMES


CONTEXT_TEST_FRAMES = 30


def stream_codes(
    decoder: RVQVAEDecoder, codes: torch.Tensor, frame_counts: list[int]
) -> torch.Tensor:
    codec = StreamingCodec(decoder, "cpu")
    parts = [codec.push(piece) for piece in codes.split(frame_counts)]
    return torch.cat([*parts, codec.flush()])


@pytest.fixture(scope="module")
def context_codec() -> tuple[RVQVAEDecoder, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    decoder = RVQVAEDecoder(SMALL_CODEC_CONFIG)
    for parameter in decoder.parameters():
        nn.init.normal_(parameter, std=0.3)
    decoder.control_codes.copy_(torch.arange(CODEBOOK_SIZE, CODEBOOK_SIZE + 3))
    decoder.silence_codes.zero_()
    decoder.eval()
    codes = torch.randint(0, CODEBOOK_SIZE, (CONTEXT_TEST_FRAMES, NUM_QUANTIZERS))
    one_frame_pushes = stream_codes(decoder, codes, [1] * CONTEXT_TEST_FRAMES)
    return decoder, codes, one_frame_pushes


@pytest.mark.parametrize("push_frame_counts", [[30], [1, 29], [16, 14]])
def test_long_pushes_keep_the_codec_context(
    context_codec: tuple[RVQVAEDecoder, torch.Tensor, torch.Tensor],
    push_frame_counts: list[int],
) -> None:
    """Splitting a long push must keep the codec's left context for unemitted audio."""
    decoder, codes, one_frame_pushes = context_codec
    # Missing context errs by tens here; float32 window-shape noise stays below 1e-4.
    torch.testing.assert_close(
        stream_codes(decoder, codes, push_frame_counts),
        one_frame_pushes,
        rtol=0,
        atol=1e-3,
    )
