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
from sglang_omni.models.nemotron_voicechat.codec_cuda_graph import (
    CodecDecodeGraphRunner,
)
from sglang_omni.platforms.device_graph import CudaDeviceGraphBackend

NUM_QUANTIZERS = 4
SAMPLES_PER_FRAME = 512
CODEBOOK_SIZE = 32
# note (Xinhao Tan): the checkpoint codec's rates, kernel and STFT sizes with
# narrow channels, so every window shape is captured at a fraction of the cost.
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_graph_replay_matches_eager_decode_for_every_window_size():
    torch.manual_seed(0)
    device = torch.device("cuda")
    decoder = RVQVAEDecoder(SMALL_CODEC_CONFIG)
    for parameter in decoder.parameters():
        nn.init.normal_(parameter, std=0.3)
    decoder.control_codes.copy_(torch.arange(CODEBOOK_SIZE, CODEBOOK_SIZE + 3))
    decoder.silence_codes.zero_()
    decoder = decoder.to(device).eval()
    runner = CodecDecodeGraphRunner(
        decoder, CudaDeviceGraphBackend(), device, DECODE_WINDOW_FRAMES
    )

    with torch.inference_mode():
        for window_frames in range(1, DECODE_WINDOW_FRAMES + 1):
            codes_TQ = torch.randint(
                0, CODEBOOK_SIZE + 3, (window_frames, NUM_QUANTIZERS), device=device
            )
            torch.testing.assert_close(
                runner(codes_TQ), decoder(codes_TQ), rtol=0, atol=0
            )
