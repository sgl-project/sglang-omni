# SPDX-License-Identifier: Apache-2.0
"""D2 playback stays on the 80 ms clock and its self feedback follows the played waveform."""

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.models.qwen3_omni_d2.waveform import (
    DEZIPPER_TRANSITION_SAMPLES,
    FIRST_FRAME_DELAY_SAMPLES,
    OUTPUT_FRAME_SAMPLES,
    PCM16_FULL_SCALE,
    Code2WavRenderer,
    PlaybackStream,
)


def pcm16_samples(pcm16: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(pcm16), dtype=torch.int16).float()


def test_feedback_resamples_the_continuous_playback_across_frames() -> None:
    generator = torch.Generator().manual_seed(0)
    raw = [
        0.01
        * torch.randn(
            OUTPUT_FRAME_SAMPLES - FIRST_FRAME_DELAY_SAMPLES, generator=generator
        )
    ]
    raw += [
        0.01 * torch.randn(OUTPUT_FRAME_SAMPLES, generator=generator) for _ in range(3)
    ]
    stream = PlaybackStream()
    outputs = [stream.append(samples, is_active=True) for samples in raw]
    first = pcm16_samples(outputs[0][0])
    assert first.numel() == OUTPUT_FRAME_SAMPLES
    assert not first[:FIRST_FRAME_DELAY_SAMPLES].any()
    playback = torch.cat((torch.zeros(FIRST_FRAME_DELAY_SAMPLES), *raw))
    # Feedback sample j reads the playback at 1.5 j - 1, linearly interpolated.
    positions = (1.5 * torch.arange(playback.numel() * 2 // 3) - 1.0).clamp_min(0.0)
    left = positions.floor().long()
    weight = positions - left
    expected = (
        playback[left] * (1 - weight)
        + playback[(left + 1).clamp_max(playback.numel() - 1)] * weight
    )
    feedback = torch.cat([frame_feedback for _, frame_feedback in outputs])
    torch.testing.assert_close(feedback, expected, rtol=0, atol=1e-7)


def test_an_inactive_frame_feeds_back_silence_and_ramps_from_the_last_sample() -> None:
    stream = PlaybackStream()
    stream.append(
        torch.full((OUTPUT_FRAME_SAMPLES - FIRST_FRAME_DELAY_SAMPLES,), 0.5),
        is_active=True,
    )
    pcm16, feedback = stream.append(
        torch.full((OUTPUT_FRAME_SAMPLES,), 0.5), is_active=False
    )
    audible = pcm16_samples(pcm16)
    assert not feedback.any()
    assert audible[0] == round(0.5 * PCM16_FULL_SCALE)
    assert (audible[:DEZIPPER_TRANSITION_SAMPLES].diff() <= 0).all()
    assert not audible[DEZIPPER_TRANSITION_SAMPLES - 1 :].any()


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@torch.no_grad()
def test_incremental_rendering_matches_one_full_decode() -> None:
    torch.manual_seed(0)
    config = Qwen3OmniMoeCode2WavConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        decoder_dim=64,
        codebook_size=64,
        attn_implementation="sdpa",
    )
    code2wav = Qwen3OmniMoeCode2Wav(config).cuda().bfloat16().eval()
    codes = torch.randint(0, config.codebook_size, (30, config.num_quantizers))
    renderer = Code2WavRenderer(code2wav, capacity_frames=codes.shape[0])
    incremental = torch.cat([renderer.append(frame) for frame in codes])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = code2wav(codes.T[None].cuda()).float().cpu().view(-1)
    assert incremental.shape == full.shape
    # One-token and whole-sequence BF16 forwards round differently: 5.1e-3 measured on H100.
    assert (incremental - full).abs().max() < 5e-2 * full.abs().max()
