# SPDX-License-Identifier: Apache-2.0
"""Folded encoder weight norm and bucketed encoder graphs keep reference latents."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.dots_tts.codec import ReferenceEncoderGraphs, fold_weight_norm

HOP_SIZE = 8
PATCH_SIZE = 4


def tiny_vocoder() -> tuple[torch.nn.Module, torch.nn.Module]:
    try:
        from sglang_omni.models.dots_tts.compat import import_dots_tts

        import_dots_tts()
        from dots_tts.modules.vocoder.bigvgan import AudioVAE
        from dots_tts.modules.vocoder.config import AudioVAEConfig
        from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
    except ImportError as exc:
        pytest.skip(f"dots_tts unavailable: {exc}")
    else:
        pass
    torch.manual_seed(0)
    config = AudioVAEConfig(
        sample_rate=1600,
        upsample_rates=[4, 2],
        upsample_kernel_sizes=[8, 4],
        upsample_initial_channel=32,
        resblock="1",
        resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5]],
        downsample_rates=[2, 4],
        downsample_channels=[12, 16, 32],
        latent_dim=8,
        causal=True,
        mi_num_layers=1,
        causal_encoder=True,
        use_bias_at_final=False,
        use_tanh_at_final=False,
    )
    vocoder = AudioVAE(config).eval()
    return vocoder, VocoderInference(vocoder)


@torch.no_grad()
def test_folded_encoder_weight_norm_keeps_latents() -> None:
    vocoder, inference = tiny_vocoder()
    generator = torch.Generator().manual_seed(1)
    audio = torch.randn(1, 1, 6 * PATCH_SIZE * HOP_SIZE, generator=generator)
    expected = inference.extract_latents(audio)

    fold_weight_norm(vocoder.audio_encoder)

    assert not any(
        hasattr(module, "weight_g") for module in vocoder.audio_encoder.modules()
    )
    torch.testing.assert_close(
        inference.extract_latents(audio), expected, rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    ("causal_encoder", "lookahead_frames", "message"),
    [(False, 2, "causal reference encoder"), (True, PATCH_SIZE + 1, "lookahead")],
)
def test_encoder_graphs_reject_unsafe_padding(
    causal_encoder: bool, lookahead_frames: int, message: str
) -> None:
    vocoder, inference = tiny_vocoder()
    vocoder.h.causal_encoder = causal_encoder
    vocoder.h.num_encoder_lookahead = lookahead_frames
    with pytest.raises(ValueError, match=message):
        ReferenceEncoderGraphs(
            inference,
            samples_per_patch=PATCH_SIZE * HOP_SIZE,
            hop_size=HOP_SIZE,
            device=torch.device("cpu"),
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@torch.no_grad()
def test_encoder_graphs_match_eager_before_the_dropped_patch() -> None:
    vocoder, inference = tiny_vocoder()
    fold_weight_norm(vocoder.audio_encoder)
    vocoder.cuda()
    graphs = ReferenceEncoderGraphs(
        inference,
        samples_per_patch=PATCH_SIZE * HOP_SIZE,
        hop_size=HOP_SIZE,
        device=torch.device("cuda"),
    )
    generator = torch.Generator(device="cuda").manual_seed(2)
    for patches in (5, 8, 13):
        audio = torch.randn(
            1, 1, patches * PATCH_SIZE * HOP_SIZE, device="cuda", generator=generator
        )
        expected = inference.extract_latents(audio)
        observed = graphs.extract_latents(audio)
        assert observed.shape == expected.shape
        kept = expected.shape[-1] - PATCH_SIZE
        torch.testing.assert_close(
            observed[..., :kept], expected[..., :kept], rtol=1e-3, atol=1e-4
        )
