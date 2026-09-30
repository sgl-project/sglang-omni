# SPDX-License-Identifier: Apache-2.0
"""Real convolution tests for chunk boundaries; no checkpoint/GPU required."""

from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.auk.vae import AuKVAEConfig, BigVGANFlowVAE


@pytest.fixture(params=[True, False])
def vae(request):
    torch.manual_seed(42)
    model = BigVGANFlowVAE(
        AuKVAEConfig(
            upsample_rates=[2, 2],
            upsample_kernel_sizes=[4, 4],
            upsample_initial_channel=16,
            latent_dim=4,
            downsample_rates=[2, 2],
            downsample_channels=[4, 8, 16],
            flow_hidden_channels=8,
            causal=request.param,
            act_causal=request.param,
        )
    ).eval()
    model.global_mean.fill_(0.2)
    model.global_log_std.fill_(1.7)
    return model


@pytest.mark.parametrize("frames,chunk_frames", [(1, 8), (25, 25), (37, 64), (401, 53)])
@torch.inference_mode()
def test_chunked_matches_full_waveform(vae, frames, chunk_frames):
    latents = torch.randn(2, frames, 4)
    expected = vae.decode(latents).unsqueeze(1)
    with patch.object(
        vae, "inference_from_latents", wraps=vae.inference_from_latents
    ) as decode:
        actual = vae.decode_chunked(latents, chunk_frames)
    assert actual.device.type == "cpu"
    assert actual.shape == (2, 1, frames * 4)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    left, right = vae.decode_context()
    assert len(decode.call_args_list) == (frames + chunk_frames - 1) // chunk_frames
    assert all(
        call.args[0].shape[-1] <= chunk_frames + left + right
        for call in decode.call_args_list
    )
    if frames == 401:
        assert max(call.args[0].shape[-1] for call in decode.call_args_list) < frames


@pytest.mark.parametrize("chunk_frames", [0, -1, True, 1.5])
def test_invalid_chunk_size(vae, chunk_frames):
    with pytest.raises(ValueError, match="positive integer"):
        vae.decode_chunked(torch.zeros(1, 2, 4), chunk_frames)


def test_empty_latents(vae):
    with pytest.raises(ValueError, match="nonempty"):
        vae.decode_chunked(torch.zeros(1, 0, 4), 8)


@torch.inference_mode()
def test_missing_halo_changes_boundary_samples(vae):
    latents = torch.randn(1, 401, 4)
    expected = vae.decode(latents).unsqueeze(1)
    with patch.object(vae, "decode_context", return_value=(0, 0)):
        broken = vae.decode_chunked(latents, 53)
    assert not torch.allclose(broken, expected, rtol=1e-5, atol=2e-6)


@torch.inference_mode()
def test_checkpoint_upsampling_geometry():
    # Preserve all six production upsampling stages and filter geometry, reducing
    # channels only. This exercises odd strides as well as the 480-sample hop.
    torch.manual_seed(9)
    vae = BigVGANFlowVAE(
        AuKVAEConfig(
            upsample_initial_channel=64,
            latent_dim=4,
            downsample_rates=[2, 2],
            downsample_channels=[4, 8, 16],
            flow_hidden_channels=8,
        )
    ).eval()
    latents = torch.randn(1, 143, 4)
    expected = vae.decode(latents).unsqueeze(1)
    actual = vae.decode_chunked(latents, 37)
    assert actual.shape == (1, 1, 143 * 480)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
