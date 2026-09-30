"""Torch CPU versus MLX Metal parity for AuK audio encoding and synthesis."""

import os

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.auk.hf_config import AuKVAEConfig
from sglang_omni.models.auk.mlx.vae import BigVGANFlowVAE
from sglang_omni.models.auk.mlx.vae_layers import Activation1d
from sglang_omni.models.auk.vae import Activation1d as TorchActivation1d
from sglang_omni.models.auk.vae import BigVGANFlowVAE as TorchVAE
from sglang_omni.models.auk.vae import SnakeBeta as TorchSnakeBeta

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
# note (Codex): M5 Metal defaults to TF32; MLX_ENABLE_TF32=0 verifies strict FP32 parity.
RELATIVE_TOLERANCE, ABSOLUTE_TOLERANCE = (
    (2e-5, 2e-6) if os.environ.get("MLX_ENABLE_TF32") == "0" else (2e-3, 5e-4)
)


def make_models(
    causal: bool = True, act_causal: bool = True, remove_weight_norm: bool = False
) -> tuple[TorchVAE, BigVGANFlowVAE]:
    torch.manual_seed(31)
    config = AuKVAEConfig(
        upsample_rates=[3, 2],
        upsample_kernel_sizes=[6, 4],
        upsample_initial_channel=32,
        resblock_kernel_sizes=[3, 5],
        resblock_dilation_sizes=[[1, 3], [1, 3]],
        downsample_rates=[2, 3],
        downsample_channels=[4, 8, 16],
        latent_dim=4,
        flow_hidden_channels=8,
        causal=causal,
        act_causal=act_causal,
    )
    reference = TorchVAE(config).eval()
    reference.global_mean.copy_(torch.tensor([-0.3, 0.7, 0.2, -0.8]))
    reference.global_log_std.copy_(torch.tensor([0.5, 1.5, 0.8, 2.0]))
    if remove_weight_norm:
        reference.remove_weight_norm()
    model = BigVGANFlowVAE(config)
    weights = {
        name: mx.array(value.detach().numpy())
        for name, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    mx.eval(model.parameters())
    return reference, model


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("act_causal", [True, False])
@pytest.mark.parametrize("remove_weight_norm", [True, False])
def test_decoder_matches_torch_at_boundaries(
    causal: bool, act_causal: bool, remove_weight_norm: bool
) -> None:
    reference, model = make_models(causal, act_causal, remove_weight_norm)
    latents = np.random.default_rng(5).normal(size=(2, 4, 7)).astype(np.float32)
    latents[:, :, 0] = 3
    latents[:, :, -1] = -3
    with torch.no_grad():
        expected = reference.inference_from_latents(torch.from_numpy(latents)).numpy()
    with mx.stream(mx.gpu):
        actual = model.inference_from_latents(mx.array(latents))
        mx.eval(actual)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        np.array(actual), expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


@pytest.mark.parametrize("samples", [54, 55, 59])
def test_encoder_posterior_and_partial_hop_lengths_match_torch(samples: int) -> None:
    reference, model = make_models()
    waveform = torch.from_numpy(
        np.random.default_rng(4).normal(size=(2, 1, samples)).astype(np.float32)
    )
    lengths = torch.tensor([samples, samples - 8])
    with torch.no_grad():
        stats = reference.audio_encoder(waveform)
        noise = torch.randn(
            (2, 4, stats.shape[-1]), generator=torch.Generator().manual_seed(9)
        )
        expected, expected_lengths = reference.encoding_and_normalization(
            waveform, lengths, generator=torch.Generator().manual_seed(9)
        )
    with mx.stream(mx.gpu):
        actual, actual_lengths = model.encode(
            mx.array(waveform.numpy()),
            mx.array(lengths.numpy()),
            noise=mx.array(noise.numpy()),
        )
        mx.eval(actual, actual_lengths)
    np.testing.assert_allclose(
        np.array(actual),
        expected.numpy(),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )
    np.testing.assert_array_equal(np.array(actual_lengths), expected_lengths.numpy())
    assert actual.shape[1] == stats.shape[-1]


def test_normalized_decode_matches_torch() -> None:
    reference, model = make_models()
    latents = np.random.default_rng(8).normal(size=(2, 5, 4)).astype(np.float32)
    with torch.no_grad():
        expected = reference.decode(torch.from_numpy(latents)).numpy()
    with mx.stream(mx.gpu):
        actual = model.decode(mx.array(latents))
        mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual), expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
    )


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("snake_logscale", [True, False])
def test_alias_free_activation_matches_torch(
    causal: bool, snake_logscale: bool
) -> None:
    reference = TorchActivation1d(
        TorchSnakeBeta(3, alpha_logscale=snake_logscale), causal=causal
    )
    model = Activation1d(3, causal, snake_logscale)
    signal = np.random.default_rng(3).normal(size=(2, 9, 3)).astype(np.float32)
    signal[:, 0, :] = 2
    signal[:, -1, :] = -2
    with torch.no_grad():
        expected = reference(torch.from_numpy(signal).transpose(1, 2))
    with mx.stream(mx.gpu):
        actual = model(mx.array(signal))
        mx.eval(actual)
    np.testing.assert_allclose(
        np.array(actual),
        expected.transpose(1, 2).numpy(),
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def test_request_key_is_reproducible_without_global_rng_changes() -> None:
    _, model = make_models()
    sample = mx.zeros((1, 1, 59))
    key = mx.random.key(17)
    first, lengths = model.encode(sample, key=key)
    mx.random.seed(999)
    second, _ = model.encode(sample, key=key)
    other, _ = model.encode(sample, key=mx.random.key(18))
    mx.eval(first, second, other, lengths)
    np.testing.assert_array_equal(np.array(first), np.array(second))
    assert not np.array_equal(np.array(first), np.array(other))
    assert lengths.tolist() == [59 // model.hop_size]


def test_bfloat16_inputs_preserve_float32_vae_computation() -> None:
    _, model = make_models()
    waveform = mx.array(
        np.random.default_rng(2).normal(size=(1, 1, 54)).astype(np.float32)
    ).astype(mx.bfloat16)
    key = mx.random.key(11)
    actual_latents, actual_lengths = model.encode(waveform, key=key)
    expected_latents, expected_lengths = model.encode(
        waveform.astype(mx.float32), key=key
    )
    latents = actual_latents.astype(mx.bfloat16)
    actual_audio = model.decode(latents)
    expected_audio = model.decode(latents.astype(mx.float32))
    mx.eval(actual_latents, expected_latents, actual_audio, expected_audio)
    assert actual_latents.dtype == mx.float32
    assert actual_audio.dtype == mx.float32
    np.testing.assert_array_equal(np.array(actual_latents), np.array(expected_latents))
    np.testing.assert_array_equal(np.array(actual_lengths), np.array(expected_lengths))
    np.testing.assert_array_equal(np.array(actual_audio), np.array(expected_audio))
