# SPDX-License-Identifier: Apache-2.0
"""Native MLX backbone and sampling parity against the Torch implementation."""

from __future__ import annotations

import os
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
pytest.importorskip("x_transformers")

from x_transformers.x_transformers import RotaryEmbedding as TorchRotary
from x_transformers.x_transformers import apply_rotary_pos_emb

from sglang_omni.models.auk.constants import FLASH_T_GRID
from sglang_omni.models.auk.dit import AuKDit as TorchDiT
from sglang_omni.models.auk.flow_matching import AuKFlowMatching as TorchFlow
from sglang_omni.models.auk.flow_matching import AuKSampleItem as TorchSampleItem
from sglang_omni.models.auk.flow_matching import build_time_grid as torch_time_grid
from sglang_omni.models.auk.flow_matching import fuse_hidden_states as torch_fuse
from sglang_omni.models.auk.mlx.blocks import RotaryEmbedding, apply_rope
from sglang_omni.models.auk.mlx.dit import AuKDit
from sglang_omni.models.auk.mlx.flow_matching import (
    AuKFlowMatching,
    AuKSampleItem,
    build_time_grid,
    fuse_hidden_states,
    request_key,
)


@pytest.fixture(params=["cpu", "gpu"])
def compute_device(request):
    if request.param == "gpu" and not mx.metal.is_available():
        pytest.skip("requires Metal")
    with mx.stream(mx.cpu if request.param == "cpu" else mx.gpu):
        yield request.param


@pytest.fixture
def parity_tolerance(compute_device):
    # note (Codex): M5 uses TF32 GPU matmuls unless MLX_ENABLE_TF32 is disabled.
    tf32 = compute_device == "gpu" and os.environ.get("MLX_ENABLE_TF32") != "0"
    return dict(rtol=2e-4, atol=6e-3 if tf32 else 2e-5)


@pytest.fixture
def models(compute_device) -> tuple[TorchFlow, AuKFlowMatching]:
    torch.manual_seed(42)
    config = dict(
        dim=32,
        heads=2,
        dim_head=16,
        latent_dim=8,
        text_hidden_dim=16,
        num_layers=1,
        num_single_layers=1,
    )
    reference = TorchFlow(TorchDiT(**config), num_llm_layers=2).eval()
    for module in reference.modules():
        if isinstance(module, torch.nn.RMSNorm):
            module.eps = torch.finfo(torch.float32).eps
    for parameter in reference.parameters():
        torch.nn.init.uniform_(parameter, -0.2, 0.2)
    reference.transformer.rotary_embed.inv_freq = (
        reference.transformer.rotary_embed.inv_freq.to(torch.bfloat16).float()
    )
    model = AuKFlowMatching(AuKDit(**config), num_llm_layers=2)
    weights = {
        key: mx.array(value.detach().numpy())
        for key, value in reference.state_dict().items()
    }
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    return reference, model


@pytest.mark.parametrize("cfg_infer", [False, True])
@pytest.mark.parametrize("reference_length", [0, 4])
def test_dit_matches_torch(models, parity_tolerance, cfg_infer, reference_length):
    reference, model = models
    arrays = {
        "x": torch.randn(2, 7, 8),
        "text": torch.randn(2, 5, 16),
        "time": torch.tensor([0.1, 0.75]),
        "mask": torch.arange(7)[None, :] < torch.tensor([7, 4])[:, None],
        "c_mask": torch.arange(5)[None, :] < torch.tensor([3, 5])[:, None],
        "ref": torch.randn(2, reference_length, 8),
        "ref_mask": torch.arange(reference_length)[None, :]
        < torch.tensor([reference_length, max(0, reference_length - 2)])[:, None],
    }
    with torch.no_grad():
        expected = reference.transformer(**arrays, cfg_infer=cfg_infer).numpy()
    actual = model.transformer(
        **{key: mx.array(value.numpy()) for key, value in arrays.items()},
        cfg_infer=cfg_infer,
    )
    np.testing.assert_allclose(np.array(actual), expected, **parity_tolerance)


def sample_items() -> (
    tuple[list[TorchSampleItem], list[AuKSampleItem], list[torch.Tensor]]
):
    reference_items, items, noises = [], [], []
    for text_length, frames, ref_length, seed in [
        (5, 19, 4, 1),
        (8, 11, 7, 2),
        (3, 15, 0, 3),
    ]:
        conditioning = torch.randn(text_length, 16)
        text_mask = torch.ones(text_length, dtype=torch.bool)
        ref_latent = torch.randn(ref_length, 8) if ref_length else None
        noise = torch.randn(frames, 8)
        reference_items.append(
            TorchSampleItem(
                conditioning,
                text_mask,
                frames,
                ref_latent,
                seed=seed,
                ref_length=max(0, ref_length - 1),
            )
        )
        items.append(
            AuKSampleItem(
                conditioning=mx.array(conditioning.numpy()),
                text_mask=mx.array(text_mask.numpy()),
                target_frames=frames,
                ref_latent=None if ref_latent is None else mx.array(ref_latent.numpy()),
                seed=seed,
                ref_length=max(0, ref_length - 1),
                noise=mx.array(noise.numpy()),
            )
        )
        noises.append(noise)
    return reference_items, items, noises


@pytest.mark.parametrize(
    "sampling",
    [
        dict(steps=3, cfg_strength=0.0),
        dict(steps=32, cfg_strength=2.0, sway_sampling_coef=-1.0),
        dict(steps=4, cfg_strength=0.0, t_grid=FLASH_T_GRID),
    ],
)
def test_sampling_matches_torch_with_fixed_noise_and_variable_lengths(
    models, parity_tolerance, sampling
):
    reference, model = models
    reference_items, items, noises = sample_items()
    with patch("sglang_omni.models.auk.flow_matching.torch.randn", side_effect=noises):
        expected = reference.sample_batch(reference_items, **sampling)
    actual = model.sample_batch(items, **sampling)
    for output, target in zip(actual, expected):
        assert output.dtype == mx.float32
        np.testing.assert_allclose(np.array(output), target.numpy(), **parity_tolerance)


@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_seeded_batch_matches_individual_and_preserves_global_rng(
    models, parity_tolerance, cfg_strength
):
    _, model = models
    _, fixed_items, _ = sample_items()
    items = [replace(item, noise=None) for item in fixed_items]
    sampling = dict(steps=3, cfg_strength=cfg_strength)
    expected = [model.sample(item, **sampling) for item in items]
    mx.random.seed(913)
    expected_random = mx.random.normal((8,))
    mx.eval(expected_random)
    mx.random.seed(913)
    actual = model.sample_batch(items, **sampling)
    actual_random = mx.random.normal((8,))
    np.testing.assert_array_equal(np.array(actual_random), np.array(expected_random))
    for output, target in zip(actual, expected):
        np.testing.assert_allclose(
            np.array(output), np.array(target), **parity_tolerance
        )
    repeated = model.sample(items[0], **sampling)
    np.testing.assert_array_equal(np.array(repeated), np.array(expected[0]))
    reordered = model.sample_batch(list(reversed(items)), **sampling)
    for output, target in zip(reordered, reversed(actual)):
        np.testing.assert_allclose(
            np.array(output), np.array(target), **parity_tolerance
        )


def test_unseeded_requests_preserve_global_rng(models):
    _, model = models
    _, fixed_items, _ = sample_items()
    item = replace(fixed_items[0], seed=None, noise=None)
    mx.random.seed(913)
    expected_random = mx.random.normal((8,))
    mx.eval(expected_random)
    mx.random.seed(913)
    first = model.sample(item, steps=1, cfg_strength=0)
    second = model.sample(item, steps=1, cfg_strength=0)
    actual_random = mx.random.normal((8,))
    np.testing.assert_array_equal(np.array(actual_random), np.array(expected_random))
    assert not np.array_equal(np.array(first), np.array(second))


def test_bfloat16_backbone_retains_float32_integration(models):
    reference, model = models
    reference_items, items, noises = sample_items()
    expected = model.sample_batch(items, steps=3, cfg_strength=2)
    model.transformer.set_dtype(mx.bfloat16)
    model.transformer.rotary_embed.inv_freq = (
        model.transformer.rotary_embed.inv_freq.astype(mx.float32)
    )
    reference.transformer.to(torch.bfloat16)
    reference.transformer.rotary_embed.inv_freq = (
        reference.transformer.rotary_embed.inv_freq.float()
    )
    with patch("sglang_omni.models.auk.flow_matching.torch.randn", side_effect=noises):
        torch_outputs = reference.sample_batch(reference_items, steps=3, cfg_strength=2)
    actual = model.sample_batch(items, steps=3, cfg_strength=2)
    for output, target, torch_output in zip(actual, expected, torch_outputs):
        assert output.dtype == mx.float32
        np.testing.assert_allclose(
            np.array(output), torch_output.numpy(), atol=0.04, rtol=0.02
        )
        output_np, target_np = np.array(output).flatten(), np.array(target).flatten()
        assert np.isfinite(output_np).all()
        cosine = np.dot(output_np, target_np) / (
            np.linalg.norm(output_np) * np.linalg.norm(target_np)
        )
        assert cosine > 0.99


def test_fusion_and_sampling_grids_match_torch():
    torch.manual_seed(17)
    hidden, weights, scale = (
        torch.randn(2, 5, 7, 16),
        torch.randn(4),
        torch.tensor([1.5]),
    )
    actual = fuse_hidden_states(
        *(mx.array(value.numpy()) for value in (hidden, weights, scale))
    )
    np.testing.assert_allclose(
        np.array(actual),
        torch_fuse(hidden, weights, scale).numpy(),
        atol=5e-7,
        rtol=2e-5,
    )
    for kwargs in [{}, {"sway_sampling_coef": -1.0}, {"t_grid": FLASH_T_GRID}]:
        np.testing.assert_allclose(
            np.array(build_time_grid(4, **kwargs)),
            torch_time_grid(4, **kwargs).numpy(),
            atol=1e-7,
        )


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": 3, "t_grid": [0.0]}])
def test_invalid_sampling_grid_is_rejected(kwargs):
    with pytest.raises(ValueError):
        build_time_grid(**kwargs)


def test_invalid_sampling_inputs_are_rejected(models):
    _, model = models
    _, items, _ = sample_items()
    with pytest.raises(ValueError, match="at least one"):
        model.sample_batch([], steps=1, cfg_strength=0)
    for invalid, message in [
        (replace(items[0], target_frames=0), "positive"),
        (replace(items[0], ref_length=100), "ref_length"),
        (replace(items[0], noise=mx.zeros((1, 8))), "noise shape"),
    ]:
        with pytest.raises(ValueError, match=message):
            model.sample(invalid, steps=1, cfg_strength=0)


def test_request_keys_keep_high_seed_bits_and_match_negative_seed_mapping():
    assert not np.array_equal(
        np.array(request_key(1)), np.array(request_key(2**32 + 1))
    )
    np.testing.assert_array_equal(
        np.array(request_key(-1)), np.array(request_key(2**64 - 1))
    )
    np.testing.assert_array_equal(
        np.array(request_key(-(2**63))), np.array(request_key(2**63))
    )
    for invalid in [-(2**63) - 1, 2**64]:
        with pytest.raises(ValueError, match="seed"):
            request_key(invalid)


def test_reference_and_generation_noise_streams_are_independent():
    target_key = request_key(42)
    reference_key = request_key(42, stream="reference")
    assert not np.array_equal(np.array(target_key), np.array(reference_key))
    expected = mx.random.normal((7, 8), key=target_key)
    mx.eval(mx.random.normal((19, 8), key=reference_key))
    mx.eval(mx.random.normal((3, 8), key=request_key(900, stream="reference")))
    actual = mx.random.normal((7, 8), key=request_key(42, stream="generation"))
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm_matches_float32_accumulation_epsilon(compute_device, dtype):
    reference = torch.nn.RMSNorm(16, eps=torch.finfo(torch.float32).eps).to(dtype)
    model = AuKDit(
        dim=16, heads=1, dim_head=8, num_layers=0, num_single_layers=0
    ).txt_norm
    model.weight = mx.array(reference.weight.detach().float().numpy())
    x = torch.linspace(-0.02, 0.03, 48).reshape(3, 16).to(dtype)
    inputs = mx.array(x.float().numpy()).astype(
        mx.bfloat16 if dtype == torch.bfloat16 else mx.float32
    )
    model.weight = model.weight.astype(inputs.dtype)
    with torch.no_grad():
        expected = reference(x).float().numpy()
    np.testing.assert_allclose(
        np.array(model(inputs).astype(mx.float32)),
        expected,
        atol=3e-3 if dtype == torch.bfloat16 else 1e-6,
        rtol=1e-3,
    )


def test_rotary_uses_checkpoint_frequencies_and_float32_positions(compute_device):
    reference = TorchRotary(8)
    reference.inv_freq = torch.tensor(
        [1.0, 0.099609375, 0.010009765625, 0.00099945068359375]
    )
    model = RotaryEmbedding(8)
    model.load_weights(
        [("inv_freq", mx.array(reference.inv_freq.numpy()))], strict=True
    )
    model.set_dtype(mx.bfloat16)
    positions = torch.tensor([[255, 256, 257, 511, 512, 513, 4097]])
    values = torch.linspace(-1, 1, 56).reshape(1, 1, 7, 8)
    angles, scale = reference(positions)
    expected = apply_rotary_pos_emb(values, angles, scale)
    actual = apply_rope(mx.array(values.numpy()), model(mx.array(positions.numpy())))
    np.testing.assert_allclose(np.array(actual), expected.numpy(), atol=1e-6, rtol=1e-5)
