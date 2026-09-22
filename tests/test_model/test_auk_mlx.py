# SPDX-License-Identifier: Apache-2.0
"""Opt-in local-checkpoint MLX parity with MLX_ENABLE_TF32=0."""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

pytestmark = pytest.mark.accelerator


@pytest.fixture
def checkpoint() -> str:
    path = os.environ.get("AUK_MLX_CHECKPOINT")
    if not path:
        pytest.skip("Set AUK_MLX_CHECKPOINT to a local AuK or AuK-Flash checkpoint")
    elif not Path(path).is_dir():
        pytest.fail(
            "AUK_MLX_CHECKPOINT must name an existing local checkpoint directory"
        )
    return str(Path(path).resolve())


@pytest.fixture
def metal_backend(checkpoint: str) -> Iterator[ModuleType]:
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("Native AuK checkpoint parity requires Apple Metal")
    elif os.environ.get("MLX_ENABLE_TF32") != "0":
        pytest.fail(
            "Run real-checkpoint parity with MLX_ENABLE_TF32=0 for strict float32 matmuls"
        )
    from sglang_omni.models.auk.mlx.loader import load_vae

    with mx.stream(mx.gpu):
        try:
            yield mx
        finally:
            load_vae.cache_clear()
            gc.collect()
            mx.clear_cache()


def assert_parity(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> None:
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    error = np.abs(actual - expected)
    relative_l2 = np.linalg.norm(error.reshape(-1)) / max(
        np.linalg.norm(expected.reshape(-1)), np.finfo(np.float32).eps
    )
    np.testing.assert_allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        err_msg=f"max_abs={error.max():.6g}, relative_l2={relative_l2:.6g}",
    )


def assert_conditioning_parity(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    reference = expected.astype(np.float64)
    squared_error = (actual.astype(np.float64) - reference) ** 2
    token_rms = np.sqrt(np.mean(reference**2, axis=-1, keepdims=True))
    token_scale = np.maximum(token_rms, np.finfo(np.float32).eps)
    token_relative_l2 = np.sqrt(np.mean(squared_error, axis=-1)) / token_scale[..., 0]

    # note (Codex): FP32 cancellation near zero needs the residual vector's scale;
    # separate token and layer bounds prevent large channels from hiding drift.
    if actual.ndim == 3:
        layer_relative_l2 = np.sqrt(
            np.sum(squared_error, axis=(-2, -1))
            / np.maximum(np.sum(reference**2, axis=(-2, -1)), np.finfo(np.float32).eps)
        )
        assert np.max(layer_relative_l2) <= 1e-4, layer_relative_l2
        token_limit, normalized_atol = 5e-4, 1e-3
    else:
        assert actual.ndim == 2
        token_limit, normalized_atol = 1e-4, 2e-4
    assert np.max(token_relative_l2) <= token_limit, token_relative_l2
    assert_parity(
        actual / token_scale,
        expected / token_scale,
        atol=normalized_atol,
        rtol=2e-3,
    )


def test_vae_checkpoint_encode_decode_matches_torch_cpu(
    checkpoint: str, metal_backend: ModuleType
) -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.auk.hf_config import AuKVAEConfig, make_runtime_config
    from sglang_omni.models.auk.mlx.loader import load_vae
    from sglang_omni.models.auk.vae import BigVGANFlowVAE
    from sglang_omni.models.auk.weight_loader import load_vae_weights

    mx = metal_backend
    config = make_runtime_config(checkpoint)
    vae_config = AuKVAEConfig.from_dict(config.vae_init_kwargs)
    rng = np.random.default_rng(42)
    sample = rng.normal(0, 0.05, (2, 1, vae_config.hop_size * 8 + 117)).astype(
        np.float32
    )
    lengths = np.array([sample.shape[-1], vae_config.hop_size * 5], dtype=np.int64)
    with torch.device("meta"):
        reference = BigVGANFlowVAE(vae_config)
    load_vae_weights(reference, checkpoint)
    reference = reference.float().eval().requires_grad_(False)
    with torch.inference_mode():
        shape = reference.audio_encoder(torch.from_numpy(sample)).shape
        noise = rng.standard_normal(
            (shape[0], vae_config.latent_dim, shape[-1])
        ).astype(np.float32)
        with patch(
            "sglang_omni.models.auk.vae.torch.randn",
            return_value=torch.from_numpy(noise),
        ):
            latent, valid_lengths = reference.encoding_and_normalization(
                torch.from_numpy(sample), torch.from_numpy(lengths)
            )
        expected_latent = latent.numpy().copy()
        expected_lengths = valid_lengths.numpy().copy()
        expected_waveform = reference.decode(latent).numpy().copy()
    del reference, latent, valid_lengths
    gc.collect()

    native = load_vae(checkpoint)
    actual_latent, actual_lengths = native.encode(
        mx.array(sample), mx.array(lengths), noise=mx.array(noise)
    )
    assert_parity(np.array(actual_latent), expected_latent, atol=2e-4, rtol=2e-3)
    np.testing.assert_array_equal(np.array(actual_lengths), expected_lengths)
    waveform = native.decode(mx.array(expected_latent))
    assert_parity(np.array(waveform), expected_waveform, atol=3e-4, rtol=3e-3)


def test_dit_checkpoint_sampling_matches_torch_cpu(
    checkpoint: str, metal_backend: ModuleType
) -> None:
    torch = pytest.importorskip("torch")
    from sglang_omni.models.auk.constants import FLASH_T_GRID
    from sglang_omni.models.auk.dit import AuKDit
    from sglang_omni.models.auk.flow_matching import AuKFlowMatching, AuKSampleItem
    from sglang_omni.models.auk.hf_config import AuKDitConfig, make_runtime_config
    from sglang_omni.models.auk.mlx.flow_matching import AuKSampleItem as MlxSampleItem
    from sglang_omni.models.auk.mlx.loader import load_flow, load_fusion
    from sglang_omni.models.auk.weight_loader import load_dit_weights

    mx = metal_backend
    config = make_runtime_config(checkpoint)
    arch = asdict(AuKDitConfig.from_dict(config.arch))
    arch["latent_dim"] = config.latent_dim
    fusion_weights, _ = load_fusion(checkpoint)
    with torch.device("meta"):
        reference = AuKFlowMatching(AuKDit(**arch), num_llm_layers=fusion_weights.size)
    load_dit_weights(reference, checkpoint)
    reference = reference.float().eval().requires_grad_(False)
    rng = np.random.default_rng(73)
    items, native_items, noises = [], [], []
    for text_length, frames, reference_frames in [(6, 11, 3), (4, 7, 0)]:
        text = rng.standard_normal((text_length, arch["text_hidden_dim"])).astype(
            np.float32
        )
        mask = np.ones(text_length, dtype=bool)
        ref = rng.standard_normal((reference_frames, config.latent_dim)).astype(
            np.float32
        )
        noise = rng.standard_normal((frames, config.latent_dim)).astype(np.float32)
        items.append(
            AuKSampleItem(
                torch.from_numpy(text),
                torch.from_numpy(mask),
                frames,
                torch.from_numpy(ref),
                ref_length=reference_frames,
            )
        )
        native_items.append(
            MlxSampleItem(
                conditioning=mx.array(text),
                text_mask=mx.array(mask),
                target_frames=frames,
                ref_latent=mx.array(ref),
                ref_length=reference_frames,
                noise=mx.array(noise),
            )
        )
        noises.append(torch.from_numpy(noise))
    sampling = dict(
        steps=4 if config.is_flash else 3,
        cfg_strength=0.0 if config.is_flash else 2.0,
        sway_sampling_coef=None if config.is_flash else -1.0,
        t_grid=FLASH_T_GRID if config.is_flash else None,
    )
    with patch("sglang_omni.models.auk.flow_matching.torch.randn", side_effect=noises):
        expected = [
            value.numpy().copy() for value in reference.sample_batch(items, **sampling)
        ]
    del reference
    gc.collect()

    native = load_flow(checkpoint, mx.float32)
    actual = native.sample_batch(native_items, **sampling)
    for output, target in zip(actual, expected):
        assert output.dtype == mx.float32
        assert_parity(np.array(output), target, atol=3e-4, rtol=2e-3)


def test_conditioner_checkpoint_text_audio_matches_torch_cpu(
    checkpoint: str, metal_backend: ModuleType
) -> None:
    path = os.environ.get("AUK_QWEN_CHECKPOINT")
    if not path:
        pytest.skip("Set AUK_QWEN_CHECKPOINT to a local Qwen2.5-Omni-3B checkpoint")
    elif not Path(path).is_dir():
        pytest.fail(
            "AUK_QWEN_CHECKPOINT must name an existing local checkpoint directory"
        )
    torch = pytest.importorskip("torch")
    from sglang_omni.models.auk.flow_matching import fuse_hidden_states as torch_fuse
    from sglang_omni.models.auk.mlx.conditioning import AuKMlxConditionEncoder
    from sglang_omni.models.auk.mlx.flow_matching import fuse_hidden_states
    from sglang_omni.models.auk.mlx.loader import load_fusion
    from sglang_omni.models.auk.reference_encode import (
        AuKConditionEncoder,
        build_messages,
    )

    mx = metal_backend
    messages = [
        build_messages('Say "Welcome home." in a warm voice.', False),
        build_messages('Say "你好，欢迎回家。" with the same voice.', True),
    ]
    audio = (0.1 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)).astype(np.float32)
    audios = [None, audio]
    weights, scale = load_fusion(checkpoint)
    reference = AuKConditionEncoder(
        str(Path(path).resolve()), device="cpu", dtype=torch.float32
    )
    encodings = reference.encode_batch(messages, audios)
    expected = [
        (hidden.numpy().copy(), mask.numpy().copy()) for hidden, mask in encodings
    ]
    expected_fused = [
        torch_fuse(
            hidden[None],
            torch.from_numpy(np.array(weights)),
            torch.from_numpy(np.array(scale)),
        )[0]
        .numpy()
        .copy()
        for hidden, _ in encodings
    ]
    del reference, encodings
    gc.collect()

    native = AuKMlxConditionEncoder(str(Path(path).resolve()), dtype=mx.float32)
    actual = native.encode_batch(messages, audios)
    for (hidden, mask), (target, target_mask), fused_target in zip(
        actual, expected, expected_fused
    ):
        assert_conditioning_parity(np.array(hidden), target)
        np.testing.assert_array_equal(np.array(mask), target_mask)
        fused = fuse_hidden_states(hidden[None], weights, scale)[0]
        assert_conditioning_parity(np.array(fused), fused_target)
