# SPDX-License-Identifier: Apache-2.0
"""Focused image-decoder tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from safetensors.torch import save_file

from sglang_omni.models.llada2_uni.components.decoder_model import (
    DecoderConfig,
    ZImageTransformer2DModelWrapper,
    decoder_config,
)
from sglang_omni.models.llada2_uni.components.image_decoder import (
    LLaDA2ImageDecoder,
    create_decoder_model_fn,
)
from sglang_omni.models.llada2_uni.components.transport import sample_velocity


def native_worker(
    rank: int, directory: str, cfg: DecoderConfig, dtype: torch.dtype
) -> None:
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    from sglang_omni.models.llada2_uni.components.decoder_runtime import (
        initialize_decoder_runtime,
    )

    path = Path(directory)
    with initialize_decoder_runtime(
        directory,
        gpu_id=0,
        dtype=dtype,
        attention_backend="torch_sdpa",
        dist_timeout=120,
    ) as runtime:
        with runtime.compute_context(), torch.inference_mode():
            for device in ("cuda", "cuda:0"):
                decoder = LLaDA2ImageDecoder(
                    directory,
                    device=device,
                    dtype=dtype,
                    backend="sglang",
                    runtime=runtime,
                )
                assert decoder.device == runtime.device
            with pytest.raises(ValueError, match="device/dtype must match"):
                LLaDA2ImageDecoder(
                    directory,
                    device="cuda:1",
                    dtype=dtype,
                    backend="sglang",
                    runtime=runtime,
                )
            inputs = torch.load(
                path / "inputs.pt", weights_only=True, map_location=runtime.device
            )
            model = ZImageTransformer2DModelWrapper(
                directory,
                cfg,
                runtime.device,
                dtype,
                backend="sglang",
                runtime=runtime,
            )
            output = torch.stack(
                model(
                    list(inputs["x"].to(dtype).unbind(0)),
                    inputs["t"],
                    list(inputs["cap"].to(dtype).unbind(0)),
                    return_dict=False,
                )[0]
            )
            torch.save(output.cpu(), path / "native.pt")
    assert not torch.distributed.is_initialized()


@pytest.mark.accelerator
def test_sglang_backend_matches_diffusers(tmp_path: Path) -> None:
    dtype = torch.bfloat16
    cfg = decoder_config(
        {
            "dim": 768,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 6,
            "n_kv_heads": 6,
            "cap_feat_dim": 16,
            "axes_dims": (32, 48, 48),
            "axes_lens": (128, 64, 64),
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        reference = ZImageTransformer2DModel(**cfg).eval()
        torch.nn.init.normal_(reference.x_pad_token, std=0.01)
        torch.nn.init.normal_(reference.cap_pad_token, std=0.01)
        inputs = {
            "x": torch.randn(2, 16, 1, 16, 16),
            "cap": torch.randn(2, 32, 16),
            "t": torch.tensor([0.125, 0.875]),
        }
    save_file(
        {
            name.replace("cap_embedder.", "semantic_embedder."): value.contiguous()
            for name, value in reference.state_dict().items()
        },
        str(tmp_path / "model.safetensors"),
    )
    torch.save(inputs, tmp_path / "inputs.pt")
    with torch.inference_mode():
        expected = torch.stack(
            reference(
                x=list(inputs["x"].unbind(0)),
                t=inputs["t"],
                cap_feats=list(inputs["cap"].unbind(0)),
                return_dict=False,
            )[0]
        )
    del reference

    mp.spawn(
        native_worker,
        args=(str(tmp_path), cfg, dtype),
        nprocs=1,
        join=True,
    )
    actual = torch.load(tmp_path / "native.pt", weights_only=True)
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


@pytest.fixture
def tiny_config():
    return decoder_config(
        {
            "dim": 32,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 4,
            "n_kv_heads": 4,
            "cap_feat_dim": 16,
            "axes_dims": (2, 2, 4),
            "axes_lens": (128, 32, 32),
        }
    )


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_config):
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = ZImageTransformer2DModel(**tiny_config).eval()
        torch.nn.init.normal_(model.x_pad_token, std=0.01)
        torch.nn.init.normal_(model.cap_pad_token, std=0.01)
    state = {
        key.replace("cap_embedder.", "semantic_embedder."): value.contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    return tmp_path, model, state


def test_diffusers_checkpoint_forward(tiny_checkpoint, tiny_config):
    path, reference, _ = tiny_checkpoint
    wrapper = ZImageTransformer2DModelWrapper(path, tiny_config, "cpu", torch.float32)
    x = [torch.randn(16, 1, 4, 6), torch.randn(16, 1, 6, 4)]
    cap = [torch.randn(7, 16), torch.randn(33, 16)]
    t = torch.tensor([0.125, 0.875])
    with torch.inference_mode():
        expected = reference(x=x, t=t, cap_feats=cap, return_dict=False)[0]
        actual = wrapper(x, t, cap, return_dict=False)[0]
    for actual_item, expected_item in zip(actual, expected):
        torch.testing.assert_close(actual_item, expected_item, rtol=0, atol=0)


def test_cfg_batched_model_fn():
    calls = []
    positive = [torch.ones(4, 16), torch.ones(5, 16) * 2]
    negative = [torch.zeros_like(cap) for cap in positive]
    positive_latent = torch.tensor([3.0, 4.0]).reshape(2, 1, 1, 1)
    negative_latent = torch.tensor([-1.0, 2.0]).reshape(2, 1, 1, 1)

    def model(**kwargs):
        calls.append(kwargs)
        return (
            [
                positive_latent,
                positive_latent * 2,
                negative_latent,
                negative_latent * 2,
            ],
        )

    model_fn = create_decoder_model_fn(
        model, positive, negative, 1.0, 2, 1, torch.bfloat16
    )
    x = torch.randn(2, 2, 1, 1, 1)
    actual = model_fn(x, torch.tensor(0.25))
    expected = positive_latent + (positive_latent - negative_latent)
    expected *= min(
        1.0,
        torch.linalg.vector_norm(positive_latent) / torch.linalg.vector_norm(expected),
    )
    torch.testing.assert_close(actual, torch.stack([expected, expected * 2]))
    assert actual.dtype == torch.float32
    assert calls[0]["cap_feats"] == positive + negative
    assert len(calls[0]["x"]) == 4


def test_turbo_sampler_matches_reference_and_preserves_rng():
    initial_latents = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)

    def velocity_model(latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return latents * 0.125 + timestep.reshape(-1, 1, 1, 1) * 0.25

    actual_generator = torch.Generator().manual_seed(29)
    actual = sample_velocity(
        initial_latents,
        velocity_model,
        num_steps=3,
        turbo=True,
        generator=actual_generator,
    )

    reference_generator = torch.Generator().manual_seed(29)
    reference_timesteps = torch.linspace(0, 1, 4, dtype=torch.float64).to(
        initial_latents
    )
    reference_latents = initial_latents.to(torch.float64)
    for current_time, next_time in zip(
        reference_timesteps[:-1], reference_timesteps[1:]
    ):
        batch_time = (
            torch.ones(reference_latents.shape[0], dtype=reference_latents.dtype)
            * current_time
        )
        prediction = velocity_model(reference_latents, batch_time)
        expanded_time = batch_time[(...,) + (None,) * (reference_latents.ndim - 1)]
        alpha, sigma = expanded_time, 1 - expanded_time
        denominator = sigma - -alpha
        clean = (sigma * prediction - -reference_latents) / denominator
        next_batch_time = torch.ones_like(batch_time) * next_time
        next_alpha = next_batch_time[(...,) + (None,) * (reference_latents.ndim - 1)]
        next_sigma = 1 - next_alpha
        noise = torch.randn(
            reference_latents.shape,
            dtype=reference_latents.dtype,
            generator=reference_generator,
        )
        reference_latents = next_alpha * clean + next_sigma * noise

    torch.testing.assert_close(actual, reference_latents)
    torch.testing.assert_close(
        torch.randn(4, generator=actual_generator),
        torch.randn(4, generator=reference_generator),
    )


@pytest.mark.parametrize(
    "turbo,steps,expected_calls", [(False, 2, 1), (False, 50, 49), (True, 1, 1)]
)
def test_sampler_forward_counts(turbo, steps, expected_calls):
    initial_latents = torch.zeros(1, 1, 1, 1)
    calls = []

    def model(latents: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        calls.append(timestep.clone())
        return torch.zeros_like(latents)

    sample_velocity(
        initial_latents,
        model,
        num_steps=steps,
        turbo=turbo,
        generator=torch.Generator().manual_seed(29),
    )
    assert len(calls) == expected_calls


def test_normal_step_bound_before_loading(tmp_path):
    with pytest.raises(ValueError, match="normal num_steps must be at least 2"):
        LLaDA2ImageDecoder(str(tmp_path), device="cpu", num_steps=1)
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")
    with pytest.raises(ValueError, match="normal num_steps must be at least 2"):
        decoder.decode([1], 1, 1, num_steps=1)
    LLaDA2ImageDecoder(
        str(tmp_path), device="cpu", decode_mode="decoder-turbo", num_steps=1
    )


@pytest.fixture
def decode_probe(monkeypatch, tmp_path):
    observed = SimpleNamespace(ids=None, latents=None)
    decoder = LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        dtype=torch.float32,
        num_steps=4,
        resolution_multiplier=1,
    )

    def sigvq(ids):
        observed.ids = ids.clone()
        return ids.float().unsqueeze(-1).expand(-1, -1, 16)

    def model(**kwargs):
        return ([torch.ones_like(latent) for latent in kwargs["x"]],)

    def vae_decode(latents, return_dict):
        observed.latents = latents.clone()
        assert not return_dict
        pixels = torch.empty(1, 3, latents.shape[-2] * 8, latents.shape[-1] * 8)
        pixels[:, 0], pixels[:, 1], pixels[:, 2] = -2, 0, 2
        return (pixels,)

    decoder.sigvq = sigvq
    decoder.diff_model = model
    decoder.diff_config = {
        "all_patch_size": [2],
        "all_f_patch_size": [1],
    }
    decoder.vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=2.0, shift_factor=3.0),
        decode=vae_decode,
    )
    monkeypatch.setattr(decoder, "ensure_diff_model", lambda mode: None)
    return decoder, observed


def test_decode_pipeline(decode_probe):
    decoder, observed = decode_probe
    image = decoder.decode([1, 2], 1, 2, seed=9)
    assert image.size == (32, 16)
    assert image.mode == "RGB" and image.getpixel((0, 0)) == (0, 127, 255)
    torch.testing.assert_close(observed.ids, torch.tensor([[1, 1, 2, 2, 1, 1, 2, 2]]))
    expected_noise = torch.randn(
        (1, 16, 1, 2, 4), generator=torch.Generator().manual_seed(9)
    )
    torch.testing.assert_close(
        observed.latents, (expected_noise.squeeze(2) + 1) / 2 + 3
    )


def test_decode_rejects_invalid_tokens_before_loading(tmp_path, monkeypatch):
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")
    monkeypatch.setattr(
        decoder, "ensure_sigvq", lambda: pytest.fail("invalid input loaded weights")
    )
    with pytest.raises(ValueError):
        decoder.decode([16384], 1, 1)
