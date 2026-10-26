# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.sensenova_u1.neo_unify import image_denoise
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOLLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import Qwen3Model
from sglang_omni.models.sensenova_u1.neo_unify.qwen_attention import Qwen3Attention
from sglang_omni.models.sensenova_u1.neo_unify.rotary_embeddings import (
    Qwen3RotaryEmbedding,
)
from sglang_omni.models.sensenova_u1.sampling import SenseNovaU1Sampling


def tiny_dense_config(num_hidden_layers=2):
    config = NEOLLMConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        attention_bias=False,
        rms_norm_eps=1e-6,
    )
    config.layer_types = ["full_attention"] * num_hidden_layers
    config._attn_implementation = (  # noqa: leading-underscore -- Transformers attention backend field
        "eager"
    )
    return config


def axis_indexes(batch_size=1, sequence_length=5):
    positions = torch.arange(sequence_length)
    indexes = torch.stack((positions, positions * 2, positions * 5))
    if batch_size == 1:
        return indexes
    return indexes.unsqueeze(0).expand(batch_size, -1, -1).clone()


@pytest.mark.parametrize("image_only", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_rope_tables_are_built_once_per_dense_forward(
    monkeypatch, image_only, batch_size
):
    model = Qwen3Model(tiny_dense_config()).eval()
    inputs = torch.randn(batch_size, 5, 8)
    indexes = axis_indexes(batch_size)
    indicators = None if image_only else torch.zeros(batch_size, 5, dtype=torch.bool)
    calls = 0
    original = Qwen3RotaryEmbedding.forward

    def count_calls(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Qwen3RotaryEmbedding, "forward", count_calls)
    with torch.no_grad():
        model(
            inputs_embeds=inputs,
            image_gen_indicators=indicators,
            image_only=image_only,
            indexes=indexes,
            attention_mask={"full_attention": None},
        )

    assert calls == 3


def test_shared_rope_tables_match_per_layer_for_batched_indexes():
    first = Qwen3Attention(tiny_dense_config(), 0)
    second = Qwen3Attention(tiny_dense_config(), 1)
    hidden_states = torch.randn(2, 5, 8)
    indexes = axis_indexes(batch_size=2)

    shared = first.resolve_rope_tables(indexes, hidden_states)
    reference = second.resolve_rope_tables(indexes, hidden_states)

    for shared_axis, reference_axis in zip(shared, reference):
        for shared_value, reference_value in zip(shared_axis, reference_axis):
            assert torch.equal(shared_value, reference_value)


def test_shared_rope_tables_rebuild_after_dtype_change():
    attention = Qwen3Attention(tiny_dense_config(), 0)
    indexes = axis_indexes()
    fp32_tables = attention.resolve_rope_tables(
        indexes, torch.randn(1, 5, 8, dtype=torch.float32)
    )

    bf16_tables = attention.resolve_rope_tables(
        indexes,
        torch.randn(1, 5, 8, dtype=torch.bfloat16),
        fp32_tables,
    )

    assert bf16_tables[0][0].dtype == torch.bfloat16
    assert bf16_tables is not fp32_tables


def test_rope_sharing_matches_per_layer_build_with_bf16_norm_dispatch(monkeypatch):
    torch.manual_seed(0)
    model = Qwen3Model(tiny_dense_config()).eval()
    inputs = torch.randn(1, 5, 8, dtype=torch.bfloat16)
    indexes = axis_indexes()

    def run():
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            return model(
                inputs_embeds=inputs,
                image_only=True,
                indexes=indexes,
                attention_mask={"full_attention": None},
            ).last_hidden_state

    shared = run()
    original = Qwen3Attention.resolve_rope_tables
    monkeypatch.setattr(
        Qwen3Attention,
        "resolve_rope_tables",
        lambda self, idx, hidden, _tables=None: original(self, idx, hidden, None),
    )
    per_layer = run()

    assert torch.equal(shared, per_layer)


def test_image_only_matches_all_true_indicators():
    torch.manual_seed(0)
    model = Qwen3Model(tiny_dense_config()).eval()
    inputs = torch.randn(1, 5, 8)
    indexes = axis_indexes()
    common = {
        "inputs_embeds": inputs,
        "indexes": indexes,
        "attention_mask": {"full_attention": None},
    }

    with torch.no_grad():
        expected = model(
            image_gen_indicators=torch.ones(1, 5, dtype=torch.bool), **common
        ).last_hidden_state
        actual = model(image_only=True, **common).last_hidden_state

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_denoising_embeds_each_timestep_once_per_step(monkeypatch):
    """CUDA denoising embedded every image token at every step; each step must still add its own timestep's embedding."""
    embedded_timestep_counts: list[int] = []

    class RecordingTimestepEmbedder(nn.Module):
        def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
            embedded_timestep_counts.append(timesteps.numel())
            return timesteps[:, None].expand(-1, 8).float()

    model = SimpleNamespace(
        downsample_ratio=0.5,
        patch_size=2,
        noise_scale=1.0,
        noise_scale_mode="constant",
        noise_scale_base_image_seq_len=1,
        noise_scale_max_value=1.0,
        add_noise_scale_embedding=False,
        fm_modules={"timestep_embedder": RecordingTimestepEmbedder()},
        extract_feature=lambda pixel_values, gen_model, grid_hw: torch.zeros(
            pixel_values.shape[0], 2
        ),
    )
    prefix = image_denoise.ImagePrefix(
        cache=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    keys=torch.zeros(1, 1, 3, 4), values=torch.zeros(1, 1, 3, 4)
                )
            ]
        ),
        image_indexes=torch.zeros(3, 64, dtype=torch.long),
        token_lengths=torch.tensor([3]),
    )
    predictor_inputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def predict_velocity(
        model, embeddings, indexes, mask, cache, timestep, latents, **kwargs
    ):
        predictor_inputs.append((embeddings, timestep))
        return torch.zeros_like(latents)

    monkeypatch.setattr(image_denoise, "t2i_predict_v", predict_velocity)
    sampling = SenseNovaU1Sampling.from_params(
        {"width": 32, "height": 32, "num_inference_steps": 3, "guidance_scale": 1.0}
    )

    images = image_denoise.denoise_images(
        model, sampling=sampling, prefixes=[prefix], seeds=[7, 8]
    )

    assert images.shape == (2, 3, 32, 32)
    assert embedded_timestep_counts == [3]
    assert len(predictor_inputs) == 3
    for embeddings, timestep in predictor_inputs:
        torch.testing.assert_close(
            embeddings, timestep.expand_as(embeddings), rtol=0, atol=0
        )
