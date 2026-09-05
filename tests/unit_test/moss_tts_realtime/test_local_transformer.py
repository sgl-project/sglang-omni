# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang_omni.models.moss_tts_realtime.local_transformer import (
    MossTTSRealtimeLocalTransformerForCausalLM,
    MossTTSRealtimeLocalTransformerModel,
    _repeat_kv,
    _rotate_half,
)


def _local_config(*, num_layers: int = 2) -> PretrainedConfig:
    return PretrainedConfig(
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=1_000_000.0,
        max_position_embeddings=33,
        attention_bias=False,
        attention_dropout=0.0,
        audio_vocab_size=1027,
        audio_pad_token=1024,
        rvq=16,
    )


def _full_forward(
    model: MossTTSRealtimeLocalTransformerModel,
    inputs: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, _ = inputs.shape
    config = model.config
    current = inputs
    cos = model.rope_cos[:sequence].view(1, sequence, 1, config.head_dim)
    sin = model.rope_sin[:sequence].view(1, sequence, 1, config.head_dim)

    for layer in model.layers:
        residual = current
        normalized = layer.input_layernorm(current)
        query = layer.self_attn.q_proj(normalized).view(
            batch,
            sequence,
            config.num_attention_heads,
            config.head_dim,
        )
        key = layer.self_attn.k_proj(normalized).view(
            batch,
            sequence,
            config.num_key_value_heads,
            config.head_dim,
        )
        value = layer.self_attn.v_proj(normalized).view(
            batch,
            sequence,
            config.num_key_value_heads,
            config.head_dim,
        )
        query = layer.self_attn.q_norm(query)
        key = layer.self_attn.k_norm(key)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        query = query.transpose(1, 2)
        key = _repeat_kv(key.transpose(1, 2), 2)
        value = _repeat_kv(value.transpose(1, 2), 2)
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=layer.self_attn.scaling,
        )
        attention = attention.transpose(1, 2).reshape(batch, sequence, -1)
        current = residual + layer.self_attn.o_proj(attention)
        residual = current
        current = residual + layer.mlp(layer.post_attention_layernorm(current))
    return model.norm(current)


@pytest.mark.parametrize("num_layers", [1, 2])
def test_local_incremental_matches_full_causal_recompute(num_layers: int) -> None:
    torch.manual_seed(0)
    model = MossTTSRealtimeLocalTransformerModel(
        _local_config(num_layers=num_layers)
    ).eval()
    inputs = torch.randn(3, 16, 64)

    expected = _full_forward(model, inputs)
    actual = torch.stack(
        [model.step(inputs[:, position], position) for position in range(16)],
        dim=1,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_local_teacher_forcing_and_decode_callback_use_codebook_order() -> None:
    torch.manual_seed(1)
    local = MossTTSRealtimeLocalTransformerForCausalLM(_local_config()).eval()
    hidden = torch.randn(2, 64)
    prefix = torch.randint(0, 1024, (2, 15))

    logits = local.teacher_forced_logits(hidden, prefix)
    visited: list[int] = []

    def sample_audio(values: torch.Tensor, codebook: int) -> torch.Tensor:
        visited.append(codebook)
        if codebook < 15:
            return prefix[:, codebook]
        return torch.argmax(values, dim=-1)

    decoded = local.decode_frame(hidden, sample_audio=sample_audio)

    assert logits.shape == (2, 16, 1027)
    assert decoded.shape == (2, 16)
    assert visited == list(range(16))
    torch.testing.assert_close(decoded[:, :15], prefix)
    torch.testing.assert_close(decoded[:, 15], torch.argmax(logits[:, 15], dim=-1))


@pytest.mark.parametrize(
    "bad_prefix",
    [
        torch.zeros((1, 15), dtype=torch.float32),
        torch.zeros((1, 15), dtype=torch.complex64),
    ],
)
def test_local_teacher_forcing_rejects_non_integer_prefix(
    bad_prefix: torch.Tensor,
) -> None:
    local = MossTTSRealtimeLocalTransformerForCausalLM(_local_config()).eval()

    with pytest.raises(TypeError, match="integer tensor"):
        local.teacher_forced_logits(torch.randn(1, 64), bad_prefix)


def test_local_decode_rejects_invalid_sampler_output() -> None:
    local = MossTTSRealtimeLocalTransformerForCausalLM(_local_config()).eval()
    hidden = torch.randn(1, 64)

    with pytest.raises(TypeError, match="integer token ids"):
        local.decode_frame(
            hidden,
            sample_audio=lambda logits, codebook: torch.zeros(1),
        )

    with pytest.raises(ValueError, match="out-of-range"):
        local.decode_frame(
            hidden,
            sample_audio=lambda logits, codebook: torch.full(
                (1,),
                1027,
                dtype=torch.long,
            ),
        )


def test_local_decode_accepts_exact_compute_provider() -> None:
    torch.manual_seed(9)
    local = MossTTSRealtimeLocalTransformerForCausalLM(_local_config()).eval()
    hidden = torch.randn(1, 64)
    prefix = torch.randint(0, 1024, (1, 15))
    visited: list[int] = []

    def compute_logits(current: torch.Tensor, codebook: int) -> torch.Tensor:
        visited.append(codebook)
        local_hidden = local.model.step(current, codebook)
        return local.local_lm_heads[codebook](local_hidden)

    def sample_audio(logits: torch.Tensor, codebook: int) -> torch.Tensor:
        if codebook < 15:
            return prefix[:, codebook]
        return torch.argmax(logits, dim=-1)

    expected = local.teacher_forced_logits(hidden, prefix)
    actual = local.decode_frame(
        hidden,
        sample_audio=sample_audio,
        compute_logits=compute_logits,
    )

    assert visited == list(range(16))
    torch.testing.assert_close(actual[:, :15], prefix)
    torch.testing.assert_close(actual[:, 15], torch.argmax(expected[:, 15], dim=-1))


def test_local_cache_grows_and_rejects_invalid_inputs() -> None:
    model = MossTTSRealtimeLocalTransformerModel(_local_config(num_layers=1))
    assert model.step(torch.randn(2, 64), 0).shape == (2, 64)
    assert model.step(torch.randn(8, 64), 0).shape == (8, 64)
    assert model._kv_capacity >= 8

    with pytest.raises(ValueError, match="local position"):
        model.step(torch.randn(1, 64), 16)
    with pytest.raises(ValueError, match="hidden size"):
        model.step(torch.randn(1, 32), 0)


def test_local_cache_freeze_preserves_captured_capacity() -> None:
    model = MossTTSRealtimeLocalTransformerModel(_local_config(num_layers=1))
    model._ensure_kv_cache(4, device=torch.device("cpu"), dtype=torch.float32)
    model.freeze_kv_cache()

    assert model.step(torch.randn(4, 64), 0).shape == (4, 64)
    with pytest.raises(RuntimeError, match="frozen after CUDA graph capture"):
        model.step(torch.randn(5, 64), 0)


def test_local_cache_matches_hf_static_shape_and_resets_per_frame() -> None:
    model = MossTTSRealtimeLocalTransformerModel(_local_config(num_layers=1))

    for position in range(16):
        model.step(torch.randn(2, 64), position)
    key_cache, value_cache = model._kv_cache[0]
    assert key_cache.shape == (2, 2, 16, 16)
    assert value_cache.shape == (2, 2, 16, 16)
    assert torch.count_nonzero(key_cache[:, :, 1:]) > 0
    assert torch.count_nonzero(value_cache[:, :, 1:]) > 0

    model.step(torch.randn(2, 64), 0)

    assert torch.count_nonzero(key_cache[:, :, 1:]) == 0
    assert torch.count_nonzero(value_cache[:, :, 1:]) == 0


def test_local_parameter_names_match_checkpoint_layout() -> None:
    local = MossTTSRealtimeLocalTransformerForCausalLM(_local_config(num_layers=1))
    names = set(dict(local.named_parameters()))

    assert "model.embed_tokens.14.weight" in names
    assert "model.layers.0.self_attn.q_proj.weight" in names
    assert "model.layers.0.self_attn.k_norm.weight" in names
    assert "model.layers.0.mlp.gate_proj.weight" in names
    assert "model.norm.weight" in names
    assert "local_lm_heads.15.weight" in names


def test_qwen_half_split_rotation_is_not_interleaved() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert _rotate_half(values).tolist() == [-3.0, -4.0, 1.0, 2.0]
