# SPDX-License-Identifier: Apache-2.0
"""Differential tests for the MLX Qwen3-TTS talker port.

The port swaps two things out of the mlx-audio reference: externally computed
interleaved MRoPE becomes ``nn.RoPE`` on the attention module (SGLang's MLX
attention contract), and the code predictor's two priming tokens are issued as
one two-token call instead of two single-token calls (SGLang's CUDA form).
Both reference algorithms are reimplemented here rather than imported, because
mlx-audio is deliberately not a dependency of this package.
"""

from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402

from sglang_omni.models.qwen3_tts.mlx.config import (  # noqa: E402
    CodePredictorConfig,
    ModelConfig,
    TalkerConfig,
)
from sglang_omni.models.qwen3_tts.mlx.model import Qwen3TTSTalkerModel  # noqa: E402
from sglang_omni.models.qwen3_tts.mlx.talker import (  # noqa: E402
    Qwen3TTSTalkerForConditionalGeneration,
    reset_code_cache,
)


def _talker_config(**overrides) -> TalkerConfig:
    """A tiny talker whose head_dim // 2 matches the MRoPE section sum."""
    predictor = CodePredictorConfig(
        vocab_size=32,
        hidden_size=12,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        head_dim=4,
        num_code_groups=4,
        rope_theta=10000.0,
    )
    config = TalkerConfig(
        code_predictor_config=predictor,
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        text_hidden_size=6,
        text_vocab_size=40,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling={
            "interleaved": True,
            # head_dim // 2 == 4 frequencies split across temporal/height/width
            "mrope_section": [2, 1, 1],
            "rope_type": "default",
        },
        num_code_groups=4,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _build(dtype=mx.float32) -> Qwen3TTSTalkerForConditionalGeneration:
    mx.random.seed(0)
    model = Qwen3TTSTalkerForConditionalGeneration(_talker_config())
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


# --------------------------------------------------------------------------
# mlx-audio reference: interleaved MRoPE computed outside attention
# --------------------------------------------------------------------------


def _reference_mrope_cos_sin(
    *,
    head_dim: int,
    base: float,
    mrope_section: list[int],
    position_ids: mx.array,
    dtype,
) -> tuple[mx.array, mx.array]:
    """mlx-audio ``TalkerRotaryEmbedding``: 3-D positions -> combined cos/sin.

    ``position_ids`` is ``[3, batch, seq_len]``.
    """
    inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    pos = mx.expand_dims(position_ids.astype(mx.float32), axis=2)
    inv = mx.broadcast_to(
        inv_freq[None, None, :, None],
        (3, position_ids.shape[1], inv_freq.shape[0], 1),
    )
    freqs = mx.swapaxes(inv @ pos, 2, 3)  # [3, batch, seq_len, head_dim // 2]

    # apply_interleaved_mrope: H takes indices 3k+1, W takes 3k+2, T keeps rest.
    indices = mx.arange(freqs.shape[-1])
    h_mask = ((indices % 3 == 1) & (indices < mrope_section[1] * 3)).reshape(1, 1, -1)
    w_mask = ((indices % 3 == 2) & (indices < mrope_section[2] * 3)).reshape(1, 1, -1)
    combined = mx.where(h_mask, freqs[1], freqs[0])
    combined = mx.where(w_mask, freqs[2], combined)

    emb = mx.concatenate([combined, combined], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class _ReferenceMRoPEAttention(nn.Module):
    """mlx-audio ``TalkerAttention``, sharing the ported module's weights."""

    def __init__(self, inner, mrope_section: list[int]) -> None:
        super().__init__()
        self.inner = inner
        self.mrope_section = mrope_section

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        inner = self.inner
        B, L, _ = x.shape
        offset = 0 if cache is None else cache.offset

        q = inner.q_proj(x).reshape(B, L, inner.num_heads, inner.head_dim)
        k = inner.k_proj(x).reshape(B, L, inner.num_kv_heads, inner.head_dim)
        v = inner.v_proj(x).reshape(B, L, inner.num_kv_heads, inner.head_dim)
        q = inner.q_norm(q).transpose(0, 2, 1, 3)
        k = inner.k_norm(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # The talker is always driven with three identical position rows.
        pos = mx.broadcast_to(mx.arange(offset, offset + L)[None, :], (B, L))
        cos, sin = _reference_mrope_cos_sin(
            head_dim=inner.head_dim,
            base=inner.config.rope_theta,
            mrope_section=self.mrope_section,
            position_ids=mx.stack([pos, pos, pos], axis=0),
            dtype=x.dtype,
        )
        cos = mx.expand_dims(cos, axis=1)
        sin = mx.expand_dims(sin, axis=1)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # mlx-audio always builds an explicit additive causal mask; comparing
        # against it also pins mlx-lm's "causal" shorthand as equivalent.
        additive = None
        if k.shape[2] > 1 and L > 1:
            additive = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(
                x.dtype
            )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=inner.scale, mask=additive
        )
        return inner.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


def _swap_in_reference_attention(model) -> None:
    section = model.config.rope_scaling["mrope_section"]
    for layer in model.model.layers:
        layer.self_attn = _ReferenceMRoPEAttention(layer.self_attn, section)


def _max_abs_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max())


# --------------------------------------------------------------------------
# RoPE equivalence
# --------------------------------------------------------------------------


def test_interleaved_mrope_collapses_to_plain_rope_for_equal_rows() -> None:
    head_dim, base = 8, 10000.0
    pos = mx.arange(5)[None, :]
    equal_rows = mx.stack([pos, pos, pos], axis=0)
    cos, sin = _reference_mrope_cos_sin(
        head_dim=head_dim,
        base=base,
        mrope_section=[2, 1, 1],
        position_ids=equal_rows,
        dtype=mx.float32,
    )

    # nn.RoPE(traditional=False) is exactly x*cos + rotate_half(x)*sin.
    mx.random.seed(1)
    x = mx.random.normal((1, 2, 5, head_dim))
    rope = nn.RoPE(head_dim, traditional=False, base=base)
    expected = (x * cos[:, None]) + (_rotate_half(x) * sin[:, None])
    assert _max_abs_diff(rope(x), expected) < 1e-5


def test_interleaved_mrope_diverges_when_rows_differ() -> None:
    """Pins the assumption: the collapse is a property of equal rows only."""
    pos = mx.arange(5)[None, :]
    diverged = mx.stack([pos, pos + 1, pos + 2], axis=0)
    cos, _ = _reference_mrope_cos_sin(
        head_dim=8,
        base=10000.0,
        mrope_section=[2, 1, 1],
        position_ids=diverged,
        dtype=mx.float32,
    )
    equal, _ = _reference_mrope_cos_sin(
        head_dim=8,
        base=10000.0,
        mrope_section=[2, 1, 1],
        position_ids=mx.broadcast_to(pos[None], (3, 1, 5)),
        dtype=mx.float32,
    )
    assert _max_abs_diff(cos, equal) > 1e-3


# --------------------------------------------------------------------------
# SGLang MLX attention contract
# --------------------------------------------------------------------------


def test_talker_attention_satisfies_the_sglang_attention_contract() -> None:
    contract = pytest.importorskip(
        "sglang.srt.hardware_backend.mlx.kv_cache.attention_contract"
    )
    model = _build()
    attn = model.model.layers[0].self_attn

    assert contract.is_attention_module(attn)
    assert contract.get_num_heads(attn) == 2
    assert contract.get_num_kv_heads(attn) == 1
    assert contract.get_head_dim(attn) == 8
    assert contract.get_attention_scale(attn) == pytest.approx(1.0 / math.sqrt(8))
    # A sliding-window marker on the container would make batched decode drop
    # KV past the window; the talker must not declare one.
    assert contract.get_container_window_size(model) is None
    assert contract.get_layer_window_sizes(model) == {}


def test_attention_discovery_covers_the_talker_and_not_the_code_predictor() -> None:
    patching = pytest.importorskip(
        "sglang.srt.hardware_backend.mlx.kv_cache.model_patching"
    )
    model = _build()

    layers, attrs = patching.find_attention_layers(model)
    assert list(layers) == list(model.model.layers)
    assert attrs == ["self_attn"] * len(model.model.layers)

    patched = patching.patch_model_attention(model)
    assert patched == len(model.model.layers)
    # The predictor keeps its own untouched attention: its KV is frame-local.
    predictor_attn = model.code_predictor.model.layers[0].self_attn
    assert not isinstance(predictor_attn, patching.MLXAttentionWrapper)
    assert patching.patch_model_attention(model) == 0


# --------------------------------------------------------------------------
# Talker numerical parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 2])
def test_prefill_and_decode_match_the_mrope_reference(batch_size: int) -> None:
    model = _build()
    config = model.config
    mx.random.seed(2)
    prompt = mx.random.normal((batch_size, 6, config.hidden_size))
    step = mx.random.normal((batch_size, 1, config.hidden_size))

    port_cache = model.make_cache()
    port_logits, port_hidden = model(prompt, cache=port_cache)
    port_step_logits, port_step_hidden = model(step, cache=port_cache)
    mx.eval(port_logits, port_hidden, port_step_logits, port_step_hidden)

    _swap_in_reference_attention(model)
    ref_cache = model.make_cache()
    ref_logits, ref_hidden = model(prompt, cache=ref_cache)
    ref_step_logits, ref_step_hidden = model(step, cache=ref_cache)
    mx.eval(ref_logits, ref_hidden, ref_step_logits, ref_step_hidden)

    assert _max_abs_diff(port_hidden, ref_hidden) < 1e-4
    assert _max_abs_diff(port_logits, ref_logits) < 1e-4
    assert _max_abs_diff(port_step_hidden, ref_step_hidden) < 1e-4
    assert _max_abs_diff(port_step_logits, ref_step_logits) < 1e-4


def test_batched_decode_through_the_sglang_wrapper_matches_per_request_decode() -> None:
    """The reason for conforming to the contract: generic ragged batched decode.

    Two requests with different prefill lengths must decode identically whether
    stepped one at a time or fused into one batched attention call.
    """
    kv = pytest.importorskip("sglang.srt.hardware_backend.mlx.kv_cache")
    model = _build()
    kv.patch_model_attention(model)
    config = model.config
    num_layers = len(model.model.layers)

    def new_caches() -> list:
        return [
            kv.ContiguousAttentionKVCache(
                n_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                max_seq_len=64,
                dtype=mx.float32,
            )
            for _ in range(num_layers)
        ]

    mx.random.seed(8)
    prompts = [
        mx.random.normal((1, 6, config.hidden_size)),
        mx.random.normal((1, 3, config.hidden_size)),
    ]
    steps = [mx.random.normal((1, 1, config.hidden_size)) for _ in prompts]

    stepwise_caches = [new_caches() for _ in prompts]
    batched_caches = [new_caches() for _ in prompts]
    for caches in (stepwise_caches, batched_caches):
        for prompt, cache in zip(prompts, caches):
            mx.eval(model(prompt, cache=cache))

    stepwise = mx.concatenate(
        [
            model(step, cache=cache)[0][:, -1, :]
            for step, cache in zip(steps, stepwise_caches)
        ],
        axis=0,
    )

    context = kv.BatchedDecodeContext(
        batch_size=len(prompts),
        seq_lens=[cache[0].offset for cache in batched_caches],
        attention_layer_caches=[
            [caches[layer_idx] for caches in batched_caches]
            for layer_idx in range(num_layers)
        ],
    )
    assert context.seq_lens == [6, 3] and context.needs_padding
    kv.set_context(context)
    try:
        # The model still needs an offset to build its mask; SGLang hands it a
        # shim while the wrapper reads real KV from the context.
        shim = [
            kv.AttentionOffsetCache(offset=max(context.seq_lens))
            for _ in range(num_layers)
        ]
        batched = model(mx.concatenate(steps, axis=0), cache=shim)[0][:, -1, :]
        mx.eval(batched)
    finally:
        kv.clear_context()

    mx.eval(stepwise)
    assert _max_abs_diff(batched, stepwise) < 1e-4
    assert [cache[0].offset for cache in batched_caches] == [7, 4]


def test_decode_offsets_come_from_the_cache() -> None:
    """One 6-token prefill must equal six single-token steps."""
    model = _build()
    mx.random.seed(3)
    prompt = mx.random.normal((1, 6, model.config.hidden_size))

    bulk_cache = model.make_cache()
    bulk_logits, _ = model(prompt, cache=bulk_cache)

    step_cache = model.make_cache()
    step_logits = [
        model(prompt[:, i : i + 1, :], cache=step_cache)[0] for i in range(6)
    ]
    stepwise = mx.concatenate(step_logits, axis=1)
    mx.eval(bulk_logits, stepwise)

    assert _max_abs_diff(bulk_logits, stepwise) < 1e-4
    assert bulk_cache[0].offset == step_cache[0].offset == 6


# --------------------------------------------------------------------------
# Code predictor
# --------------------------------------------------------------------------


def _reference_predict_codes(model, first_code: mx.array, talker_hidden: mx.array):
    """SGLang's CUDA form: two single-token priming calls, then one per group."""
    predictor = model.code_predictor
    num_groups = model.config.code_predictor_config.num_code_groups
    cache = predictor.make_cache()

    codes = [first_code]
    summed = model.get_input_embeddings()(first_code)

    predictor.model(predictor.project_input(talker_hidden[:, -1:, :]), cache=cache)
    layer0_embed = model.get_input_embeddings()(first_code)
    hidden = predictor.model(predictor.project_input(layer0_embed), cache=cache)

    for group_index in range(num_groups - 1):
        logits = predictor.lm_head[group_index](hidden)
        code = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        codes.append(code)
        embed = predictor.codec_embedding[group_index](code)
        summed = summed + embed
        if group_index < num_groups - 2:
            hidden = predictor.model(predictor.project_input(embed), cache=cache)

    return mx.concatenate(codes, axis=1), summed


@pytest.mark.parametrize("batch_size", [1, 2])
def test_predict_codes_matches_the_sequential_reference(batch_size: int) -> None:
    model = _build()
    mx.random.seed(4)
    hidden = mx.random.normal((batch_size, 3, model.config.hidden_size))
    first_code = mx.array([[5]] * batch_size, dtype=mx.int32)

    codes, summed = model.predict_codes(first_code, hidden)
    ref_codes, ref_summed = _reference_predict_codes(model, first_code, hidden)
    mx.eval(codes, summed, ref_codes, ref_summed)

    assert codes.shape == (batch_size, model.config.num_code_groups)
    assert summed.shape == (batch_size, 1, model.config.hidden_size)
    assert (codes == ref_codes).all().item()
    assert _max_abs_diff(summed, ref_summed) < 1e-4


def test_predict_codes_keeps_group_zero_and_echoes_the_first_code() -> None:
    model = _build()
    mx.random.seed(5)
    hidden = mx.random.normal((1, 1, model.config.hidden_size))
    first_code = mx.array([[7]], dtype=mx.int32)

    codes, summed = model.predict_codes(first_code, hidden)
    mx.eval(codes, summed)

    assert codes[0, 0].item() == 7
    # Group 0 embeds through the talker's table, groups 1.. through the
    # predictor's; the sum is the codec half of the next talker input.
    expected = model.get_input_embeddings()(codes[:, 0:1])
    for index in range(1, model.config.num_code_groups):
        expected = expected + model.code_predictor.codec_embedding[index - 1](
            codes[:, index : index + 1]
        )
    mx.eval(expected)
    assert _max_abs_diff(summed, expected) < 1e-4


def test_reused_predictor_cache_is_reset_between_frames() -> None:
    model = _build()
    mx.random.seed(6)
    hidden = mx.random.normal((1, 1, model.config.hidden_size))
    first_code = mx.array([[3]], dtype=mx.int32)

    fresh, _ = model.predict_codes(first_code, hidden)
    cache = model.code_predictor.make_cache()
    first, _ = model.predict_codes(first_code, hidden, cache=cache)
    second, _ = model.predict_codes(first_code, hidden, cache=cache)
    mx.eval(fresh, first, second)

    assert (fresh == first).all().item()
    assert (first == second).all().item()

    reset_code_cache(cache)
    assert all(entry.offset == 0 for entry in cache)
    assert all(entry.keys is None for entry in cache)


def test_code_sampler_receives_last_position_logits_and_group_index() -> None:
    model = _build()
    mx.random.seed(7)
    hidden = mx.random.normal((1, 1, model.config.hidden_size))
    seen: list[tuple[int, tuple[int, ...]]] = []

    def sampler(logits: mx.array, group_index: int) -> mx.array:
        seen.append((group_index, tuple(logits.shape)))
        return mx.zeros((logits.shape[0], 1), dtype=mx.int32)

    codes, _ = model.predict_codes(
        mx.array([[1]], dtype=mx.int32), hidden, sampler=sampler
    )
    mx.eval(codes)

    vocab = model.config.code_predictor_config.vocab_size
    assert seen == [(i, (1, vocab)) for i in range(model.config.num_code_groups - 1)]
    assert codes[0, 0].item() == 1
    assert [codes[0, i].item() for i in range(1, model.config.num_code_groups)] == [
        0
    ] * (model.config.num_code_groups - 1)


# --------------------------------------------------------------------------
# dtype and weights
# --------------------------------------------------------------------------


def test_bfloat16_is_preserved_through_the_talker_and_its_cache() -> None:
    model = _build(dtype=mx.bfloat16)
    prompt = mx.random.normal((1, 4, model.config.hidden_size)).astype(mx.bfloat16)

    cache = model.make_cache()
    logits, hidden = model(prompt, cache=cache)
    codes, summed = model.predict_codes(
        mx.array([[2]], dtype=mx.int32), hidden, cache=model.code_predictor.make_cache()
    )
    mx.eval(logits, hidden, codes, summed)

    assert hidden.dtype == mx.bfloat16
    assert logits.dtype == mx.bfloat16
    assert summed.dtype == mx.bfloat16
    assert cache[0].keys.dtype == mx.bfloat16
    assert cache[0].values.dtype == mx.bfloat16


def test_sanitize_scopes_weights_to_the_talker_subtree() -> None:
    sanitize = Qwen3TTSTalkerForConditionalGeneration.sanitize
    weights = {
        "talker.model.layers.0.self_attn.q_proj.weight": mx.zeros((2, 2)),
        "talker.codec_head.weight": mx.zeros((2, 2)),
        "speech_tokenizer.decoder.conv.weight": mx.zeros((2, 2)),
        "speaker_encoder.fc.weight": mx.zeros((2, 2)),
    }
    assert set(sanitize(weights)) == {
        "model.layers.0.self_attn.q_proj.weight",
        "codec_head.weight",
    }

    already_scoped = {"codec_head.weight": mx.zeros((2, 2))}
    assert set(sanitize(already_scoped)) == {"codec_head.weight"}


def test_loader_model_accepts_a_whole_checkpoint_config() -> None:
    config = ModelConfig(
        talker_config=_talker_config(),
        tts_model_type="custom_voice",
        speaker_encoder_config={"mel_dim": 128},
    )
    model = Qwen3TTSTalkerModel(config)

    assert model.tts_model_type == "custom_voice"
    assert len(model.model.layers) == config.talker_config.num_hidden_layers
    assert len(model.code_predictor.lm_head) == config.talker_config.num_code_groups - 1


def test_talker_weight_keys_match_the_checkpoint_layout() -> None:
    from mlx.utils import tree_flatten

    model = _build()
    keys = {key for key, _ in tree_flatten(model.parameters())}

    for expected in (
        "model.codec_embedding.weight",
        "model.text_embedding.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.norm.weight",
        "text_projection.linear_fc1.weight",
        "text_projection.linear_fc2.bias",
        "codec_head.weight",
        "code_predictor.model.layers.0.self_attn.q_proj.weight",
        "code_predictor.model.codec_embedding.0.weight",
        "code_predictor.lm_head.0.weight",
    ):
        assert expected in keys, expected
    # The predictor is wider than the talker here, so the CustomVoice-style
    # projection exists and carries checkpoint weights of its own.
    assert model.code_predictor.small_to_mtp_projection is not None
    assert "code_predictor.small_to_mtp_projection.bias" in keys


def test_matched_hidden_sizes_omit_the_predictor_projection() -> None:
    config = _talker_config()
    config.code_predictor_config.hidden_size = config.hidden_size
    model = Qwen3TTSTalkerForConditionalGeneration(config)

    assert model.code_predictor.small_to_mtp_projection is None
    hidden = mx.random.normal((1, 1, config.hidden_size))
    codes, _ = model.predict_codes(mx.array([[1]], dtype=mx.int32), hidden)
    mx.eval(codes)
    assert codes.shape == (1, config.num_code_groups)
