# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from sglang_omni_mlx.voxtral_tts.audio_tokenizer import (  # noqa: E402
    ATTENTION_QUERY_CHUNK_SIZE,
    AudioAttention,
)
from sglang_omni_mlx.voxtral_tts.config import AudioTokenizerConfig  # noqa: E402


@pytest.fixture(autouse=True)
def use_cpu_device() -> Iterator[None]:
    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous_device)


@pytest.mark.parametrize("causal", [True, False])
def test_chunked_audio_attention_matches_dense_attention(causal: bool) -> None:
    config = AudioTokenizerConfig(
        sampling_rate=24000,
        patch_size=2,
        patch_projection_kernel_size=3,
        semantic_codebook_size=8,
        semantic_dimension=4,
        acoustic_codebook_size=5,
        acoustic_dimension=3,
        causal=causal,
        sliding_window_size=16,
        half_attention_window_on_downsample=True,
        encoder_convolution_strides=(2, 2, 2, 1),
        dimension=8,
        hidden_dimension=16,
        head_dimension=4,
        head_count=2,
        kv_head_count=1,
        qk_norm_epsilon=1e-6,
        qk_norm=True,
        use_biases=True,
        norm_epsilon=1e-5,
        layer_scale=False,
        layer_scale_initial=None,
        transformer_lengths=(1,),
        convolution_kernels=(3,),
        convolution_strides=(1,),
    )
    attention = AudioAttention(config, window_size=16)
    sequence_length = ATTENTION_QUERY_CHUNK_SIZE + 33
    mx.random.seed(0)
    hidden_states = mx.random.normal((1, sequence_length, config.dimension))
    actual = attention(hidden_states)

    queries = attention.wq(hidden_states)
    keys = attention.wk(hidden_states)
    values = attention.wv(hidden_states)
    queries = (
        attention.q_norm(queries)
        .reshape(1, sequence_length, config.head_count, config.head_dimension)
        .transpose(0, 2, 1, 3)
    )
    keys = (
        attention.k_norm(keys)
        .reshape(1, sequence_length, config.kv_head_count, config.head_dimension)
        .transpose(0, 2, 1, 3)
    )
    values = values.reshape(
        1, sequence_length, config.kv_head_count, config.head_dimension
    ).transpose(0, 2, 1, 3)
    keys = mx.repeat(keys, config.head_count // config.kv_head_count, axis=1)
    values = mx.repeat(values, config.head_count // config.kv_head_count, axis=1)
    sequence_positions = mx.arange(sequence_length)
    relative_positions = sequence_positions[None, :] - sequence_positions[:, None]
    alibi_slopes = mx.array(attention.get_alibi_slopes(config.head_count))
    right_window_tokens = 0 if causal else attention.window_size
    attention_bias = alibi_slopes.astype(hidden_states.dtype)[:, None, None] * (
        relative_positions[None, :, :].astype(hidden_states.dtype)
    )
    if causal:
        attention_bias = mx.where(
            relative_positions[None, :, :] > 0,
            -float("inf"),
            attention_bias,
        )
    else:
        pass
    outside_window = (relative_positions < -attention.window_size) | (
        relative_positions > right_window_tokens
    )
    attention_bias = mx.where(outside_window[None, :, :], -float("inf"), attention_bias)
    dense_attended = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=config.head_dimension**-0.5,
        mask=attention_bias[None, :, :, :],
    )
    expected = attention.wo(
        dense_attended.transpose(0, 2, 1, 3).reshape(
            1, sequence_length, config.head_count * config.head_dimension
        )
    )

    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=1e-5)
