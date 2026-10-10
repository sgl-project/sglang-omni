# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni_mlx.qwen3_asr.audio import AudioLayout, swift_token_count  # noqa: E402
from sglang_omni_mlx.qwen3_asr.model import (  # noqa: E402
    AudioEncoderConfig,
    Qwen3ASR,
    conv_output_frames,
    load_qwen3_asr,
)
from sglang_omni_mlx.text_decoder import (  # noqa: E402
    KV_CACHE_STEP_TOKENS,
    TextDecoderConfig,
)

AUDIO_CONFIG = {
    "num_mel_bins": 16,
    "encoder_layers": 2,
    "encoder_attention_heads": 2,
    "encoder_ffn_dim": 32,
    "d_model": 16,
    "max_source_positions": 1500,
    "n_window": 50,
    "n_window_infer": 200,
    "downsample_hidden_size": 4,
    "output_dim": 32,
}
TEXT_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
}


@pytest.fixture(autouse=True)
def exact_mlx_device() -> Iterator[None]:
    # The CPU device keeps the comparisons below free of Metal accumulation order.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def tiny_model() -> Qwen3ASR:
    mx.random.seed(0)
    model = Qwen3ASR(
        AudioEncoderConfig(**AUDIO_CONFIG), TextDecoderConfig(**TEXT_CONFIG)
    )
    mx.eval(model.parameters())
    return model


def block_mask_encoder(model: Qwen3ASR, mel: mx.array, layout: AudioLayout) -> mx.array:
    """The encoder written as one sequence under a block-diagonal mask."""
    tower = model.audio_tower
    chunk_frame_count = AUDIO_CONFIG["n_window"] * 2
    frame_count = mel.shape[-1]
    starts = list(range(0, frame_count, chunk_frame_count))
    lengths = [min(chunk_frame_count, frame_count - start) for start in starts]
    longest = max(lengths)
    chunks = mx.stack(
        [
            mx.pad(mel[:, s : s + n], [(0, 0), (0, longest - n)])
            for s, n in zip(starts, lengths)
        ]
    )
    x = nn.gelu(
        tower.conv2d3(nn.gelu(tower.conv2d2(nn.gelu(tower.conv2d1(chunks[..., None])))))
    )
    count, bins, frames, channels = x.shape
    x = tower.conv_out(x.transpose(0, 2, 3, 1).reshape(count, frames, channels * bins))
    half = AUDIO_CONFIG["d_model"] // 2
    scaled = (
        mx.arange(frames)[:, None]
        * mx.exp(-np.log(10000.0) / (half - 1) * mx.arange(half))[None]
    )
    x = x + mx.concatenate([mx.sin(scaled), mx.cos(scaled)], axis=1)[None]
    credited = [
        (
            conv_output_frames(n)
            if layout is AudioLayout.REFERENCE
            else swift_token_count(n)
        )
        for n in lengths
    ]
    hidden = mx.concatenate([x[i, : min(n, frames)] for i, n in enumerate(credited)])
    per_window = AUDIO_CONFIG["n_window_infer"] // chunk_frame_count
    mask = np.full((hidden.shape[0], hidden.shape[0]), -np.inf, dtype=np.float32)
    start = 0
    for first in range(0, count, per_window):
        end = min(start + sum(credited[first : first + per_window]), hidden.shape[0])
        mask[start:end, start:end] = 0.0
        start = end
    mask[start:, start:] = 0.0
    hidden = hidden[None]
    for layer in tower.layers:
        attention = layer.self_attn
        normed = layer.self_attn_layer_norm(hidden)
        q, k, v = (
            p(normed)
            .reshape(1, -1, attention.head_count, attention.head_dim)
            .transpose(0, 2, 1, 3)
            for p in (attention.q_proj, attention.k_proj, attention.v_proj)
        )
        attended = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=attention.head_dim**-0.5, mask=mx.array(mask)[None, None]
        )
        hidden = hidden + attention.out_proj(
            attended.transpose(0, 2, 1, 3).reshape(hidden.shape)
        )
        hidden = hidden + layer.fc2(nn.gelu(layer.fc1(layer.final_layer_norm(hidden))))
    return tower.proj2(nn.gelu(tower.proj1(tower.ln_post(hidden[0]))))


@pytest.mark.parametrize("layout", [AudioLayout.REFERENCE, AudioLayout.VOXT_SWIFT])
@pytest.mark.parametrize("frame_count", [37, 100, 335, 679])
def test_windowed_encoder_matches_one_masked_sequence(
    layout: AudioLayout, frame_count: int
) -> None:
    model = tiny_model()
    mel = mx.random.normal((AUDIO_CONFIG["num_mel_bins"], frame_count))
    windowed = model.audio_tower(mel, layout)
    expected = block_mask_encoder(model, mel, layout)
    assert windowed.shape == expected.shape
    np.testing.assert_allclose(np.array(windowed), np.array(expected), atol=1e-5)


def test_a_lone_partial_chunk_keeps_only_the_rows_its_convolutions_produce() -> None:
    model = tiny_model()
    mel = mx.random.normal((AUDIO_CONFIG["num_mel_bins"], 37))
    # Chunks pad only to the longest chunk, so Swift's 9 credited rows are cut to
    # the 5 the convolutions produce, as in the Swift encoder.
    assert model.audio_tower(mel, AudioLayout.VOXT_SWIFT).shape[0] == 5


def test_swift_layout_keeps_extra_rows_of_a_partial_chunk() -> None:
    model = tiny_model()
    mel = mx.random.normal((AUDIO_CONFIG["num_mel_bins"], 335))
    # Chunks of 100, 100, 100 and 35 frames: 13 rows each, then 5 (reference) or
    # 9 (Swift: 35 / 100 * 13 credited on top of the 5).
    assert model.audio_tower(mel, AudioLayout.REFERENCE).shape[0] == 3 * 13 + 5
    assert model.audio_tower(mel, AudioLayout.VOXT_SWIFT).shape[0] == 3 * 13 + 9


def test_cached_decoding_matches_a_full_forward_across_cache_growth() -> None:
    model = tiny_model()
    prompt_length = KV_CACHE_STEP_TOKENS - 3
    tokens = mx.random.randint(0, TEXT_CONFIG["vocab_size"], (1, prompt_length + 8))
    caches = model.new_caches()
    model.model(model.model.embed_tokens(tokens[:, :prompt_length]), caches)
    for position in range(prompt_length, tokens.shape[1]):
        stepped = model.model(
            model.model.embed_tokens(tokens[:, position : position + 1]), caches
        )
        full = model.model(
            model.model.embed_tokens(tokens[:, : position + 1]), model.new_caches()
        )
        np.testing.assert_allclose(np.array(stepped), np.array(full), atol=1e-4)
    assert caches[0].offset == tokens.shape[1]


def test_load_rebuilds_a_quantized_checkpoint(tmp_path: Path) -> None:
    model = tiny_model()
    nn.quantize(model.model, group_size=32, bits=4)
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "thinker_config": {
                    "audio_config": AUDIO_CONFIG,
                    "text_config": TEXT_CONFIG,
                },
                "quantization": {"group_size": 32, "bits": 4},
            }
        )
    )
    loaded = load_qwen3_asr(tmp_path)
    assert isinstance(loaded.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(loaded.audio_tower.layers[0].fc1, nn.Linear)
    tokens = mx.array([[1, 2, 3, 4]])
    np.testing.assert_array_equal(
        np.array(loaded.model(loaded.model.embed_tokens(tokens), loaded.new_caches())),
        np.array(model.model(model.model.embed_tokens(tokens), model.new_caches())),
    )
