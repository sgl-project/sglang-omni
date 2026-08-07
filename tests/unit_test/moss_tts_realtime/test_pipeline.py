# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoConfig

from sglang_omni.models.moss_tts_realtime.hf_config import (
    MossTTSRealtimeConfig,
    register_moss_tts_realtime_hf_config,
)
from sglang_omni.models.moss_tts_realtime.local_transformer import (
    MossTTSRealtimeLocalTransformer,
    _rotate_half,
)
from sglang_omni.models.moss_tts_realtime.model_runner import MossTTSRealtimeModelRunner
from sglang_omni.models.moss_tts_realtime.payload_types import (
    AUDIO_BOS_TOKEN,
    AUDIO_PAD_TOKEN,
    N_CODEBOOKS,
    TEXT_PAD_TOKEN,
)
from sglang_omni.models.moss_tts_realtime.processor import (
    MossTTSRealtimePromptProcessor,
)
from sglang_omni.models.moss_tts_realtime.request_builders import _generation_kwargs
from sglang_omni.models.moss_tts_realtime.state_pool import (
    MossTTSRealtimeDecodeStatePool,
)


class _Tokenizer:
    audio_pad_id = 151654

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|audio_pad|>"
        return self.audio_pad_id

    def __call__(self, text: str) -> dict[str, list[int]]:
        ids = []
        audio_pad = "<|audio_pad|>"
        index = 0
        while index < len(text):
            if text.startswith(audio_pad, index):
                ids.append(self.audio_pad_id)
                index += len(audio_pad)
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return {"input_ids": ids}


def _local_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        rvq=4,
        audio_vocab_size=11,
        audio_pad_token=8,
    )


def _full_local_forward(
    module: MossTTSRealtimeLocalTransformer, inputs: torch.Tensor
) -> torch.Tensor:
    batch, seq, hidden = inputs.shape
    config = module.config
    head_dim = int(config.head_dim)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    repeats = num_heads // num_kv_heads
    inv_freq = 1.0 / (
        float(config.rope_theta)
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.arange(seq, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().view(1, seq, 1, head_dim)
    sin = emb.sin().view(1, seq, 1, head_dim)
    values = inputs
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    for layer in module.model.layers:
        residual = values
        normalized = layer.input_layernorm(values)
        query = layer.self_attn.q_norm(
            layer.self_attn.q_proj(normalized).view(batch, seq, num_heads, head_dim)
        )
        key = layer.self_attn.k_norm(
            layer.self_attn.k_proj(normalized).view(batch, seq, num_kv_heads, head_dim)
        )
        value = layer.self_attn.v_proj(normalized).view(
            batch, seq, num_kv_heads, head_dim
        )
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        query = query.transpose(1, 2)
        key = key.transpose(1, 2).repeat_interleave(repeats, dim=1)
        value = value.transpose(1, 2).repeat_interleave(repeats, dim=1)
        scores = torch.matmul(query, key.transpose(-1, -2)) / head_dim**0.5
        scores = scores.masked_fill(~causal, float("-inf"))
        attended = torch.matmul(torch.softmax(scores, dim=-1), value)
        attended = attended.transpose(1, 2).reshape(batch, seq, hidden)
        values = residual + layer.self_attn.o_proj(attended)
        values = values + layer.mlp(layer.post_attention_layernorm(values))
    return module.model.norm(values)


def test_prompt_uses_12_text_tokens_and_audio_bos():
    processor = MossTTSRealtimePromptProcessor(_Tokenizer())
    reference = torch.arange(3 * N_CODEBOOKS).reshape(3, N_CODEBOOKS) % 1024
    rows, text_ids, prefill_count = processor.build_generation_prompt(
        "abcdefghijklmnop",
        reference,
    )

    assert prefill_count == 12
    assert len(text_ids) == 16
    assert torch.equal(rows[-12:, 0], torch.tensor(text_ids[:12]))
    assert int(rows[-1, 1]) == AUDIO_BOS_TOKEN
    assert torch.all(rows[-12:-1, 1:] == AUDIO_PAD_TOKEN)
    reference_rows = rows[(rows[:, 1:] != AUDIO_PAD_TOKEN).any(dim=1)]
    assert torch.equal(reference_rows[:3, 1:], reference)


def test_local_incremental_sdpa_matches_full_causal_attention():
    torch.manual_seed(7)
    module = MossTTSRealtimeLocalTransformer(_local_config()).eval()
    inputs = torch.randn(2, 4, 16)
    expected = _full_local_forward(module, inputs)
    actual = torch.stack(
        [module.step(inputs[:, position], position) for position in range(4)],
        dim=1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-5)


def test_local_attention_backend_validation_and_cpu_fallback():
    with pytest.raises(ValueError, match="attention_backend"):
        MossTTSRealtimeLocalTransformer(
            _local_config(),
            attention_backend="unknown",
        )

    module = MossTTSRealtimeLocalTransformer(
        _local_config(),
        attention_backend="auto",
    )
    module.ensure_kv_cache(2, torch.device("cpu"), torch.float32)
    assert module.model._resolved_attention_backend == "sdpa"
    assert module.model._kv_cache[0][0].shape == (2, 2, 4, 4)


def test_fused_qkv_projection_matches_split_projection():
    torch.manual_seed(9)
    module = MossTTSRealtimeLocalTransformer(
        _local_config(),
        attention_backend="sdpa",
    )
    attention = module.model.layers[0].self_attn
    hidden_states = torch.randn(3, 16)
    expected = attention.project(hidden_states)
    attention.refresh_fused_qkv()
    actual = attention.project(hidden_states)

    assert not attention._fused_qkv_weight.requires_grad
    for fused, split in zip(actual, expected, strict=True):
        torch.testing.assert_close(fused, split)


def test_local_checkpoint_names_match_upstream_layout():
    keys = set(MossTTSRealtimeLocalTransformer(_local_config()).state_dict())
    assert "model.embed_tokens.0.weight" in keys
    assert "model.layers.0.self_attn.q_proj.weight" in keys
    assert "model.layers.0.self_attn.q_norm.weight" in keys
    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.norm.weight" in keys
    assert "local_lm_heads.0.weight" in keys


def test_hf_config_registration_preserves_nested_configs():
    register_moss_tts_realtime_hf_config()
    config = AutoConfig.for_model(
        "moss_tts_realtime",
        language_config={"hidden_size": 2048},
        local_config={"num_hidden_layers": 4},
    )
    assert isinstance(config, MossTTSRealtimeConfig)
    assert config.language_config.hidden_size == 2048
    assert config.local_config.num_hidden_layers == 4


def _pool_model() -> SimpleNamespace:
    return SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.zeros(1, 8, dtype=torch.bfloat16)
        ),
        config=SimpleNamespace(
            n_vq=N_CODEBOOKS,
            audio_vocab_size=1027,
            repetition_window=50,
        ),
    )


def test_repetition_history_expires_after_50_frames():
    pool = MossTTSRealtimeDecodeStatePool(_pool_model())
    row = pool.acquire_row("request")
    row_t = torch.tensor([row])
    for token in range(51):
        codes = torch.full((1, N_CODEBOOKS + 1), token, dtype=torch.long)
        pool.update_audio_history(row_t, codes)
    assert not bool(pool.audio_token_presence[row, 0, 0])
    assert bool(pool.audio_token_presence[row, 0, 1])
    assert bool(pool.audio_token_presence[row, 0, 50])


def test_runner_advances_text_and_uses_audio_eos_as_stop():
    runner = object.__new__(MossTTSRealtimeModelRunner)
    runner.model = SimpleNamespace(
        config=SimpleNamespace(audio_end_token_id=151645, n_vq=N_CODEBOOKS)
    )
    data = SimpleNamespace(
        prefill_text_tokens=12,
        generation_steps=0,
        text_token_ids=list(range(20)),
    )
    codes = torch.zeros(1, N_CODEBOOKS, dtype=torch.long)
    rows, next_text, end_id = runner._compose_frame_rows(
        codes=codes,
        stop_choice=torch.tensor([0]),
        requests=[SimpleNamespace(data=data)],
        device=torch.device("cpu"),
    )
    assert int(next_text[0]) == 12
    assert int(rows[0, 0]) == 12
    assert end_id == 151645

    data.generation_steps = 20
    _, next_text, _ = runner._compose_frame_rows(
        codes=codes,
        stop_choice=torch.tensor([0]),
        requests=[SimpleNamespace(data=data)],
        device=torch.device("cpu"),
    )
    assert int(next_text[0]) == TEXT_PAD_TOKEN

    _, next_text, _ = runner._compose_frame_rows(
        codes=codes,
        stop_choice=torch.tensor([1]),
        requests=[SimpleNamespace(data=data)],
        device=torch.device("cpu"),
    )
    assert int(next_text[0]) == 151645


def test_generation_defaults_match_model_card():
    kwargs = _generation_kwargs({}, {})
    assert kwargs["audio_temperature"] == 0.8
    assert kwargs["audio_top_p"] == 0.6
    assert kwargs["audio_top_k"] == 30
    assert kwargs["audio_repetition_penalty"] == 1.1
    assert kwargs["repetition_window"] == 50


def test_reference_code_orientation_is_normalized():
    processor = MossTTSRealtimePromptProcessor(_Tokenizer())
    values = np.arange(N_CODEBOOKS * 3).reshape(N_CODEBOOKS, 3)
    normalized = processor._normalize_reference_codes(values)
    assert normalized.shape == (3, N_CODEBOOKS)
    np.testing.assert_array_equal(normalized, values.T)
