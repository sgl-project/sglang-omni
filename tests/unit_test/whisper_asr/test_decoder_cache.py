# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr import sglang_model
from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperSGLangCrossAttention,
    WhisperSGLangSelfAttention,
)


def _tiny_whisper_config() -> WhisperConfig:
    return WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        vocab_size=32,
        max_source_positions=8,
        max_target_positions=8,
        num_mel_bins=4,
    )


def test_whisper_model_exposes_decoder_body() -> None:
    model = WhisperModel(_tiny_whisper_config())

    assert model.layers is model.decoder.layers
    assert "input_embeds" in inspect.signature(model.forward).parameters


def test_whisper_cross_attention_caches_fused_encoder_kv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = WhisperSGLangCrossAttention(_tiny_whisper_config(), layer_id=1)
    encoder_states = torch.randn(3, attention.embed_dim)
    cache_loc = torch.arange(3, dtype=torch.int64)
    writes: list[tuple[object, object, torch.Tensor, torch.Tensor]] = []
    token_to_kv_pool = SimpleNamespace(
        set_kv_buffer=lambda layer, loc, key, value: writes.append(
            (layer, loc, key, value)
        )
    )
    monkeypatch.setattr(
        sglang_model,
        "get_attn_backend",
        lambda: SimpleNamespace(token_to_kv_pool=token_to_kv_pool),
    )

    attention.cache_encoder_states(encoder_states, cache_loc)

    layer, write_loc, key, value = writes[0]
    expected_key, expected_value = attention.kv_proj(encoder_states).chunk(2, dim=-1)
    assert layer is attention.attn
    assert write_loc.loc is cache_loc
    torch.testing.assert_close(
        key, expected_key.view(-1, attention.num_heads, attention.head_dim)
    )
    torch.testing.assert_close(
        value, expected_value.view(-1, attention.num_heads, attention.head_dim)
    )


@pytest.mark.parametrize(
    "attention_cls",
    [WhisperSGLangSelfAttention, WhisperSGLangCrossAttention],
)
def test_whisper_attention_flattens_head_shaped_backend_output(
    attention_cls: type[torch.nn.Module],
) -> None:
    class _HeadShapedAttention(torch.nn.Module):
        def forward(self, query, key, value, forward_batch):
            del key, value, forward_batch
            return query

    attention = attention_cls(_tiny_whisper_config(), layer_id=0)
    attention.attn = _HeadShapedAttention()
    attention.out_proj = torch.nn.Identity()
    hidden_states = torch.randn(3, attention.embed_dim)

    output = attention(hidden_states, SimpleNamespace())

    assert output.shape == hidden_states.shape


@pytest.mark.parametrize("use_precomputed_states", [True, False])
def test_whisper_forward_caches_encoder_kv_before_decoder(
    monkeypatch: pytest.MonkeyPatch,
    use_precomputed_states: bool,
) -> None:
    class _LogitsProcessor(torch.nn.Module):
        def forward(self, *args):
            calls.append("logits")
            return args[1]

    calls: list[str] = []
    monkeypatch.setattr(
        sglang_model,
        "LogitsProcessor",
        lambda _config: _LogitsProcessor(),
    )
    model = WhisperForConditionalGeneration(_tiny_whisper_config())
    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    encoder_states = torch.randn(3, model.config.d_model)
    forward_batch = SimpleNamespace(
        encoder_out_cache_loc=torch.arange(3, dtype=torch.int64)
    )

    monkeypatch.setattr(
        model,
        "_batch_precomputed_encoder_states",
        lambda batch: encoder_states if use_precomputed_states else None,
    )
    monkeypatch.setattr(
        model,
        "_batch_audio_inputs",
        lambda batch: (torch.randn(1, 4, 4), [3]),
    )
    monkeypatch.setattr(
        model,
        "_run_encoder",
        lambda features: encoder_states.unsqueeze(0),
    )
    monkeypatch.setattr(
        model.model,
        "cache_encoder_states",
        lambda states, loc: calls.append("cache"),
    )
    monkeypatch.setattr(
        model.model.decoder,
        "embed_input_ids",
        lambda ids, pos: torch.randn(ids.shape[0], model.config.d_model),
    )
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda *args, **kwargs: calls.append("decoder")
        or torch.randn(input_ids.shape[0], model.config.d_model),
    )

    model(input_ids, positions, forward_batch)

    assert calls == ["cache", "decoder", "logits"]
