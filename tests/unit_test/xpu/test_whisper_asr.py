# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for Whisper ASR (no accelerator required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import WhisperConfig

from sglang_omni import platforms
from sglang_omni.models.whisper_asr import sglang_model
from sglang_omni.models.whisper_asr.engine_builder import WhisperASREngineBuilder
from sglang_omni.platforms.xpu import XPUOmniPlatform


def _builder() -> WhisperASREngineBuilder:
    return WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
    )


def _server_args(attention_backend: str) -> SimpleNamespace:
    return SimpleNamespace(
        quantization=None,
        moe_runner_backend="auto",
        attention_backend=attention_backend,
    )


def test_whisper_selects_torch_native_attention(monkeypatch) -> None:
    monkeypatch.setattr(platforms, "current_platform", XPUOmniPlatform())

    defaults = _builder().generation_defaults(dtype="float16")

    assert defaults["attention_backend"] == "torch_native"


def test_whisper_rejects_unsafe_attention_backend() -> None:
    with pytest.raises(ValueError, match="requires attention_backend='torch_native'"):
        XPUOmniPlatform().apply_model_worker_backend_policy(
            _server_args("triton"),
            SimpleNamespace(quantization=None),
            "WhisperForConditionalGeneration",
        )


def test_whisper_accepts_torch_native_attention_backend() -> None:
    effective_quantization = XPUOmniPlatform().apply_model_worker_backend_policy(
        _server_args("torch_native"),
        SimpleNamespace(quantization=None),
        "WhisperForConditionalGeneration",
    )

    assert effective_quantization is None


@pytest.mark.parametrize("output_rank", [2, 3])
@pytest.mark.parametrize("attention_kind", ["self", "cross"])
def test_whisper_decoder_attention_flattens_backend_output(
    monkeypatch, output_rank: int, attention_kind: str
) -> None:
    class _StubRadixAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

        def forward(self, query, key, value, forward_batch):
            del key, value, forward_batch
            if output_rank == 2:
                return query.reshape(query.shape[0], -1)
            return query

    monkeypatch.setattr(sglang_model, "RadixAttention", _StubRadixAttention)
    config = WhisperConfig(d_model=8, decoder_attention_heads=2)
    hidden_states = torch.randn(3, config.d_model)

    if attention_kind == "self":
        attention = sglang_model.WhisperSGLangSelfAttention(config, layer_id=0)
        output = attention(hidden_states, forward_batch=object())
    else:
        attention = sglang_model.WhisperSGLangCrossAttention(config, layer_id=0)
        output = attention(
            hidden_states,
            torch.randn(4, config.d_model),
            forward_batch=object(),
        )

    assert output.shape == hidden_states.shape
