from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import WhisperConfig

import sglang_omni.models.whisper_asr.sglang_model as sglang_model
from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperDecoder,
    WhisperDecoderLayerNorm,
    WhisperEncoder,
)


def _config() -> WhisperConfig:
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
    )


def test_flashinfer_layer_norm_is_decoder_only() -> None:
    config = _config()
    encoder = WhisperEncoder(config)
    decoder = WhisperDecoder(config)
    decoder_layer = decoder.layers[0]

    assert type(encoder.layers[0].self_attn_layer_norm) is torch.nn.LayerNorm
    assert type(encoder.layers[0].final_layer_norm) is torch.nn.LayerNorm
    assert type(encoder.layer_norm) is torch.nn.LayerNorm
    assert isinstance(decoder_layer.self_attn_layer_norm, WhisperDecoderLayerNorm)
    assert isinstance(decoder_layer.encoder_attn_layer_norm, WhisperDecoderLayerNorm)
    assert isinstance(decoder_layer.final_layer_norm, WhisperDecoderLayerNorm)
    assert isinstance(decoder.layer_norm, WhisperDecoderLayerNorm)


def test_decoder_layer_norm_cpu_fallback_matches_pytorch() -> None:
    layer_norm = WhisperDecoderLayerNorm(8)
    hidden_states = torch.randn(3, 8)

    actual = layer_norm(hidden_states)
    expected = F.layer_norm(
        hidden_states,
        layer_norm.normalized_shape,
        layer_norm.weight,
        layer_norm.bias,
        layer_norm.eps,
    )

    torch.testing.assert_close(actual, expected)


def test_decoder_layer_norm_falls_back_when_flashinfer_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_norm = WhisperDecoderLayerNorm(8)
    hidden_states = torch.randn(3, 8)
    monkeypatch.setattr(sglang_model, "flashinfer_layer_norm", None)

    actual = layer_norm(hidden_states)
    expected = F.layer_norm(
        hidden_states,
        layer_norm.normalized_shape,
        layer_norm.weight,
        layer_norm.bias,
        layer_norm.eps,
    )

    torch.testing.assert_close(actual, expected)


def test_decoder_layer_norm_falls_back_for_cuda_norm_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer_norm = WhisperDecoderLayerNorm(8)
    hidden_states = torch.randn(3, 8)
    monkeypatch.setenv("FLASHINFER_USE_CUDA_NORM", "1")
    monkeypatch.setattr(
        sglang_model,
        "flashinfer_layer_norm",
        lambda *_args: pytest.fail("incompatible FlashInfer CUDA norm was called"),
    )

    actual = layer_norm(hidden_states)
    expected = F.layer_norm(
        hidden_states,
        layer_norm.normalized_shape,
        layer_norm.weight,
        layer_norm.bias,
        layer_norm.eps,
    )

    torch.testing.assert_close(actual, expected)
