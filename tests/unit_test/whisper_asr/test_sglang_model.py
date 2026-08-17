# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.srt.runtime_context import get_context, get_parallel
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperDecoderLayerNorm,
    WhisperForConditionalGeneration,
)


def _small_whisper_config() -> WhisperConfig:
    return WhisperConfig(
        vocab_size=32,
        num_mel_bins=4,
        d_model=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        max_source_positions=10,
        max_target_positions=10,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
    )


def test_whisper_fuses_attention_projections_and_loads_bias_shards() -> None:
    with (
        get_context().override_server_args(enable_dp_lm_head=False),
        get_parallel().override(
            tp_rank=0,
            tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
        ),
    ):
        model = WhisperForConditionalGeneration(_small_whisper_config())

    decoder_layer = model.model.decoder.layers[0]
    encoder_layer = model.model.encoder.layers[0]
    assert isinstance(encoder_layer.self_attn.q_proj, torch.nn.Linear)
    assert isinstance(decoder_layer.self_attn.qkv_proj, torch.nn.Linear)
    assert isinstance(decoder_layer.self_attn_layer_norm, WhisperDecoderLayerNorm)
    assert isinstance(decoder_layer.encoder_attn_layer_norm, WhisperDecoderLayerNorm)
    assert isinstance(decoder_layer.final_layer_norm, WhisperDecoderLayerNorm)
    assert isinstance(model.model.decoder.layer_norm, WhisperDecoderLayerNorm)
    assert isinstance(decoder_layer.encoder_attn.q_proj, torch.nn.Linear)
    assert isinstance(decoder_layer.encoder_attn.k_proj, torch.nn.Linear)

    self_prefix = "model.decoder.layers.0.self_attn"
    model.load_weights(
        [
            (f"{self_prefix}.q_proj.bias", torch.full((16,), 1.0)),
            (f"{self_prefix}.v_proj.bias", torch.full((16,), 3.0)),
        ]
    )

    params = dict(model.named_parameters())
    self_bias = params[f"{self_prefix}.qkv_proj.bias"].detach()
    assert [part.mean().item() for part in self_bias.chunk(3)] == [1.0, 0.0, 3.0]


def test_whisper_configures_static_suppress_tokens() -> None:
    with (
        get_context().override_server_args(enable_dp_lm_head=False),
        get_parallel().override(
            tp_rank=0,
            tp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
        ),
    ):
        model = WhisperForConditionalGeneration(_small_whisper_config())

    model.set_suppress_tokens([-1, 11, 7, 11, 32])

    assert model._suppress_tokens.tolist() == [7, 11]
    assert model._suppress_tokens.device == model.proj_out.weight.device
