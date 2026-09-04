from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import WhisperConfig

from sglang_omni.models.whisper_asr import sglang_model
from sglang_omni.models.whisper_asr.sglang_model import WhisperForConditionalGeneration


def _make_model(monkeypatch) -> tuple[WhisperConfig, WhisperForConditionalGeneration]:
    monkeypatch.setattr(sglang_model, "LogitsProcessor", lambda _config: None)
    config = WhisperConfig(
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
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=3,
    )
    return config, WhisperForConditionalGeneration(config)


def test_fused_projection_checkpoint_shards(monkeypatch) -> None:
    config, model = _make_model(monkeypatch)
    weights = []
    expected = {}
    shard_names = tuple(
        f"model.{stack}.layers.0.{attention}.{projection}.{parameter}"
        for stack, attention, projections in (
            ("encoder", "self_attn", ("q_proj", "k_proj", "v_proj")),
            ("decoder", "self_attn", ("q_proj", "k_proj", "v_proj")),
            ("decoder", "encoder_attn", ("k_proj", "v_proj")),
        )
        for projection in projections
        for parameter in ("weight", "bias")
        if projection != "k_proj" or parameter != "bias"
    )
    for index, name in enumerate(shard_names, start=1):
        shape = (
            (config.d_model,)
            if name.endswith("bias")
            else (config.d_model, config.d_model)
        )
        tensor = torch.full(shape, index, dtype=torch.float32)
        weights.append((name, tensor))
        expected[name] = tensor

    model.load_weights(weights)

    fused_projections = (
        (model.model.encoder.layers[0].self_attn.qkv_proj, "encoder", "self_attn"),
        (model.model.decoder.layers[0].self_attn.qkv_proj, "decoder", "self_attn"),
        (model.model.decoder.layers[0].encoder_attn.kv_proj, "decoder", "encoder_attn"),
    )
    hidden_states = torch.randn(3, config.d_model)
    for fused, stack, attention in fused_projections:
        projections = (
            ("q_proj", "k_proj", "v_proj")
            if attention == "self_attn"
            else ("k_proj", "v_proj")
        )
        for shard, projection in enumerate(projections):
            for parameter in ("weight", "bias"):
                actual = getattr(fused, parameter).narrow(
                    0, shard * config.d_model, config.d_model
                )
                name = f"model.{stack}.layers.0.{attention}.{projection}.{parameter}"
                if projection == "k_proj" and parameter == "bias":
                    assert torch.count_nonzero(actual) == 0
                else:
                    assert torch.equal(actual, expected[name])

            prefix = f"model.{stack}.layers.0.{attention}.{projection}"
            bias = None if projection == "k_proj" else expected[f"{prefix}.bias"]
            reference = F.linear(hidden_states, expected[f"{prefix}.weight"], bias)
            actual = fused(hidden_states).chunk(len(projections), dim=-1)[shard]
            torch.testing.assert_close(actual, reference)


def test_fused_projection_rejects_wrong_checkpoint_shape(monkeypatch) -> None:
    config, model = _make_model(monkeypatch)
    broadcastable_weight = torch.ones(1, config.d_model)

    with pytest.raises(AssertionError, match="Attempted to load weight"):
        model.load_weights(
            [("model.encoder.layers.0.self_attn.q_proj.weight", broadcastable_weight)]
        )
