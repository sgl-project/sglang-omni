# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from sglang_omni.models.qwen3_tts import sglang_model


def _decoder_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        rope_theta=10000.0,
        rope_scaling=None,
        max_position_embeddings=128,
        head_dim=4,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )


def test_qwen3_tts_fp8_is_routed_only_to_talker_mlp(monkeypatch) -> None:
    quant_config = object()
    seen: dict[str, object] = {}

    class RecordingAttention(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            seen["attention"] = kwargs.get("quant_config")

    class RecordingMLP(nn.Module):
        def __init__(
            self,
            hidden_size,
            intermediate_size,
            quant_config=None,
            prefix="",
        ) -> None:
            super().__init__()
            del hidden_size, intermediate_size, prefix
            seen["mlp"] = quant_config

    monkeypatch.setattr(
        sglang_model, "Qwen3OmniMoeThinkerTextAttention", RecordingAttention
    )
    monkeypatch.setattr(sglang_model, "Qwen3OmniMoeTalkerDenseMLP", RecordingMLP)

    sglang_model.Qwen3TTSTalkerDecoderLayer(
        _decoder_config(),
        layer_id=0,
        quant_config=quant_config,
    )

    assert seen == {"attention": None, "mlp": quant_config}


def test_qwen3_tts_fp8_reaches_main_and_predictor_layers(monkeypatch) -> None:
    quant_config = object()
    seen: list[object] = []

    class RecordingDecoderLayer(nn.Module):
        def __init__(self, config, layer_id, quant_config=None, prefix="") -> None:
            super().__init__()
            del config, layer_id, prefix
            seen.append(quant_config)
            self.self_attn = SimpleNamespace(num_kv_heads=1, head_dim=4)

    class FakeReplicatedLinear(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    predictor_config = SimpleNamespace(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
    )
    config = SimpleNamespace(
        vocab_size=32,
        text_vocab_size=32,
        text_hidden_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        rms_norm_eps=1e-6,
        num_code_groups=2,
        code_predictor_config=predictor_config,
    )
    monkeypatch.setattr(
        sglang_model, "Qwen3TTSTalkerDecoderLayer", RecordingDecoderLayer
    )
    monkeypatch.setattr(sglang_model, "ReplicatedLinear", FakeReplicatedLinear)
    monkeypatch.setattr(
        sglang_model,
        "RMSNorm",
        lambda hidden_size, eps: nn.Identity(),
    )
    monkeypatch.setattr(
        sglang_model,
        "get_global_server_args",
        lambda: SimpleNamespace(max_running_requests=1),
    )

    sglang_model.Qwen3TTSTalkerTextModel(config, quant_config=quant_config)
    sglang_model.Qwen3TTSCodePredictor(config, quant_config=quant_config)

    assert seen == [quant_config, quant_config]
