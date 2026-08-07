# SPDX-License-Identifier: Apache-2.0
"""Hugging Face config registration for MOSS-TTS-Realtime."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3 import Qwen3Config


def _as_config(
    value: Any,
    config_class: type[PretrainedConfig],
) -> PretrainedConfig:
    if isinstance(value, config_class):
        return value
    if value is None:
        return config_class()
    if isinstance(value, dict):
        return config_class(**value)
    raise TypeError(
        f"Unsupported config type for {config_class.__name__}: {type(value)}"
    )


class MossTTSRealtimeLocalTransformerConfig(PretrainedConfig):
    model_type = "moss_tts_realtime_local_transformer"

    def __init__(
        self,
        head_dim: int = 128,
        use_cache: bool = True,
        hidden_size: int = 2048,
        rms_norm_eps: float = 1e-6,
        num_hidden_layers: int = 4,
        intermediate_size: int = 6144,
        num_attention_heads: int = 16,
        initializer_range: float = 0.02,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 33,
        num_key_value_heads: int = 8,
        hidden_act: str = "silu",
        rope_theta: int = 1_000_000,
        rope_type: str = "linear",
        pad_token_id: int = 1024,
        rope_parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.head_dim = head_dim
        self.use_cache = use_cache
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.rope_parameters = rope_parameters or {
            "rope_type": rope_type,
            "rope_theta": rope_theta,
            "factor": 1.0,
        }
        self.audio_pad_token = 1024
        self.audio_vocab_size = 1027
        self.rvq = 16


class MossTTSRealtimeConfig(PretrainedConfig):
    model_type = "moss_tts_realtime"

    def __init__(
        self,
        language_config: Qwen3Config | dict[str, Any] | None = None,
        local_config: (
            MossTTSRealtimeLocalTransformerConfig | dict[str, Any] | None
        ) = None,
        rvq: int = 16,
        audio_pad_token: int = 1024,
        audio_vocab_size: int = 1027,
        reference_audio_pad: int = 151654,
        text_pad: int = 151655,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rvq = rvq
        self.audio_pad_token = audio_pad_token
        self.audio_vocab_size = audio_vocab_size
        self.reference_audio_pad = reference_audio_pad
        self.text_pad = text_pad
        self.initializer_range = initializer_range
        self.language_config = _as_config(language_config, Qwen3Config)
        self.local_config = _as_config(
            local_config,
            MossTTSRealtimeLocalTransformerConfig,
        )


def register_moss_tts_realtime_hf_config() -> None:
    AutoConfig.register(
        MossTTSRealtimeLocalTransformerConfig.model_type,
        MossTTSRealtimeLocalTransformerConfig,
        exist_ok=True,
    )
    AutoConfig.register(
        MossTTSRealtimeConfig.model_type,
        MossTTSRealtimeConfig,
        exist_ok=True,
    )


__all__ = [
    "MossTTSRealtimeConfig",
    "MossTTSRealtimeLocalTransformerConfig",
    "register_moss_tts_realtime_hf_config",
]
