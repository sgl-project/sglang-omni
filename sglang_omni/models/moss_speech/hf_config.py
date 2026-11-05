# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration with separate shared and modality layer counts."""

from __future__ import annotations

from typing import ClassVar

from transformers import AutoConfig, PretrainedConfig

MAX_CONTEXT_TOKENS = 40960
KV_RESERVE_TOKENS = 2


class MossSpeechConfig(PretrainedConfig):
    model_type: ClassVar[str] = "moss_speech"
    keys_to_ignore_at_inference: ClassVar[list[str]] = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 151680,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = MAX_CONTEXT_TOKENS,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, str | float | int | list[float]] | None = None,
        attention_bias: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        max_window_layers: int | None = None,
        layer_types: list[str] | None = None,
        attention_dropout: float = 0.0,
        audio_vocab_size: int | None = None,
        modality_pad_token_id: int = 0,
        sosp_token_id: int | None = None,
        eosp_token_id: int | None = None,
        num_shared_layers: int = 32,
        num_modality_layers: int = 4,
        moss_decode_buffer_bs: int = 128,
        **kwargs: object,
    ) -> None:
        assert num_shared_layers + num_modality_layers == num_hidden_layers
        self.vocab_size: int = vocab_size
        self.hidden_size: int = hidden_size
        self.intermediate_size: int = intermediate_size
        self.num_hidden_layers: int = num_hidden_layers
        self.num_attention_heads: int = num_attention_heads
        self.num_key_value_heads: int = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim: int = head_dim
        self.hidden_act: str = hidden_act
        self.max_position_embeddings: int = max_position_embeddings
        self.initializer_range: float = initializer_range
        self.rms_norm_eps: float = rms_norm_eps
        self.use_cache: bool = use_cache
        self.rope_theta: float = rope_theta
        self.rope_scaling: dict[str, str | float | int | list[float]] | None = (
            rope_scaling
        )
        self.rope_parameters: dict[str, str | float | int | list[float]] = dict(
            rope_scaling or {}
        )
        self.rope_parameters.setdefault("rope_theta", rope_theta)
        self.attention_bias: bool = attention_bias
        self.use_sliding_window: bool = use_sliding_window
        self.sliding_window: int | None = sliding_window if use_sliding_window else None
        self.max_window_layers: int | None = max_window_layers
        self.attention_dropout: float = attention_dropout
        self.audio_vocab_size: int | None = audio_vocab_size
        self.modality_pad_token_id: int = modality_pad_token_id
        self.sosp_token_id: int | None = sosp_token_id
        self.eosp_token_id: int | None = eosp_token_id
        self.num_shared_layers: int = num_shared_layers
        self.num_modality_layers: int = num_modality_layers
        self.moss_decode_buffer_bs: int = moss_decode_buffer_bs
        if layer_types is None:
            self.layer_types: list[str] = [
                (
                    "sliding_attention"
                    if self.sliding_window is not None
                    and max_window_layers is not None
                    and i >= max_window_layers
                    else "full_attention"
                )
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types
        if len(self.layer_types) != num_hidden_layers or set(self.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("layer_types must describe every checkpoint layer")
        else:
            pass
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.standardize_rope_params()
        self.validate_rope()


def ensure_moss_speech_config_registered() -> None:
    """Register explicitly during engine construction, without import effects."""
    AutoConfig.register("moss_speech", MossSpeechConfig, exist_ok=True)


__all__ = ["MossSpeechConfig", "ensure_moss_speech_config_registered"]
