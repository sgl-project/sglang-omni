# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config adapter for Chatterbox-Turbo checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import GPT2Config

CHATTERBOX_T3_MODEL_ARCH_OVERRIDE = "ChatterboxT3SGLangModel"

_chatterbox_hf_config_registered = False


class ChatterboxT3Config(GPT2Config):
    """GPT-2 backbone config plus T3 dual-vocab and conditioning fields.

    The T3 backbone is a 24-layer GPT-2 (hidden 1024, MLP 4096, 16 heads,
    learned positional embeddings, LayerNorm with bias). Text and speech use
    separate embedding/head tables, and a speaker embedding (256) conditions
    the sequence via cond_enc.spkr_enc.
    """

    model_type = "chatterbox_t3"

    def __init__(
        self,
        text_tokens_dict_size: int = 50276,
        speech_tokens_dict_size: int = 6563,
        speaker_embed_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            vocab_size=50276,
            n_positions=8196,
            n_ctx=8196,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function="gelu_new",
            layer_norm_epsilon=1e-5,
            **kwargs,
        )
        self.text_tokens_dict_size = text_tokens_dict_size
        self.speech_tokens_dict_size = speech_tokens_dict_size
        self.speaker_embed_size = speaker_embed_size


def register_chatterbox_hf_config() -> None:
    """Register the local config before SGLang builds its ModelConfig."""

    global _chatterbox_hf_config_registered
    if _chatterbox_hf_config_registered:
        return

    from transformers import AutoConfig

    AutoConfig.register("chatterbox_t3", ChatterboxT3Config, exist_ok=True)
    _chatterbox_hf_config_registered = True
