# SPDX-License-Identifier: MIT
# Derived from mlx-audio Qwen3-TTS (Copyright 2025 Prince Canuma and contributors).
"""Loader-facing MLX Qwen3-TTS talker model."""

from __future__ import annotations

from .config import ModelConfig
from .talker import Qwen3TTSTalkerForConditionalGeneration


class Qwen3TTSTalkerModel(Qwen3TTSTalkerForConditionalGeneration):
    """The talker, built from a whole-checkpoint Qwen3-TTS config.

    ``mlx_lm.utils.load_model`` calls ``ModelClass(ModelArgs.from_dict(config))``
    with the checkpoint's full ``config.json``, while the talker itself only
    needs ``talker_config``. Subclassing rather than wrapping keeps the module
    tree at ``model.layers``, which is where SGLang's MLX backend discovers
    attention layers and installs its per-request KV caches.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.talker_config)
        self.model_config = config

    @property
    def tts_model_type(self) -> str:
        return self.model_config.tts_model_type


Model = Qwen3TTSTalkerModel
ModelArgs = ModelConfig
