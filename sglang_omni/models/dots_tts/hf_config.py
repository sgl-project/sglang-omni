# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config adapters for dots.tts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoConfig, PretrainedConfig, Qwen2Config

from sglang_omni.models.dots_tts.components.model_config import ModelConfig

DOTS_TTS_MODEL_ARCH_OVERRIDE = "DotsTTSSGLangModel"
DOTS_TTS_MODEL_TYPE = "dots_tts"
DOTS_TTS_CONFIG_FILENAME = "config.json"
DOTS_TTS_LLM_CONFIG_FILENAME = "llm_config.json"
DOTS_TTS_LATENT_STATS_FILENAME = "latent_stats.pt"
DOTS_TTS_MODEL_FILENAME = "model.safetensors"
DOTS_TTS_VOCODER_FILENAME = "vocoder.safetensors"
DOTS_TTS_SPEAKER_ENCODER_FILENAME = "speaker_encoder.safetensors"

# note (chenyang): the checkpoint keeps the Qwen2 backbone config in a separate
# llm_config.json, so AutoConfig cannot reach it from config.json alone. The
# engine registers the owning checkpoint directory before SGLang builds its
# ModelConfig, and from_dict splices the backbone config in.
_checkpoint_dir: str | None = None

_MODEL_CONFIG_FIELDS = (
    "model_type",
    "latent_dim",
    "patch_size",
    "cfg_droprate",
    "PatchEncoder",
    "DiT",
    "vocoder",
    "fm_sigma",
    "xvec_drop_rate",
    "campplus_embedding_size",
    "xvec_max_audio_seconds",
    "meanflow",
)


def load_dots_tts_model_config(checkpoint_dir: str | Path) -> ModelConfig:
    """Structured dots.tts component config parsed from ``config.json``."""
    path = Path(checkpoint_dir) / DOTS_TTS_CONFIG_FILENAME
    return ModelConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_dots_tts_llm_config(checkpoint_dir: str | Path) -> Qwen2Config:
    """Qwen2 backbone config parsed from ``llm_config.json``."""
    path = Path(checkpoint_dir) / DOTS_TTS_LLM_CONFIG_FILENAME
    return Qwen2Config.from_dict(json.loads(path.read_text(encoding="utf-8")))


class DotsTTSConfig(PretrainedConfig):
    """Top-level dots.tts composite config with a nested Qwen2 backbone."""

    model_type = DOTS_TTS_MODEL_TYPE
    sub_configs = {"llm_config": Qwen2Config}

    def __init__(
        self,
        llm_config: Qwen2Config | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(llm_config, dict):
            llm_config = Qwen2Config(**llm_config)
        self.llm_config = llm_config
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "DotsTTSConfig":
        if config_dict.get("llm_config") is None and _checkpoint_dir is not None:
            config_dict = {
                **config_dict,
                "llm_config": load_dots_tts_llm_config(_checkpoint_dir).to_dict(),
            }
        return super().from_dict(config_dict, **kwargs)

    def dots_model_config(self) -> ModelConfig:
        """Structured component config rebuilt from the flat HF attributes."""
        return ModelConfig.model_validate(
            {name: getattr(self, name) for name in _MODEL_CONFIG_FIELDS}
        )


def register_dots_tts_hf_config(checkpoint_dir: str) -> None:
    """Bind the dots.tts checkpoint and register its local HF config class."""
    global _checkpoint_dir

    _checkpoint_dir = str(checkpoint_dir)
    AutoConfig.register(DOTS_TTS_MODEL_TYPE, DotsTTSConfig, exist_ok=True)


__all__ = [
    "DOTS_TTS_CONFIG_FILENAME",
    "DOTS_TTS_LATENT_STATS_FILENAME",
    "DOTS_TTS_LLM_CONFIG_FILENAME",
    "DOTS_TTS_MODEL_ARCH_OVERRIDE",
    "DOTS_TTS_MODEL_FILENAME",
    "DOTS_TTS_MODEL_TYPE",
    "DOTS_TTS_SPEAKER_ENCODER_FILENAME",
    "DOTS_TTS_VOCODER_FILENAME",
    "DotsTTSConfig",
    "load_dots_tts_llm_config",
    "load_dots_tts_model_config",
    "register_dots_tts_hf_config",
]
