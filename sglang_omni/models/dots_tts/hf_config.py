# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config adapter for dots.tts checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig, Qwen2Config

DOTS_TTS_MODEL_ARCH_OVERRIDE = "DotsTTSForConditionalGeneration"

_dots_tts_hf_config_registered = False


class DotsTTSConfig(PretrainedConfig):
    """Top-level dots.tts config with its Qwen2 config attached in memory."""

    model_type = "dots_tts"
    sub_configs = {"llm_config": Qwen2Config}

    def __init__(
        self,
        llm_config: Qwen2Config | dict[str, Any] | None = None,
        dots_tts_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(llm_config, dict):
            llm_config = Qwen2Config(**llm_config)
        self.llm_config = llm_config
        self.dots_tts_config = dict(dots_tts_config or {})
        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs: Any) -> "DotsTTSConfig":
        merged = dict(config_dict)
        if "llm_config" not in merged:
            checkpoint_dir = kwargs.get("name_or_path")
            if not checkpoint_dir:
                raise ValueError("dots.tts config requires a checkpoint path")
            llm_path = Path(str(checkpoint_dir)) / "llm_config.json"
            merged["llm_config"] = json.loads(llm_path.read_text(encoding="utf-8"))
        merged.setdefault("dots_tts_config", dict(config_dict))
        return super().from_dict(merged, **kwargs)


def register_dots_tts_hf_config() -> None:
    """Register the local dots.tts config before SGLang loads ModelConfig."""

    global _dots_tts_hf_config_registered
    if _dots_tts_hf_config_registered:
        return

    from transformers import AutoConfig

    AutoConfig.register("dots_tts", DotsTTSConfig, exist_ok=True)
    _dots_tts_hf_config_registered = True


__all__ = [
    "DOTS_TTS_MODEL_ARCH_OVERRIDE",
    "DotsTTSConfig",
    "register_dots_tts_hf_config",
]
