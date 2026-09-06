# SPDX-License-Identifier: Apache-2.0
"""SGLang config parser for MOSS-TTS-Nano checkpoints."""

from __future__ import annotations

from typing import Any

from transformers import GPT2Config, PretrainedConfig

MOSS_TTS_NANO_MODEL_CONFIG_PARSER = "moss_tts_nano"
MOSS_TTS_NANO_MODEL_ARCH_OVERRIDE = "MossTTSNanoSGLangModel"

_moss_tts_nano_parser_registered = False


def adapt_moss_tts_nano_hf_config(config: PretrainedConfig) -> PretrainedConfig:
    """Expose the nested GPT-2 config before SGLang derives model shapes."""

    if getattr(config, "model_type", None) != "moss_tts_nano":
        raise ValueError(
            "MOSS-TTS-Nano requires model_type='moss_tts_nano'; got "
            f"{getattr(config, 'model_type', None)!r}"
        )
    gpt2_config: Any = getattr(config, "gpt2_config", None)
    if isinstance(gpt2_config, dict):
        gpt2_config = GPT2Config(**gpt2_config)
        config.gpt2_config = gpt2_config
    if gpt2_config is None:
        raise ValueError("MOSS-TTS-Nano config is missing gpt2_config")

    config.language_config = gpt2_config
    config.architectures = [MOSS_TTS_NANO_MODEL_ARCH_OVERRIDE]
    return config


def select_moss_tts_nano_model_config_parser(overrides: dict[str, Any]) -> None:
    """Resolve the required parser before SGLang freezes ``ServerArgs``."""

    selected_parser = overrides.get("model_config_parser")
    if selected_parser not in (
        None,
        "auto",
        MOSS_TTS_NANO_MODEL_CONFIG_PARSER,
    ):
        raise ValueError(
            "MOSS-TTS-Nano requires model_config_parser="
            f"{MOSS_TTS_NANO_MODEL_CONFIG_PARSER!r}; got {selected_parser!r}"
        )
    register_moss_tts_nano_model_config_parser()
    overrides["model_config_parser"] = MOSS_TTS_NANO_MODEL_CONFIG_PARSER


def register_moss_tts_nano_model_config_parser() -> None:
    """Register the parser before SGLang constructs ``ModelConfig``."""

    global _moss_tts_nano_parser_registered
    if _moss_tts_nano_parser_registered:
        return

    from sglang.srt.configs.model_config_parser_registry import (
        register_model_config_parser,
    )
    from sglang.srt.utils.hf_transformers.config import HfModelConfigParser

    @register_model_config_parser(MOSS_TTS_NANO_MODEL_CONFIG_PARSER)
    class MossTTSNanoModelConfigParser(HfModelConfigParser):
        def parse(self, *args: Any, **kwargs: Any) -> PretrainedConfig:
            return adapt_moss_tts_nano_hf_config(super().parse(*args, **kwargs))

    _moss_tts_nano_parser_registered = True


__all__ = [
    "MOSS_TTS_NANO_MODEL_ARCH_OVERRIDE",
    "MOSS_TTS_NANO_MODEL_CONFIG_PARSER",
    "adapt_moss_tts_nano_hf_config",
    "register_moss_tts_nano_model_config_parser",
    "select_moss_tts_nano_model_config_parser",
]
