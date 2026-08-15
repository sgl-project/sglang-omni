# SPDX-License-Identifier: Apache-2.0
"""Native-only config loader for SenseNova U1 SGLang serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import PretrainedConfig, Qwen3Config

_parser_registered = False

_U1_LLM_EXTRA_FIELDS = (
    "rope_theta",
    "rope_theta_hw",
    "max_position_embeddings_hw",
    "pure_llm",
    "use_deepep",
)


def _build_u1_llm_config(raw: Qwen3Config | dict[str, Any] | None) -> Qwen3Config | None:
    if raw is None or isinstance(raw, Qwen3Config):
        return raw
    cfg = Qwen3Config(**raw)
    for key in _U1_LLM_EXTRA_FIELDS:
        if key in raw:
            setattr(cfg, key, raw[key])
    return cfg


class SenseNovaU1NativeConfig(PretrainedConfig):
    """Top-level U1 config without importing official HF modeling code."""

    model_type = "neo_chat"
    sub_configs = {"llm_config": Qwen3Config}

    def __init__(
        self,
        llm_config: Qwen3Config | dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("auto_map", None)
        llm_config = _build_u1_llm_config(llm_config)
        self.llm_config = llm_config
        self.vision_config = dict(vision_config or {})
        super().__init__(**kwargs)
        self.architectures = ["SenseNovaU1NativeForCausalLM"]

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        **kwargs: Any,
    ) -> "SenseNovaU1NativeConfig":
        clean = dict(config_dict)
        clean.pop("auto_map", None)
        return super().from_dict(clean, **kwargs)


def register_sensenova_u1_native_config_parser() -> None:
    """Register a parser that reads U1 config.json without remote code."""

    global _parser_registered
    if _parser_registered:
        return

    from sglang.srt.configs.model_config_parser_registry import (
        ModelConfigParserBase,
        register_model_config_parser,
    )

    @register_model_config_parser("sensenova_u1_native")
    class SenseNovaU1NativeModelConfigParser(ModelConfigParserBase):
        def parse(
            self,
            model,
            trust_remote_code: bool,
            revision: str | None = None,
            **kwargs: Any,
        ):
            del trust_remote_code, revision
            config_path = Path(str(model)) / "config.json"
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            config = SenseNovaU1NativeConfig.from_dict(
                raw,
                name_or_path=str(model),
                **kwargs,
            )
            text_config = config.llm_config
            for key, value in vars(text_config).items():
                if not hasattr(config, key) and value is not None:
                    setattr(config, key, value)
            return config

    _parser_registered = True


__all__ = [
    "SenseNovaU1NativeConfig",
    "register_sensenova_u1_native_config_parser",
]
