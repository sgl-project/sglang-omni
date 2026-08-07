# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json


def test_dots_hf_config_attaches_llm_config_without_checkpoint_view(tmp_path) -> None:
    from transformers import AutoConfig

    from sglang_omni.models.dots_tts.hf_config import (
        DotsTTSConfig,
        register_dots_tts_hf_config,
    )

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "dots_tts",
                "architectures": ["DotsTTSForConditionalGeneration"],
                "latent_dim": 128,
                "patch_size": 4,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "llm_config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 1536,
                "num_hidden_layers": 28,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "vocab_size": 151672,
            }
        ),
        encoding="utf-8",
    )

    register_dots_tts_hf_config()
    config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)

    assert isinstance(config, DotsTTSConfig)
    assert config.name_or_path == str(tmp_path)
    assert config.llm_config.hidden_size == 1536
    assert config.dots_tts_config["patch_size"] == 4


def test_dots_arch_uses_nested_llm_config() -> None:
    from sglang_omni.model_runner.model_worker import _ARCH_CONFIG_MAP

    assert _ARCH_CONFIG_MAP["DotsTTSForConditionalGeneration"] == (
        "llm_config",
        None,
    )
