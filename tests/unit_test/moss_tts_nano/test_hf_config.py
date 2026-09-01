# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from transformers import GPT2Config, PretrainedConfig

from sglang_omni.models.moss_tts_nano.hf_config import (
    MOSS_TTS_NANO_MODEL_ARCH_OVERRIDE,
    MOSS_TTS_NANO_MODEL_CONFIG_PARSER,
    adapt_moss_tts_nano_hf_config,
    select_moss_tts_nano_model_config_parser,
)


def test_config_adapter_exposes_gpt2_shapes_to_sglang() -> None:
    config = PretrainedConfig(architectures=["MossTTSNanoForCausalLM"])
    config.model_type = "moss_tts_nano"
    config.gpt2_config = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "vocab_size": 16384,
    }

    adapted = adapt_moss_tts_nano_hf_config(config)

    assert isinstance(adapted.gpt2_config, GPT2Config)
    assert adapted.language_config is adapted.gpt2_config
    assert adapted.language_config.num_attention_heads == 12
    assert adapted.language_config.num_hidden_layers == 12
    assert adapted.architectures == [MOSS_TTS_NANO_MODEL_ARCH_OVERRIDE]


def test_config_adapter_rejects_missing_gpt2_config() -> None:
    config = PretrainedConfig()
    config.model_type = "moss_tts_nano"
    with pytest.raises(ValueError, match="missing gpt2_config"):
        adapt_moss_tts_nano_hf_config(config)


def test_config_adapter_rejects_wrong_model_type() -> None:
    config = PretrainedConfig()
    config.gpt2_config = GPT2Config()
    with pytest.raises(ValueError, match="requires model_type='moss_tts_nano'"):
        adapt_moss_tts_nano_hf_config(config)


def test_parser_selection_happens_before_server_args(monkeypatch) -> None:
    from sglang_omni.models.moss_tts_nano import hf_config

    calls: list[str] = []
    monkeypatch.setattr(
        hf_config,
        "register_moss_tts_nano_model_config_parser",
        lambda: calls.append("register"),
    )
    overrides = {"model_config_parser": "auto"}

    select_moss_tts_nano_model_config_parser(overrides)

    assert calls == ["register"]
    assert overrides["model_config_parser"] == MOSS_TTS_NANO_MODEL_CONFIG_PARSER


def test_parser_selection_rejects_an_operator_selected_parser() -> None:
    with pytest.raises(ValueError, match="requires model_config_parser"):
        select_moss_tts_nano_model_config_parser({"model_config_parser": "hf"})


def test_model_worker_registers_nano_parser_before_model_config(monkeypatch) -> None:
    platforms_module = ModuleType("sglang_omni.platforms")
    platforms_module.current_platform = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "sglang_omni.platforms", platforms_module)

    from sglang_omni.model_runner.model_worker import ModelWorker
    from sglang_omni.models.moss_tts_nano import hf_config

    call_order: list[str] = []
    monkeypatch.setattr(
        hf_config,
        "register_moss_tts_nano_model_config_parser",
        lambda: call_order.append("register"),
    )

    model_config_module = ModuleType("sglang.srt.configs.model_config")

    class FakeModelConfig:
        @classmethod
        def from_server_args(cls, *, server_args, **kwargs):
            del kwargs
            assert server_args.model_config_parser == MOSS_TTS_NANO_MODEL_CONFIG_PARSER
            call_order.append("from_server_args")
            return SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=[],
                    gpt2_config=SimpleNamespace(
                        hidden_size=768,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        vocab_size=16384,
                    ),
                )
            )

    model_config_module.ModelConfig = FakeModelConfig
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs",
        ModuleType("sglang.srt.configs"),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.configs.model_config",
        model_config_module,
    )

    worker = object.__new__(ModelWorker)
    worker.server_args = SimpleNamespace(
        model_path="dummy",
        revision=None,
        model_config_parser=MOSS_TTS_NANO_MODEL_CONFIG_PARSER,
    )
    worker.model_arch_override = MOSS_TTS_NANO_MODEL_ARCH_OVERRIDE

    worker._init_model_config()

    assert call_order == ["register", "from_server_args"]
    assert (
        worker.model_config.hf_text_config is worker.model_config.hf_config.gpt2_config
    )


def test_sglang_model_config_parses_nested_nano_shapes(tmp_path) -> None:
    pytest.importorskip("sglang")
    from sglang.srt.configs.model_config import ModelConfig

    from sglang_omni.models.moss_tts_nano.hf_config import (
        register_moss_tts_nano_model_config_parser,
    )

    (tmp_path / "configuration_moss_tts_nano.py").write_text(
        """
from transformers import GPT2Config, PretrainedConfig


class MossTTSNanoConfig(PretrainedConfig):
    model_type = "moss_tts_nano"

    def __init__(self, gpt2_config=None, **kwargs):
        self.gpt2_config = GPT2Config(**(gpt2_config or {}))
        super().__init__(**kwargs)
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "moss_tts_nano",
                "architectures": ["MossTTSNanoForCausalLM"],
                "auto_map": {
                    "AutoConfig": ("configuration_moss_tts_nano.MossTTSNanoConfig")
                },
                "dtype": "float32",
                "pad_token_id": 3,
                "gpt2_config": {
                    "hidden_size": 768,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 12,
                    "n_positions": 32768,
                    "vocab_size": 16384,
                },
            }
        ),
        encoding="utf-8",
    )

    register_moss_tts_nano_model_config_parser()
    model_config = ModelConfig(
        model_path=str(tmp_path),
        trust_remote_code=True,
        context_length=128,
        dtype="float32",
        model_config_parser=MOSS_TTS_NANO_MODEL_CONFIG_PARSER,
    )

    assert model_config.hf_config.architectures == [MOSS_TTS_NANO_MODEL_ARCH_OVERRIDE]
    assert model_config.hf_text_config is model_config.hf_config.gpt2_config
    assert model_config.hidden_size == 768
    assert model_config.num_attention_heads == 12
    assert model_config.num_key_value_heads == 12
    assert model_config.num_hidden_layers == 12
    assert model_config.vocab_size == 16384
