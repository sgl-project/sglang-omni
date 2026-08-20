# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

from sglang_omni.config.manager import ConfigManager
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.kimi_audio.config import KimiAudioPipelineConfig
from sglang_omni.models.kimi_audio.engine_builder import KimiAudioEngineBuilder
from sglang_omni.models.kimi_audio.stages import create_kimi_audio_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_kimi_audio_pipeline_is_registered() -> None:
    config = KimiAudioPipelineConfig(model_path="moonshotai/Kimi-Audio-7B-Instruct")

    assert config.entry_stage == "generation"
    assert config.terminal_stages == ["generation"]
    assert config.stages[0].factory.endswith("create_kimi_audio_executor")
    assert config.stages[0].factory_args["max_running_requests"] == 8
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("MoonshotKimiaForCausalLM")
        is KimiAudioPipelineConfig
    )
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config_for_model_id(
            "moonshotai/Kimi-Audio-7B-Instruct"
        )
        is KimiAudioPipelineConfig
    )


def test_canonical_model_id_resolves_without_remote_config(monkeypatch) -> None:
    def fail_remote_config(*args, **kwargs):
        del args, kwargs
        raise AssertionError("canonical Kimi model id should not access remote config")

    monkeypatch.setattr(
        "sglang_omni.config.manager.AutoConfig.from_pretrained", fail_remote_config
    )

    manager = ConfigManager.from_model_path("moonshotai/Kimi-Audio-7B-Instruct")

    assert isinstance(manager.config, KimiAudioPipelineConfig)


def test_kimi_audio_stage_defaults_are_text_output_focused() -> None:
    signature = inspect.signature(create_kimi_audio_executor)

    assert signature.parameters["dtype"].default == "bfloat16"
    assert signature.parameters["max_new_tokens"].default == 512
    assert signature.parameters["audio_tokenizer_path"].default == (
        "THUDM/glm-4-voice-tokenizer"
    )


def test_kimi_audio_prefill_budget_is_capped_to_model_context() -> None:
    builder = KimiAudioEngineBuilder(
        max_running_requests=8,
        max_new_tokens=512,
        mem_fraction_static=None,
        enable_torch_compile=False,
        request_build_max_workers=2,
        request_build_max_pending=8,
        audio_tokenizer_path="THUDM/glm-4-voice-tokenizer",
    )
    builder.context_length = 8192

    defaults = builder.generation_defaults(dtype="bfloat16")

    assert defaults["chunked_prefill_size"] == -1
    assert defaults["max_prefill_tokens"] == 8192


def test_kimi_audio_arch_override_uses_text_vocab_and_main_model_layers() -> None:
    hf_config = SimpleNamespace(
        architectures=["MoonshotKimiaForCausalLM"],
        kimia_text_output_vocab=152064,
        num_hidden_layers=30,
        kimia_mimo_transformer_from_layer_index=21,
        kimia_mimo_layers=6,
    )
    model_config = SimpleNamespace(
        hf_config=hf_config,
        vocab_size=168448,
        num_hidden_layers=28,
        num_attention_layers=28,
    )

    ModelWorker._apply_arch_override(model_config, "KimiAudioForTextGeneration")

    assert model_config.vocab_size == 152064
    assert model_config.num_hidden_layers == 30
    assert model_config.num_attention_layers == 30
