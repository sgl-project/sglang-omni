# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

from sglang_omni.models.qwen3_asr.config import Qwen3ASRPipelineConfig
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


def test_qwen3_asr_config_uses_batched_stage_with_32_running_requests() -> None:
    config = Qwen3ASRPipelineConfig(model_path="Qwen/Qwen3-ASR-1.7B")

    assert config.entry_stage == "asr"
    assert [stage.name for stage in config.stages] == ["asr"]
    assert config.terminal_stages == ["asr"]
    assert config.gpu_placement == {"asr": 0}
    assert config.stages[0].factory.endswith("create_sglang_qwen3_asr_executor")
    assert config.stages[0].factory_args["device"] == "cuda:0"
    assert config.stages[0].factory_args["max_running_requests"] == 32
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("Qwen3ASRForConditionalGeneration")
        is Qwen3ASRPipelineConfig
    )


def test_qwen3_asr_stage_default_allows_32_running_requests() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["max_running_requests"].default == 32


def test_qwen3_asr_stage_default_uses_auto_static_kv_budget() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mem_fraction_static"].default is None


def test_qwen3_asr_stage_default_disables_multimodal_embedding_cache() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["mm_embedding_cache_size_bytes"].default == 0


def test_qwen3_asr_stage_default_disables_torch_compile() -> None:
    signature = inspect.signature(create_sglang_qwen3_asr_executor)

    assert signature.parameters["enable_torch_compile"].default is False
