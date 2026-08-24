# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.models.llada2_uni import config, stages
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState


def test_thinker_factory_keeps_legacy_text_defaults() -> None:
    signature = inspect.signature(
        stages.create_sglang_dllm_thinker_executor_from_config
    )

    assert signature.parameters["dllm_algorithm"].default == "LowConfidence"


def test_native_variant_enables_cfg_and_cuda_graph_explicitly() -> None:
    native = config.Variants["omni"](model_path="checkpoint")
    stage_by_name = {stage.name: stage for stage in native.stages}
    thinker_args = stage_by_name[config.THINKER_STAGE].factory_args

    assert thinker_args["dllm_algorithm"] == "LowConfidenceCFG"
    assert thinker_args["server_args_overrides"] == {
        "disable_cuda_graph": False,
        "max_running_requests": 3,
    }


def test_thinker_registers_cfg_runtime_before_building_server_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered: dict[str, object] = {}
    calls: list[str] = []
    algorithm = object()

    algorithm_module = ModuleType("sglang.srt.dllm.algorithm")
    algorithm_module.algo_name_to_cls = registered
    monkeypatch.setitem(sys.modules, "sglang.srt.dllm.algorithm", algorithm_module)

    local_algorithm_module = ModuleType(
        "sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg"
    )
    local_algorithm_module.LowConfidenceCFG = algorithm
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg",
        local_algorithm_module,
    )

    attention_module = ModuleType("sglang_omni.models.llada2_uni.cfg_attention_backend")
    attention_module.register_llada2_cfg_flashinfer_backend = lambda: calls.append(
        "attention"
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang_omni.models.llada2_uni.cfg_attention_backend",
        attention_module,
    )

    bootstrap_module = ModuleType("sglang_omni.models.llada2_uni.bootstrap")
    bootstrap_module.create_dllm_thinker_scheduler = lambda args, gpu_id: (
        args,
        gpu_id,
    )
    monkeypatch.setitem(
        sys.modules, "sglang_omni.models.llada2_uni.bootstrap", bootstrap_module
    )

    backend_module = ModuleType("sglang_omni.scheduling.sglang_backend")

    def _build_args(model_path, **kwargs):
        del model_path
        calls.append("build")
        assert kwargs["disable_cuda_graph"] is False
        assert kwargs["max_running_requests"] >= 3
        return SimpleNamespace(
            dllm_algorithm=kwargs["dllm_algorithm"], mem_fraction_static=0.5
        )

    backend_module.build_sglang_server_args = _build_args
    monkeypatch.setitem(
        sys.modules, "sglang_omni.scheduling.sglang_backend", backend_module
    )

    result = stages.create_sglang_dllm_thinker_executor_from_config(
        "checkpoint",
        dllm_algorithm="LowConfidenceCFG",
        server_args_overrides={
            "disable_cuda_graph": False,
            "max_running_requests": 3,
        },
    )

    assert registered == {"LowConfidenceCFG": algorithm}
    assert calls == ["attention", "build"]
    assert result[0].dllm_algorithm == "LowConfidenceCFG"


def test_thinker_rejects_attention_backend_without_cfg_padding_support() -> None:
    with pytest.raises(ValueError, match="flashinfer"):
        stages.create_sglang_dllm_thinker_executor_from_config(
            "checkpoint",
            dllm_algorithm="LowConfidenceCFG",
            server_args_overrides={"attention_backend": "triton"},
        )


def test_native_variant_has_dedicated_image_decoder_terminal() -> None:
    native = config.Variants["omni"](model_path="checkpoint")
    stage_by_name = {stage.name: stage for stage in native.stages}

    assert config.IMAGE_DECODE_STAGE in stage_by_name
    assert stage_by_name[config.IMAGE_DECODE_STAGE].terminal is True
    assert set(stage_by_name[config.THINKER_STAGE].next) == {
        config.DECODE_STAGE,
        config.IMAGE_DECODE_STAGE,
    }
    assert native.terminal_stages_fn == (
        "sglang_omni.models.llada2_uni.routing.resolve_terminal_stages"
    )


def test_thinker_router_sends_native_images_to_image_decoder() -> None:
    from sglang_omni.models.llada2_uni.routing import thinker_next

    payload = SimpleNamespace(data=LLaDA2UniPipelineState(task_kind="t2i").to_dict())
    assert thinker_next("request-0", payload) == config.IMAGE_DECODE_STAGE

    payload.data = LLaDA2UniPipelineState(task_kind="chat").to_dict()
    assert thinker_next("request-0", payload) == config.DECODE_STAGE


def test_terminal_resolver_matches_task_specific_thinker_route() -> None:
    from sglang_omni.models.llada2_uni.routing import resolve_terminal_stages

    text_request = SimpleNamespace(metadata={"output_modalities": ["text"]})
    image_request = SimpleNamespace(metadata={"output_modalities": ["image"]})

    assert resolve_terminal_stages(text_request) == [config.DECODE_STAGE]
    assert resolve_terminal_stages(image_request) == [config.IMAGE_DECODE_STAGE]
