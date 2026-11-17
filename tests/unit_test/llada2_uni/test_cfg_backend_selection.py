# SPDX-License-Identifier: Apache-2.0
"""Platform routing for the LLaDA2-Uni DLLM attention backend."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from sglang.srt.model_executor.cuda_graph_config import Backend as CudaGraphBackend

from sglang_omni.models.llada2_uni import bootstrap, stages
from sglang_omni.platforms import current_platform

CFG_MODULE = "sglang_omni.models.llada2_uni.cfg_attention_backend"
CFG_ASCEND_MODULE = "sglang_omni.models.llada2_uni.cfg_attention_ascend"


def stub_cfg_module(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = types.ModuleType(CFG_MODULE)
    stub.CFG_ATTENTION_BACKEND = "llada2_uni_cfg_flashinfer"
    stub.register_llada2_cfg_flashinfer_backend = lambda: None
    monkeypatch.setitem(sys.modules, CFG_MODULE, stub)


def stub_cfg_ascend_module(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = types.ModuleType(CFG_ASCEND_MODULE)
    stub.register_llada2_cfg_ascend_backend = lambda: None
    monkeypatch.setitem(sys.modules, CFG_ASCEND_MODULE, stub)


def capture_thinker_overrides(
    monkeypatch: pytest.MonkeyPatch, *, npu: bool, algorithm: str
) -> dict[str, object]:
    captured: dict[str, object] = {}
    monkeypatch.setattr(current_platform, "is_npu", lambda: npu, raising=False)

    def fake_build(model_path: str, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(dllm_algorithm=algorithm, mem_fraction_static=0.8)

    monkeypatch.setattr(stages, "build_sglang_server_args", fake_build)
    monkeypatch.setattr(stages, "pin_resolved_device_type", lambda *args: None)
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda device, gpu_id: SimpleNamespace(type="npu" if npu else "cuda", index=0),
    )
    monkeypatch.setattr(
        stages, "create_dllm_thinker_scheduler", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        stages,
        "register_llada2_uni_cfg",
        lambda: captured.setdefault("registered", True),
    )
    monkeypatch.setattr(
        stages,
        "resolved_view",
        lambda server_args: SimpleNamespace(
            dllm_algorithm=algorithm, mem_fraction_static=0.8
        ),
    )
    stages.create_sglang_dllm_thinker_executor_from_config(
        "fake/model", dllm_algorithm=algorithm
    )
    return captured


def test_cfg_thinker_uses_ascend_backend_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = capture_thinker_overrides(
        monkeypatch, npu=True, algorithm="LowConfidenceCFG"
    )

    # NPU argument resolution pins the backend name; the CFG variant is
    # injected by wrapping the ascend factory instead of renaming it.
    assert overrides["attention_backend"] == "ascend"
    assert overrides["disable_cuda_graph"] is True
    assert overrides["dllm_fdfo"] is False
    assert overrides["registered"] is True


def test_cfg_thinker_keeps_flashinfer_backend_off_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_cfg_module(monkeypatch)

    overrides = capture_thinker_overrides(
        monkeypatch, npu=False, algorithm="LowConfidenceCFG"
    )

    assert overrides["attention_backend"] == "llada2_uni_cfg_flashinfer"


def test_plain_cfg_algorithm_defaults_to_platform_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = capture_thinker_overrides(
        monkeypatch, npu=True, algorithm="LowConfidence"
    )

    assert overrides["attention_backend"] == "ascend"
    assert "registered" not in overrides


def test_plain_cfg_algorithm_keeps_flashinfer_off_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = capture_thinker_overrides(
        monkeypatch, npu=False, algorithm="LowConfidence"
    )

    assert overrides["attention_backend"] == "flashinfer"
    assert "registered" not in overrides


def run_validate_cfg(
    monkeypatch: pytest.MonkeyPatch, *, npu: bool, backends: list[str]
) -> None:
    monkeypatch.setattr(current_platform, "is_npu", lambda: npu, raising=False)
    monkeypatch.setattr(bootstrap, "register_llada2_uni_cfg", lambda: None)
    monkeypatch.setattr(
        bootstrap,
        "resolved_view",
        lambda server_args: SimpleNamespace(
            dllm_algorithm="LowConfidenceCFG",
            disable_cuda_graph=True,
            dllm_fdfo=False,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=CudaGraphBackend.DISABLED)
            ),
        ),
    )
    monkeypatch.setattr(
        bootstrap,
        "attention_backends_of",
        lambda cfg: backends,
    )

    bootstrap.validate_cfg(object())


def test_validate_cfg_accepts_ascend_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_validate_cfg(monkeypatch, npu=True, backends=["ascend"])


def test_validate_cfg_rejects_flashinfer_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="requires ascend"):
        run_validate_cfg(monkeypatch, npu=True, backends=["flashinfer"])


def test_validate_cfg_rejects_ascend_off_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_cfg_module(monkeypatch)

    with pytest.raises(ValueError, match="llada2_uni_cfg_flashinfer"):
        run_validate_cfg(monkeypatch, npu=False, backends=["ascend"])


def test_register_cfg_does_not_touch_flashinfer_on_npu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(current_platform, "is_npu", lambda: True, raising=False)
    # Importing the CFG backend module here raises ImportError by construction.
    monkeypatch.setitem(sys.modules, CFG_MODULE, None)
    stub_cfg_ascend_module(monkeypatch)

    bootstrap.register_llada2_uni_cfg()

    from sglang.srt.dllm.algorithm import algo_name_to_cls

    from sglang_omni.models.llada2_uni.low_confidence_cfg import LowConfidenceCFG

    assert algo_name_to_cls["LowConfidenceCFG"] is LowConfidenceCFG
    assert bootstrap.cfg_attention_backend_name() == "ascend"
