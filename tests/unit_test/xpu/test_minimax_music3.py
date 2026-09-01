# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni import platforms
from sglang_omni.models.minimax_music3 import acoustic, engine_builder, stages
from sglang_omni.models.minimax_music3.config import MiniMaxMusic3PipelineConfig
from sglang_omni.models.minimax_music3.platform_policy import (
    Qualification,
    get_minimax_music3_platform_policy,
)


def _mock_xpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(platforms.current_platform, "is_musa", lambda: False)
    monkeypatch.setattr(platforms.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(platforms.current_platform, "device_type", "xpu")


def test_minimax_music3_xpu_defaults_to_eager_dual_device_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeXPU:
        @staticmethod
        def device_count() -> int:
            return 2

    _mock_xpu_platform(monkeypatch)
    monkeypatch.setattr(torch, "get_device_module", lambda _device: _FakeXPU)

    config = MiniMaxMusic3PipelineConfig(model_path="/models/minimax")
    configured_stages = {stage.name: stage for stage in config.stages}

    assert configured_stages["dit_dav"].gpu == 1
    assert configured_stages["dit_dav"].factory.compile_acoustic is False
    policy = get_minimax_music3_platform_policy()
    assert policy.qualification is Qualification.UNVERIFIED
    assert "real-XPU" in policy.reason
    assert engine_builder.MiniMaxMusic3EngineBuilder().generation_defaults(
        dtype="bfloat16"
    )["disable_cuda_graph"]


def test_minimax_music3_xpu_ar_stage_uses_platform_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeBuilder:
        def __init__(self, *, max_running_requests: int) -> None:
            self.max_running_requests = max_running_requests

        def build(self, model_path: str, **kwargs: object) -> object:
            captured.update(model_path=model_path, **kwargs)
            return object()

    _mock_xpu_platform(monkeypatch)
    monkeypatch.setattr(
        stages,
        "resolve_device_spec",
        lambda device, gpu_id: f"xpu:{gpu_id}",
    )
    monkeypatch.setattr(engine_builder, "MiniMaxMusic3EngineBuilder", _FakeBuilder)

    stages.create_ar_executor("/models/minimax", gpu_id=3, max_concurrency=4)

    assert captured["device"] == "xpu:3"
    assert captured["gpu_id"] == 3
    assert captured["dtype"] == "bfloat16"


def test_minimax_music3_xpu_acoustic_stage_defaults_to_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeDecoder:
        def __init__(self, _model_path: str, **kwargs: object) -> None:
            captured.update(kwargs)
            self.device = kwargs["device"]
            self.dtype = kwargs["dtype"]
            self.dit_steps = kwargs["dit_steps"]
            self.dit_cfg_scale = kwargs["dit_cfg_scale"]
            self.attention_backend = kwargs["attention_backend"]
            self.compile_acoustic = kwargs["compile_acoustic"]

    _mock_xpu_platform(monkeypatch)
    monkeypatch.setattr(
        stages,
        "resolve_device_spec",
        lambda device, gpu_id: f"xpu:{gpu_id}",
    )
    monkeypatch.setattr(stages, "MiniMaxMusic3AcousticDecoder", _FakeDecoder)

    stages.create_dit_dav_executor("/models/minimax", gpu_id=1)

    assert captured["device"] == "xpu:1"
    assert captured["compile_acoustic"] is False
    assert captured["attention_backend"] == "torch_sdpa"


def test_minimax_music3_xpu_skips_rvq_cuda_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)

    engine_builder.MiniMaxMusic3EngineBuilder().setup_model_resources(
        object(), object(), generation_cuda_graph_enabled=False
    )


def test_minimax_music3_xpu_rejects_breakable_cuda_graph_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_devices: list[str | None] = []
    _mock_xpu_platform(monkeypatch)
    monkeypatch.setattr(
        acoustic,
        "resolve_device_spec",
        lambda device: resolved_devices.append(device) or "xpu:0",
    )

    with pytest.raises(ValueError, match="unavailable on xpu"):
        acoustic.MiniMaxMusic3AcousticDecoder(
            "/models/minimax", breakable_cuda_graph=True
        )
    assert resolved_devices == [None]
