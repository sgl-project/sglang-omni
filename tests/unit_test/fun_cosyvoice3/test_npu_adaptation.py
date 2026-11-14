# SPDX-License-Identifier: Apache-2.0
"""NPU platform adaptation tests for Fun-CosyVoice3."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sglang_omni.models.fun_cosyvoice3.config as config_module
import sglang_omni.models.fun_cosyvoice3.engine_builder as engine_builder_module
from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.platforms.cpu import CPUOmniPlatform
from sglang_omni.platforms.cuda import CUDAOmniPlatform
from sglang_omni.platforms.interface import OmniPlatform
from sglang_omni.platforms.npu import NPUOmniPlatform
from sglang_omni.scheduling.generation_batch_policy import CudaGraphBackend

MODEL_PATH = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


class FakePlatform:
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    def is_npu(self) -> bool:
        return self.device_type == "npu"


def test_the_npu_keeps_the_dit_eager() -> None:
    assert stages.dit_torch_compile_allowed(True, FakePlatform("npu")) is False


def test_other_devices_keep_the_requested_flag() -> None:
    assert stages.dit_torch_compile_allowed(True, FakePlatform("cuda")) is True
    assert stages.dit_torch_compile_allowed(True, FakePlatform("cpu")) is True


def test_an_explicit_opt_out_never_compiles() -> None:
    """A caller that turned compile off keeps it off, whatever the device says."""
    assert stages.dit_torch_compile_allowed(False, FakePlatform("cuda")) is False


def test_only_the_npu_drops_float64() -> None:
    assert NPUOmniPlatform().is_float64_supported() is False
    for platform_type in (OmniPlatform, CUDAOmniPlatform, CPUOmniPlatform):
        assert platform_type().is_float64_supported() is True


def npu_config(monkeypatch: pytest.MonkeyPatch) -> FunCosyVoice3PipelineConfig:
    monkeypatch.setattr(config_module, "current_platform", FakePlatform("npu"))
    return FunCosyVoice3PipelineConfig(model_path=MODEL_PATH)


def test_the_npu_gets_per_stream_host_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared Host queue deadlocks the scheduler once a decode graph replays."""
    config = npu_config(monkeypatch)

    assert config.resolved_stage_env_defaults("tts_engine")["PER_STREAM_QUEUE"] == "1"


def test_a_written_per_stream_queue_value_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "current_platform", FakePlatform("npu"))
    config = FunCosyVoice3PipelineConfig(
        model_path=MODEL_PATH, env_defaults={"PER_STREAM_QUEUE": "0"}
    )

    assert config.resolved_stage_env_defaults("tts_engine")["PER_STREAM_QUEUE"] == "0"


def test_other_devices_get_no_per_stream_queue_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "current_platform", FakePlatform("cuda"))
    config = FunCosyVoice3PipelineConfig(model_path=MODEL_PATH)

    assert "PER_STREAM_QUEUE" not in config.resolved_stage_env_defaults("tts_engine")


def npu_decode_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: CudaGraphBackend,
    max_bs: int,
    per_stream_queue: str | None,
) -> FunCosyVoice3EngineBuilder:
    monkeypatch.setattr(engine_builder_module, "current_platform", FakePlatform("npu"))
    decode = SimpleNamespace(backend=backend, max_bs=max_bs)
    monkeypatch.setattr(
        engine_builder_module,
        "resolved_view",
        lambda _: SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode)),
    )
    if per_stream_queue is None:
        monkeypatch.delenv("PER_STREAM_QUEUE", raising=False)
    else:
        monkeypatch.setenv("PER_STREAM_QUEUE", per_stream_queue)
    return FunCosyVoice3EngineBuilder()


def test_the_npu_rejects_multi_row_decode_graphs_on_a_shared_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup must fail instead of stalling after the first batch."""
    builder = npu_decode_graph(
        monkeypatch, backend=CudaGraphBackend.FULL, max_bs=16, per_stream_queue=None
    )

    with pytest.raises(ValueError, match="PER_STREAM_QUEUE=1"):
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=16))


def test_per_stream_queues_clear_the_multi_row_decode_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = npu_decode_graph(
        monkeypatch, backend=CudaGraphBackend.FULL, max_bs=16, per_stream_queue="1"
    )

    builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=16))


def test_an_eager_or_single_row_decode_graph_needs_no_per_stream_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = ((CudaGraphBackend.DISABLED, 16), (CudaGraphBackend.FULL, 1))
    for backend, max_bs in cases:
        builder = npu_decode_graph(
            monkeypatch, backend=backend, max_bs=max_bs, per_stream_queue=None
        )
        builder.validate_before_infrastructure(SimpleNamespace(max_running_requests=16))
