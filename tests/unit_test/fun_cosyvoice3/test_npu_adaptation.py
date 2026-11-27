# SPDX-License-Identifier: Apache-2.0
"""NPU platform adaptation tests for Fun-CosyVoice3."""

from __future__ import annotations

import pytest

import sglang_omni.models.fun_cosyvoice3.config as config_module
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig

MODEL_PATH = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


class FakePlatform:
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type

    def is_npu(self) -> bool:
        return self.device_type == "npu"


def npu_config(monkeypatch: pytest.MonkeyPatch) -> FunCosyVoice3PipelineConfig:
    monkeypatch.setattr(config_module, "current_platform", FakePlatform("npu"))
    return FunCosyVoice3PipelineConfig(model_path=MODEL_PATH)


def test_the_npu_gets_per_stream_host_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared Host queue deadlocks the scheduler once a decode graph replays."""
    config = npu_config(monkeypatch)

    assert config.resolved_stage_env_defaults("tts_engine")["PER_STREAM_QUEUE"] == "1"


def test_stages_without_an_engine_get_no_per_stream_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the engine starts graphs, so the other stages must stay untouched."""
    config = npu_config(monkeypatch)

    assert "PER_STREAM_QUEUE" not in config.resolved_stage_env_defaults("preprocessing")


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
