# SPDX-License-Identifier: Apache-2.0
"""A pipeline sets its startup budget; SGLANG_OMNI_STARTUP_TIMEOUT overrides it."""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig
from sglang_omni.serve.launcher import startup_timeout_s


class SlowStartPipelineConfig(PipelineConfig):
    startup_timeout_s: ClassVar[float] = 3600.0


def make(config_cls: type[PipelineConfig]) -> PipelineConfig:
    # The budget is a class attribute; skip validating a real stage list.
    return config_cls.__new__(config_cls)


def test_pipelines_wait_ten_minutes_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STARTUP_TIMEOUT", raising=False)
    assert startup_timeout_s(make(PipelineConfig)) == 600.0


def test_a_pipeline_can_raise_its_budget(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STARTUP_TIMEOUT", raising=False)
    assert startup_timeout_s(make(SlowStartPipelineConfig)) == 3600.0


def test_the_environment_overrides_the_pipeline(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_STARTUP_TIMEOUT", "90")
    assert startup_timeout_s(make(SlowStartPipelineConfig)) == 90.0
