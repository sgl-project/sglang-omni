# SPDX-License-Identifier: Apache-2.0

import inspect

import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.auk.config import AuKPipelineConfig
from sglang_omni.models.auk.stages import (
    create_auk_engine_executor,
    create_conditioning_executor,
    scheduler,
)


def test_conditioning_idle_wait_is_opt_in():
    parameter = inspect.signature(create_conditioning_executor).parameters[
        "batch_wait_when_idle"
    ]
    assert parameter.default is False
    assert (
        "batch_wait_when_idle"
        not in inspect.signature(create_auk_engine_executor).parameters
    )

    stages = AuKPipelineConfig.model_fields["stages"].default
    by_name = {stage.name: stage for stage in stages}
    assert by_name["conditioning"].factory.batch_wait_when_idle is False
    assert "batch_wait_when_idle" not in (
        by_name["auk_engine"].factory.model_extra or {}
    )


def test_scheduler_forwards_idle_wait_policy():
    instance = scheduler(lambda payloads: payloads, torch.device("cpu"), 16, 10)
    assert instance._batch_wait_when_idle is False

    instance = scheduler(
        lambda payloads: payloads,
        torch.device("cpu"),
        16,
        10,
        batch_wait_when_idle=True,
    )
    assert instance._batch_wait_when_idle is True
    assert instance._max_batch_wait_s == 0.01


def test_conditioning_idle_wait_can_be_enabled_from_cli():
    config = AuKPipelineConfig(model_path="model")
    manager = ConfigManager(config)
    patches = manager.parse_extra_args(
        ["--conditioning.factory.batch_wait_when_idle", "true"]
    )
    merged = manager.merge_config(patches)
    stages = {stage.name: stage for stage in merged.stages}

    assert stages["conditioning"].factory.batch_wait_when_idle is True
    assert "batch_wait_when_idle" not in (
        stages["auk_engine"].factory.model_extra or {}
    )
