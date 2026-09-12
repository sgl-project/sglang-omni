# SPDX-License-Identifier: Apache-2.0

import inspect

import torch

from sglang_omni.models.auk.config import AuKPipelineConfig
from sglang_omni.models.auk.stages import (
    _scheduler,
    create_auk_engine_executor,
    create_conditioning_executor,
)


def test_batch_wait_is_opt_in_for_conditioning_and_engine():
    for factory in (create_conditioning_executor, create_auk_engine_executor):
        parameter = inspect.signature(factory).parameters["batch_wait_when_idle"]
        assert parameter.default is False

    stages = AuKPipelineConfig.model_fields["stages"].default
    for name in ("conditioning", "auk_engine"):
        stage = next(stage for stage in stages if stage.name == name)
        assert stage.factory.batch_wait_when_idle is False


def test_scheduler_forwards_idle_wait_policy():
    scheduler = _scheduler(lambda payloads: payloads, torch.device("cpu"), 16, 10)
    assert scheduler._batch_wait_when_idle is False

    scheduler = _scheduler(
        lambda payloads: payloads,
        torch.device("cpu"),
        16,
        10,
        batch_wait_when_idle=True,
    )
    assert scheduler._batch_wait_when_idle is True
    assert scheduler._max_batch_wait_s == 0.01
