# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from sglang_omni_v1.config.schema import StageConfig
from sglang_omni_v1.pipeline.stage.runtime import Stage
from sglang_omni_v1.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    fake_factory_path,
)

FACTORY = fake_factory_path("make_scheduler")


def stage(name: str, **kwargs: Any) -> StageConfig:
    kwargs.setdefault("factory", FACTORY)
    return StageConfig(name=name, **kwargs)


def make_stage(
    *,
    name: str = "stage",
    role: str = "single",
    get_next=None,
    endpoints: dict[str, str] | None = None,
    scheduler: FakeScheduler | None = None,
    relay: FakeRelay | None = None,
    control_plane: RecordingStageControlPlane | None = None,
    **kwargs: Any,
) -> Stage:
    return Stage(
        name=name,
        role=role,
        get_next=get_next or (lambda request_id, output: None),
        gpu_id=None,
        endpoints=endpoints or {},
        control_plane=control_plane or RecordingStageControlPlane(),
        relay=relay or FakeRelay(),
        scheduler=scheduler or FakeScheduler(),
        **kwargs,
    )


def run_scheduler(
    scheduler: Any,
    messages: list[IncomingMessage],
    *,
    output_count: int,
    before_collect: Callable[[], None] | None = None,
) -> list[Any]:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for message in messages:
            scheduler.inbox.put(message)
        if before_collect is not None:
            before_collect()
        return [scheduler.outbox.get(timeout=2.0) for _ in range(output_count)]
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)
