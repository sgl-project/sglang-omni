# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sglang_omni.config.placement import build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import (
    ProcessTopologyPlan,
    build_process_topology_plan,
    compile_logical_processes,
)
from sglang_omni.pipeline.replicas import expand_replica_stages
from sglang_omni.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    fake_factory_path,
)

FACTORY = fake_factory_path("make_scheduler")

if TYPE_CHECKING:
    from sglang_omni.pipeline.stage.runtime import Stage


def stage(name: str, **kwargs: Any) -> StageConfig:
    kwargs.setdefault("factory", FACTORY)
    if kwargs.get("tp_size", 1) == 1:
        kwargs.setdefault("process", "pipeline")
    return StageConfig(name=name, **kwargs)


def build_compiled_process_topology(
    config: PipelineConfig,
) -> ProcessTopologyPlan:
    logical_plan, stages = compile_logical_processes(config)
    stages, replica_topology = expand_replica_stages(stages, logical_plan)
    placement = build_stage_placement_plan(
        config,
        stages_cfg=stages,
        replica_instances=replica_topology.replicas,
    )
    return build_process_topology_plan(
        config,
        placement,
        stages_cfg=stages,
    )


def make_stage(
    *,
    name: str = "stage",
    role: str = "single",
    get_next=None,
    endpoints: dict[str, str] | None = None,
    gpu_id: int | None = None,
    scheduler: FakeScheduler | None = None,
    relay: FakeRelay | None = None,
    control_plane: RecordingStageControlPlane | None = None,
    **kwargs: Any,
) -> Stage:
    from sglang_omni.pipeline.stage.runtime import Stage

    return Stage(
        name=name,
        role=role,
        get_next=get_next or (lambda request_id, output: None),
        gpu_id=gpu_id,
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
