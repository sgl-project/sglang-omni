# SPDX-License-Identifier: Apache-2.0
"""Subprocess specification and entrypoint for pipeline stages.

"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from sglang_omni.pipeline.control_plane import StageControlPlane
from sglang_omni.pipeline.stage.input import AggregatedInput, DirectInput
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.pipeline.tp_control import TPFollowerControlPlane, TPLeaderFanout
from sglang_omni.utils.gpu_memory import gpu_startup_lock
from sglang_omni.utils.imports import import_string


@dataclass
class StageProcessSpec:
    """Everything a stage subprocess needs — no re-compilation required.

    All string references (factory, merge_fn) are dotted import
    paths resolved by the child via :func:`import_string`.
    """

    # Identity
    stage_name: str
    role: Literal["single", "leader", "follower"] = "single"
    tp_rank: int = 0
    tp_size: int = 1
    gpu_id: int | None = None
    nccl_port: int | None = None

    # Factory
    factory: str = ""
    factory_args: dict[str, Any] = field(default_factory=dict)

    # Routing: static next stage(s)
    next_stages: str | list[str] | None = None
    route_fn: str | None = None
    is_terminal: bool = False

    # Fan-in
    wait_for: list[str] | None = None
    merge_fn: str | None = None
    project_payload: dict[str, str] = field(default_factory=dict)

    # Relay
    relay_config: dict[str, Any] = field(default_factory=dict)

    # Endpoints
    recv_endpoint: str = ""
    coordinator_endpoint: str = ""
    abort_endpoint: str = ""
    stage_endpoints: dict[str, str] = field(default_factory=dict)

    # Stream wiring
    stream_targets: list[str] = field(default_factory=list)
    stream_done_to_fn: str | None = None
    same_gpu_targets: set[str] = field(default_factory=set)
    is_stream_receiver: bool = False
    can_accept_stream_before_payload: bool = False

    # Fusion name map
    name_map: dict[str, str] = field(default_factory=dict)

    # TP internal control (leader -> followers)
    follower_work_queues: list[Any] = field(default_factory=list)
    follower_abort_queues: list[Any] = field(default_factory=list)
    internal_work_queue: Any | None = None
    internal_abort_queue: Any | None = None

    @property
    def owns_external_io(self) -> bool:
        return self.role in {"single", "leader"}

    @property
    def is_leader(self) -> bool:
        return self.role == "leader"

    @property
    def is_follower(self) -> bool:
        return self.role == "follower"


@dataclass
class StageWorkerProcessSpec:
    """Everything one OS process needs to run one or more stages."""

    process_name: str
    stage_specs: list[StageProcessSpec]
    gpu_id: int | None = None


def stage_process_main(
    spec: StageWorkerProcessSpec,
    ready_event: multiprocessing.Event,
) -> None:
    """Subprocess entrypoint: construct stage(s) from *spec* and run them."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if not spec.stage_specs:
        raise ValueError(f"Process {spec.process_name!r} requires at least one stage")
    log = logging.getLogger(f"stage_process.{spec.process_name}")

    try:
        for stage_spec in spec.stage_specs:
            _prepare_cuda_environment(stage_spec, log)
        _run_process(spec, ready_event, log)
    except Exception:
        import traceback

        log.error(
            "Stage process %s failed:\n%s",
            spec.process_name,
            traceback.format_exc(),
        )
        sys.exit(1)


def _run_process(
    spec: StageWorkerProcessSpec,
    ready_event: multiprocessing.Event,
    log: logging.Logger,
) -> None:
    """Construct and drive all stages owned by one OS process.

    Multi-stage semantics (since the topology PR):
    - All stages in ``spec.stage_specs`` share this OS process and one asyncio
      event loop. ``asyncio.gather`` runs them concurrently; **if any stage
      raises, the whole process exits** and ``MultiProcessPipelineRunner``'s
      ``_monitor_children`` will fail-all in-flight requests on the
      coordinator. There is no per-stage failure isolation inside one process
      group.
    - Scheduler construction is serialized by :func:`gpu_startup_lock` per GPU
      inside :func:`_construct_scheduler` — so when N stages on the same GPU
      live in this process, cold-start time degrades from ``max`` to ``sum``
      across them.
    """
    stages = [_construct_stage(stage_spec, log) for stage_spec in spec.stage_specs]

    async def _start_and_run():
        tasks: list[asyncio.Task] = []
        try:
            for stage in stages:
                await stage.start()
            log.info(
                "Process %s ready with stages=%s",
                spec.process_name,
                [stage.name for stage in stages],
            )
            ready_event.set()
            tasks = [asyncio.create_task(stage.run()) for stage in stages]
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            for stage in stages:
                if getattr(stage, "_running", False):
                    await stage.stop()

    asyncio.run(_start_and_run())


def _construct_stage(
    spec: StageProcessSpec,
    log: logging.Logger,
) -> Stage:
    gpu_id = spec.relay_config.get("gpu_id")
    if gpu_id is None:
        gpu_id = spec.factory_args.get("gpu_id")
    if gpu_id is None and _factory_args_use_cuda(spec.factory_args):
        gpu_id = spec.gpu_id
    if gpu_id is not None:
        import torch

        torch.cuda.set_device(int(gpu_id))
        log.info("Set current CUDA device to %s for stage %s", gpu_id, spec.stage_name)

    # --- Build scheduler via factory ---
    log.info(
        "Building scheduler for %s (tp_rank=%d/%d) ...",
        spec.stage_name,
        spec.tp_rank,
        spec.tp_size,
    )

    scheduler = _construct_scheduler(spec, gpu_id, log)

    def _target_list(targets: str | list[str] | None) -> list[str]:
        if targets is None:
            return []
        if isinstance(targets, str):
            return [targets]
        if isinstance(targets, list):
            return list(targets)
        raise ValueError(
            f"Dynamic route function for stage {spec.stage_name!r} returned "
            f"unsupported target value {targets!r}"
        )

    def _map_target_list(targets: str | list[str] | None) -> list[str]:
        return [spec.name_map.get(t, t) for t in _target_list(targets)]

    def _target_result(
        targets: str | list[str] | None,
        *,
        allowed_targets: set[str],
        allow_empty: bool,
        hook_name: str,
    ) -> str | list[str] | None:
        mapped_targets = _map_target_list(targets)
        if not mapped_targets:
            if allow_empty:
                return None
            raise ValueError(
                f"{hook_name} for stage {spec.stage_name!r} returned no targets; "
                "dynamic route functions must return downstream stage(s)"
            )
        unknown = set(mapped_targets) - allowed_targets
        if unknown:
            raise ValueError(
                f"{hook_name} for stage {spec.stage_name!r} returned targets "
                f"outside the static topology: {sorted(unknown)}. "
                f"Allowed targets: {sorted(allowed_targets)}"
            )
        return mapped_targets[0] if isinstance(targets, str) else mapped_targets

    # --- Build routing ---
    if spec.is_terminal:
        get_next = lambda request_id, output: None
    elif spec.route_fn:
        route_fn = import_string(spec.route_fn)
        allowed_route_targets = set(_map_target_list(spec.next_stages))

        def get_next(request_id, output, _fn=route_fn):
            return _target_result(
                _fn(request_id, output),
                allowed_targets=allowed_route_targets,
                allow_empty=False,
                hook_name="route_fn",
            )

    else:
        target = spec.next_stages
        if isinstance(target, str):
            mapped = spec.name_map.get(target, target)
            get_next = lambda request_id, output, _t=mapped: _t
        elif isinstance(target, list):
            mapped = [spec.name_map.get(t, t) for t in target]
            get_next = lambda request_id, output, _t=mapped: _t
        else:
            get_next = lambda request_id, output: None

    if spec.stream_done_to_fn:
        stream_done_to_fn = import_string(spec.stream_done_to_fn)
        allowed_stream_targets = set(_map_target_list(spec.stream_targets))
        get_stream_done_targets = (
            lambda request_id, output, _fn=stream_done_to_fn: _target_result(
                _fn(request_id, output),
                allowed_targets=allowed_stream_targets,
                allow_empty=True,
                hook_name="stream_done_to_fn",
            )
        )
    else:
        get_stream_done_targets = None

    # --- Build input handler ---
    if spec.wait_for and spec.merge_fn:
        merge_fn = import_string(spec.merge_fn)
        sources = {spec.name_map.get(n, n) for n in spec.wait_for}
        input_handler = AggregatedInput(sources=sources, merge=merge_fn)
    else:
        input_handler = DirectInput()
    project_payload = {
        target: import_string(dotted_path)
        for target, dotted_path in spec.project_payload.items()
    }

    if spec.owns_external_io:
        control_plane = StageControlPlane(
            stage_name=spec.stage_name,
            recv_endpoint=spec.recv_endpoint,
            coordinator_endpoint=spec.coordinator_endpoint,
            abort_endpoint=spec.abort_endpoint,
        )
    else:
        control_plane = TPFollowerControlPlane(
            stage_name=spec.stage_name,
            recv_endpoint=spec.recv_endpoint,
            work_queue=spec.internal_work_queue,
            abort_queue=spec.internal_abort_queue,
        )

    tp_fanout = None
    if spec.is_leader:
        tp_fanout = TPLeaderFanout(
            stage_name=spec.stage_name,
            follower_work_queues=spec.follower_work_queues,
            follower_abort_queues=spec.follower_abort_queues,
        )

    # --- Construct Stage ---
    stage = Stage(
        name=spec.stage_name,
        role=spec.role,
        get_next=get_next,
        gpu_id=spec.gpu_id,
        endpoints=spec.stage_endpoints,
        control_plane=control_plane,
        input_handler=input_handler,
        relay_config=spec.relay_config,
        scheduler=scheduler,
        project_payload=project_payload or None,
        stream_targets=spec.stream_targets or None,
        get_stream_done_targets=get_stream_done_targets,
        same_gpu_targets=spec.same_gpu_targets or None,
        can_accept_stream_before_payload=spec.can_accept_stream_before_payload,
        tp_fanout=tp_fanout,
        is_terminal=spec.is_terminal,
    )

    if spec.is_stream_receiver:
        stage._stream_queue = StreamQueue(max_pending=4096)

    return stage


def _construct_scheduler(
    spec: StageProcessSpec,
    gpu_id: int | None,
    log: logging.Logger,
) -> Any:
    """Build a scheduler, serializing GPU factory work per visible device."""

    factory = import_string(spec.factory)
    if gpu_id is None:
        return factory(**spec.factory_args)

    with gpu_startup_lock(int(gpu_id)) as lock_path:
        log.info(f"Acquired GPU startup lock for stage {spec.stage_name}: {lock_path}")
        return factory(**spec.factory_args)


def _factory_args_use_cuda(factory_args: Mapping[str, Any]) -> bool:
    for value in factory_args.values():
        if isinstance(value, str) and value.startswith("cuda"):
            return True
    return False


def get_stage_process_env(
    spec: StageProcessSpec,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return per-process env overrides needed before TP child startup."""
    if spec.tp_size <= 1:
        return {}

    source_env = env if env is not None else os.environ
    original_visible = source_env.get("CUDA_VISIBLE_DEVICES")
    if spec.gpu_id is None:
        raise ValueError(f"tp stage {spec.stage_name!r} requires a GPU id")
    if original_visible:
        visible_devices = [item.strip() for item in original_visible.split(",")]
        if spec.gpu_id >= len(visible_devices):
            raise ValueError(
                f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                f"but CUDA_VISIBLE_DEVICES only exposes {visible_devices}"
            )
        mapped_gpu = visible_devices[spec.gpu_id]
    else:
        mapped_gpu = str(spec.gpu_id)

    return {
        "CUDA_VISIBLE_DEVICES": mapped_gpu,
        "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "true",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "false",
    }


def _prepare_cuda_environment(
    spec: StageProcessSpec,
    log: logging.Logger,
) -> None:
    """Map TP rank processes to one visible CUDA device before torch init."""
    if os.environ.get("SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS") == "true":
        mapped_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", str(spec.gpu_id))
        _normalize_spec_gpu_id_to_local_device(spec)
        log.info(
            "TP stage %s rank %d sees CUDA_VISIBLE_DEVICES=%s (local gpu_id=0)",
            spec.stage_name,
            spec.tp_rank,
            mapped_gpu,
        )
        return

    env_updates = get_stage_process_env(spec)
    if not env_updates:
        return

    mapped_gpu = env_updates["CUDA_VISIBLE_DEVICES"]
    for key, value in env_updates.items():
        os.environ[key] = value

    _normalize_spec_gpu_id_to_local_device(spec)
    log.info(
        "Mapped TP stage %s rank %d to CUDA_VISIBLE_DEVICES=%s (local gpu_id=0)",
        spec.stage_name,
        spec.tp_rank,
        mapped_gpu,
    )


def _normalize_spec_gpu_id_to_local_device(spec: StageProcessSpec) -> None:
    if "gpu_id" in spec.factory_args:
        spec.factory_args["gpu_id"] = 0
    if "gpu_id" in spec.relay_config:
        spec.relay_config["gpu_id"] = 0
    spec.gpu_id = 0
