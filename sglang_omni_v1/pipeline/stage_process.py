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

from sglang_omni_v1.pipeline.control_plane import StageControlPlane
from sglang_omni_v1.pipeline.stage.input import AggregatedInput, DirectInput
from sglang_omni_v1.pipeline.stage.runtime import Stage
from sglang_omni_v1.pipeline.stage.stream_queue import StreamQueue
from sglang_omni_v1.pipeline.tp_control import TPFollowerControlPlane, TPLeaderFanout
from sglang_omni_v1.utils import import_string


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
    gpu_id: int = 0
    nccl_port: int | None = None

    # Factory
    factory: str = ""
    factory_args: dict[str, Any] = field(default_factory=dict)

    # Routing: static next stage(s)
    next_stages: str | list[str] | None = None
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
    same_gpu_targets: set[str] = field(default_factory=set)
    is_stream_receiver: bool = False

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


def stage_process_main(
    spec: StageProcessSpec,
    ready_event: multiprocessing.Event,
) -> None:
    """Subprocess entrypoint: construct a Stage from *spec* and run it."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    tp_suffix = f"-tp{spec.tp_rank}" if spec.tp_size > 1 else ""
    log = logging.getLogger(f"stage.{spec.stage_name}{tp_suffix}")

    try:
        _prepare_cuda_environment(spec, log)
        _run_stage(spec, ready_event, log)
    except Exception:
        import traceback

        log.error("Stage process failed:\n%s", traceback.format_exc())
        sys.exit(1)


def _run_stage(
    spec: StageProcessSpec,
    ready_event: multiprocessing.Event,
    log: logging.Logger,
) -> None:

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

    factory = import_string(spec.factory)
    scheduler = factory(**spec.factory_args)

    # --- Build routing ---
    if spec.is_terminal:
        get_next = lambda request_id, output: None
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
        same_gpu_targets=spec.same_gpu_targets or None,
        tp_fanout=tp_fanout,
    )

    if spec.is_stream_receiver:
        stage._stream_queue = StreamQueue(max_pending=4096)

    # --- Run ---
    async def _start_and_run():
        await stage.start()
        log.info("Stage %s (tp_rank=%d) ready", spec.stage_name, spec.tp_rank)
        ready_event.set()
        await stage.run()

    asyncio.run(_start_and_run())


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
    if original_visible:
        visible_devices = [item.strip() for item in original_visible.split(",")]
        if len(visible_devices) == 1:
            mapped_gpu = visible_devices[0]
        elif spec.gpu_id >= len(visible_devices):
            raise ValueError(
                f"tp stage {spec.stage_name!r} assigned gpu_id={spec.gpu_id}, "
                f"but CUDA_VISIBLE_DEVICES only exposes {visible_devices}"
            )
        else:
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
    env_updates = get_stage_process_env(spec)
    if not env_updates:
        return

    mapped_gpu = env_updates["CUDA_VISIBLE_DEVICES"]
    for key, value in env_updates.items():
        os.environ[key] = value

    if "gpu_id" in spec.factory_args:
        spec.factory_args["gpu_id"] = 0
    if "gpu_id" in spec.relay_config:
        spec.relay_config["gpu_id"] = 0
    spec.gpu_id = 0

    log.info(
        "Mapped TP stage %s rank %d to CUDA_VISIBLE_DEVICES=%s (local gpu_id=0)",
        spec.stage_name,
        spec.tp_rank,
        mapped_gpu,
    )
