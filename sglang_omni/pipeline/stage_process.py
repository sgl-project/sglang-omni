# SPDX-License-Identifier: Apache-2.0
"""Subprocess specification and entrypoint for pipeline stages.

The main process builds a fully-resolved :class:`StageProcessSpec` for each
subprocess so the child never re-compiles the pipeline config.
"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Picklable specification — built in the main process
# ---------------------------------------------------------------------------


@dataclass
class StageProcessSpec:
    """Everything a stage subprocess needs — no re-compilation required.

    All string references (factory, merge_fn) are dotted import
    paths resolved by the child via :func:`import_string`.
    """

    # Identity
    stage_name: str
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


# ---------------------------------------------------------------------------
# Subprocess entrypoint
# ---------------------------------------------------------------------------


def stage_process_main(
    spec: StageProcessSpec,
    ready_event: multiprocessing.Event,
) -> None:
    """Subprocess entrypoint: construct a Stage from *spec* and run it.

    No pipeline re-compilation happens here — *spec* already contains every
    resolved parameter the Stage needs.
    """
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    tp_suffix = f"-tp{spec.tp_rank}" if spec.tp_size > 1 else ""
    log = logging.getLogger(f"stage.{spec.stage_name}{tp_suffix}")

    try:
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
    from sglang_omni.pipeline.stage.input import AggregatedInput, DirectInput
    from sglang_omni.pipeline.stage.runtime import Stage
    from sglang_omni.pipeline.stage.stream_queue import StreamQueue
    from sglang_omni.utils import import_string

    gpu_id = spec.relay_config.get("gpu_id")
    if gpu_id is None:
        gpu_id = spec.factory_args.get("gpu_id")
    if gpu_id is not None:
        import torch

        torch.cuda.set_device(int(gpu_id))
        log.info("Set current CUDA device to %s for stage %s", gpu_id, spec.stage_name)

    # --- TP: initialise torch.distributed for this stage's NCCL group ---
    if spec.tp_size > 1:
        _init_torch_distributed(spec, log)

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

    # --- Construct Stage ---
    stage = Stage(
        name=spec.stage_name,
        get_next=get_next,
        gpu_id=spec.gpu_id,
        recv_endpoint=spec.recv_endpoint,
        coordinator_endpoint=spec.coordinator_endpoint,
        abort_endpoint=spec.abort_endpoint,
        endpoints=spec.stage_endpoints,
        input_handler=input_handler,
        relay_config=spec.relay_config,
        scheduler=scheduler,
        stream_targets=spec.stream_targets or None,
        same_gpu_targets=spec.same_gpu_targets or None,
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


def _init_torch_distributed(spec: StageProcessSpec, log: logging.Logger) -> None:
    """Initialise a per-stage NCCL process group for tensor parallelism."""
    import os

    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    if spec.nccl_port is not None:
        os.environ["MASTER_PORT"] = str(spec.nccl_port)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.gpu_id)
    torch.cuda.set_device(0)  # after CUDA_VISIBLE_DEVICES, local device is 0

    log.info(
        "Initialising torch.distributed for stage=%s tp_rank=%d/%d nccl_port=%s",
        spec.stage_name,
        spec.tp_rank,
        spec.tp_size,
        spec.nccl_port,
    )
    dist.init_process_group(
        backend="nccl",
        world_size=spec.tp_size,
        rank=spec.tp_rank,
    )
