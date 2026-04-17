# SPDX-License-Identifier: Apache-2.0
"""Multi-process pipeline runner.

Spawns each pipeline stage (possibly with multiple TP ranks) in its own OS
process(es).  The main process runs only the Coordinator.

Architecture
------------
``PipelineConfig`` → ``_build_stage_groups()`` → ``list[StageGroup]``

Each :class:`StageGroup` holds *tp_size* :class:`StageProcessSpec` objects
(one per TP rank) and manages the corresponding OS processes.  The
:class:`MultiProcessPipelineRunner` orchestrates startup, monitoring, and
shutdown across all groups.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import socket
from typing import Any

from sglang_omni.config.compiler import (
    _allocate_endpoints,
    _build_relay_config,
    _detect_same_gpu_targets,
)
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.pipeline import Coordinator
from sglang_omni.pipeline.stage_group import StageGroup
from sglang_omni.pipeline.stage_process import StageProcessSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec builder — turns PipelineConfig into a list of StageGroups
# ---------------------------------------------------------------------------


def _build_stage_groups(config: PipelineConfig) -> list[StageGroup]:
    """Compile *config* into one :class:`StageGroup` per logical stage.

    This runs in the **main process** so that subprocesses never need to
    re-compile the pipeline configuration.
    """
    stages_cfg, name_map, _ = config.apply_fusion()
    endpoints = _allocate_endpoints(config, stages=stages_cfg)
    stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}
    cfg_map = {s.name: s for s in stages_cfg}

    # Pre-compute which stages receive streams
    stream_receivers: set[str] = set()
    for scfg in stages_cfg:
        for target in scfg.stream_to:
            stream_receivers.add(target)

    nccl_port_counter = _NcclPortAllocator()

    groups: list[StageGroup] = []
    for stage_cfg in stages_cfg:
        tp_size = stage_cfg.tp_size
        gpu_ids = _resolve_gpu_ids(stage_cfg, config)
        nccl_port = nccl_port_counter.allocate() if tp_size > 1 else None

        # Pre-resolve stream targets
        same_gpu_targets: set[str] = set()
        if stage_cfg.stream_to:
            same_gpu_targets = _detect_same_gpu_targets(
                stage_cfg,
                stage_cfg.stream_to,
                gpu_placement=config.gpu_placement,
                cfg_map=cfg_map,
            )

        # Pre-resolve factory args (inject model_path, gpu_id)
        base_factory_args = _resolve_factory_args(stage_cfg, config)

        specs: list[StageProcessSpec] = []
        for tp_rank in range(tp_size):
            gpu_id = gpu_ids[tp_rank] if tp_rank < len(gpu_ids) else gpu_ids[0]

            # Per-rank factory args: override gpu_id and inject tp_rank/tp_size
            factory_args = dict(base_factory_args)
            if "gpu_id" in base_factory_args:
                factory_args["gpu_id"] = gpu_id
            if tp_size > 1:
                factory_args["tp_rank"] = tp_rank
                factory_args["tp_size"] = tp_size

            # Per-rank relay config
            relay_config = _resolve_relay_config(stage_cfg, config, gpu_id=gpu_id)

            specs.append(
                StageProcessSpec(
                    stage_name=stage_cfg.name,
                    tp_rank=tp_rank,
                    tp_size=tp_size,
                    gpu_id=gpu_id,
                    nccl_port=nccl_port,
                    factory=stage_cfg.factory,
                    factory_args=factory_args,
                    next_stages=stage_cfg.next,
                    is_terminal=stage_cfg.terminal,
                    wait_for=stage_cfg.wait_for,
                    merge_fn=stage_cfg.merge_fn,
                    relay_config=relay_config,
                    recv_endpoint=stage_endpoints[stage_cfg.name],
                    coordinator_endpoint=endpoints["completion"],
                    abort_endpoint=endpoints["abort"],
                    stage_endpoints=stage_endpoints,
                    stream_targets=list(stage_cfg.stream_to),
                    same_gpu_targets=same_gpu_targets,
                    is_stream_receiver=stage_cfg.name in stream_receivers,
                    name_map=name_map,
                )
            )

        groups.append(StageGroup(stage_cfg.name, specs))

    return groups


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_gpu_ids(stage_cfg: StageConfig, config: PipelineConfig) -> list[int]:
    """Return the list of GPU ids for *stage_cfg* (one per TP rank)."""
    placement = config.gpu_placement.get(stage_cfg.name)
    if placement is None:
        return [0] * stage_cfg.tp_size
    if isinstance(placement, int):
        return [placement] * stage_cfg.tp_size
    # list[int] — one gpu per tp rank
    if len(placement) != stage_cfg.tp_size:
        raise ValueError(
            f"Stage {stage_cfg.name!r}: gpu_placement has {len(placement)} "
            f"entries but tp_size={stage_cfg.tp_size}"
        )
    return list(placement)


def _resolve_factory_args(
    stage_cfg: StageConfig, config: PipelineConfig
) -> dict[str, Any]:
    """Pre-resolve factory args: inject model_path and gpu_id if accepted."""
    from sglang_omni.utils import import_string

    args = dict(stage_cfg.factory_args)
    factory = import_string(stage_cfg.factory)
    sig = inspect.signature(factory)

    if "model_path" in sig.parameters and "model_path" not in args:
        args["model_path"] = config.model_path

    if "gpu_id" in sig.parameters and "gpu_id" not in args:
        placement = config.gpu_placement.get(stage_cfg.name, 0)
        gpu_id = placement[0] if isinstance(placement, list) else placement
        args["gpu_id"] = gpu_id

    return args


def _resolve_relay_config(
    stage_cfg: StageConfig,
    config: PipelineConfig,
    *,
    gpu_id: int,
) -> dict[str, Any]:
    """Build relay config, overriding gpu_id from placement."""
    relay_config = _build_relay_config(stage_cfg, config)
    # Override gpu_id for GPU stages
    if stage_cfg.gpu is not None:
        relay_config["gpu_id"] = gpu_id
    return relay_config


class _NcclPortAllocator:
    """Allocate unique NCCL ports for per-stage TP groups."""

    def __init__(self, base_port: int = 29500):
        self._next = base_port

    def allocate(self) -> int:
        """Return an available port, incrementing the counter."""
        while True:
            port = self._next
            self._next += 1
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue


# ---------------------------------------------------------------------------
# MultiProcessPipelineRunner  (public API — kept compatible)
# ---------------------------------------------------------------------------


class MultiProcessPipelineRunner:
    """Run each pipeline stage in its own OS process(es).

    Main process runs only the Coordinator.  Each logical stage is backed
    by a :class:`StageGroup` that may contain multiple processes when
    ``tp_size > 1``.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._coordinator: Coordinator | None = None
        self._groups: list[StageGroup] = []
        self._completion_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._started = False

    @property
    def coordinator(self) -> Coordinator:
        if self._coordinator is None:
            raise RuntimeError("Runner not started")
        return self._coordinator

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(self, timeout: float = 120.0) -> None:
        if self._started:
            raise RuntimeError("Already started")

        try:
            groups = _build_stage_groups(self._config)

            # Coordinator
            stages_cfg, _, entry_stage = self._config.apply_fusion()
            endpoints = _allocate_endpoints(self._config, stages=stages_cfg)

            self._coordinator = Coordinator(
                completion_endpoint=endpoints["completion"],
                abort_endpoint=endpoints["abort"],
                entry_stage=entry_stage,
                terminal_stages=self._config.terminal_stages or None,
            )
            await self._coordinator.start()
            self._completion_task = asyncio.create_task(
                self._coordinator.run_completion_loop()
            )

            # Spawn all stage groups
            ctx = multiprocessing.get_context("spawn")
            for group in groups:
                group.spawn(ctx)
            self._groups = groups

            # Wait for all groups to be ready
            await asyncio.gather(*(g.wait_ready(timeout) for g in self._groups))

            # Verify no early deaths
            for group in self._groups:
                if group.any_dead():
                    raise RuntimeError(
                        f"Stage process(es) died during startup: "
                        f"{group.dead_summary()}"
                    )

            # Register stages (coordinator talks to rank 0 of each group)
            for group in self._groups:
                self._coordinator.register_stage(
                    group.stage_name, group.leader_endpoint
                )

            self._started = True
            self._monitor_task = asyncio.create_task(self._monitor_children())

            total_procs = sum(g.tp_size for g in self._groups)
            logger.info(
                "MultiProcessPipelineRunner started: %d stage(s), %d process(es)",
                len(self._groups),
                total_procs,
            )

        except Exception:
            await self._cleanup_on_failure()
            raise

    # ------------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------------

    async def _monitor_children(self) -> None:
        while self._started:
            for group in self._groups:
                if group.any_dead():
                    logger.error(
                        "Dead stage process(es) detected: %s",
                        group.dead_summary(),
                    )
                    await self.stop()
                    return
            await asyncio.sleep(5.0)

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False

        if self._monitor_task is not None:
            current = asyncio.current_task()
            if current != self._monitor_task:
                self._monitor_task.cancel()
            self._monitor_task = None

        # Send shutdown to stages via coordinator
        try:
            await self._coordinator.shutdown_stages()
        except Exception as e:
            logger.warning("shutdown_stages error: %s", e)

        # Shutdown all groups
        await asyncio.gather(
            *(g.shutdown() for g in self._groups),
            return_exceptions=True,
        )

        if self._completion_task is not None:
            self._completion_task.cancel()
            try:
                await self._completion_task
            except asyncio.CancelledError:
                pass

        await self._coordinator.stop()
        self._groups.clear()

    # ------------------------------------------------------------------
    # Internal cleanup
    # ------------------------------------------------------------------

    async def _cleanup_on_failure(self) -> None:
        """Best-effort cleanup after a failed start()."""
        for group in self._groups:
            for p in group.processes:
                if p.is_alive():
                    p.terminate()
            for p in group.processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
        self._groups.clear()

        if self._completion_task is not None:
            self._completion_task.cancel()
            try:
                await self._completion_task
            except asyncio.CancelledError:
                pass
            self._completion_task = None

        if self._coordinator is not None:
            try:
                await self._coordinator.stop()
            except Exception:
                pass
            self._coordinator = None
