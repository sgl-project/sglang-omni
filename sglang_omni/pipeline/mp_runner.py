# SPDX-License-Identifier: Apache-2.0
"""Multi-process pipeline runner.

The runner owns the single serving path. It can start one OS process containing
multiple non-TP stages, multiple OS processes on the same GPU, and the existing
one-process-per-rank TP topology.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import socket
from typing import Any

from sglang_omni.config.placement import (
    StagePlacementPlan,
    resolve_same_gpu_stream_targets,
    resolve_stage_gpu_ids,
)
from sglang_omni.config.runtime import (
    StageLaunchMode,
    build_stage_launch_modes,
    resolve_stage_factory_args,
)
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import ProcessTopologyPlan
from sglang_omni.pipeline import Coordinator
from sglang_omni.pipeline.runtime_config import (
    IpcRuntimeDir,
    PipelineRuntimePrep,
    build_relay_config,
    prepare_pipeline_runtime,
)
from sglang_omni.pipeline.stage_group import StageGroup
from sglang_omni.pipeline.stage_process import StageProcessSpec, StageWorkerProcessSpec
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)

_TP_LAUNCH_PARAMS = frozenset({"tp_rank", "tp_size", "nccl_port"})


def any_sglang_backend_stage(config: PipelineConfig) -> bool:
    return any(
        mode.requires_sglang_launch
        for mode in build_stage_launch_modes(config).values()
    )


def _run_tp_preflight(
    stages_cfg: list[StageConfig],
    *,
    launch_modes: dict[str, StageLaunchMode],
) -> None:
    for stage_cfg in stages_cfg:
        factory = import_string(stage_cfg.factory)
        params = inspect.signature(factory).parameters
        mode = launch_modes[stage_cfg.name]
        requires_tp_kwargs = stage_cfg.tp_size > 1 or mode.requires_sglang_launch
        if not requires_tp_kwargs:
            continue

        accepts_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD
            for param in params.values()
        )
        missing = [] if accepts_kwargs else sorted(_TP_LAUNCH_PARAMS - params.keys())
        if missing:
            raise ValueError(
                f"Stage {stage_cfg.name!r}: factory {stage_cfg.factory!r} is not "
                f"TP-capable; missing launch parameters {missing}"
            )

        if (
            stage_cfg.tp_size > 1
            and mode.has_backend_parameter
            and not mode.is_sglang_execution
        ):
            raise ValueError(
                f"Stage {stage_cfg.name!r}: tp_size={stage_cfg.tp_size} "
                "requires backend='sglang' or backend='auto' resolving to "
                f"SGLang (got requested={mode.requested_backend!r}, "
                f"execution={mode.execution_backend!r})"
            )


def _build_stage_groups(
    config: PipelineConfig,
    ctx: multiprocessing.context.BaseContext | None = None,
    *,
    stages_cfg: list[StageConfig] | None = None,
    name_map: dict[str, str] | None = None,
    endpoints: dict[str, str] | None = None,
    placement_plan: StagePlacementPlan | None = None,
    process_plan: ProcessTopologyPlan | None = None,
    launch_modes: dict[str, StageLaunchMode] | None = None,
) -> list[StageGroup]:
    """Build lifecycle groups from prepared endpoints and process topology.

    The caller owns endpoint allocation and IPC runtime-dir lifecycle. This
    helper only converts prepared runtime state into subprocess specs.
    """
    if ctx is None:
        ctx = multiprocessing.get_context("spawn")
    if (
        stages_cfg is None
        or name_map is None
        or endpoints is None
        or placement_plan is None
        or process_plan is None
    ):
        prep = prepare_pipeline_runtime(config)
        stages_cfg = prep.stages_cfg
        name_map = prep.name_map
        endpoints = prep.endpoints
        placement_plan = prep.placement_plan
        process_plan = prep.process_plan
        launch_modes = prep.launch_modes
    if launch_modes is None:
        launch_modes = build_stage_launch_modes(config, stages_cfg=stages_cfg)
    _run_tp_preflight(stages_cfg, launch_modes=launch_modes)

    stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}
    stream_receivers: set[str] = set()
    for scfg in stages_cfg:
        for target in scfg.stream_to:
            stream_receivers.add(target)
    stage_cfg_by_name = {stage.name: stage for stage in stages_cfg}

    nccl_port_counter = _NcclPortAllocator()

    single_stage_specs: dict[str, StageProcessSpec] = {}
    tp_groups: list[StageGroup] = []
    for stage_cfg in stages_cfg:
        tp_size = stage_cfg.tp_size
        gpu_ids = resolve_stage_gpu_ids(placement_plan, stage_cfg)

        same_gpu_targets = resolve_same_gpu_stream_targets(
            placement_plan,
            stage_cfg,
        )
        same_process_targets = _resolve_same_process_targets(
            stage_cfg,
            stage_cfg_by_name,
            name_map,
            process_plan,
        )

        base_factory_args = resolve_stage_factory_args(stage_cfg, config)
        sglang_launch_mode = launch_modes[stage_cfg.name].requires_sglang_launch
        nccl_port = (
            nccl_port_counter.allocate()
            if tp_size > 1 or sglang_launch_mode
            else None
        )

        stage_kwargs = dict(
            stage_name=stage_cfg.name,
            factory=stage_cfg.factory,
            next_stages=stage_cfg.next,
            route_fn=stage_cfg.route_fn,
            is_terminal=stage_cfg.terminal,
            env_defaults=dict(config.env_defaults),
            wait_for=stage_cfg.wait_for,
            wait_for_fn=stage_cfg.wait_for_fn,
            merge_fn=stage_cfg.merge_fn,
            project_payload={
                name_map.get(target, target): dotted_path
                for target, dotted_path in stage_cfg.project_payload.items()
            },
            coordinator_endpoint=endpoints["completion"],
            abort_endpoint=endpoints["abort"],
            stage_endpoints=stage_endpoints,
            stream_targets=list(stage_cfg.stream_to),
            stream_done_to_fn=stage_cfg.stream_done_to_fn,
            same_gpu_targets=same_gpu_targets,
            same_process_targets=same_process_targets,
            is_stream_receiver=stage_cfg.name in stream_receivers,
            can_accept_stream_before_payload=stage_cfg.can_accept_stream_before_payload,
            name_map=name_map,
        )
        if tp_size == 1:
            single_stage_specs[stage_cfg.name] = _build_single_stage_spec(
                stage_cfg=stage_cfg,
                config=config,
                gpu_id=gpu_ids[0],
                recv_endpoint=stage_endpoints[stage_cfg.name],
                base_factory_args=base_factory_args,
                stage_kwargs=stage_kwargs,
                sglang_launch_mode=sglang_launch_mode,
                nccl_port=nccl_port,
            )
        else:
            specs = _build_tp_stage_specs(
                ctx=ctx,
                stage_cfg=stage_cfg,
                config=config,
                gpu_ids=gpu_ids,
                nccl_port=nccl_port,
                recv_endpoint=stage_endpoints[stage_cfg.name],
                base_factory_args=base_factory_args,
                stage_kwargs=stage_kwargs,
            )
            process_specs = [
                StageWorkerProcessSpec(
                    process_name=process_plan.tp_stage_to_processes[stage_cfg.name][
                        spec.tp_rank
                    ],
                    stage_specs=[spec],
                    gpu_id=spec.gpu_id,
                )
                for spec in specs
            ]
            tp_groups.append(StageGroup(stage_cfg.name, process_specs))

    groups: list[StageGroup] = []
    for group in process_plan.groups:
        groups.append(
            StageGroup(
                group.name,
                [
                    StageWorkerProcessSpec(
                        process_name=group.name,
                        stage_specs=[
                            single_stage_specs[stage_name]
                            for stage_name in group.stage_names
                        ],
                        gpu_id=group.gpu_id,
                    )
                ],
            )
        )
    groups.extend(tp_groups)

    return groups


def _resolve_same_process_targets(
    stage_cfg: StageConfig,
    stage_cfg_by_name: dict[str, StageConfig],
    name_map: dict[str, str],
    process_plan: ProcessTopologyPlan,
) -> set[str]:
    if stage_cfg.tp_size > 1:
        return set()
    source_process = process_plan.stage_to_process.get(stage_cfg.name)
    if source_process is None:
        return set()

    raw_targets: list[str] = []
    if stage_cfg.next is not None:
        raw_targets.extend(
            [stage_cfg.next] if isinstance(stage_cfg.next, str) else stage_cfg.next
        )
    raw_targets.extend(stage_cfg.stream_to)

    same_process_targets: set[str] = set()
    for raw_target in raw_targets:
        target = name_map.get(raw_target, raw_target)
        target_cfg = stage_cfg_by_name.get(target)
        if target_cfg is None or target_cfg.tp_size > 1:
            continue
        if process_plan.stage_to_process.get(target) == source_process:
            same_process_targets.add(target)
    return same_process_targets


def _build_single_stage_spec(
    *,
    stage_cfg: StageConfig,
    config: PipelineConfig,
    gpu_id: int | None,
    recv_endpoint: str,
    base_factory_args: dict[str, Any],
    stage_kwargs: dict[str, Any],
    sglang_launch_mode: bool,
    nccl_port: int | None,
) -> StageProcessSpec:
    factory_args = dict(base_factory_args)
    if "gpu_id" in base_factory_args:
        factory_args["gpu_id"] = gpu_id
    if sglang_launch_mode:
        if nccl_port is None:
            raise AssertionError(
                f"SGLang-backed stage {stage_cfg.name!r} requires an NCCL port"
            )
        factory_args["tp_rank"] = 0
        factory_args["tp_size"] = 1
        factory_args["nccl_port"] = nccl_port
    relay_config = _resolve_relay_config(stage_cfg, config, gpu_id=gpu_id)
    return StageProcessSpec(
        role="single",
        tp_rank=0,
        tp_size=1,
        gpu_id=gpu_id,
        nccl_port=factory_args.get("nccl_port"),
        factory_args=factory_args,
        relay_config=relay_config,
        recv_endpoint=recv_endpoint,
        **stage_kwargs,
    )


def _build_tp_stage_specs(
    *,
    ctx: multiprocessing.context.BaseContext,
    stage_cfg: StageConfig,
    config: PipelineConfig,
    gpu_ids: list[int | None],
    nccl_port: int | None,
    recv_endpoint: str,
    base_factory_args: dict[str, Any],
    stage_kwargs: dict[str, Any],
) -> list[StageProcessSpec]:
    if nccl_port is None:
        raise AssertionError(f"TP stage {stage_cfg.name!r} requires an NCCL port")
    follower_work_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    follower_abort_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    specs: list[StageProcessSpec] = []

    for tp_rank in range(stage_cfg.tp_size):
        gpu_id = gpu_ids[tp_rank] if tp_rank < len(gpu_ids) else gpu_ids[0]
        if gpu_id is None:
            raise ValueError(f"TP stage {stage_cfg.name!r} requires GPU placement")
        factory_args = dict(base_factory_args)
        if "gpu_id" in base_factory_args:
            factory_args["gpu_id"] = gpu_id
        factory_args["tp_rank"] = tp_rank
        factory_args["tp_size"] = stage_cfg.tp_size
        factory_args["nccl_port"] = nccl_port

        relay_config = _resolve_relay_config(stage_cfg, config, gpu_id=gpu_id)

        if tp_rank == 0:
            specs.append(
                StageProcessSpec(
                    role="leader",
                    tp_rank=tp_rank,
                    tp_size=stage_cfg.tp_size,
                    gpu_id=gpu_id,
                    nccl_port=nccl_port,
                    factory_args=factory_args,
                    relay_config=relay_config,
                    recv_endpoint=recv_endpoint,
                    follower_work_queues=follower_work_queues,
                    follower_abort_queues=follower_abort_queues,
                    **stage_kwargs,
                )
            )
            continue

        idx = tp_rank - 1
        specs.append(
            StageProcessSpec(
                role="follower",
                tp_rank=tp_rank,
                tp_size=stage_cfg.tp_size,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                factory_args=factory_args,
                relay_config=relay_config,
                recv_endpoint="",
                internal_work_queue=follower_work_queues[idx],
                internal_abort_queue=follower_abort_queues[idx],
                **stage_kwargs,
            )
        )

    return specs


def _resolve_relay_config(
    stage_cfg: StageConfig,
    config: PipelineConfig,
    *,
    gpu_id: int | None,
) -> dict[str, Any]:
    """Build relay config, overriding gpu_id from placement."""
    relay_config = build_relay_config(stage_cfg, config)
    # shm copies into host shared memory, so CUDA staging only creates extra
    # GPU allocator pressure.
    if stage_cfg.gpu is not None and config.relay_backend != "shm":
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


class MultiProcessPipelineRunner:

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._coordinator: Coordinator | None = None
        self._ipc_runtime_dir: IpcRuntimeDir | None = None
        self._groups: list[StageGroup] = []
        self._completion_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._fatal_event: asyncio.Event | None = None
        self._fatal_error: BaseException | None = None
        self._prep: PipelineRuntimePrep | None = None
        self._started = False

    @property
    def coordinator(self) -> Coordinator:
        if self._coordinator is None:
            raise RuntimeError("Runner not started")
        return self._coordinator

    @property
    def prep(self) -> PipelineRuntimePrep:
        """Return the resolved runtime prep (placement plan, process plan,
        endpoints, fused stages). Valid only after :meth:`start`."""
        if self._prep is None:
            raise RuntimeError("Runner not started")
        return self._prep

    @property
    def stage_control_endpoints(self) -> dict[str, str]:
        if not self._started:
            raise RuntimeError("Runner not started")
        endpoints: dict[str, str] = {}
        for group in self._groups:
            endpoints.update(group.stage_control_endpoints)
        return endpoints

    async def start(self, timeout: float = 120.0) -> None:
        if self._started:
            raise RuntimeError("Already started")

        try:
            ctx = multiprocessing.get_context("spawn")
            self._fatal_event = asyncio.Event()
            self._fatal_error = None
            prep = prepare_pipeline_runtime(
                self._config,
                ipc_runtime_dir=self._ipc_runtime_dir,
            )
            self._prep = prep
            self._ipc_runtime_dir = prep.runtime_dir
            groups = _build_stage_groups(
                self._config,
                ctx,
                stages_cfg=prep.stages_cfg,
                name_map=prep.name_map,
                endpoints=prep.endpoints,
                placement_plan=prep.placement_plan,
                process_plan=prep.process_plan,
                launch_modes=prep.launch_modes,
            )

            terminal_stages_resolver = (
                import_string(self._config.terminal_stages_fn)
                if self._config.terminal_stages_fn
                else None
            )
            self._coordinator = Coordinator(
                completion_endpoint=prep.endpoints["completion"],
                abort_endpoint=prep.endpoints["abort"],
                entry_stage=prep.entry_stage,
                terminal_stages=self._config.terminal_stages or None,
                terminal_stages_resolver=terminal_stages_resolver,
            )
            await self._coordinator.start()
            self._completion_task = asyncio.create_task(
                self._coordinator.run_completion_loop()
            )

            self._groups = groups
            if self._config.env_defaults:
                env_names = ", ".join(sorted(self._config.env_defaults))
                logger.info(f"Configured stage process env defaults: {env_names}")
            for group in self._groups:
                group.spawn(ctx)
                # Keep GPU-heavy scheduler construction serialized at the
                # StageGroup level. If two TP encoder groups share the same
                # GPU pair and start concurrently, rank-local per-GPU startup
                # locks can be acquired in opposite orders and block distributed
                # initialization. Waiting group-by-group preserves concurrent
                # rank startup within one TP group while avoiding cross-group
                # lock inversion on colocated deployments.
                await group.wait_ready(timeout)

            for group in self._groups:
                if group.any_dead():
                    raise RuntimeError(
                        f"Stage process(es) died during startup: "
                        f"{group.dead_summary()}"
                    )

            for group in self._groups:
                for stage_name, endpoint in group.stage_control_endpoints.items():
                    self._coordinator.register_stage(stage_name, endpoint)

            self._started = True
            self._monitor_task = asyncio.create_task(self._monitor_children())

            total_stages = sum(
                len(group.stage_control_endpoints) for group in self._groups
            )
            total_procs = sum(g.process_count for g in self._groups)
            logger.info(
                "MultiProcessPipelineRunner started: %d stage(s), %d process(es)",
                total_stages,
                total_procs,
            )

        except Exception:
            await self._cleanup_on_failure()
            raise

    async def _monitor_children(self) -> None:
        while self._started:
            for group in self._groups:
                if group.any_dead():
                    group_name = getattr(group, "group_name", None) or getattr(
                        group, "stage_name", "unknown"
                    )
                    error = RuntimeError(
                        f"Dead stage process(es) detected in {group_name}: "
                        f"{group.dead_summary()}"
                    )
                    logger.error("%s", error)
                    await self._fail_runtime(error)
                    return
            await asyncio.sleep(5.0)

    async def _fail_runtime(self, error: BaseException) -> None:
        self._fatal_error = error
        if self._coordinator is not None:
            self._coordinator.fail_all_active(str(error))
        fatal_event = getattr(self, "_fatal_event", None)
        if fatal_event is not None:
            fatal_event.set()
        await self.stop()

    async def wait_failed(self) -> None:
        if self._fatal_event is None:
            raise RuntimeError("Runner not started")
        await self._fatal_event.wait()
        if self._fatal_error is not None:
            raise self._fatal_error
        raise RuntimeError("Pipeline runtime failed")

    async def _cancel_completion_task(self) -> None:
        if self._completion_task is None:
            return
        self._completion_task.cancel()
        try:
            await self._completion_task
        except asyncio.CancelledError:
            pass
        self._completion_task = None

    def _close_runtime_dir(self) -> None:
        if self._ipc_runtime_dir is None:
            return
        self._ipc_runtime_dir.close()
        self._ipc_runtime_dir = None

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

        await self._cancel_completion_task()

        await self._coordinator.stop()
        self._groups.clear()
        self._coordinator = None

        self._close_runtime_dir()

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
            group.close_control_channels()
        self._groups.clear()

        await self._cancel_completion_task()

        if self._coordinator is not None:
            try:
                await self._coordinator.stop()
            except Exception:
                pass
            self._coordinator = None

        self._close_runtime_dir()
