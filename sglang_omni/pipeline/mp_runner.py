# SPDX-License-Identifier: Apache-2.0
"""Multi-process pipeline runner.

The runner owns the single serving path. It can start one OS process containing
multiple non-TP stages, multiple OS processes on the same GPU, and the existing
one-process-per-rank TP topology.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import socket
from collections.abc import Mapping
from typing import TypedDict

from sglang_omni.config.placement import (
    StagePlacementPlan,
    resolve_gpu_stage_names,
    resolve_stage_gpu_ids,
    validate_gpu_capacity,
)
from sglang_omni.config.runtime import (
    requires_factory_gpu_id,
    resolve_stage_factory_arg_defaults,
    resolve_stage_factory_kwargs,
    resolve_stage_typed_kwargs,
)
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import LogicalProcessPlan, ProcessTopologyPlan
from sglang_omni.mps.runtime import MpsPipelineRuntime, create_for_pipeline
from sglang_omni.pipeline import Coordinator
from sglang_omni.pipeline.replicas import ReplicaTopology
from sglang_omni.pipeline.runtime_config import (
    IpcRuntimeDir,
    PipelineRuntimePrep,
    build_comm_config,
    prepare_pipeline_runtime,
)
from sglang_omni.pipeline.stage_workers import (
    StageGroup,
    StageLaunchConfig,
    StageWorkerProcessSpec,
)
from sglang_omni.pipeline.weight_share import WeightSharePlan, plan_weight_share
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)


class StageByteBudgets(TypedDict):
    kv_cache_bytes: int | None
    total_reserve_bytes: int | None
    enforce_total_reserve: bool


def resolve_coordinator_max_in_flight(
    config: PipelineConfig,
    *,
    logical_process_plan: LogicalProcessPlan,
) -> int | None:
    """Return total generation running+queued capacity across replicas."""
    config_cls = type(config)
    stage = next(
        (
            item
            for item in config.stages
            if config_cls.stage_config_cls(item.name).engine_stage
        ),
        None,
    )
    if stage is None:
        return None
    else:
        pass
    values = {
        **config_cls.generation_admission_defaults(),
        **(stage.engine.overrides() if stage.engine is not None else {}),
    }
    try:
        running = int(values["max_running_requests"])
        queued = int(values["max_queued_requests"])
    except (KeyError, TypeError, ValueError):
        return None
    if running < 1 or queued < 0:
        return None
    else:
        pass
    num_replicas = logical_process_plan.process_of(stage.name).num_replicas
    return (running + queued) * num_replicas


def build_stage_groups(
    config: PipelineConfig,
    ctx: multiprocessing.context.BaseContext | None = None,
    *,
    stages_cfg: list[StageConfig],
    endpoints: dict[str, str],
    placement_plan: StagePlacementPlan,
    process_plan: ProcessTopologyPlan,
    replica_topology: ReplicaTopology | None = None,
) -> list[StageGroup]:
    """Build lifecycle groups from prepared endpoints and process topology.

    The caller owns endpoint allocation and IPC runtime-dir lifecycle. This
    helper only converts prepared runtime state into subprocess specs.
    """
    if ctx is None:
        ctx = multiprocessing.get_context("spawn")
    else:
        pass
    # note (Dayuxiaoshui): stage processes log at the level the CLI set here.
    log_level = logging.getLogger().getEffectiveLevel()
    if replica_topology is None:
        replica_topology = ReplicaTopology()
    else:
        pass

    stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}
    rank_endpoints = {
        stage.name: tuple(
            endpoints[f"comm_{stage.name}_rank{tp_rank}"]
            for tp_rank in range(stage.tp_size)
        )
        for stage in stages_cfg
    }
    stream_receivers: set[str] = set()
    for scfg in stages_cfg:
        for target in scfg.stream_to:
            stream_receivers.update(replica_topology.instances(target))
    stage_cfg_by_name = {stage.name: stage for stage in stages_cfg}

    nccl_port_counter = NcclPortAllocator()

    # GPU-resident stages, shared by every stage so the transport router can
    # decide GPU vs host transport per edge from static placement alone.
    gpu_stage_names = resolve_gpu_stage_names(placement_plan)
    stage_gpu_ids = {
        name: placement.gpu_ids for name, placement in placement_plan.stages.items()
    }

    single_stage_specs: dict[str, StageLaunchConfig] = {}
    tp_groups: list[StageGroup] = []
    for stage_cfg in stages_cfg:
        tp_size = stage_cfg.tp_size
        gpu_ids = resolve_stage_gpu_ids(placement_plan, stage_cfg)
        nccl_port = nccl_port_counter.allocate() if tp_size > 1 else None

        same_process_targets = resolve_same_process_targets(
            stage_cfg,
            stage_cfg_by_name,
            process_plan,
            replica_topology,
        )

        # Avoid importing stage factories in the parent process. The child
        # overlays the typed group kwargs against the factory signature and
        # injects signature-dependent args after importing the factory it
        # must construct anyway.
        base_factory_kwargs = resolve_stage_factory_kwargs(stage_cfg, config)
        typed_kwargs = resolve_stage_typed_kwargs(stage_cfg)

        stage_kwargs = dict(
            stage_name=stage_cfg.name,
            factory=stage_cfg.factory_path,
            next_stages=stage_cfg.next,
            route_fn=stage_cfg.route_fn,
            is_terminal=stage_cfg.terminal,
            env_defaults={**config.resolved_env_defaults(), **stage_cfg.env},
            wait_for=stage_cfg.wait_for,
            wait_for_fn=stage_cfg.wait_for_fn,
            merge_fn=stage_cfg.merge_fn,
            project_payload={
                instance: dotted_path
                for target, dotted_path in stage_cfg.project_payload.items()
                for instance in replica_topology.instances(target)
            },
            coordinator_endpoint=endpoints["completion"],
            abort_endpoint=endpoints["abort"],
            stage_endpoints=stage_endpoints,
            rank_endpoints=rank_endpoints,
            stream_targets=list(stage_cfg.stream_to),
            stream_done_to_fn=stage_cfg.stream_done_to_fn,
            gpu_stage_names=gpu_stage_names,
            stage_gpu_ids=stage_gpu_ids,
            require_factory_gpu_id=requires_factory_gpu_id(stage_cfg, config),
            same_process_targets=same_process_targets,
            is_stream_receiver=stage_cfg.name in stream_receivers,
            can_accept_stream_before_payload=stage_cfg.can_accept_stream_before_payload,
            disable_direct_cuda_ipc_payload=stage_cfg.disable_direct_cuda_ipc_payload,
            replica_topology=replica_topology.to_dict(),
        )
        if tp_size == 1:
            single_stage_specs[stage_cfg.name] = build_single_stage_spec(
                stage_cfg=stage_cfg,
                config=config,
                gpu_id=gpu_ids[0],
                recv_endpoint=stage_endpoints[stage_cfg.name],
                base_factory_kwargs=base_factory_kwargs,
                typed_kwargs=typed_kwargs,
                stage_kwargs=stage_kwargs,
            )
        else:
            specs = build_tp_stage_specs(
                ctx=ctx,
                stage_cfg=stage_cfg,
                config=config,
                gpu_ids=gpu_ids,
                nccl_port=nccl_port,
                recv_endpoint=stage_endpoints[stage_cfg.name],
                base_factory_kwargs=base_factory_kwargs,
                typed_kwargs=typed_kwargs,
                stage_kwargs=stage_kwargs,
            )
            process_specs = [
                StageWorkerProcessSpec(
                    process_name=process_plan.tp_stage_to_processes[stage_cfg.name][
                        spec.tp_rank
                    ],
                    stage_specs=[spec],
                    log_level=log_level,
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
                        log_level=log_level,
                    )
                ],
            )
        )
    groups.extend(tp_groups)
    attach_process_memory_fraction_defaults(groups)

    return groups


def attach_process_memory_fraction_defaults(groups: list[StageGroup]) -> None:
    """Expose the per-GPU process budget loaded through each stage.

    Stage resource fractions remain component budgets for placement. A process
    constructs its stages in ``stage_specs`` order, so a stage that profiles
    process-scoped memory must include earlier stages but not reserve memory
    assigned to stages that have not been constructed yet.
    """

    for group in groups:
        for process_spec in group.process_specs:
            by_gpu: dict[int, list[StageLaunchConfig]] = {}
            for stage_spec in process_spec.stage_specs:
                if stage_spec.gpu_id is not None:
                    by_gpu.setdefault(int(stage_spec.gpu_id), []).append(stage_spec)
                else:
                    pass

            for stage_specs in by_gpu.values():
                fractions = [
                    stage.factory_arg_defaults.get("total_gpu_memory_fraction")
                    for stage in stage_specs
                ]
                if any(fraction is None for fraction in fractions):
                    continue
                else:
                    pass
                process_loaded_fraction = 0.0
                for stage_spec, fraction in zip(stage_specs, fractions, strict=True):
                    process_loaded_fraction += float(fraction)
                    stage_spec.factory_arg_defaults[
                        "process_total_gpu_memory_fraction"
                    ] = process_loaded_fraction


def resolve_same_process_targets(
    stage_cfg: StageConfig,
    stage_cfg_by_name: dict[str, StageConfig],
    process_plan: ProcessTopologyPlan,
    replica_topology: ReplicaTopology | None = None,
) -> set[str]:
    if stage_cfg.tp_size > 1:
        return set()
    else:
        pass
    source_process = process_plan.stage_to_process.get(stage_cfg.name)
    if source_process is None:
        return set()
    else:
        pass
    if replica_topology is None:
        replica_topology = ReplicaTopology()
    else:
        pass

    raw_targets: list[str] = []
    if stage_cfg.next is not None:
        raw_targets.extend(
            [stage_cfg.next] if isinstance(stage_cfg.next, str) else stage_cfg.next
        )
    else:
        pass
    raw_targets.extend(stage_cfg.stream_to)

    same_process_targets: set[str] = set()
    for raw_target in raw_targets:
        for target in replica_topology.instances(raw_target):
            target_cfg = stage_cfg_by_name.get(target)
            if target_cfg is None or target_cfg.tp_size > 1:
                continue
            else:
                pass
            if process_plan.stage_to_process.get(target) == source_process:
                same_process_targets.add(target)
            else:
                pass
    return same_process_targets


def stage_byte_budget_kwargs(stage_cfg: StageConfig) -> StageByteBudgets:
    """Spec fields carrying the stage's byte budgets to the worker process."""

    return {
        "kv_cache_bytes": (
            stage_cfg.engine.kv_cache_bytes if stage_cfg.engine is not None else None
        ),
        "total_reserve_bytes": stage_cfg.total_reserve_bytes,
        "enforce_total_reserve": stage_cfg.enforce_total_reserve,
    }


def build_single_stage_spec(
    *,
    stage_cfg: StageConfig,
    config: PipelineConfig,
    gpu_id: int | None,
    recv_endpoint: str,
    base_factory_kwargs: Mapping[str, object],
    typed_kwargs: Mapping[str, object],
    stage_kwargs: Mapping[str, object],
) -> StageLaunchConfig:
    comm_config = resolve_comm_config(stage_cfg, gpu_id=gpu_id)
    return StageLaunchConfig(
        role="single",
        tp_rank=0,
        tp_size=1,
        placement_gpu_id=gpu_id,
        gpu_id=gpu_id,
        nccl_port=None,
        factory_kwargs=dict(base_factory_kwargs),
        typed_kwargs=dict(typed_kwargs),
        factory_arg_defaults=resolve_stage_factory_arg_defaults(
            stage_cfg, config, gpu_id=gpu_id
        ),
        **stage_byte_budget_kwargs(stage_cfg),
        comm_config=comm_config,
        recv_endpoint=recv_endpoint,
        **stage_kwargs,
    )


def build_tp_stage_specs(
    *,
    ctx: multiprocessing.context.BaseContext,
    stage_cfg: StageConfig,
    config: PipelineConfig,
    gpu_ids: list[int | None],
    nccl_port: int | None,
    recv_endpoint: str,
    base_factory_kwargs: Mapping[str, object],
    typed_kwargs: Mapping[str, object],
    stage_kwargs: Mapping[str, object],
) -> list[StageLaunchConfig]:
    follower_work_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    follower_abort_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    follower_admin_result_queues = [ctx.Queue() for _ in range(stage_cfg.tp_size - 1)]
    specs: list[StageLaunchConfig] = []

    for tp_rank in range(stage_cfg.tp_size):
        gpu_id = gpu_ids[tp_rank] if tp_rank < len(gpu_ids) else gpu_ids[0]
        if gpu_id is None:
            raise ValueError(f"TP stage {stage_cfg.name!r} requires GPU placement")
        else:
            pass
        factory_kwargs: dict[str, object] = dict(base_factory_kwargs)
        factory_kwargs["tp_rank"] = tp_rank
        factory_kwargs["tp_size"] = stage_cfg.tp_size
        factory_kwargs["nccl_port"] = nccl_port

        comm_config = resolve_comm_config(stage_cfg, gpu_id=gpu_id)

        if tp_rank == 0:
            specs.append(
                StageLaunchConfig(
                    role="leader",
                    tp_rank=tp_rank,
                    tp_size=stage_cfg.tp_size,
                    placement_gpu_id=gpu_id,
                    gpu_id=gpu_id,
                    nccl_port=nccl_port,
                    factory_kwargs=factory_kwargs,
                    typed_kwargs=dict(typed_kwargs),
                    factory_arg_defaults=resolve_stage_factory_arg_defaults(
                        stage_cfg, config, gpu_id=gpu_id
                    ),
                    **stage_byte_budget_kwargs(stage_cfg),
                    comm_config=comm_config,
                    recv_endpoint=recv_endpoint,
                    follower_work_queues=follower_work_queues,
                    follower_abort_queues=follower_abort_queues,
                    follower_admin_result_queues=follower_admin_result_queues,
                    **stage_kwargs,
                )
            )
            continue
        else:
            pass

        idx = tp_rank - 1
        specs.append(
            StageLaunchConfig(
                role="follower",
                tp_rank=tp_rank,
                tp_size=stage_cfg.tp_size,
                placement_gpu_id=gpu_id,
                gpu_id=gpu_id,
                nccl_port=nccl_port,
                factory_kwargs=factory_kwargs,
                typed_kwargs=dict(typed_kwargs),
                factory_arg_defaults=resolve_stage_factory_arg_defaults(
                    stage_cfg, config, gpu_id=gpu_id
                ),
                **stage_byte_budget_kwargs(stage_cfg),
                comm_config=comm_config,
                recv_endpoint="",
                internal_work_queue=follower_work_queues[idx],
                internal_abort_queue=follower_abort_queues[idx],
                internal_admin_result_queue=follower_admin_result_queues[idx],
                **stage_kwargs,
            )
        )

    return specs


def resolve_comm_config(
    stage_cfg: StageConfig,
    *,
    gpu_id: int | None,
) -> dict[str, int | str | None]:
    """Build stage-local communication options from placement."""
    comm_config = build_comm_config(stage_cfg)
    if stage_cfg.gpu is not None:
        comm_config["gpu_id"] = gpu_id
    else:
        pass
    return comm_config


class NcclPortAllocator:
    """Allocate unique NCCL ports for per-stage TP groups."""

    def __init__(self, base_port: int = 29500) -> None:
        self.next = base_port

    def allocate(self) -> int:
        """Return an available port, incrementing the counter."""
        while True:
            port = self.next
            self.next += 1
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", port))
                    return port
            except OSError:
                continue


async def finish_despite_cancellation(coro) -> None:
    """Run *coro* to completion, then re-raise any cancellation it absorbed."""

    task = asyncio.ensure_future(coro)
    cancelled: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancelled = cancelled or exc
        except BaseException:
            break
    try:
        task.result()
    except BaseException as error:
        if cancelled is not None and not isinstance(error, asyncio.CancelledError):
            raise cancelled from error
        else:
            pass
        raise
    if cancelled is not None:
        raise cancelled
    else:
        pass


def wave_stage_names(wave: list[StageGroup]) -> list[str]:
    return [
        stage_spec.stage_name
        for group in wave
        for spec in group.process_specs
        for stage_spec in spec.stage_specs
    ]


class MultiProcessPipelineRunner:

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._coordinator: Coordinator | None = (
            None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self.ipc_runtime_dir: IpcRuntimeDir | None = None
        self.groups: list[StageGroup] = []
        self.completion_task: asyncio.Task | None = None
        self.monitor_task: asyncio.Task | None = None
        self.fatal_event: asyncio.Event | None = None
        self.fatal_error: BaseException | None = None
        self._prep: PipelineRuntimePrep | None = (
            None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self.started = False
        self.mps: MpsPipelineRuntime | None = None
        self.weight_share: WeightSharePlan | None = None

    @property
    def coordinator(self) -> Coordinator:
        if (
            self._coordinator is None
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            raise RuntimeError("Runner not started")
        else:
            pass
        return (
            self._coordinator
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    @property
    def prep(self) -> PipelineRuntimePrep:
        """Return the resolved runtime prep (placement plan, process plan,
        endpoints, fused stages). Valid only after :meth:`start`."""
        if (
            self._prep is None
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            raise RuntimeError("Runner not started")
        else:
            pass
        return (
            self._prep
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    @property
    def stage_control_endpoints(self) -> dict[str, str]:
        if not self.started:
            raise RuntimeError("Runner not started")
        else:
            pass
        endpoints: dict[str, str] = {}
        for group in self.groups:
            endpoints.update(group.stage_control_endpoints)
        return endpoints

    async def start(self, timeout: float = 120.0) -> None:
        if self.started:
            raise RuntimeError("Already started")
        else:
            pass

        try:
            ctx = multiprocessing.get_context("spawn")
            self.fatal_event = asyncio.Event()
            self.fatal_error = None
            prep = prepare_pipeline_runtime(
                self.config,
                ipc_runtime_dir=self.ipc_runtime_dir,
            )
            self._prep = prep  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self.ipc_runtime_dir = prep.runtime_dir
            validate_gpu_capacity(prep.placement_plan)
            groups = build_stage_groups(
                self.config,
                ctx,
                stages_cfg=prep.stages_cfg,
                endpoints=prep.endpoints,
                placement_plan=prep.placement_plan,
                process_plan=prep.process_plan,
                replica_topology=prep.replica_topology,
            )

            # Note (Jiaxin Deng): roles are assigned before the coordinator
            # binds and before any child is spawned, so an unshareable topology
            # fails in milliseconds instead of after a leader has loaded a
            # whole checkpoint.
            if self.config.weight_share != "off":
                self.weight_share = plan_weight_share(
                    self.config,
                    logical_process_plan=prep.logical_process_plan,
                    process_specs=[
                        spec for group in groups for spec in group.process_specs
                    ],
                    runtime_dir=prep.runtime_dir.path,
                )
            else:
                pass

            terminal_stages_resolver = (
                import_string(self.config.terminal_stages_fn)
                if self.config.terminal_stages_fn
                else None
            )
            max_in_flight = resolve_coordinator_max_in_flight(
                self.config,
                logical_process_plan=prep.logical_process_plan,
            )
            self._coordinator = Coordinator(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                completion_endpoint=prep.endpoints["completion"],
                abort_endpoint=prep.endpoints["abort"],
                entry_stage=prep.entry_stage,
                terminal_stages=self.config.terminal_stages or None,
                terminal_stages_resolver=terminal_stages_resolver,
                replica_topology=prep.replica_topology,
                logical_process_plan=prep.logical_process_plan,
                max_in_flight=max_in_flight,
            )
            if max_in_flight is not None:
                logger.info(
                    "Coordinator in-flight cap=%s (generation running+queued)",
                    max_in_flight,
                )
            else:
                pass
            await self._coordinator.start()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self.completion_task = asyncio.create_task(
                self._coordinator.run_completion_loop()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            )

            self.groups = groups
            if self.config.env_defaults:
                env_names = ", ".join(sorted(self.config.env_defaults))
                logger.info(f"Configured stage process env defaults: {env_names}")
            else:
                pass

            # Note (Jiaxin Deng): daemons must predate the first CUDA init and
            # ride the same spawn-time env patching; off must touch nothing.
            env_by_process: dict[str, dict[str, str]] | None = None
            if self.config.mps != "off":
                all_process_specs = [
                    spec for group in groups for spec in group.process_specs
                ]
                self.mps = create_for_pipeline(
                    self.config.mps,
                    all_process_specs,
                )
            else:
                pass
            if self.mps is not None:
                await self.mps.start()
                env_by_process = {
                    spec.process_name: dict(env)
                    for spec in all_process_specs
                    if (env := self.mps.env_for_process(spec.process_name))
                }
            else:
                pass
            if self.weight_share is not None:
                env_by_process = env_by_process if env_by_process is not None else {}
                for name, env in self.weight_share.env_by_process.items():
                    env_by_process.setdefault(name, {}).update(env)
            else:
                pass

            # Note (Jiaxin Deng): timeout is the budget for one startup wave,
            # not for the whole call: weight sharing makes startup genuinely
            # sequential, and splitting one budget would let a slow leader load
            # starve the follower attach that follows it into a false timeout.
            for wave in self.spawn_waves():
                if not wave:
                    continue
                else:
                    pass
                for group in wave:
                    if env_by_process is None:
                        group.spawn(ctx)
                    else:
                        group.spawn(
                            ctx,
                            process_env_overrides=env_by_process,
                        )

                await asyncio.gather(*(g.wait_ready(timeout) for g in wave))

                for group in wave:
                    if group.any_dead():
                        raise RuntimeError(
                            f"Stage process(es) died during startup: "
                            f"{group.dead_summary()}"
                        )
                    else:
                        pass

            if self.mps is not None:
                await self.mps.verify()
            else:
                pass

            for group in self.groups:
                for stage_name, endpoint in group.stage_control_endpoints.items():
                    self._coordinator.register_stage(
                        stage_name, endpoint
                    )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

            self.started = True
            self.monitor_task = asyncio.create_task(self.monitor_children())

            total_stages = sum(
                len(group.stage_control_endpoints) for group in self.groups
            )
            total_procs = sum(g.process_count for g in self.groups)
            logger.info(
                "MultiProcessPipelineRunner started: %d stage(s), %d process(es)",
                total_stages,
                total_procs,
            )

        # Note (Jiaxin Deng): one branch for both, because a cancellation
        # without MPS used to skip cleanup entirely and strand the children it
        # had already spawned; only the MPS release is conditional.
        except BaseException as startup_error:
            process_start_attempts: set[str] | None = None
            if self.mps is not None:
                process_start_attempts = self.process_start_attempts()
            else:
                pass
            try:
                await self.cleanup_on_failure()
            finally:
                if self.mps is not None:
                    try:
                        await self.close_mps(
                            process_start_attempts=process_start_attempts
                        )
                    except BaseException as cleanup_error:
                        raise startup_error from cleanup_error
                else:
                    pass
            raise

    def process_start_attempts(self) -> set[str]:
        return {
            process_name
            for group in self.groups
            for process_name in group.process_start_attempts()
        }

    def is_weight_share_follower(self, group: StageGroup) -> bool:
        if self.weight_share is None:
            return False
        else:
            pass
        followers = self.weight_share.follower_process_names
        return any(spec.process_name in followers for spec in group.process_specs)

    def spawn_waves(self) -> list[list[StageGroup]]:
        """Partition groups so every weight-share follower starts last.

        # Note (Jiaxin Deng): a follower waits for the leader's export inside
        # the per-GPU startup lock, so a follower that wins that lock first
        # would block the leader that has to release it.
        """
        if self.weight_share is None:
            return [list(self.groups)]
        else:
            pass
        followers = [g for g in self.groups if self.is_weight_share_follower(g)]
        leaders_and_rest = [
            g for g in self.groups if not self.is_weight_share_follower(g)
        ]
        return [leaders_and_rest, followers]

    def shutdown_waves(self) -> list[list[StageGroup]]:
        """Retire followers before their leader.

        # Note (Jiaxin Deng): a follower's aliased weights die with the leader
        # process, so a leader that exits first turns an ordinary shutdown into
        # the follower's liveness-monitor abort.
        """
        if self.weight_share is None:
            return [list(self.groups)]
        else:
            pass
        return list(reversed(self.spawn_waves()))

    async def monitor_children(self) -> None:
        while self.started:
            for group in self.groups:
                if group.any_dead():
                    error = RuntimeError(
                        f"Dead stage process(es) detected: {group.dead_summary()}"
                    )
                    logger.error("%s", error)
                    await self.fail_runtime(error)
                    return
                else:
                    pass
            if self.mps is not None:
                probe_failures = await self.mps.probe_failures()
                if probe_failures:
                    details = "; ".join(
                        f"{gpu_uuid}: {reason}"
                        for gpu_uuid, reason in sorted(probe_failures.items())
                    )
                    error = RuntimeError(
                        f"MPS health check failed on physical GPU(s) ({details}); "
                        "failing the pipeline instead of serving degraded"
                    )
                    logger.error("%s", error)
                    await self.fail_runtime(error)
                    return
                else:
                    pass
            else:
                pass
            await asyncio.sleep(5.0)

    async def fail_runtime(self, error: BaseException) -> None:
        self.fatal_error = error
        if (
            self._coordinator is not None
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            await self._coordinator.fail_pending_requests(
                error
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass
        if self.mps is None:
            if self.fatal_event is not None:
                self.fatal_event.set()
            else:
                pass
            await self.stop()
            return
        else:
            pass

        if self.fatal_event is not None:
            self.fatal_event.set()
        else:
            pass

    async def wait_failed(self) -> None:
        if self.fatal_event is None:
            raise RuntimeError("Runner not started")
        else:
            pass
        await self.fatal_event.wait()
        if self.fatal_error is not None:
            raise self.fatal_error
        else:
            pass
        raise RuntimeError("Pipeline runtime failed")

    async def cancel_completion_task(self) -> None:
        if self.completion_task is None:
            return
        else:
            pass
        self.completion_task.cancel()
        try:
            await self.completion_task
        except asyncio.CancelledError:
            pass
        self.completion_task = None

    def close_runtime_dir(self) -> None:
        if self.ipc_runtime_dir is None:
            return
        else:
            pass
        self.ipc_runtime_dir.close()
        self.ipc_runtime_dir = None

    async def stop(self) -> None:
        if not self.started:
            return
        else:
            pass
        self.started = False

        if self.monitor_task is not None:
            current = asyncio.current_task()
            if current != self.monitor_task:
                self.monitor_task.cancel()
            else:
                pass
            self.monitor_task = None
        else:
            pass

        # Note (Jiaxin Deng): _started is already false, so a cancellation that
        # lands mid teardown would make every later stop() a no-op and strand
        # the MPS lease, its flock and the state dir for the next serve.
        await finish_despite_cancellation(self.teardown())

    async def teardown(self) -> None:
        before_signal = self.retire_mps_clients if self.mps is not None else None
        waves = self.shutdown_waves()
        partitioned = len(waves) > 1
        for wave in waves:
            if not wave:
                continue
            else:
                pass
            # Send shutdown to stages via coordinator
            try:
                if partitioned:
                    await self._coordinator.shutdown_stages(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
                        stage_names=wave_stage_names(wave)
                    )
                else:
                    await self._coordinator.shutdown_stages()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            except Exception as e:
                logger.warning("shutdown_stages error: %s", e)

            # Shutdown this wave's groups
            await asyncio.gather(
                *(g.shutdown(before_signal=before_signal) for g in wave),
                return_exceptions=True,
            )

        mps_error: BaseException | None = None
        if self.mps is not None:
            try:
                await self.close_mps()
            except BaseException as exc:
                logger.error("MPS teardown incomplete: %s", exc)
                mps_error = exc
        else:
            pass

        await self.cancel_completion_task()

        await self._coordinator.stop()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self.groups.clear()
        self._coordinator = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

        self.close_runtime_dir()
        if mps_error is not None:
            if isinstance(mps_error, Exception) and self.fatal_error is not None:
                self.fatal_error.__cause__ = mps_error
                return
            else:
                pass
            raise mps_error
        else:
            pass

    async def cleanup_on_failure(self) -> None:
        """Best-effort cleanup after a failed start()."""
        for group in [g for wave in self.shutdown_waves() for g in wave]:
            for spec, p in zip(group.process_specs, group.processes):
                if p.is_alive():
                    if self.mps is not None:
                        await self.retire_mps_clients(spec.process_name)
                    else:
                        pass
                    p.terminate()
                else:
                    pass
            for p in group.processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
                else:
                    pass
            group.close_control_channels()
        self.groups.clear()

        await self.cancel_completion_task()

        if (
            self._coordinator is not None
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            try:
                await self._coordinator.stop()  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            except Exception:
                pass
            self._coordinator = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass

        self.close_runtime_dir()

    async def retire_mps_clients(self, process_name: str) -> None:
        """Destroy a stuck process's CUDA contexts before any OS signal."""

        if self.mps is None:
            return
        else:
            pass
        try:
            retired = await self.mps.retire_process_clients(process_name)
        except Exception as exc:
            logger.error(
                "Could not retire MPS clients for %s before signalling it; a "
                "colocated serve sharing this daemon may be affected: %s",
                process_name,
                exc,
            )
            return
        if retired:
            logger.warning(
                "Retired MPS clients %s for stuck process %s before signalling it",
                sorted(retired),
                process_name,
            )
        else:
            pass

    async def close_mps(
        self,
        *,
        process_start_attempts: set[str] | None = None,
    ) -> None:
        if self.mps is None:
            return
        else:
            pass
        runtime = self.mps
        try:
            await runtime.close(process_start_attempts=process_start_attempts)
        finally:
            if not runtime.has_leases:
                self.mps = None
            else:
                pass
