# SPDX-License-Identifier: Apache-2.0
"""Stage worker process specifications, entrypoints, and lifecycle groups."""
from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing
import os
import queue
import sys
import time
from collections.abc import Iterable
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from sglang_omni.config.runtime import resolve_factory_signature_args
from sglang_omni.pipeline.control_plane import StageControlPlane
from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
from sglang_omni.pipeline.stage.input import AggregatedInput, DirectInput
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.pipeline.tp_control import TPFollowerControlPlane, TPLeaderFanout
from sglang_omni.utils.gpu_compat import (
    apply_gpu_compat_env_defaults,
    get_gpu_compat_env_defaults,
)
from sglang_omni.utils.gpu_memory import gpu_startup_lock
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)


@dataclass
class StageLaunchConfig:
    """Resolved launch metadata for one logical stage instance.

    ``StageWorkerProcessSpec`` is the OS-process payload. A worker process may
    carry multiple launch configs for colocated non-TP stages, while TP ranks
    each get their own launch config and process.

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
    factory_arg_defaults: dict[str, Any] = field(default_factory=dict)
    env_defaults: dict[str, str] = field(default_factory=dict)

    # Routing: static next stage(s)
    next_stages: str | list[str] | None = None
    route_fn: str | None = None
    is_terminal: bool = False

    # Fan-in
    wait_for: list[str] | None = None
    wait_for_fn: str | None = None
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

    # Same-process full payload wiring
    same_process_targets: set[str] = field(default_factory=set)

    # Fusion name map
    name_map: dict[str, str] = field(default_factory=dict)

    # TP internal control (leader -> followers)
    follower_work_queues: list[Any] = field(default_factory=list)
    follower_abort_queues: list[Any] = field(default_factory=list)
    follower_admin_result_queues: list[Any] = field(default_factory=list)
    internal_work_queue: Any | None = None
    internal_abort_queue: Any | None = None
    internal_admin_result_queue: Any | None = None

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
    stage_specs: list[StageLaunchConfig]


def _get_worker_process_env(spec: StageWorkerProcessSpec) -> dict[str, str]:
    """Return the spawn-time env overrides for *spec*.

    Hard invariant: a TP stage (``tp_size > 1``) must own its OS process
    exclusively. Its CUDA env remap and NCCL settings depend on being the sole
    tenant, so mixing a TP stage with any other stage in the same process group
    is a placement bug.
    """
    tp_stages = [s for s in spec.stage_specs if s.tp_size > 1]
    if not tp_stages:
        return {}
    if len(tp_stages) > 1 or len(spec.stage_specs) > 1:
        raise AssertionError(
            f"Process {spec.process_name!r} mixes a TP stage with other "
            "stages; TP stages must own their OS process exclusively. "
            f"stage_specs={[s.stage_name for s in spec.stage_specs]}"
        )
    return get_stage_process_env(tp_stages[0])


@contextmanager
def _patched_spawn_env(spec: StageWorkerProcessSpec):
    env_default_updates: dict[str, str] = {}
    for stage_spec in spec.stage_specs:
        for key, value in stage_spec.env_defaults.items():
            existing = env_default_updates.get(key)
            if existing is not None and existing != value:
                raise AssertionError(
                    f"Process {spec.process_name!r} has conflicting env default "
                    f"for {key!r}: {existing!r} != {value!r}"
                )
            if key not in os.environ:
                env_default_updates[key] = value

    worker_process_env = _get_worker_process_env(spec)
    compat_env_defaults = get_gpu_compat_env_defaults(
        {
            **os.environ,
            **env_default_updates,
            **worker_process_env,
        }
    )
    updates = {
        **env_default_updates,
        **compat_env_defaults,
        **worker_process_env,
    }
    if not updates:
        yield
        return

    backup = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class StageGroup:
    """Lifecycle manager for one or more OS processes in a topology group."""

    def __init__(
        self,
        group_name: str,
        process_specs: Sequence[StageWorkerProcessSpec],
    ):
        if not process_specs:
            raise ValueError(
                f"StageGroup requires at least one process spec (group={group_name})"
            )
        self.group_name = group_name
        self.process_specs = list(process_specs)
        self._processes: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.Event] = []
        self._startup_error_channels: list[object] = []

    @property
    def process_count(self) -> int:
        return len(self.process_specs)

    @property
    def specs(self) -> list[StageLaunchConfig]:
        return [
            stage_spec
            for process_spec in self.process_specs
            for stage_spec in process_spec.stage_specs
        ]

    @property
    def leader_spec(self) -> StageLaunchConfig:
        for spec in self.specs:
            if spec.role in {"single", "leader"}:
                return spec
        raise RuntimeError(f"StageGroup {self.group_name} has no leader-owned spec")

    @property
    def leader_endpoint(self) -> str:
        """Control-plane recv endpoint for tp_rank 0 (used by Coordinator)."""
        return self.leader_spec.recv_endpoint

    @property
    def stage_control_endpoints(self) -> dict[str, str]:
        return {
            spec.stage_name: spec.recv_endpoint
            for spec in self.specs
            if spec.owns_external_io
        }

    @property
    def processes(self) -> list[multiprocessing.Process]:
        return list(self._processes)

    def spawn(self, ctx: multiprocessing.context.SpawnContext) -> None:
        """Spawn the OS process(es) owned by this group."""
        for spec in self.process_specs:
            event = ctx.Event()
            startup_error_channel = ctx.Queue()
            proc_name = _process_name(spec)
            proc = ctx.Process(
                target=stage_process_main,
                args=(spec, event, startup_error_channel),
                name=proc_name,
                daemon=True,
            )
            try:
                with _patched_spawn_env(spec):
                    proc.start()
            except Exception:
                _close_queue(startup_error_channel)
                raise
            self._processes.append(proc)
            self._ready_events.append(event)
            self._startup_error_channels.append(startup_error_channel)

        logger.info(
            "StageGroup %s: spawned %d process(es) (pids=%s)",
            self.group_name,
            len(self._processes),
            [p.pid for p in self._processes],
        )

    async def wait_ready(self, timeout: float) -> None:
        """Block until every TP rank signals ready or *timeout* expires."""
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout

        for i, event in enumerate(self._ready_events):
            proc = self._processes[i]
            spec = self.process_specs[i]
            process_label = spec.process_name
            startup_error_channel = self._startup_error_channels[i]

            while not event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    details = ""
                    try:
                        traceback_text = startup_error_channel.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        details = f"\nStartup failure detail:\n{traceback_text}"
                    raise TimeoutError(
                        f"Process {process_label} did not become ready "
                        f"within {timeout:.0f}s{details}"
                    )
                if not proc.is_alive():
                    details = ""
                    try:
                        traceback_text = startup_error_channel.get(timeout=0.2)
                    except queue.Empty:
                        pass
                    else:
                        details = f"\nStartup failure detail:\n{traceback_text}"
                    raise RuntimeError(
                        f"Process {process_label} died during startup "
                        f"(exit code {proc.exitcode}){details}"
                    )
                await loop.run_in_executor(None, event.wait, min(remaining, 1.0))

            logger.info("Process %s ready", process_label)

    def any_dead(self) -> bool:
        """Return True if any process in the group exited while runner is active."""
        return any(not p.is_alive() for p in self._processes)

    def dead_summary(self) -> str:
        """Human-readable summary of dead processes (for error messages)."""
        parts = []
        for i, p in enumerate(self._processes):
            if not p.is_alive():
                process_spec = self.process_specs[i]
                parts.append(
                    f"{process_spec.process_name} " f"(pid={p.pid}, exit={p.exitcode})"
                )
        return ", ".join(parts) if parts else "(none)"

    def close_control_channels(self) -> None:
        for q in self._startup_error_channels:
            _close_queue(q)
        for stage_spec in self.specs:
            for q in (
                stage_spec.follower_work_queues
                + stage_spec.follower_abort_queues
                + stage_spec.follower_admin_result_queues
            ):
                _close_queue(q)

    async def shutdown(self, join_timeout: float = 30.0) -> None:
        try:
            for p in self._processes:
                p.join(timeout=join_timeout)
                if p.is_alive():
                    logger.warning(
                        "Terminating stuck process %s (pid=%s)",
                        p.name,
                        p.pid,
                    )
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=2)
        finally:
            self.close_control_channels()
            self._processes.clear()
            self._ready_events.clear()
            self._startup_error_channels.clear()


def stage_process_main(
    spec: StageWorkerProcessSpec,
    ready_event: multiprocessing.Event,
    startup_error_channel: Any | None = None,
) -> None:
    """Subprocess entrypoint: construct stage(s) from *spec* and run them."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    if not spec.stage_specs:
        raise ValueError(f"Process {spec.process_name!r} requires at least one stage")
    log = logging.getLogger(f"stage_workers.{spec.process_name}")

    try:
        for stage_spec in spec.stage_specs:
            _prepare_cuda_environment(stage_spec, log)
        apply_gpu_compat_env_defaults()
        _run_process(spec, ready_event, log)
    except (KeyboardInterrupt, SystemExit):
        _destroy_torch_distributed_process_group(log)
        _reclaim_process_cuda_memory(
            _stage_gpu_ids(spec.stage_specs),
            log,
            reason=f"stage process {spec.process_name} terminated during startup",
        )
        raise
    except Exception as exc:
        import traceback

        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        with suppress(Exception):
            traceback.clear_frames(exc.__traceback__)
        log.error("Stage process %s failed\n%s", spec.process_name, traceback_text)
        _destroy_torch_distributed_process_group(log)
        _reclaim_process_cuda_memory(
            _stage_gpu_ids(spec.stage_specs),
            log,
            reason=f"stage process {spec.process_name} exit after failure",
        )
        if startup_error_channel is not None:
            startup_error_channel.put(traceback_text)
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
    local_dispatcher = LocalStageDispatcher()
    stages: list[Stage] = []

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
                if stage._running:
                    await stage.stop()

    try:
        for stage_spec in spec.stage_specs:
            stages.append(
                _construct_stage(
                    stage_spec,
                    log,
                    local_dispatcher=local_dispatcher,
                )
            )
        local_dispatcher.register_many(stages)
        asyncio.run(_start_and_run())
    except BaseException:
        _cleanup_constructed_stages(
            stages,
            log,
            reason=f"stage process {spec.process_name} failure",
        )
        raise


def _cleanup_constructed_stages(
    stages: list[Stage],
    log: logging.Logger,
    *,
    reason: str,
) -> None:
    if stages:
        log.warning(
            "Cleaning up %d constructed stage(s) after %s",
            len(stages),
            reason,
        )
    for stage in reversed(stages):
        try:
            asyncio.run(stage.stop())
        except Exception as exc:
            log.warning(
                "Stage %s cleanup failed after process failure: %s",
                stage.name,
                exc,
                exc_info=True,
            )
        finally:
            stage.scheduler = None


def _stage_gpu_ids(stage_specs: Iterable[StageLaunchConfig]) -> list[int]:
    return sorted(
        {
            int(stage_spec.gpu_id)
            for stage_spec in stage_specs
            if stage_spec.gpu_id is not None
        }
    )


def _destroy_torch_distributed_process_group(log: logging.Logger) -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            log.warning("Destroying torch.distributed process group after failure")
            dist.destroy_process_group()
    except Exception as exc:
        log.warning(
            "torch.distributed cleanup failed after stage process failure: %s",
            exc,
            exc_info=True,
        )


def _reclaim_process_cuda_memory(
    gpu_ids: Iterable[int],
    log: logging.Logger,
    *,
    reason: str,
) -> None:
    gpu_id_list = list(gpu_ids)
    if not gpu_id_list:
        return
    gc.collect()
    try:
        import torch

        if not torch.cuda.is_available():
            return
        log.warning(
            "Reclaiming CUDA memory after %s on gpu_ids=%s",
            reason,
            gpu_id_list,
        )
        for gpu_id in gpu_id_list:
            try:
                torch.cuda.set_device(int(gpu_id))
                with suppress(Exception):
                    torch.cuda.synchronize()
                torch.cuda.empty_cache()
                with suppress(Exception):
                    torch.cuda.ipc_collect()
            except Exception as exc:
                log.warning(
                    "CUDA memory reclaim failed for gpu_id=%s after %s: %s",
                    gpu_id,
                    reason,
                    exc,
                    exc_info=True,
                )
        gc.collect()
        log.warning(
            "CUDA memory reclaim complete after %s on gpu_ids=%s",
            reason,
            gpu_id_list,
        )
    except Exception as exc:
        log.warning(
            "CUDA memory reclaim skipped after %s: %s",
            reason,
            exc,
            exc_info=True,
        )


def _construct_stage(
    spec: StageLaunchConfig,
    log: logging.Logger,
    local_dispatcher: LocalStageDispatcher | None = None,
) -> Stage:
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

    def _map_wait_source_list(sources: str | Iterable[str] | None) -> list[Any] | None:
        if sources is None:
            return None
        if isinstance(sources, str):
            return [spec.name_map.get(sources, sources)]
        if isinstance(sources, Iterable):
            return [
                spec.name_map.get(source, source) if isinstance(source, str) else source
                for source in sources
            ]
        raise ValueError(
            f"wait_for_fn for stage {spec.stage_name!r} returned unsupported "
            f"source value {sources!r}"
        )

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
        expected_sources_fn = None
        if spec.wait_for_fn:
            wait_for_fn = import_string(spec.wait_for_fn)

            def expected_sources_fn(request_id, from_stage, data, _fn=wait_for_fn):
                resolved_sources = _fn(request_id, from_stage, data)
                return _map_wait_source_list(resolved_sources)

        input_handler = AggregatedInput(
            sources=sources,
            merge=merge_fn,
            expected_sources_fn=expected_sources_fn,
        )
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
            admin_result_queue=spec.internal_admin_result_queue,
        )

    tp_fanout = None
    if spec.is_leader:
        tp_fanout = TPLeaderFanout(
            stage_name=spec.stage_name,
            follower_work_queues=spec.follower_work_queues,
            follower_abort_queues=spec.follower_abort_queues,
            follower_admin_result_queues=spec.follower_admin_result_queues,
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
        same_process_targets=spec.same_process_targets or None,
        local_dispatcher=local_dispatcher,
        can_accept_stream_before_payload=spec.can_accept_stream_before_payload,
        tp_fanout=tp_fanout,
        is_terminal=spec.is_terminal,
    )

    if spec.is_stream_receiver:
        stage._stream_queue = StreamQueue(max_pending=4096)

    return stage


def _construct_scheduler(
    spec: StageLaunchConfig,
    gpu_id: int | None,
    log: logging.Logger,
) -> Any:
    """Build a scheduler, serializing GPU factory work per visible device."""

    factory = import_string(spec.factory)
    factory_args = resolve_factory_signature_args(
        factory,
        spec.factory_args,
        defaults=spec.factory_arg_defaults,
    )
    if gpu_id is None:
        return factory(**factory_args)

    with gpu_startup_lock(int(gpu_id)) as lock_path:
        log.info(f"Acquired GPU startup lock for stage {spec.stage_name}: {lock_path}")
        return factory(**factory_args)


def get_stage_process_env(
    spec: StageLaunchConfig,
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
    spec: StageLaunchConfig,
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


def _normalize_spec_gpu_id_to_local_device(spec: StageLaunchConfig) -> None:
    spec.gpu_id = 0
    if "gpu_id" in spec.factory_arg_defaults:
        spec.factory_arg_defaults["gpu_id"] = 0
    if "gpu_id" in spec.relay_config:
        spec.relay_config["gpu_id"] = 0


def _process_name(spec: StageWorkerProcessSpec) -> str:
    if len(spec.stage_specs) > 1:
        return f"process-{spec.process_name}"
    stage_spec = spec.stage_specs[0]
    if stage_spec.role == "single":
        return f"stage-{stage_spec.stage_name}"
    if stage_spec.role == "leader":
        return f"stage-{stage_spec.stage_name}-leader"
    return f"stage-{stage_spec.stage_name}-tp{stage_spec.tp_rank}-follower"


def _close_queue(q: object) -> None:
    q.close()
    join_thread = getattr(q, "join_thread", None)
    if callable(join_thread):
        join_thread()
