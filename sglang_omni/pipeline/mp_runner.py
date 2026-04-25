# SPDX-License-Identifier: Apache-2.0
"""Multi-process pipeline runner.

Spawns each pipeline stage in its own OS process. Main process runs only
the Coordinator. Stages communicate via ZMQ (control plane) and relay
(data plane) — same protocols as single-process, now cross-process.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import time
from contextlib import suppress
from typing import Any

from sglang_omni.config.compiler import (
    IpcRuntimeDir,
    _build_relay_config,
    _create_input_handler,
    _wrap_get_next,
    prepare_pipeline_runtime,
)
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.pipeline import Coordinator, Stage, Worker
from sglang_omni.utils import import_string

logger = logging.getLogger(__name__)


def _noop_executor_factory(model_path: str = "", **kwargs):
    """No-op executor factory for testing."""
    from sglang_omni.executors import PreprocessingExecutor

    return PreprocessingExecutor(lambda payload: payload)


def _noop_get_next(request_id: str, output: Any) -> None:
    """No-op get_next for testing — terminal stage."""
    return None


def _build_stage_process_config(
    *,
    pipeline_config: PipelineConfig,
    stage_name: str,
    stage_endpoints: dict[str, str],
    all_endpoints: dict[str, str],
    name_map: dict[str, str],
) -> dict[str, Any]:
    """Build a picklable config dict for a stage subprocess."""
    return {
        "pipeline_config": pipeline_config.model_dump(),
        "stage_name": stage_name,
        "stage_endpoints": stage_endpoints,
        "all_endpoints": all_endpoints,
        "name_map": name_map,
    }


def _resolve_relay_config(
    stage_cfg: StageConfig, global_cfg: PipelineConfig
) -> dict[str, Any]:
    """Build relay config with gpu_id from gpu_placement (not relay.device).

    The base _build_relay_config uses relay.device to determine gpu_id,
    which defaults to 0 for "cuda". For multi-process deployment, we
    override with the actual gpu_placement value.
    """
    relay_config = _build_relay_config(stage_cfg, global_cfg)

    # Override gpu_id from gpu_placement when relay is on CUDA
    if stage_cfg.relay.device != "cpu":
        placement_gpu = global_cfg.gpu_placement.get(stage_cfg.name)
        if placement_gpu is not None:
            relay_config["gpu_id"] = placement_gpu

    return relay_config


def _compile_stage_local(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
    stage_endpoints: dict[str, str],
    all_endpoints: dict[str, str],
    name_map: dict[str, str],
) -> Stage:
    """Compile a single stage in the current process.

    Same logic as compiler._compile_stage but uses _resolve_relay_config
    for correct GPU placement in multi-process mode.
    """
    factory = import_string(stage_cfg.executor.factory)
    if not callable(factory):
        raise TypeError(f"Executor factory not callable: {stage_cfg.executor.factory}")

    get_next = import_string(stage_cfg.get_next)
    if not callable(get_next):
        raise TypeError(f"get_next not callable: {stage_cfg.get_next}")
    get_next = _wrap_get_next(get_next, name_map)

    payload_filter = None
    if stage_cfg.payload_filter:
        payload_filter = import_string(stage_cfg.payload_filter)
        if not callable(payload_filter):
            raise TypeError(f"payload_filter not callable: {stage_cfg.payload_filter}")

    input_handler = _create_input_handler(stage_cfg.input_handler, name_map=name_map)

    stage = Stage(
        name=stage_cfg.name,
        get_next=get_next,
        recv_endpoint=stage_endpoints[stage_cfg.name],
        coordinator_endpoint=all_endpoints["completion"],
        abort_endpoint=all_endpoints["abort"],
        endpoints=stage_endpoints,
        input_handler=input_handler,
        payload_filter=payload_filter,
        relay_config=_resolve_relay_config(stage_cfg, global_cfg),
    )

    # Inject model_path and gpu_id into executor args (same as compiler)
    if (
        "model_path" in inspect.signature(factory).parameters
        and "model_path" not in stage_cfg.executor.args
    ):
        stage_cfg.executor.args["model_path"] = global_cfg.model_path

    if (
        "gpu_id" in inspect.signature(factory).parameters
        and "gpu_id" not in stage_cfg.executor.args
    ):
        gpu_id = global_cfg.gpu_placement.get(stage_cfg.name, 0)
        stage_cfg.executor.args["gpu_id"] = gpu_id

    # Also translate gpu_placement to device string for stages using "device"
    # instead of "gpu_id" (e.g., talker, audio_encoder).
    if "device" in inspect.signature(
        factory
    ).parameters and stage_cfg.executor.args.get("device", "").startswith("cuda"):
        gpu_id = global_cfg.gpu_placement.get(stage_cfg.name, 0)
        stage_cfg.executor.args["device"] = f"cuda:{gpu_id}"

    for _ in range(stage_cfg.num_workers):
        executor = factory(**stage_cfg.executor.args)
        stage.add_worker(Worker(executor=executor))

    return stage


def _wire_stream_targets_local(
    stage: Stage,
    stage_cfg: StageConfig,
    all_stages_cfg: list[StageConfig],
    stage_endpoints: dict[str, str],
    *,
    gpu_placement: dict[str, int] | None = None,
) -> None:
    """Wire stream_to targets for a single stage (sender + receiver sides).

    Uses the same StreamQueue-based pattern as compiler._wire_stream_targets
    but works with only the local Stage and config for all stages.
    """
    from sglang_omni.config.compiler import _detect_same_gpu_targets
    from sglang_omni.pipeline.stage.stream_queue import StreamQueue

    # --- Sender side: set stream targets and wire stream_fn ---
    targets = stage_cfg.stream_to
    if targets:
        all_targets = [t.to_stage for t in targets]
        bootstrap_targets = {t.to_stage for t in targets if t.bootstrap}

        cfg_map = {s.name: s for s in all_stages_cfg}
        same_gpu_targets = _detect_same_gpu_targets(
            stage_cfg,
            targets,
            gpu_placement=gpu_placement,
            cfg_map=cfg_map,
        )

        for worker in stage.workers:
            worker._stream_targets = all_targets
            worker._bootstrap_targets = bootstrap_targets
            worker._same_gpu_targets = same_gpu_targets
            # Wire stream_fn: executor calls worker._enqueue_stream
            set_fn = getattr(worker.executor, "set_stream_fn", None)
            if callable(set_fn):
                set_fn(worker._enqueue_stream)

    # --- Receiver side: other stages stream to this stage ---
    is_receiver = any(
        any(t.to_stage == stage.name for t in other_cfg.stream_to)
        for other_cfg in all_stages_cfg
    )

    if is_receiver:
        if stage._stream_queue is None:
            queue = StreamQueue(max_pending=4096)
            stage._stream_queue = queue
        else:
            queue = stage._stream_queue

        for worker in stage.workers:
            worker.executor._stream_queue = queue
            set_feedback_mailbox = getattr(
                worker.executor, "set_feedback_mailbox", None
            )
            if callable(set_feedback_mailbox):
                set_feedback_mailbox(queue)


def _stage_process_entry(
    config_dict: dict[str, Any],
    ready_event: multiprocessing.Event,
) -> None:
    """Subprocess entrypoint: reconstruct and run a single Stage.

    1. Deserialize PipelineConfig from dict
    2. Find this stage's StageConfig
    3. Create Stage (relay, executor, workers)
    4. Wire chunk transfers (sender + receiver)
    5. Signal ready
    6. Run stage.run() until shutdown
    """
    import logging
    import sys

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    log = logging.getLogger(f"stage.{config_dict['stage_name']}")

    try:
        stage_name = config_dict["stage_name"]
        stage_endpoints = config_dict["stage_endpoints"]
        all_endpoints = config_dict["all_endpoints"]
        name_map = config_dict["name_map"]

        # Reconstruct PipelineConfig from serialized dict
        pipeline_config = PipelineConfig(**config_dict["pipeline_config"])

        # Set default CUDA device early based on gpu_placement, before
        # any CUDA initialization.  This prevents PyTorch from allocating
        # the context on the wrong GPU in multi-GPU deployments.
        gpu_id = pipeline_config.gpu_placement.get(stage_name, 0)
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > gpu_id:
            torch.cuda.set_device(gpu_id)
            log.info("Set CUDA device to cuda:%d for stage %s", gpu_id, stage_name)

        # Apply fusion to get actual stage configs
        stages_cfg, fused_name_map, _ = pipeline_config.apply_fusion()
        name_map.update(fused_name_map)

        # Find this stage's config
        stage_cfg = next((s for s in stages_cfg if s.name == stage_name), None)
        if stage_cfg is None:
            log.error("Stage %s not found in config", stage_name)
            return

        log.info("Compiling stage %s...", stage_name)

        # Compile stage (creates relay, loads executor/model, adds workers)
        stage = _compile_stage_local(
            stage_cfg, pipeline_config, stage_endpoints, all_endpoints, name_map
        )

        # Wire stream targets
        _wire_stream_targets_local(
            stage,
            stage_cfg,
            stages_cfg,
            stage_endpoints,
            gpu_placement=pipeline_config.gpu_placement,
        )

        # Start the stage (opens connections, binds ZMQ sockets) before
        # signalling ready.  stage.run() calls start() internally but it
        # is idempotent, so this is safe.
        async def _start_and_run() -> None:
            await stage.start()
            log.info("Stage %s ready", stage_name)
            ready_event.set()
            await stage.run()

        asyncio.run(_start_and_run())

    except Exception:
        import traceback

        log.error("Stage process failed:\n%s", traceback.format_exc())
        # Exit with non-zero code so the parent monitor can detect the failure.
        sys.exit(1)


class MultiProcessPipelineRunner:
    """Run each pipeline stage in its own OS process.

    Main process runs only the Coordinator. Each stage is spawned as a
    separate multiprocessing.Process that reconstructs its Stage from
    serialized PipelineConfig.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._coordinator: Coordinator | None = None
        self._ipc_runtime_dir: IpcRuntimeDir | None = None
        self._processes: list[multiprocessing.Process] = []
        self._completion_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._started = False

    @property
    def coordinator(self) -> Coordinator:
        if self._coordinator is None:
            raise RuntimeError("Runner not started")
        return self._coordinator

    async def start(self, timeout: float = 120.0) -> None:
        """Start coordinator and spawn stage subprocesses.

        Args:
            timeout: Max seconds to wait for all stages to be ready.
        """
        if self._started:
            raise RuntimeError("Already started")

        (
            stages_cfg,
            name_map,
            entry_stage,
            endpoints,
            self._ipc_runtime_dir,
            _,
        ) = prepare_pipeline_runtime(
            self._config,
            ipc_runtime_dir=self._ipc_runtime_dir,
        )

        try:
            stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}

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

            ctx = multiprocessing.get_context("spawn")
            ready_events: list[multiprocessing.Event] = []

            for stage_cfg in stages_cfg:
                ready = ctx.Event()
                config_dict = _build_stage_process_config(
                    pipeline_config=self._config,
                    stage_name=stage_cfg.name,
                    stage_endpoints=stage_endpoints,
                    all_endpoints=endpoints,
                    name_map=name_map,
                )
                # Stages that spawn TP follower subprocesses (tp_size > 1)
                # cannot be daemon — Python forbids daemon children.
                needs_children = (
                    stage_cfg.executor
                    and stage_cfg.executor.args
                    and stage_cfg.executor.args.get("server_args_overrides", {}).get(
                        "tp_size", 1
                    )
                    > 1
                )
                p = ctx.Process(
                    target=_stage_process_entry,
                    args=(config_dict, ready),
                    name=f"stage-{stage_cfg.name}",
                    daemon=not needs_children,
                )
                p.start()
                self._processes.append(p)
                ready_events.append(ready)

            loop = asyncio.get_running_loop()
            for i, event in enumerate(ready_events):
                stage_name = stages_cfg[i].name
                p = self._processes[i]
                deadline = time.monotonic() + timeout
                while not event.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"Stage {stage_name} did not become ready within {timeout}s"
                        )
                    # Check if process died before signalling ready
                    if not p.is_alive():
                        raise RuntimeError(
                            f"Stage {stage_name} process died during startup "
                            f"(exit code {p.exitcode})"
                        )
                    await loop.run_in_executor(None, event.wait, min(remaining, 1.0))
                logger.info("Stage %s ready", stage_name)

            for i, p in enumerate(self._processes):
                if not p.is_alive() and p.exitcode != 0:
                    raise RuntimeError(
                        f"Stage {stages_cfg[i].name} exited with code {p.exitcode}"
                    )

            for stage_cfg in stages_cfg:
                self._coordinator.register_stage(
                    stage_cfg.name, stage_endpoints[stage_cfg.name]
                )

            self._started = True
            self._monitor_task = asyncio.create_task(self._monitor_children())
            logger.info(
                "MultiProcessPipelineRunner started: %d stages", len(self._processes)
            )

        except Exception:
            # Rollback: kill any spawned processes to avoid leaks
            for p in self._processes:
                if p.is_alive():
                    p.terminate()
            for p in self._processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)
            self._processes.clear()

            # Cancel completion loop if started
            if self._completion_task is not None:
                self._completion_task.cancel()
                try:
                    await self._completion_task
                except asyncio.CancelledError:
                    pass
                self._completion_task = None

            # Stop coordinator if started
            if self._coordinator is not None:
                try:
                    await self._coordinator.stop()
                except Exception:
                    pass
                self._coordinator = None

            if self._ipc_runtime_dir is not None:
                self._ipc_runtime_dir.close()
                self._ipc_runtime_dir = None

            raise

    async def _monitor_children(self) -> None:
        """Periodically check that all stage processes are alive."""
        while self._started:
            for i, p in enumerate(self._processes):
                if not p.is_alive():
                    logger.error(
                        "Stage process %d (pid=%d) died with exitcode=%s",
                        i,
                        p.pid,
                        p.exitcode,
                    )
                    # Trigger shutdown
                    await self.stop()
                    return
            await asyncio.sleep(5.0)

    async def stop(self) -> None:
        """Gracefully stop all stage processes and coordinator."""
        if not self._started:
            return
        self._started = False

        # Cancel the monitor task only if we are not being called from it
        if self._monitor_task is not None:
            current_task = asyncio.current_task()
            if current_task != self._monitor_task:
                self._monitor_task.cancel()
            self._monitor_task = None

        try:
            await self._coordinator.shutdown_stages()
        except Exception as e:
            logger.warning("shutdown_stages error: %s", e)

        for p in self._processes:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning("Terminating stuck process %s", p.name)
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)

        if self._completion_task is not None:
            self._completion_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._completion_task
            self._completion_task = None

        try:
            await self._coordinator.stop()
            self._processes.clear()
        finally:
            if self._ipc_runtime_dir is not None:
                self._ipc_runtime_dir.close()
                self._ipc_runtime_dir = None
