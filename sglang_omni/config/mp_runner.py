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
from typing import Any

from sglang_omni.config.compiler import (
    _allocate_endpoints,
    _build_relay_config,
    _create_input_handler,
    _wrap_get_next,
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

    input_handler = _create_input_handler(stage_cfg.input_handler, name_map=name_map)

    stage = Stage(
        name=stage_cfg.name,
        get_next=get_next,
        recv_endpoint=stage_endpoints[stage_cfg.name],
        coordinator_endpoint=all_endpoints["completion"],
        abort_endpoint=all_endpoints["abort"],
        endpoints=stage_endpoints,
        input_handler=input_handler,
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

    for _ in range(stage_cfg.num_workers):
        executor = factory(**stage_cfg.executor.args)
        stage.add_worker(Worker(executor=executor))

    return stage


def _wire_chunk_transfers_local(
    stage: Stage,
    stage_cfg: StageConfig,
    all_stages_cfg: list[StageConfig],
    stage_endpoints: dict[str, str],
) -> None:
    """Wire chunk transfers for a single stage (sender + receiver sides).

    Unlike compiler._wire_chunk_transfers which needs all Stage objects,
    this only needs the local Stage and config for all stages.
    """
    from sglang_omni.pipeline.chunk.mailbox import ChunkMailbox
    from sglang_omni.pipeline.chunk.receiver import ChunkReceiver
    from sglang_omni.pipeline.chunk.sender import ChunkTransferAdapter
    from sglang_omni.pipeline.worker.data_plane import DataPlaneAdapter

    data_plane = DataPlaneAdapter(stage.relay)

    # --- Sender side: this stage sends chunks to other stages ---
    stage.chunk_transfer_targets = {
        ct["to_stage"] for ct in stage_cfg.chunk_transfers if ct.get("bootstrap", True)
    }
    senders = []
    for ct in stage_cfg.chunk_transfers:
        to_stage_name = ct["to_stage"]
        to_stage_endpoint = stage_endpoints.get(to_stage_name, "")

        chunk_sender = ChunkTransferAdapter(
            data_plane=data_plane,
            control_plane=stage.control_plane,
            to_stage=to_stage_name,
            to_stage_endpoint=to_stage_endpoint,
            from_stage=stage.name,
        )
        senders.append(chunk_sender)

        for worker in stage.workers:
            set_fn = getattr(worker.executor, "set_chunk_enqueue_fn", None)
            if callable(set_fn):
                set_fn(chunk_sender.enqueue)
            # Store sender reference for EOS signaling
            senders_list = getattr(worker.executor, "_chunk_senders", None)
            if senders_list is not None:
                senders_list.append(chunk_sender)
            else:
                worker.executor._chunk_senders = [chunk_sender]

    stage.chunk_senders = senders

    # --- Receiver side: other stages send chunks to this stage ---
    is_receiver = any(
        ct.get("to_stage") == stage.name
        for other_cfg in all_stages_cfg
        for ct in other_cfg.chunk_transfers
    )

    if is_receiver:
        mailbox = ChunkMailbox(max_pending=4096)
        stage.chunk_mailbox = mailbox

        for worker in stage.workers:
            if worker.chunk_receiver is None:
                worker.chunk_receiver = ChunkReceiver(
                    data_plane=data_plane,
                    mailbox=mailbox,
                )
            worker.executor._chunk_mailbox = mailbox
            if (
                not hasattr(worker.executor, "_chunk_prefetch_count")
                or worker.executor._chunk_prefetch_count is None
            ):
                worker.executor._chunk_prefetch_count = 4096
            set_feedback_mailbox = getattr(
                worker.executor, "set_feedback_mailbox", None
            )
            if callable(set_feedback_mailbox):
                set_feedback_mailbox(mailbox)


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

        # Wire chunk transfers
        _wire_chunk_transfers_local(stage, stage_cfg, stages_cfg, stage_endpoints)

        log.info("Stage %s ready", stage_name)
        ready_event.set()

        # Run stage event loop (blocks until shutdown message)
        asyncio.run(stage.run())

    except Exception:
        import traceback

        log.error("Stage process failed:\n%s", traceback.format_exc())
        ready_event.set()  # Unblock main process even on failure


class MultiProcessPipelineRunner:
    """Run each pipeline stage in its own OS process.

    Main process runs only the Coordinator. Each stage is spawned as a
    separate multiprocessing.Process that reconstructs its Stage from
    serialized PipelineConfig.
    """

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._coordinator: Coordinator | None = None
        self._processes: list[multiprocessing.Process] = []
        self._completion_task: asyncio.Task | None = None
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

        # 1. Apply fusion, allocate endpoints
        stages_cfg, name_map, entry_stage = self._config.apply_fusion()
        endpoints = _allocate_endpoints(self._config, stages=stages_cfg)

        stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}

        # 2. Create Coordinator in main process (binds ZMQ first)
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

        # 3. Spawn one subprocess per stage
        ready_events: list[multiprocessing.Event] = []

        for stage_cfg in stages_cfg:
            ready = multiprocessing.Event()
            config_dict = _build_stage_process_config(
                pipeline_config=self._config,
                stage_name=stage_cfg.name,
                stage_endpoints=stage_endpoints,
                all_endpoints=endpoints,
                name_map=name_map,
            )
            p = multiprocessing.Process(
                target=_stage_process_entry,
                args=(config_dict, ready),
                name=f"stage-{stage_cfg.name}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            ready_events.append(ready)

        # 4. Wait for all stages to be ready
        for i, event in enumerate(ready_events):
            stage_name = stages_cfg[i].name
            if not event.wait(timeout=timeout):
                raise TimeoutError(
                    f"Stage {stage_name} did not become ready within {timeout}s"
                )
            logger.info("Stage %s ready", stage_name)

        # 5. Check for early process failures
        for i, p in enumerate(self._processes):
            if not p.is_alive() and p.exitcode != 0:
                raise RuntimeError(
                    f"Stage {stages_cfg[i].name} exited with code {p.exitcode}"
                )

        # 6. Register stages with coordinator
        for stage_cfg in stages_cfg:
            self._coordinator.register_stage(
                stage_cfg.name, stage_endpoints[stage_cfg.name]
            )

        self._started = True
        logger.info(
            "MultiProcessPipelineRunner started: %d stages", len(self._processes)
        )

    async def stop(self) -> None:
        """Gracefully stop all stage processes and coordinator."""
        if not self._started:
            return

        # 1. Send shutdown to all stages via coordinator
        try:
            await self._coordinator.shutdown_stages()
        except Exception as e:
            logger.warning("shutdown_stages error: %s", e)

        # 2. Wait for processes to exit
        for p in self._processes:
            p.join(timeout=30)
            if p.is_alive():
                logger.warning("Terminating stuck process %s", p.name)
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)

        # 3. Cancel completion loop and stop coordinator
        if self._completion_task is not None:
            self._completion_task.cancel()
            try:
                await self._completion_task
            except asyncio.CancelledError:
                pass

        await self._coordinator.stop()
        self._processes.clear()
        self._started = False
