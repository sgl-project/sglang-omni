# SPDX-License-Identifier: Apache-2.0
"""Compile pipeline configuration into runtime objects."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from sglang_omni.config.schema import InputHandlerConfig, PipelineConfig, StageConfig
from sglang_omni.executors.interface import Executor
from sglang_omni.pipeline import (
    AggregatedInput,
    Coordinator,
    DirectInput,
    Stage,
    Worker,
)
from sglang_omni.pipeline.stage.input import InputHandler
from sglang_omni.utils import import_string


def compile_pipeline(config: PipelineConfig) -> tuple[Coordinator, list[Stage]]:
    """
    Build the coordinator and stage objects from the pipeline configuration.
    """
    # 1. apply stage fusion if enabled
    stages_cfg, name_map, entry_stage = config.apply_fusion()

    # 3. allocate ZMQ endpoints
    endpoints = _allocate_endpoints(config, stages=stages_cfg)

    # 4. create coordinator
    coordinator = Coordinator(
        completion_endpoint=endpoints["completion"],
        abort_endpoint=endpoints["abort"],
        entry_stage=entry_stage,
        terminal_stages=config.terminal_stages or None,
    )

    # 5. create each stage in order
    stage_endpoints = {
        stage_cfg.name: endpoints[f"stage_{stage_cfg.name}"] for stage_cfg in stages_cfg
    }

    stages: list[Stage] = []
    for stage_cfg in stages_cfg:
        stage = _compile_stage(
            stage_cfg, config, stage_endpoints, endpoints, name_map=name_map
        )
        coordinator.register_stage(stage.name, stage.control_plane.recv_endpoint)
        stages.append(stage)

    # 6. wire chunk transfers
    stage_map = {stage.name: stage for stage in stages}
    for stage_cfg in stages_cfg:
        stage = stage_map.get(stage_cfg.name)
        if stage is None:
            continue
        _wire_chunk_transfers(stage, stage_cfg, stage_map, stage_endpoints)

    return coordinator, stages


def _compile_stage(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
    stage_endpoints: dict[str, str],
    endpoints: dict[str, str],
    *,
    name_map: dict[str, str],
) -> Stage:
    factory = import_string(stage_cfg.executor.factory)
    if not callable(factory):
        raise TypeError(
            f"Executor factory is not callable: {stage_cfg.executor.factory}"
        )

    get_next = import_string(stage_cfg.get_next)
    if not callable(get_next):
        raise TypeError(f"get_next is not callable: {stage_cfg.get_next}")
    get_next = _wrap_get_next(get_next, name_map)

    input_handler = _create_input_handler(stage_cfg.input_handler, name_map=name_map)

    stage = Stage(
        name=stage_cfg.name,
        get_next=get_next,
        recv_endpoint=stage_endpoints[stage_cfg.name],
        coordinator_endpoint=endpoints["completion"],
        abort_endpoint=endpoints["abort"],
        endpoints=stage_endpoints,
        input_handler=input_handler,
        relay_config=_build_relay_config(stage_cfg, global_cfg),
    )

    # check if factory has the signature of model_path and the user does not provide the model path
    # if yes, use the one in global config
    if (
        "model_path" in inspect.signature(factory).parameters
        and "model_path" not in stage_cfg.executor.args
    ):
        stage_cfg.executor.args["model_path"] = global_cfg.model_path

    # Inject gpu_id from gpu_placement map
    if (
        "gpu_id" in inspect.signature(factory).parameters
        and "gpu_id" not in stage_cfg.executor.args
    ):
        gpu_id = global_cfg.gpu_placement.get(stage_cfg.name, 0)
        stage_cfg.executor.args["gpu_id"] = gpu_id

    for _ in range(stage_cfg.num_workers):
        executor = factory(**stage_cfg.executor.args)
        if not isinstance(executor, Executor):
            raise TypeError(
                f"Executor factory {stage_cfg.executor.factory} returned "
                f"{type(executor)}"
            )
        stage.add_worker(Worker(executor=executor))

    return stage


def _create_input_handler(
    config: InputHandlerConfig, *, name_map: dict[str, str]
) -> InputHandler:
    if config.type == "direct":
        return DirectInput()

    if not config.sources:
        raise ValueError("Aggregated input handler requires sources")
    if not config.merge_fn:
        raise ValueError("Aggregated input handler requires merge_fn")

    merge_fn = import_string(config.merge_fn)
    if not callable(merge_fn):
        raise TypeError(f"merge_fn is not callable: {config.merge_fn}")

    sources = [_map_stage_name(name_map, name) for name in config.sources]
    sources = _dedupe_list(sources)
    return AggregatedInput(sources=set(sources), merge=merge_fn)


def _build_relay_config(
    stage_cfg: StageConfig, global_cfg: PipelineConfig
) -> dict[str, Any]:
    relay_cfg = stage_cfg.relay
    return {
        "relay_type": global_cfg.relay_backend,
        "slot_size_mb": relay_cfg.slot_size_mb,
        "credits": relay_cfg.credits,
        "rank": relay_cfg.rank,
        "world_size": relay_cfg.world_size,
        "gpu_id": _parse_gpu_id(relay_cfg.device),
    }


def _parse_gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        index = device.split(":", 1)[1]
        if not index:
            raise ValueError("CUDA device index is required after 'cuda:'")
        return int(index)
    raise ValueError(f"Unsupported device string: {device}")


def _allocate_endpoints(
    config: PipelineConfig, *, stages: list[StageConfig]
) -> dict[str, str]:
    endpoints: dict[str, str] = {}

    if config.completion_endpoint:
        endpoints["completion"] = config.completion_endpoint
    if config.abort_endpoint:
        endpoints["abort"] = config.abort_endpoint

    if config.endpoints.scheme == "ipc":
        base_dir = Path(config.endpoints.base_path) / config.name
        base_dir.mkdir(parents=True, exist_ok=True)

        endpoints.setdefault("completion", f"ipc://{base_dir}/completion.sock")
        endpoints.setdefault("abort", f"ipc://{base_dir}/abort.sock")

        for stage_cfg in stages:
            endpoints[f"stage_{stage_cfg.name}"] = (
                f"ipc://{base_dir}/stage_{stage_cfg.name}.sock"
            )
        return endpoints

    if config.endpoints.scheme == "tcp":
        port = config.endpoints.base_port
        if "completion" not in endpoints:
            endpoints["completion"] = f"tcp://127.0.0.1:{port}"
            port += 1
        if "abort" not in endpoints:
            endpoints["abort"] = f"tcp://127.0.0.1:{port}"
            port += 1

        for stage_cfg in stages:
            endpoints[f"stage_{stage_cfg.name}"] = f"tcp://127.0.0.1:{port}"
            port += 1
        return endpoints

    raise ValueError(f"Unknown endpoint scheme: {config.endpoints.scheme}")


def _wrap_get_next(get_next: Any, name_map: dict[str, str]):
    def _wrapped(request_id: str, output: Any):
        result = get_next(request_id, output)
        return _remap_next(result, name_map)

    return _wrapped


def _remap_next(value: Any, name_map: dict[str, str]) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return _map_stage_name(name_map, value)
    if isinstance(value, list):
        remapped = [_map_stage_name(name_map, item) for item in value]
        return _dedupe_list(remapped)
    return value


def _map_stage_name(name_map: dict[str, str], name: str) -> str:
    return name_map.get(name, name)


def _dedupe_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _wire_chunk_transfers(
    sender_stage: Stage,
    sender_cfg: StageConfig,
    stage_map: dict[str, Stage],
    stage_endpoints: dict[str, str],
) -> None:
    """Wire chunk transfer infrastructure between stages."""
    from sglang_omni.pipeline.chunk.mailbox import ChunkMailbox
    from sglang_omni.pipeline.chunk.receiver import ChunkReceiver
    from sglang_omni.pipeline.chunk.sender import ChunkTransferAdapter
    from sglang_omni.pipeline.worker.data_plane import DataPlaneAdapter

    transfers = sender_cfg.chunk_transfers
    if not transfers:
        sender_stage.chunk_senders = []
        sender_stage.chunk_transfer_targets = set()
        return

    sender_stage.chunk_transfer_targets = {
        ct["to_stage"] for ct in transfers if ct.get("bootstrap", True)
    }
    senders = []
    for ct in transfers:
        to_stage_name = ct["to_stage"]
        receiver_stage = stage_map.get(to_stage_name)
        if receiver_stage is None:
            continue

        to_stage_endpoint = stage_endpoints.get(to_stage_name, "")

        # Create sender-side adapter
        sender_data_plane = DataPlaneAdapter(sender_stage.relay)
        chunk_sender = ChunkTransferAdapter(
            data_plane=sender_data_plane,
            control_plane=sender_stage.control_plane,
            to_stage=to_stage_name,
            to_stage_endpoint=to_stage_endpoint,
            from_stage=sender_stage.name,
        )
        senders.append(chunk_sender)

        # Wire sender's enqueue to each worker executor that has set_chunk_enqueue_fn
        for worker in sender_stage.workers:
            set_fn = getattr(worker.executor, "set_chunk_enqueue_fn", None)
            if callable(set_fn):
                set_fn(chunk_sender.enqueue)
            # Store sender reference for EOS signaling
            senders_list = getattr(worker.executor, "_chunk_senders", None)
            if senders_list is not None:
                senders_list.append(chunk_sender)
            else:
                worker.executor._chunk_senders = [chunk_sender]

        # Create receiver-side: mailbox + receiver per worker
        if not hasattr(receiver_stage, "chunk_mailbox") or receiver_stage.chunk_mailbox is None:
            mailbox = ChunkMailbox(max_pending=4096)
            receiver_stage.chunk_mailbox = mailbox
        else:
            mailbox = receiver_stage.chunk_mailbox

        for worker in receiver_stage.workers:
            if worker.chunk_receiver is None:
                receiver_data_plane = DataPlaneAdapter(receiver_stage.relay)
                worker.chunk_receiver = ChunkReceiver(
                    data_plane=receiver_data_plane,
                    mailbox=mailbox,
                )
            # Set chunk_mailbox on executor for async prefetch
            worker.executor._chunk_mailbox = mailbox
            if not hasattr(worker.executor, "_chunk_prefetch_count") or worker.executor._chunk_prefetch_count is None:
                worker.executor._chunk_prefetch_count = 4096  # default: drain until EOS
            set_feedback_mailbox = getattr(worker.executor, "set_feedback_mailbox", None)
            if callable(set_feedback_mailbox):
                set_feedback_mailbox(mailbox)

    sender_stage.chunk_senders = senders
