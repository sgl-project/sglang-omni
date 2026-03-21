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

    # 6. wire stream targets
    stage_map = {stage.name: stage for stage in stages}
    cfg_map = {s.name: s for s in stages_cfg}
    for stage_cfg in stages_cfg:
        stage = stage_map.get(stage_cfg.name)
        if stage is None:
            continue
        _wire_stream_targets(
            stage,
            stage_cfg,
            stage_map,
            gpu_placement=config.gpu_placement,
            cfg_map=cfg_map,
        )

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


def _wire_stream_targets(
    sender_stage: Stage,
    sender_cfg: StageConfig,
    stage_map: dict[str, Stage],
    *,
    gpu_placement: dict[str, int] | None = None,
    cfg_map: dict[str, StageConfig] | None = None,
) -> None:
    """Wire stream_to targets between stages.

    For each stream_to entry on the sender:
    1. Set worker._stream_targets and worker._bootstrap_targets
    2. Detect same-GPU targets and set worker._same_gpu_targets (CUDA IPC)
    3. Create StreamQueue on receiver stages
    4. Set executor._stream_queue on receiver executors
    5. Wire set_stream_fn on sender executors
    """
    from sglang_omni.pipeline.stage.stream_queue import StreamQueue

    targets = sender_cfg.stream_to
    if not targets:
        return

    # Collect target stage names and bootstrap targets
    all_targets = [t.to_stage for t in targets]
    bootstrap_targets = {t.to_stage for t in targets if t.bootstrap}

    # Detect same-GPU targets for CUDA IPC zero-copy
    same_gpu_targets = _detect_same_gpu_targets(
        sender_cfg,
        targets,
        gpu_placement=gpu_placement,
        cfg_map=cfg_map,
    )

    # Set stream targets on sender workers and wire stream_fn.
    for worker in sender_stage.workers:
        worker._stream_targets = all_targets
        worker._bootstrap_targets = bootstrap_targets
        worker._same_gpu_targets = same_gpu_targets
        set_target = getattr(worker.executor, "set_stream_target", None)
        if callable(set_target):
            if len(all_targets) != 1:
                raise ValueError(
                    f"Executor for stage {sender_stage.name!r} requires exactly one "
                    "stream_to target"
                )
            set_target(all_targets[0])
        # Wire stream_fn: executor calls worker._enqueue_stream
        set_fn = getattr(worker.executor, "set_stream_fn", None)
        if callable(set_fn):
            set_fn(worker._enqueue_stream)

    # Set up receiver side: create StreamQueue on receiver stages
    for target_cfg in targets:
        receiver_stage = stage_map.get(target_cfg.to_stage)
        if receiver_stage is None:
            continue

        # Create a shared StreamQueue for the receiver stage if not already present
        if receiver_stage._stream_queue is None:
            queue = StreamQueue(max_pending=4096)
            receiver_stage._stream_queue = queue
        else:
            queue = receiver_stage._stream_queue

        # Wire stream queue to receiver executors
        for worker in receiver_stage.workers:
            worker.executor._stream_queue = queue
            # Wire feedback mailbox for executors that support it
            set_feedback_mailbox = getattr(
                worker.executor, "set_feedback_mailbox", None
            )
            if callable(set_feedback_mailbox):
                set_feedback_mailbox(queue)


def _detect_same_gpu_targets(
    sender_cfg: StageConfig,
    targets: list,
    *,
    gpu_placement: dict[str, int] | None = None,
    cfg_map: dict[str, StageConfig] | None = None,
) -> set[str]:
    """Return the set of target stage names that share a GPU with the sender.

    Same-GPU streaming uses CUDA IPC (zero data copy) instead of the relay.
    Both the sender and receiver must use CUDA relays and be placed on the
    same GPU (per ``gpu_placement``) for this to activate.
    """
    if not gpu_placement or not cfg_map:
        return set()

    sender_gpu = gpu_placement.get(sender_cfg.name)
    if sender_gpu is None:
        return set()

    # Sender must have a CUDA relay
    if sender_cfg.relay.device == "cpu":
        return set()

    same: set[str] = set()
    for target in targets:
        receiver_cfg = cfg_map.get(target.to_stage)
        if receiver_cfg is None:
            continue
        # Receiver must also have a CUDA relay
        if receiver_cfg.relay.device == "cpu":
            continue
        receiver_gpu = gpu_placement.get(target.to_stage)
        if receiver_gpu is not None and receiver_gpu == sender_gpu:
            same.add(target.to_stage)

    return same
