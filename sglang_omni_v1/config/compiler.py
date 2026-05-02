# SPDX-License-Identifier: Apache-2.0
"""Compile pipeline configuration into runtime objects."""

from __future__ import annotations

import inspect
import socket
from pathlib import Path
from typing import Any

from sglang_omni_v1.config.schema import PipelineConfig, StageConfig
from sglang_omni_v1.pipeline import AggregatedInput, Coordinator, DirectInput, Stage
from sglang_omni_v1.pipeline.control_plane import StageControlPlane
from sglang_omni_v1.pipeline.stage.input import InputHandler
from sglang_omni_v1.utils import import_string


def compile_pipeline(config: PipelineConfig) -> tuple[Coordinator, list[Stage]]:
    """Build the coordinator and stage objects from the pipeline configuration."""
    stages_cfg, name_map, entry_stage = config.apply_fusion()
    endpoints = _allocate_endpoints(config, stages=stages_cfg)

    coordinator = Coordinator(
        completion_endpoint=endpoints["completion"],
        abort_endpoint=endpoints["abort"],
        entry_stage=entry_stage,
        terminal_stages=config.terminal_stages or None,
    )

    stage_endpoints = {s.name: endpoints[f"stage_{s.name}"] for s in stages_cfg}

    stages: list[Stage] = []
    for stage_cfg in stages_cfg:
        stage = _compile_stage(
            stage_cfg, config, stage_endpoints, endpoints, name_map=name_map
        )
        coordinator.register_stage(stage.name, stage.control_plane.recv_endpoint)
        stages.append(stage)

    # Wire streaming targets
    stage_map = {stage.name: stage for stage in stages}
    cfg_map = {s.name: s for s in stages_cfg}
    for stage_cfg in stages_cfg:
        stage = stage_map.get(stage_cfg.name)
        if stage is not None:
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
    factory = import_string(stage_cfg.factory)
    get_next = _resolve_get_next(stage_cfg, name_map)
    input_handler = _create_input_handler(stage_cfg, name_map=name_map)
    factory_args = _resolve_factory_args(stage_cfg, global_cfg)
    project_payload = _resolve_project_payload(stage_cfg, name_map=name_map)
    relay_config = _build_relay_config(stage_cfg, global_cfg)

    scheduler = factory(**factory_args)

    control_plane = StageControlPlane(
        stage_name=stage_cfg.name,
        recv_endpoint=stage_endpoints[stage_cfg.name],
        coordinator_endpoint=endpoints["completion"],
        abort_endpoint=endpoints["abort"],
    )

    return Stage(
        name=stage_cfg.name,
        role="single",
        get_next=get_next,
        gpu_id=relay_config["gpu_id"],
        endpoints=stage_endpoints,
        control_plane=control_plane,
        input_handler=input_handler,
        relay_config=relay_config,
        scheduler=scheduler,
        project_payload=project_payload or None,
    )


# ------------------------------------------------------------------
# Routing
# ------------------------------------------------------------------


def _resolve_get_next(stage_cfg: StageConfig, name_map: dict[str, str]):
    """Build a get_next callable from static ``next``."""
    if stage_cfg.terminal:
        return lambda request_id, output: None

    # Static routing from `next` field
    target = stage_cfg.next
    if isinstance(target, str):
        mapped = name_map.get(target, target)
        return lambda request_id, output, _t=mapped: _t
    if isinstance(target, list):
        mapped = [name_map.get(t, t) for t in target]
        return lambda request_id, output, _t=mapped: _t

    return lambda request_id, output: None


# ------------------------------------------------------------------
# Input handler
# ------------------------------------------------------------------


def _create_input_handler(
    stage_cfg: StageConfig, *, name_map: dict[str, str]
) -> InputHandler:
    if not stage_cfg.wait_for:
        return DirectInput()

    merge_fn = import_string(stage_cfg.merge_fn)
    sources = [name_map.get(n, n) for n in stage_cfg.wait_for]
    return AggregatedInput(sources=set(sources), merge=merge_fn)


# ------------------------------------------------------------------
# Factory args
# ------------------------------------------------------------------


def _resolve_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    """Resolve factory args, injecting model_path and gpu_id if accepted."""
    args = dict(stage_cfg.factory_args)
    stage_overrides = global_cfg.runtime_overrides.get(stage_cfg.name, {})
    if stage_overrides:
        args.update(stage_overrides)
    factory = import_string(stage_cfg.factory)
    sig = inspect.signature(factory)

    if "model_path" in sig.parameters and "model_path" not in args:
        args["model_path"] = global_cfg.model_path

    if "gpu_id" in sig.parameters and "gpu_id" not in args:
        placement = global_cfg.gpu_placement.get(stage_cfg.name, 0)
        gpu_id = placement[0] if isinstance(placement, list) else placement
        args["gpu_id"] = gpu_id

    return args


def _resolve_project_payload(
    stage_cfg: StageConfig,
    *,
    name_map: dict[str, str],
) -> dict[str, Any]:
    project_payload: dict[str, Any] = {}
    for target, dotted_path in stage_cfg.project_payload.items():
        mapped_target = name_map.get(target, target)
        project_payload[mapped_target] = import_string(dotted_path)
    return project_payload


# ------------------------------------------------------------------
# Relay config
# ------------------------------------------------------------------


def _build_relay_config(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    relay_cfg = stage_cfg.relay
    if relay_cfg is not None:
        # Explicit relay config
        return {
            "relay_type": global_cfg.relay_backend,
            "slot_size_mb": relay_cfg.slot_size_mb,
            "credits": relay_cfg.credits,
            "rank": relay_cfg.rank,
            "world_size": relay_cfg.world_size,
            "gpu_id": _parse_gpu_id(relay_cfg.device),
        }

    # Auto-infer from gpu field. For shm, keep transport buffers on CPU:
    # ShmRelay copies tensors to host shared memory anyway, so CUDA staging
    # only inflates GPU allocator pressure.
    if global_cfg.relay_backend == "shm":
        gpu_id = None
    else:
        gpu = stage_cfg.gpu
        if gpu is None:
            gpu_id = None  # CPU stage
        elif isinstance(gpu, list):
            gpu_id = gpu[0]
        else:
            gpu_id = gpu

    return {
        "relay_type": global_cfg.relay_backend,
        "slot_size_mb": 512,
        "credits": 2,
        "rank": None,
        "world_size": None,
        "gpu_id": gpu_id,
    }


def _parse_gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Unsupported device string: {device}")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


def _find_free_tcp_ports(start: int, count: int) -> list[int]:
    """Find *count* available TCP ports starting from *start*.

    Does NOT set SO_REUSEADDR so that TIME_WAIT ports are skipped.
    """
    ports: list[int] = []
    port = start
    while len(ports) < count:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                ports.append(port)
        except OSError:
            pass
        port += 1
    return ports


def _allocate_endpoints(
    config: PipelineConfig,
    *,
    stages: list[StageConfig],
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
        for s in stages:
            endpoints[f"stage_{s.name}"] = f"ipc://{base_dir}/stage_{s.name}.sock"
        return endpoints

    if config.endpoints.scheme == "tcp":
        needed = 2 + len(stages)
        ports = _find_free_tcp_ports(config.endpoints.base_port, needed)
        idx = 0
        if "completion" not in endpoints:
            endpoints["completion"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        if "abort" not in endpoints:
            endpoints["abort"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        for s in stages:
            endpoints[f"stage_{s.name}"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        return endpoints

    raise ValueError(f"Unknown endpoint scheme: {config.endpoints.scheme}")


# ------------------------------------------------------------------
# Stream target wiring
# ------------------------------------------------------------------


def _wire_stream_targets(
    sender_stage: Stage,
    sender_cfg: StageConfig,
    stage_map: dict[str, Stage],
    *,
    gpu_placement: dict[str, int | list[int]] | None = None,
    cfg_map: dict[str, StageConfig] | None = None,
) -> None:
    from sglang_omni_v1.pipeline.stage.stream_queue import StreamQueue

    targets = sender_cfg.stream_to
    if not targets:
        return

    same_gpu = _detect_same_gpu_targets(
        sender_cfg,
        targets,
        gpu_placement=gpu_placement,
        cfg_map=cfg_map,
    )

    sender_stage._stream_targets = targets
    sender_stage._same_gpu_targets = same_gpu

    for target_name in targets:
        receiver = stage_map.get(target_name)
        if receiver is not None and receiver._stream_queue is None:
            receiver._stream_queue = StreamQueue(max_pending=4096)


def _detect_same_gpu_targets(
    sender_cfg: StageConfig,
    targets: list[str],
    *,
    gpu_placement: dict[str, int | list[int]] | None = None,
    cfg_map: dict[str, StageConfig] | None = None,
) -> set[str]:
    if not gpu_placement or not cfg_map:
        return set()
    sender_gpu = _primary_gpu(sender_cfg, gpu_placement)
    if sender_gpu is None:
        return set()
    same: set[str] = set()
    for target_name in targets:
        receiver_cfg = cfg_map.get(target_name)
        if receiver_cfg is None:
            continue
        receiver_gpu = _primary_gpu(receiver_cfg, gpu_placement)
        if receiver_gpu is not None and receiver_gpu == sender_gpu:
            same.add(target_name)
    return same


def _primary_gpu(
    stage_cfg: StageConfig,
    gpu_placement: dict[str, int | list[int]],
) -> int | None:
    """Return the primary (rank 0) GPU id for a stage, or None for CPU stages."""
    raw = gpu_placement.get(stage_cfg.name)
    if raw is None:
        return None
    return raw[0] if isinstance(raw, list) else raw
