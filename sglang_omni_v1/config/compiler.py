# SPDX-License-Identifier: Apache-2.0
"""Compile pipeline configuration into runtime objects."""

from __future__ import annotations

import inspect
import logging
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang_omni_v1.config.schema import PipelineConfig, StageConfig
from sglang_omni_v1.pipeline import AggregatedInput, Coordinator, DirectInput, Stage
from sglang_omni_v1.pipeline.control_plane import StageControlPlane
from sglang_omni_v1.pipeline.stage.input import InputHandler
from sglang_omni_v1.utils import import_string

logger = logging.getLogger(__name__)


class IpcRuntimeDir:
    """Runtime-owned IPC directory for one pipeline instance."""

    def __init__(self, path: Path):
        self.path = path
        self._closed = False

    def __enter__(self) -> IpcRuntimeDir:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"IpcRuntimeDir(path={self.path!r}, closed={self._closed})"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to remove IPC runtime dir %s: %s", self.path, exc)


@dataclass(frozen=True)
class PipelineRuntimePrep:
    """Prepared stage and endpoint state for one pipeline runtime."""

    stages_cfg: list[StageConfig]
    name_map: dict[str, str]
    entry_stage: str
    endpoints: dict[str, str]
    runtime_dir: IpcRuntimeDir | None
    runtime_dir_created_here: bool


@dataclass(frozen=True)
class CompiledPipeline:
    """Compiled coordinator, stages, and optional managed IPC runtime dir."""

    coordinator: Coordinator
    stages: list[Stage]
    runtime_dir: IpcRuntimeDir | None


def create_ipc_runtime_dir(config: PipelineConfig) -> IpcRuntimeDir | None:
    """Create a per-run IPC namespace for one pipeline instance."""
    if config.endpoints.scheme != "ipc":
        return None

    base_root = Path(config.endpoints.base_path)
    base_root.mkdir(parents=True, exist_ok=True)

    namespace_prefix = re.sub(r"[^0-9a-z]+", "-", config.name.lower()).strip("-")
    if not namespace_prefix:
        namespace_prefix = "pipeline"
    path = Path(tempfile.mkdtemp(prefix=f"{namespace_prefix}-", dir=base_root))
    return IpcRuntimeDir(path)


def prepare_pipeline_runtime(
    config: PipelineConfig,
    *,
    ipc_runtime_dir: IpcRuntimeDir | None = None,
) -> PipelineRuntimePrep:
    """Prepare fused stages and endpoint allocation for one runtime.

    Caller-provided IPC runtime dirs stay caller-owned. This helper only
    closes runtime dirs it creates internally.
    """
    runtime_dir = ipc_runtime_dir
    created_runtime_dir = None
    if runtime_dir is None:
        runtime_dir = create_ipc_runtime_dir(config)
        created_runtime_dir = runtime_dir
    runtime_dir_created_here = created_runtime_dir is not None

    try:
        stages_cfg, name_map, entry_stage = config.apply_fusion()
        endpoints = _allocate_endpoints(
            config,
            stages=stages_cfg,
            ipc_base_dir=runtime_dir.path if runtime_dir else None,
        )
    except Exception:
        if created_runtime_dir is not None:
            created_runtime_dir.close()
        raise

    return PipelineRuntimePrep(
        stages_cfg=stages_cfg,
        name_map=name_map,
        entry_stage=entry_stage,
        endpoints=endpoints,
        runtime_dir=runtime_dir,
        runtime_dir_created_here=runtime_dir_created_here,
    )


def compile_pipeline_core(
    config: PipelineConfig,
    *,
    ipc_runtime_dir: IpcRuntimeDir | None = None,
) -> CompiledPipeline:
    """Build coordinator and stages, returning any managed runtime dir.

    Note (Chenyang, Ratish):
        1. If the caller passes ipc_runtime_dir, that dir is caller-owned for
           the full lifetime — this function never closes it, on success or on
           failure. The same dir is returned back so callers can keep using it.
        2. If the caller does not pass one, this function may create a new
           runtime dir internally; that dir is closed automatically on failure
           and returned to the caller on success. The caller MUST close it.
    """
    prep = prepare_pipeline_runtime(
        config,
        ipc_runtime_dir=ipc_runtime_dir,
    )

    try:
        coordinator = Coordinator(
            completion_endpoint=prep.endpoints["completion"],
            abort_endpoint=prep.endpoints["abort"],
            entry_stage=prep.entry_stage,
            terminal_stages=config.terminal_stages or None,
        )

        stage_endpoints = {
            s.name: prep.endpoints[f"stage_{s.name}"] for s in prep.stages_cfg
        }

        stages: list[Stage] = []
        for stage_cfg in prep.stages_cfg:
            stage = _compile_stage(
                stage_cfg,
                config,
                stage_endpoints,
                prep.endpoints,
                name_map=prep.name_map,
            )
            coordinator.register_stage(stage.name, stage.control_plane.recv_endpoint)
            stages.append(stage)

        stage_map = {stage.name: stage for stage in stages}
        cfg_map = {s.name: s for s in prep.stages_cfg}
        for stage_cfg in prep.stages_cfg:
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
    except Exception:
        if prep.runtime_dir_created_here and prep.runtime_dir is not None:
            prep.runtime_dir.close()
        raise

    return CompiledPipeline(
        coordinator=coordinator,
        stages=stages,
        runtime_dir=prep.runtime_dir,
    )


def compile_pipeline(config: PipelineConfig) -> tuple[Coordinator, list[Stage]]:
    """Build coordinator and stages directly from a pipeline config.

    This thin helper is TCP-only. For IPC, use compile_pipeline_core(...)
    with explicit runtime-dir cleanup, or MultiProcessPipelineRunner.
    """
    if config.endpoints.scheme == "ipc":
        raise ValueError(
            "compile_pipeline() does not manage IPC runtime-dir ownership. "
            "Use MultiProcessPipelineRunner, or compile_pipeline_core(...) "
            "directly: either let it self-manage the runtime dir, or pair it "
            "with create_ipc_runtime_dir(...) for caller-managed ownership."
        )

    compiled = compile_pipeline_core(config)
    return compiled.coordinator, compiled.stages


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
    ipc_base_dir: Path | None = None,
) -> dict[str, str]:
    endpoints: dict[str, str] = {}

    if config.completion_endpoint:
        endpoints["completion"] = config.completion_endpoint
    if config.abort_endpoint:
        endpoints["abort"] = config.abort_endpoint

    if config.endpoints.scheme == "ipc":
        if ipc_base_dir is None:
            raise ValueError("IPC endpoint allocation requires an IPC runtime dir")
        base_dir = ipc_base_dir
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
