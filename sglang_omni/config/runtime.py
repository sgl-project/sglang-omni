# SPDX-License-Identifier: Apache-2.0
"""Resolve typed runtime config into stage factory arguments."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.utils.imports import import_string

logger = logging.getLogger(__name__)

_MAPPED_STAGE_RUNTIME_FIELDS = ("max_seq_len", "video_fps")
_TP_LAUNCH_PARAMS = frozenset({"tp_rank", "tp_size", "nccl_port"})
_ENCODER_ACTIVATION_BUDGET_KEY = "encoder_activation_budget_bytes"
_ENCODER_MAX_BATCH_SIZE_KEY = "encoder_max_batch_size"


@dataclass(frozen=True)
class StageLaunchMode:
    """Launcher-visible backend decision for one stage.

    `requested_backend` is the value present in resolved factory kwargs after
    `factory_args` and `runtime_overrides` merge. Factory signature defaults
    are intentionally ignored so launch decisions cannot diverge from config.
    """

    stage_name: str
    requested_backend: str
    execution_backend: str
    has_backend_parameter: bool

    @property
    def requires_sglang_launch(self) -> bool:
        return self.execution_backend == "sglang"

    @property
    def is_sglang_execution(self) -> bool:
        return self.execution_backend == "sglang"


def resolve_stage_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """Resolve final factory kwargs for a stage.

    Values are built from stage.factory_args, runtime_overrides, and typed
    stage.runtime fields, with typed runtime owning V1 resource contracts.
    Placement budgets are injected only when the factory declares them.
    """

    args = dict(stage_cfg.factory_args)
    runtime_overrides = global_cfg.runtime_overrides.get(stage_cfg.name, {})
    _validate_runtime_sources(stage_cfg, args, runtime_overrides)
    _merge_factory_arg_overrides(args, runtime_overrides)
    _apply_typed_runtime_args(args, stage_cfg)

    factory = import_string(stage_cfg.factory)
    sig = inspect.signature(factory)

    if "model_path" in sig.parameters and "model_path" not in args:
        args["model_path"] = global_cfg.model_path

    if "gpu_id" in sig.parameters and "gpu_id" not in args:
        args["gpu_id"] = (
            gpu_id
            if gpu_id is not None
            else _resolve_primary_gpu_id(stage_cfg, global_cfg)
        )

    total_gpu_memory_fraction = stage_cfg.runtime.resources.total_gpu_memory_fraction
    if (
        total_gpu_memory_fraction is not None
        and "total_gpu_memory_fraction" in sig.parameters
        and "total_gpu_memory_fraction" not in args
    ):
        args["total_gpu_memory_fraction"] = total_gpu_memory_fraction

    encoder_activation_budget_bytes = (
        stage_cfg.runtime.resources.encoder_activation_budget_bytes
    )
    if (
        encoder_activation_budget_bytes is not None
        and _ENCODER_ACTIVATION_BUDGET_KEY in sig.parameters
        and _ENCODER_ACTIVATION_BUDGET_KEY not in args
    ):
        args[_ENCODER_ACTIVATION_BUDGET_KEY] = encoder_activation_budget_bytes
    encoder_max_batch_size = stage_cfg.runtime.resources.encoder_max_batch_size
    if (
        encoder_max_batch_size is not None
        and _ENCODER_MAX_BATCH_SIZE_KEY in sig.parameters
        and _ENCODER_MAX_BATCH_SIZE_KEY not in args
    ):
        args[_ENCODER_MAX_BATCH_SIZE_KEY] = encoder_max_batch_size

    return args


def build_stage_launch_modes(
    config: PipelineConfig,
    *,
    stages_cfg: list[StageConfig] | None = None,
) -> dict[str, StageLaunchMode]:
    """Build launcher backend decisions before process topology is resolved."""

    stages = stages_cfg if stages_cfg is not None else config.stages
    modes: dict[str, StageLaunchMode] = {}
    for stage_cfg in stages:
        factory = import_string(stage_cfg.factory)
        params = inspect.signature(factory).parameters
        args = resolve_stage_factory_args(stage_cfg, config)
        requested = str(args.get("backend", "local"))
        execution = _resolve_execution_backend(
            stage_cfg,
            config,
            requested_backend=requested,
            factory=factory,
        )
        modes[stage_cfg.name] = StageLaunchMode(
            stage_name=stage_cfg.name,
            requested_backend=requested,
            execution_backend=execution,
            has_backend_parameter="backend" in params,
        )
    return modes


def _resolve_execution_backend(
    stage_cfg: StageConfig,
    config: PipelineConfig,
    *,
    requested_backend: str,
    factory: Any,
) -> str:
    if requested_backend != "auto":
        return requested_backend

    resolver = getattr(inspect.getmodule(factory), "_resolve_backend", None)
    if callable(resolver):
        try:
            resolved = resolver(
                requested_backend,
                config.model_path,
                stage=stage_cfg.name,
            )
        except TypeError as exc:
            logger.debug(
                "Backend resolver for stage %s did not match the standard "
                "_resolve_backend(backend, model_path, *, stage=...) shape: %s",
                stage_cfg.name,
                exc,
            )
        else:
            if resolved not in {"local", "sglang"}:
                raise ValueError(
                    f"Stage {stage_cfg.name!r}: backend='auto' resolved to "
                    f"unsupported execution backend {resolved!r}"
                )
            return str(resolved)

    # Conservative fallback: auto is SGLang-launch-capable. Backend-aware TP
    # factories without an explicit resolver must advertise TP launch params
    # and receive a parent-owned NCCL port.
    return "sglang"


def reject_untyped_total_gpu_memory_fraction(
    stage_name: str,
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    if (
        factory_args.get("total_gpu_memory_fraction") is None
        and runtime_overrides.get("total_gpu_memory_fraction") is None
    ):
        return
    raise ValueError(
        f"Stage {stage_name!r} sets total_gpu_memory_fraction through "
        "factory_args/runtime_overrides; set "
        "runtime.resources.total_gpu_memory_fraction instead"
    )


def reject_untyped_encoder_activation_budget_bytes(
    stage_name: str,
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    if (
        _ENCODER_ACTIVATION_BUDGET_KEY not in factory_args
        and _ENCODER_ACTIVATION_BUDGET_KEY not in runtime_overrides
    ):
        return
    raise ValueError(
        f"Stage {stage_name!r} sets encoder_activation_budget_bytes through "
        "factory_args/runtime_overrides; set "
        "runtime.resources.encoder_activation_budget_bytes instead"
    )


def reject_untyped_encoder_max_batch_size(
    stage_name: str,
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    if (
        _ENCODER_MAX_BATCH_SIZE_KEY not in factory_args
        and _ENCODER_MAX_BATCH_SIZE_KEY not in runtime_overrides
    ):
        return
    raise ValueError(
        f"Stage {stage_name!r} sets encoder_max_batch_size through "
        "factory_args/runtime_overrides; set "
        "runtime.resources.encoder_max_batch_size instead"
    )


def _validate_runtime_sources(
    stage_cfg: StageConfig,
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    """Validate ownership of runtime fields."""

    typed_mem_fraction = stage_cfg.runtime.sglang_server_args.mem_fraction_static
    if typed_mem_fraction is not None and _server_args_mem_fraction_static_is_set(
        factory_args,
        runtime_overrides,
    ):
        raise ValueError(
            f"Stage {stage_cfg.name!r} sets mem_fraction_static through both "
            "server_args_overrides and typed "
            "runtime.sglang_server_args.mem_fraction_static"
        )

    reject_untyped_total_gpu_memory_fraction(
        stage_cfg.name,
        factory_args,
        runtime_overrides,
    )
    reject_untyped_encoder_activation_budget_bytes(
        stage_cfg.name,
        factory_args,
        runtime_overrides,
    )
    reject_untyped_encoder_max_batch_size(
        stage_cfg.name,
        factory_args,
        runtime_overrides,
    )

    leaked = sorted(
        _TP_LAUNCH_PARAMS & (set(factory_args.keys()) | set(runtime_overrides.keys()))
    )
    if leaked:
        raise ValueError(
            f"Stage {stage_cfg.name!r}: factory_args/runtime_overrides cannot "
            f"set {leaked}. These keys are managed by the pipeline runner "
            "from StageConfig.tp_size and the per-stage NCCL port allocator."
        )

    for field_name in _MAPPED_STAGE_RUNTIME_FIELDS:
        value = getattr(stage_cfg.runtime, field_name)
        if value is None:
            continue
        target_arg = stage_cfg.runtime_arg_map.get(field_name)
        if target_arg and target_arg in runtime_overrides:
            raise ValueError(
                f"Stage {stage_cfg.name!r} sets {target_arg!r} through both "
                f"runtime_overrides and typed runtime.{field_name}"
            )


def _server_args_mem_fraction_static_is_set(
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> bool:
    for source in (factory_args, runtime_overrides):
        server_args = source.get("server_args_overrides")
        if (
            isinstance(server_args, dict)
            and server_args.get("mem_fraction_static") is not None
        ):
            return True
    return False


def _merge_factory_arg_overrides(
    args: dict[str, Any],
    overrides: dict[str, Any],
) -> None:
    for key, value in overrides.items():
        if (
            key == "server_args_overrides"
            and isinstance(value, dict)
            and isinstance(args.get(key), dict)
        ):
            merged = dict(args[key])
            merged.update(value)
            args[key] = merged
            continue
        args[key] = value


def _apply_typed_runtime_args(args: dict[str, Any], stage_cfg: StageConfig) -> None:
    runtime = stage_cfg.runtime

    for field_name in _MAPPED_STAGE_RUNTIME_FIELDS:
        value = getattr(runtime, field_name)
        if value is None:
            continue
        target_arg = stage_cfg.runtime_arg_map.get(field_name)
        if not target_arg:
            raise ValueError(
                f"Stage {stage_cfg.name!r} sets runtime.{field_name} but does not "
                f"define runtime_arg_map[{field_name!r}]"
            )
        args[target_arg] = value

    mem_fraction_static = runtime.sglang_server_args.mem_fraction_static
    if mem_fraction_static is not None:
        overrides = dict(args.get("server_args_overrides") or {})
        overrides["mem_fraction_static"] = mem_fraction_static
        args["server_args_overrides"] = overrides


def _resolve_primary_gpu_id(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> int | None:
    placement = global_cfg.gpu_placement.get(stage_cfg.name)
    if placement is None:
        return None
    if isinstance(placement, list):
        return placement[0]
    return int(placement)
