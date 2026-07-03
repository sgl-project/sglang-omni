# SPDX-License-Identifier: Apache-2.0
"""Resolve typed runtime config into stage factory arguments."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.utils.imports import import_string


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

    args = resolve_stage_static_factory_args(stage_cfg, global_cfg)
    factory = import_string(stage_cfg.factory)
    return resolve_factory_signature_args(
        factory,
        args,
        defaults=resolve_stage_factory_arg_defaults(
            stage_cfg,
            global_cfg,
            gpu_id=gpu_id,
        ),
    )


def resolve_stage_static_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    """Resolve factory kwargs that do not require importing the factory."""

    args = dict(stage_cfg.factory_args)
    runtime_overrides = global_cfg.runtime_overrides.get(stage_cfg.name, {})
    _validate_runtime_sources(stage_cfg, args, runtime_overrides)
    _merge_factory_arg_overrides(args, runtime_overrides)
    _apply_typed_runtime_args(args, stage_cfg)
    return args


def resolve_stage_factory_arg_defaults(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """Return standard factory kwargs used only when the factory declares them."""

    defaults: dict[str, Any] = {"model_path": global_cfg.model_path}
    if gpu_id is None:
        gpu_id = _resolve_primary_gpu_id(stage_cfg, global_cfg)
    defaults["gpu_id"] = gpu_id
    total_gpu_memory_fraction = stage_cfg.runtime.resources.total_gpu_memory_fraction
    if total_gpu_memory_fraction is not None:
        defaults["total_gpu_memory_fraction"] = total_gpu_memory_fraction
    return defaults


def resolve_factory_signature_args(
    factory: Callable[..., Any],
    args: dict[str, Any],
    *,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """Inject standard factory kwargs when the resolved factory declares them."""

    args = dict(args)
    sig = inspect.signature(factory)

    for name, value in defaults.items():
        if name in sig.parameters and name not in args:
            args[name] = value

    return args


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


def reject_gpu_id_in_factory_args(
    stage_name: str,
    factory_args: dict[str, Any],
    runtime_overrides: dict[str, Any],
) -> None:
    if "gpu_id" not in factory_args and "gpu_id" not in runtime_overrides:
        return
    raise ValueError(
        f"Stage {stage_name!r} sets gpu_id through factory_args/runtime_overrides; "
        "gpu_id is owned by placement — set the device via stage.gpu instead"
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

    reject_gpu_id_in_factory_args(
        stage_cfg.name,
        factory_args,
        runtime_overrides,
    )

    for field_name, value in _mapped_stage_runtime_values(stage_cfg).items():
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

    for field_name, value in _mapped_stage_runtime_values(stage_cfg).items():
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


def _mapped_stage_runtime_values(stage_cfg: StageConfig) -> dict[str, Any]:
    runtime = stage_cfg.runtime
    return {
        "max_seq_len": runtime.max_seq_len,
        "video_fps": runtime.video_fps,
    }


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
