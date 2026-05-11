# SPDX-License-Identifier: Apache-2.0
"""Resolve typed v1 runtime config into stage factory arguments."""

from __future__ import annotations

import inspect
from typing import Any

from sglang_omni_v1.config.schema import PipelineConfig, StageConfig
from sglang_omni_v1.utils import import_string


_MAPPED_STAGE_RUNTIME_FIELDS = ("max_seq_len", "video_fps")


def resolve_stage_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """Resolve final factory args for a stage.

    Resolution order is:
    1. static ``stage.factory_args`` from the model config,
    2. legacy ``pipeline.runtime_overrides`` compatibility overlay,
    3. typed ``stage.runtime`` values.

    The typed runtime layer intentionally wins because it is the canonical v1
    surface. Placement resource budgets are not translated here; backend memory
    controls such as SGLang ``mem_fraction_static`` stay backend namespaced.
    """

    args = dict(stage_cfg.factory_args)
    _merge_factory_arg_overrides(
        args,
        global_cfg.runtime_overrides.get(stage_cfg.name, {}),
    )
    _apply_typed_runtime_args(args, stage_cfg)

    factory = import_string(stage_cfg.factory)
    sig = inspect.signature(factory)

    if "model_path" in sig.parameters and "model_path" not in args:
        args["model_path"] = global_cfg.model_path

    if "gpu_id" in sig.parameters and "gpu_id" not in args:
        args["gpu_id"] = (
            gpu_id if gpu_id is not None else _resolve_primary_gpu_id(stage_cfg, global_cfg)
        )

    return args


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


def _resolve_primary_gpu_id(stage_cfg: StageConfig, global_cfg: PipelineConfig) -> int:
    placement = global_cfg.gpu_placement.get(stage_cfg.name, 0)
    if isinstance(placement, list):
        return placement[0]
    return int(placement)
