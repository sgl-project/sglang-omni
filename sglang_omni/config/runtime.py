# SPDX-License-Identifier: Apache-2.0
"""Resolve typed stage config groups into stage factory kwargs.

Factory kwargs for a stage come from exactly two sources:

* :meth:`PipelineConfig.stage_factory_kwargs` — wiring the pipeline author
  passes to the factory in code. Not a configuration path.
* the stage's typed consumer groups — ``factory.*`` fields pass through
  under their own names, and the set keys of ``engine.*`` travel together
  as one ``server_args_overrides`` mapping. Every set value
  is passed to the factory; a factory that does not declare the parameter is
  an error, because the value the user set would otherwise be dropped on the
  floor.

The parent process resolves both without importing the factory; the stage
worker overlays them against the factory's signature just before calling it
(:func:`apply_typed_stage_kwargs`), and injects the standard placement
kwargs only when the factory declares them
(:func:`resolve_factory_signature_args`).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from sglang_omni.config.schema import (
    PipelineConfig,
    StageConfig,
    parse_replica_instance_name,
    stage_process_name,
)
from sglang_omni.utils.imports import import_string

# Kwargs owned by placement and process construction. They are injected from
# ``factory_arg_defaults`` (and only when the factory declares them), so a
# pipeline author returning them from ``stage_factory_kwargs`` would fight
# the launch planner.
_PLACEMENT_OWNED_KWARGS = frozenset(
    {
        "gpu_id",
        "total_gpu_memory_fraction",
        "process_total_gpu_memory_fraction",
    }
)


def resolve_stage_factory_kwargs(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    """Return the pipeline author's constructor kwargs for one stage."""

    logical_stage_name, _ = parse_replica_instance_name(stage_cfg.name)
    kwargs = dict(global_cfg.stage_factory_kwargs(logical_stage_name) or {})
    reserved = _PLACEMENT_OWNED_KWARGS & kwargs.keys()
    if reserved:
        raise ValueError(
            f"stage_factory_kwargs({logical_stage_name!r}) returns "
            f"{sorted(reserved)}; these kwargs are owned by placement and are "
            "injected from stage.gpu_memory_fraction and stage.gpu"
        )
    return kwargs


def resolve_stage_typed_kwargs(stage_cfg: StageConfig) -> dict[str, Any]:
    """Return the stage's set group values, keyed by factory kwarg.

    ``factory.*`` entries pass through under their own names -- declared
    fields when set, and free-form keys always, because writing one is
    itself the intent. The set keys of ``engine.*`` travel together as one
    ``server_args_overrides`` mapping.
    """

    out: dict[str, Any] = {}
    group = stage_cfg.factory
    for name in type(group).model_fields:
        value = getattr(group, name)
        if value is not None:
            out[name] = value
    out.update(group.model_extra or {})
    if stage_cfg.engine is not None:
        server_args_overrides = stage_cfg.engine.overrides()
        if server_args_overrides:
            out["server_args_overrides"] = server_args_overrides
    return out


def typed_stage_kwarg_path(name: str) -> str:
    """The stage-config path that produced a typed factory kwarg."""

    if name == "server_args_overrides":
        return "engine"
    return f"factory.{name}"


def apply_typed_stage_kwargs(
    factory: Callable[..., Any],
    kwargs: dict[str, Any],
    typed_kwargs: Mapping[str, Any],
    *,
    stage_name: str,
) -> dict[str, Any]:
    """Overlay typed group values onto author kwargs, by factory signature.

    A typed value overrides the author's kwarg of the same name. A typed
    value the factory does not accept is refused: silently dropping it would
    turn a set configuration path into a no-op.
    """

    parameters = inspect.signature(factory).parameters
    accepts_any_kwarg = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    out = dict(kwargs)
    for name, value in typed_kwargs.items():
        if name not in parameters and not accepts_any_kwarg:
            raise ValueError(
                f"Stage {stage_name!r} sets {typed_stage_kwarg_path(name)} "
                f"but factory {getattr(factory, '__module__', '?')}."
                f"{getattr(factory, '__qualname__', str(factory))} does not "
                f"accept a {name!r} parameter"
            )
        if (
            name == "server_args_overrides"
            and isinstance(value, Mapping)
            and isinstance(out.get(name), Mapping)
        ):
            merged = dict(out[name])
            merged.update(value)
            out[name] = merged
            continue
        out[name] = value
    return out


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
    if stage_cfg.gpu_memory_fraction is not None:
        defaults["total_gpu_memory_fraction"] = stage_cfg.gpu_memory_fraction
    return defaults


def resolve_factory_signature_args(
    factory: Callable[..., Any],
    args: dict[str, Any],
    *,
    defaults: Mapping[str, Any],
    require_gpu_id: bool = False,
    stage_name: str | None = None,
) -> dict[str, Any]:
    """Inject standard factory kwargs when the resolved factory declares them."""

    args = dict(args)
    sig = inspect.signature(factory)
    if require_gpu_id and "gpu_id" not in sig.parameters:
        factory_name = f"{factory.__module__}.{factory.__qualname__}"
        stage_context = f"Stage {stage_name!r} " if stage_name is not None else "Stage "
        raise ValueError(
            f"{stage_context}uses processes.replica_devices, but factory "
            f"{factory_name!r} does not declare a gpu_id parameter"
        )

    for name, value in defaults.items():
        if name in sig.parameters and name not in args:
            args[name] = value

    return args


def requires_factory_gpu_id(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> bool:
    """Return whether replica placement requires an explicit gpu_id contract."""

    if stage_cfg.gpu is None:
        return False
    process_name, _ = parse_replica_instance_name(stage_process_name(stage_cfg))
    process_cfg = global_cfg.processes.get(process_name)
    return process_cfg is not None and process_cfg.replica_devices is not None


def resolve_stage_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
    *,
    gpu_id: int | None = None,
) -> dict[str, Any]:
    """Resolve final factory kwargs for a stage, importing its factory.

    One-process equivalent of the parent/worker split: author kwargs plus
    the typed group overlay, then placement defaults where the signature
    asks for them.
    """

    factory = import_string(stage_cfg.factory_path)
    args = apply_typed_stage_kwargs(
        factory,
        resolve_stage_factory_kwargs(stage_cfg, global_cfg),
        resolve_stage_typed_kwargs(stage_cfg),
        stage_name=stage_cfg.name,
    )
    return resolve_factory_signature_args(
        factory,
        args,
        defaults=resolve_stage_factory_arg_defaults(
            stage_cfg,
            global_cfg,
            gpu_id=gpu_id,
        ),
        require_gpu_id=requires_factory_gpu_id(stage_cfg, global_cfg),
        stage_name=stage_cfg.name,
    )


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
