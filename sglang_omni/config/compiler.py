# SPDX-License-Identifier: Apache-2.0
"""Compile pipeline configuration into runtime objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sglang_omni.config.imports import import_string
from sglang_omni.config.schema import (
    ExecutorConfig,
    InputHandlerConfig,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.executors.interface import Executor
from sglang_omni.pipeline import (
    AggregatedInput,
    Coordinator,
    DirectInput,
    Stage,
    Worker,
)
from sglang_omni.pipeline.stage.input import InputHandler


def compile_pipeline(config: PipelineConfig) -> tuple[Coordinator, list[Stage]]:
    _validate_pipeline(config)
    stages_cfg, name_map, entry_stage = _apply_fusion(config)
    endpoints = _allocate_endpoints(config, stages=stages_cfg)

    coordinator = Coordinator(
        completion_endpoint=endpoints["completion"],
        abort_endpoint=endpoints["abort"],
        entry_stage=entry_stage,
    )

    stage_endpoints = {
        stage_cfg.name: endpoints[f"stage_{stage_cfg.name}"] for stage_cfg in stages_cfg
    }

    stages: list[Stage] = []
    for stage_cfg in stages_cfg:
        stage = _compile_stage(stage_cfg, stage_endpoints, endpoints, name_map=name_map)
        coordinator.register_stage(stage.name, stage.control_plane.recv_endpoint)
        stages.append(stage)

    return coordinator, stages


def _compile_stage(
    stage_cfg: StageConfig,
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
        relay_config=_build_relay_config(stage_cfg),
    )

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


def _build_relay_config(stage_cfg: StageConfig) -> dict[str, Any]:
    relay_cfg = stage_cfg.relay
    return {
        "relay_type": relay_cfg.type,
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


def _validate_pipeline(config: PipelineConfig) -> None:
    if not config.name:
        raise ValueError("Pipeline name is required")

    stage_names = [stage_cfg.name for stage_cfg in config.stages]
    if not stage_names:
        raise ValueError("Pipeline must define at least one stage")

    if len(stage_names) != len(set(stage_names)):
        raise ValueError("Stage names must be unique")

    if config.entry_stage not in stage_names:
        raise ValueError(f"entry_stage {config.entry_stage!r} is not defined")

    for stage_cfg in config.stages:
        if stage_cfg.num_workers < 1:
            raise ValueError(f"Stage {stage_cfg.name!r} must have at least one worker")
        if not stage_cfg.executor.factory:
            raise ValueError(f"Stage {stage_cfg.name!r} missing executor factory")
        if not stage_cfg.get_next:
            raise ValueError(f"Stage {stage_cfg.name!r} missing get_next")
        if stage_cfg.input_handler.type == "aggregated":
            sources = stage_cfg.input_handler.sources or []
            unknown = set(sources) - set(stage_names)
            if unknown:
                raise ValueError(
                    f"Stage {stage_cfg.name!r} has unknown sources: {sorted(unknown)}"
                )

    _validate_fusion(config, stage_names)


def _validate_fusion(config: PipelineConfig, stage_names: list[str]) -> None:
    fused = config.fused_stages or []
    if not fused:
        return
    index_map = {name: idx for idx, name in enumerate(stage_names)}
    seen: set[str] = set()
    for group in fused:
        if not group or len(group) < 2:
            raise ValueError("fused_stages groups must have at least 2 stage names")
        for name in group:
            if name not in index_map:
                raise ValueError(f"fused stage {name!r} is not defined")
            if name in seen:
                raise ValueError(f"stage {name!r} appears in multiple fused groups")
            seen.add(name)

        indices = [index_map[name] for name in group]
        if indices != sorted(indices):
            raise ValueError(f"fused group is out of order: {group}")
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError(f"fused group must be adjacent: {group}")
        ordered = [stage_names[i] for i in indices]
        if ordered != group:
            raise ValueError(f"fused group order mismatch: {group}")


def _apply_fusion(
    config: PipelineConfig,
) -> tuple[list[StageConfig], dict[str, str], str]:
    stage_by_name = {stage.name: stage for stage in config.stages}
    fused_groups = config.fused_stages or []

    name_map = {name: name for name in stage_by_name}
    group_by_last: dict[str, list[str]] = {}

    for group in fused_groups:
        last = group[-1]
        group_by_last[last] = group
        for name in group:
            name_map[name] = last

    stages_out: list[StageConfig] = []
    for stage in config.stages:
        mapped = name_map.get(stage.name, stage.name)
        if mapped != stage.name:
            continue  # fused into another stage
        if stage.name in group_by_last:
            group = group_by_last[stage.name]
            first = stage_by_name[group[0]]
            executors = [
                {
                    "factory": stage_by_name[name].executor.factory,
                    "args": stage_by_name[name].executor.args,
                }
                for name in group
            ]
            fused_stage = StageConfig(
                name=stage.name,
                executor=ExecutorConfig(
                    factory="sglang_omni.executors.fused_executor.create_fused_executor",
                    args={"executors": executors},
                ),
                get_next=stage.get_next,
                input_handler=first.input_handler,
                relay=first.relay,
                num_workers=first.num_workers,
            )
            stages_out.append(fused_stage)
        else:
            stages_out.append(stage)

    entry_stage = _map_stage_name(name_map, config.entry_stage)
    return stages_out, name_map, entry_stage


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
