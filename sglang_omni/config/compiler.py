# SPDX-License-Identifier: Apache-2.0
"""Compile pipeline configuration into runtime objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sglang_omni.config.imports import import_string
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


def compile_pipeline(config: PipelineConfig) -> tuple[Coordinator, list[Stage]]:
    _validate_pipeline(config)
    endpoints = _allocate_endpoints(config)

    coordinator = Coordinator(
        completion_endpoint=endpoints["completion"],
        abort_endpoint=endpoints["abort"],
        entry_stage=config.entry_stage,
    )

    stage_endpoints = {
        stage_cfg.name: endpoints[f"stage_{stage_cfg.name}"]
        for stage_cfg in config.stages
    }

    stages: list[Stage] = []
    for stage_cfg in config.stages:
        stage = _compile_stage(stage_cfg, stage_endpoints, endpoints)
        coordinator.register_stage(stage.name, stage.control_plane.recv_endpoint)
        stages.append(stage)

    return coordinator, stages


def _compile_stage(
    stage_cfg: StageConfig,
    stage_endpoints: dict[str, str],
    endpoints: dict[str, str],
) -> Stage:
    factory = import_string(stage_cfg.executor.factory)
    if not callable(factory):
        raise TypeError(
            f"Executor factory is not callable: {stage_cfg.executor.factory}"
        )

    get_next = import_string(stage_cfg.get_next)
    if not callable(get_next):
        raise TypeError(f"get_next is not callable: {stage_cfg.get_next}")

    input_handler = _create_input_handler(stage_cfg.input_handler)

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


def _create_input_handler(config: InputHandlerConfig) -> InputHandler:
    if config.type == "direct":
        return DirectInput()

    if not config.sources:
        raise ValueError("Aggregated input handler requires sources")
    if not config.merge_fn:
        raise ValueError("Aggregated input handler requires merge_fn")

    merge_fn = import_string(config.merge_fn)
    if not callable(merge_fn):
        raise TypeError(f"merge_fn is not callable: {config.merge_fn}")

    return AggregatedInput(sources=set(config.sources), merge=merge_fn)


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


def _allocate_endpoints(config: PipelineConfig) -> dict[str, str]:
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

        for stage_cfg in config.stages:
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

        for stage_cfg in config.stages:
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
