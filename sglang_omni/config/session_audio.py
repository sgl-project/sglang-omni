# SPDX-License-Identifier: Apache-2.0
"""Shared stage assembly and realtime contracts for audio Session graphs."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import ClassVar

from pydantic import Field, model_validator

from sglang_omni.config.schema import PipelineConfig, SessionNodeConfig, StageConfig
from sglang_omni.models.turn_detection.node import SESSION_PORTS as TURN_DETECTION_PORTS
from sglang_omni.models.turn_detection.node import (
    TURN_DETECTION_STAGE,
    TurnDetectionStageConfig,
    turn_detection_stage,
)
from sglang_omni.models.vad.node import SESSION_PORTS as VAD_PORTS
from sglang_omni.models.vad.node import VAD_STAGE, VADStageConfig, vad_stage
from sglang_omni.proto.session import SessionLimits
from sglang_omni.proto.streams import SESSION_UNIT_MS, SessionPorts

REALTIME_OPERATION_TIMEOUT_S = 300
REALTIME_IDLE_TIMEOUT_S = 6 * 3600
REALTIME_MAX_PENDING_CHUNKS = 64
MAX_PENDING_OUTPUT_CHUNKS = 1024
MAX_PENDING_OUTPUT_BYTES = 32 * 1024 * 1024
MODEL_PIPELINE_FIELDS = ("placement", "terminal_stages_fn", "env_defaults")


@dataclass(frozen=True)
class StageGroup:
    """Stages a stage-route node names, added to the pipeline together.

    `pipeline` is the config class of the model serving them: its factory
    kwargs for these stages and its pipeline defaults apply here.
    `model_path` is their checkpoint when it is not the pipeline's
    `model_path`; `ports` is what a node over them accepts and publishes.
    """

    stages: Callable[[], list[StageConfig]]
    pipeline: type[PipelineConfig] | None
    ports: SessionPorts | None
    model_path: str | None = None


class AudioSessionPipelineConfig(PipelineConfig):
    """Assemble model stage groups and combine their Session ports."""

    realtime_deployment_factory: ClassVar[str] = (
        "sglang_omni.serve.realtime.session_deployment.build_realtime_deployment"
    )
    stage_groups: ClassVar[tuple[StageGroup, ...]] = (
        StageGroup(lambda: [vad_stage()], None, VAD_PORTS),
        StageGroup(lambda: [turn_detection_stage()], None, TURN_DETECTION_PORTS),
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        VAD_STAGE: VADStageConfig,
        TURN_DETECTION_STAGE: TurnDetectionStageConfig,
    }
    session_nodes: dict[str, SessionNodeConfig]
    session_publish: dict[str, list[str]]
    unit_ms: int = Field(default=SESSION_UNIT_MS, gt=0)
    max_pending_output_chunks: int = Field(default=MAX_PENDING_OUTPUT_CHUNKS, gt=0)
    max_pending_output_bytes: int = Field(default=MAX_PENDING_OUTPUT_BYTES, gt=0)

    @classmethod
    def groups_by_stage(cls) -> dict[str, StageGroup]:
        return {
            stage.name: group for group in cls.stage_groups for stage in group.stages()
        }

    @model_validator(mode="before")
    @classmethod
    def add_stages(cls, data: object) -> object:
        """Add the stages the declared nodes need, with the pipeline defaults of their models, when no stages are given."""
        if (
            not isinstance(data, dict)
            or "stages" in data
            or "session_nodes" not in data
        ):
            return data
        else:
            pass
        nodes = [
            SessionNodeConfig.model_validate(n) for n in data["session_nodes"].values()
        ]
        groups = cls.groups_by_stage()
        named = [stage for node in nodes for stage in node.stages or ()]
        unknown = sorted(set(named) - set(groups))
        if unknown:
            raise ValueError(f"no stage group serves the stages {unknown}")
        else:
            pass
        ordered: list[StageGroup] = []
        for stage in named:
            if groups[stage] not in ordered:
                ordered.append(groups[stage])
            else:
                pass
        models = [group.pipeline for group in ordered if group.pipeline is not None]
        defaults = {
            field: value
            for model in models
            for field in MODEL_PIPELINE_FIELDS
            if (
                value := model.model_fields[field].get_default(
                    call_default_factory=True
                )
            )
            != PipelineConfig.model_fields[field].get_default(call_default_factory=True)
        }
        return {
            **defaults,
            "entry_stage": (
                next(
                    (g.stages()[0].name for g in ordered if g.pipeline),
                    next((g.stages()[0].name for g in ordered), None),
                )
            ),
            **data,
            "stages": [
                *(stage for group in ordered for stage in group.stages()),
            ],
        }

    def check_local_conflicts(self) -> None:
        super().check_local_conflicts()
        self.session_graph()

    def model_configs(self) -> list[PipelineConfig]:
        """The config of each model whose stages a node names, over this pipeline's stages of that model."""
        stage_names = {stage.name for stage in self.stages}
        groups = [
            group
            for group in self.stage_groups
            if group.pipeline is not None
            and {stage.name for stage in group.stages()} <= stage_names
        ]
        return [
            group.pipeline(
                model_path=group.model_path or self.model_path,
                stages=[self.stage_named(stage.name) for stage in group.stages()],
            )
            for group in groups
        ]

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        """Delegate factory arguments to the model owning these stages."""
        model = next(
            (
                m
                for m in self.model_configs()
                if stage_name in {s.name for s in m.stages}
            ),
            None,
        )
        if model is None:
            return {}
        elif model.model_path != self.model_path:
            return {
                **model.stage_factory_kwargs(stage_name),
                "model_path": model.model_path,
            }
        else:
            return model.stage_factory_kwargs(stage_name)

    def node_ports(self, name: str, node: SessionNodeConfig) -> SessionPorts | None:
        return self.groups_by_stage()[node.stages[0]].ports

    def session_ports(self) -> SessionPorts:
        by_node = {
            name: self.node_ports(name, node)
            for name, node in self.session_nodes.items()
        }
        first, *others = [ports for ports in by_node.values() if ports is not None]
        for ports in others:
            first = first.with_node(ports)
        return replace(first, unit_ms=self.unit_ms)

    def session_limits(self) -> SessionLimits:
        return SessionLimits(
            operation_timeout_s=REALTIME_OPERATION_TIMEOUT_S,
            idle_timeout_s=REALTIME_IDLE_TIMEOUT_S,
            max_pending_chunks=REALTIME_MAX_PENDING_CHUNKS,
            max_output_chunks=self.max_pending_output_chunks,
            max_output_bytes=self.max_pending_output_bytes,
        )
