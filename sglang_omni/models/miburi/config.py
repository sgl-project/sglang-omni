# SPDX-License-Identifier: Apache-2.0
"""MIBURI served from its checkpoint: a full-duplex speech session with a gesture node.

The speech node is the PersonaPlex session route on the speech backbone's checkpoint; its code2wav also relays each unit's agent frames, which the gesture node turns into SMPL-X motion beside the reply audio.
"""

from collections.abc import Mapping
from dataclasses import replace
from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import FactoryArgs, SessionNodeConfig, StageConfig
from sglang_omni.models.miburi.formats import SMPLX_MOTION, SMPLX_MOTION_PORT
from sglang_omni.models.personaplex.config import (
    CODE2WAV_STAGE,
    LM_STAGE,
    MIMI_ENCODE_STAGE,
    PREPROCESSING_STAGE,
    PersonaPlexSessionPipelineConfig,
    personaplex_session_stages_factory,
)
from sglang_omni.models.personaplex.session_hooks import AGENT_FRAMES, PERSONAPLEX_REPLY
from sglang_omni.proto.session import GRAPH_INPUT, SessionGraph
from sglang_omni.proto.streams import PluginPort

MIBURI_ARCH = "MiburiForGestureGeneration"
# The speech LM the MIBURI demo server runs and the gesture checkpoint was trained on.
MOSHIKO_MODEL_PATH = "kyutai/moshiko-pytorch-bf16"
PERSONAPLEX_MODEL_PATH = "nvidia/personaplex-7b-v1"
GESTURE_STAGE = "gesture"
SPEECH_NODE = "speech"
GESTURE_NODE = "gesture"


class GestureFactoryArgs(FactoryArgs):
    """The gesture LM's experiment, speaker and sampling; the defaults are the released demo's launch values."""

    experiment: str = "demoexp_release_gtdm3_goodspk"
    speaker_id: int = 3
    cfg_coef: float = 1.3
    temporal_temperature: float = Field(default=0.9, ge=0.0)
    temporal_top_p: float = Field(default=0.8, gt=0.0, le=1.0)
    depth_temperature: float = Field(default=0.9, ge=0.0)
    depth_top_p: float = Field(default=0.95, gt=0.0, le=1.0)


class GestureStageConfig(StageConfig):
    factory: GestureFactoryArgs = Field(default_factory=GestureFactoryArgs)


def miburi_stages_factory() -> list[StageConfig]:
    return personaplex_session_stages_factory() + [
        GestureStageConfig(
            name=GESTURE_STAGE,
            process=GESTURE_STAGE,
            factory_path="sglang_omni.models.miburi.stages.create_gesture_executor",
            gpu=0,
            terminal=True,
        )
    ]


class MiburiPipelineConfig(PersonaPlexSessionPipelineConfig):
    """MIBURI on kyutai/moshiko-pytorch-bf16, which ships no voices and takes no role prompt."""

    architecture: ClassVar[str] = MIBURI_ARCH
    realtime_deployment_factory: ClassVar[str | None] = (
        "sglang_omni.models.miburi.realtime.create_realtime_deployment"
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        **PersonaPlexSessionPipelineConfig.stage_config_types,
        GESTURE_STAGE: GestureStageConfig,
    }

    speech_model_path: str = MOSHIKO_MODEL_PATH
    "The speech backbone's checkpoint, a Hub repo or a local copy; model_path is the MIBURI checkpoint."
    voice: str | None = ""
    instructions: str | None = ""
    stages: list[StageConfig] = Field(default_factory=miburi_stages_factory)
    session_nodes: dict[str, SessionNodeConfig] = Field(
        default_factory=lambda: {
            SPEECH_NODE: SessionNodeConfig(
                stages=[
                    PREPROCESSING_STAGE,
                    MIMI_ENCODE_STAGE,
                    LM_STAGE,
                    CODE2WAV_STAGE,
                ],
                inputs=[GRAPH_INPUT],
            ),
            GESTURE_NODE: SessionNodeConfig(
                stages=[GESTURE_STAGE], inputs=[f"{SPEECH_NODE}:{AGENT_FRAMES}"]
            ),
        }
    )
    session_publish: dict[str, list[str]] = Field(
        default_factory=lambda: {
            SPEECH_NODE: [PERSONAPLEX_REPLY],
            GESTURE_NODE: [SMPLX_MOTION],
        }
    )

    def plugin_output_ports(self) -> tuple[PluginPort, ...]:
        return (SMPLX_MOTION_PORT,)

    def session_graph(self) -> SessionGraph:
        """The speech and gesture nodes' graph, with motion as its only plugin format."""
        return replace(
            super().session_graph(),
            plugin_formats=frozenset(
                port.format for port in self.plugin_output_ports()
            ),
        )

    def stage_factory_kwargs(self, stage_name: str) -> Mapping[str, object]:
        kwargs = super().stage_factory_kwargs(stage_name)
        if stage_name == GESTURE_STAGE:
            return kwargs
        elif stage_name == CODE2WAV_STAGE:
            return {
                **kwargs,
                "model_path": self.speech_model_path,
                "relays_agent_frames": True,
            }
        else:
            return {**kwargs, "model_path": self.speech_model_path}


class MiburiPersonaPlexPipelineConfig(MiburiPipelineConfig):
    """MIBURI on PersonaPlex with its voices and role prompt; the gesture LM was trained on moshiko's tokens, so this pairing is not quality-verified."""

    speech_model_path: str = PERSONAPLEX_MODEL_PATH
    voice: str | None = None
    instructions: str | None = None


EntryClass = MiburiPipelineConfig

Variants = {
    "moshiko": MiburiPipelineConfig,
    "personaplex": MiburiPersonaPlexPipelineConfig,
}
