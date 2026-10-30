# SPDX-License-Identifier: Apache-2.0
"""Ming 3 text-to-speech pipeline configuration."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.utils.device import resolve_device_spec
from sglang_omni.utils.hf import try_resolve_ming3_dense_tts

PACKAGE = "sglang_omni.models.ming3_tts"


def default_stages(dense: bool = False) -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="preprocessing",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            factory=FactoryArgs(
                max_concurrency=4,
                reference_cache_entries=256,
                reference_cache_bytes=64 * 1024 * 1024,
                max_reference_seconds=120.0 if dense else None,
            ),
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="tts_engine",
            gpu=0,
            factory_path=f"{PACKAGE}.stages.create_tts_engine_executor",
            factory=FactoryArgs(
                max_seq_len=4096,
                enable_acoustic_graph=True,
                enable_fused_flow_head=dense,
                prefill_graph_token_counts=(
                    [
                        64,
                        96,
                        128,
                        160,
                        192,
                        224,
                        256,
                        288,
                        320,
                        384,
                        448,
                        512,
                    ]
                ),
                prefill_graph_max_requests=32,
                enable_async_decode=True,
            ),
            engine=EngineArgs(
                mem_fraction_static=0.75,
                max_running_requests=32,
                cuda_graph_max_bs=32,
                disable_cuda_graph=False,
                enable_torch_compile=False,
            ),
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="vocoder",
            gpu=0,
            factory_path=f"{PACKAGE}.vocoder.create_vocoder_executor",
            factory=FactoryArgs(graph_latent_steps=384, stream_chunk_steps=8),
            can_accept_stream_before_payload=True,
            terminal=True,
        ),
    ]


DENSE_ARCHITECTURE = "Ming3DenseTTSForConditionalGeneration"


class Ming3TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "LingV3TTSForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (DENSE_ARCHITECTURE,)
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }
    model_path: str
    entry_stage: str = "preprocessing"
    mps: Literal["off", "on", "auto"] = "auto"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=default_stages)

    @model_validator(mode="before")
    @classmethod
    def select_checkpoint_defaults(cls, data: object) -> object:
        """Select checkpoint defaults while preserving explicit overrides."""
        if not isinstance(data, dict) or not isinstance(data.get("model_path"), str):
            return data
        elif "stages" in data and "mps" in data:
            return data
        else:
            pass
        model_path = data["model_path"]
        if Path(model_path).is_dir():
            repo_id, revision = model_path, None
        else:
            repo_id, separator, pinned_revision = model_path.partition("@")
            revision = pinned_revision if separator else None
        dense = (
            try_resolve_ming3_dense_tts(repo_id, revision=revision)
            == DENSE_ARCHITECTURE
        )
        return {
            "stages": default_stages(dense),
            "mps": "auto",
            **data,
        }

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, bool]:
        if stage_name == "tts_engine":
            vocoder = self.stage_named("vocoder")
            return {
                "use_streaming_vocoder": (
                    vocoder.factory_path
                    not in {
                        f"{PACKAGE}.stages.create_vocoder_executor",
                        f"{PACKAGE}.vocoder.create_vocoder_executor",
                    }
                    or resolve_device_spec(vocoder.factory.device) != "cpu"
                )
            }
        else:
            return {}


EntryClass = Ming3TTSPipelineConfig
