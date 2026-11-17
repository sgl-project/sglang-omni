# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for ARK-ASR-3B."""

from __future__ import annotations

from typing import ClassVar, Self

from pydantic import Field, model_validator

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.platforms import current_platform
from sglang_omni.speculative.draft_registry import DRAFT_REGISTRY

_PKG = "sglang_omni.models.arkasr"


class ArkasrFactoryArgs(FactoryArgs):
    """ARK-ASR's own constructor knobs, typed like the shared ones."""

    encoder_max_batch_size: int | None = Field(default=None, ge=1)
    enable_encoder_cuda_graph: bool | None = None
    enable_pre_lm_encoder: bool | None = None
    pre_lm_cache_max_entries: int | None = Field(default=None, ge=1)
    pre_lm_cache_size_bytes: int | None = Field(default=None, ge=1)
    pre_lm_max_batch_size: int | None = Field(default=None, ge=1)
    pre_lm_max_batch_wait_ms: int | None = Field(default=None, ge=0)
    pre_lm_max_pending: int | None = Field(default=None, ge=1)


class ArkasrStageConfig(EngineStageConfig):
    factory: ArkasrFactoryArgs = Field(default_factory=ArkasrFactoryArgs)


class ArkasrPipelineConfig(PipelineConfig):
    """Single-stage batched ASR pipeline for ARK-ASR-3B checkpoints."""

    architecture: ClassVar[str] = "ArkasrForConditionalGeneration"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "asr": ArkasrStageConfig,
    }

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"asr": "asr"}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": "asr"}

    model_path: str
    entry_stage: str = "asr"
    stages: list[StageConfig] = [
        ArkasrStageConfig(
            name="asr",
            process="asr",
            factory_path=f"{_PKG}.stages.create_sglang_arkasr_executor",
            factory=ArkasrFactoryArgs(
                max_new_tokens=256,
                encoder_max_batch_size=8,
                enable_encoder_cuda_graph=True,
                prefill_coalesce_requests=16,
                prefill_coalesce_when_idle=True,
                prefill_coalesce_requires_pending_builds=True,
                enable_pre_lm_encoder=True,
                pre_lm_cache_max_entries=4096,
                pre_lm_cache_size_bytes=2 * 1024**3,
                # Note (Akazaakane): One drained group maps to exactly one
                # encoder microbatch at the matching encoder_max_batch_size.
                pre_lm_max_batch_size=8,
                pre_lm_max_batch_wait_ms=0,
                pre_lm_max_pending=32,
                request_build_max_pending=16,
            ),
            engine=EngineArgs(max_running_requests=32),
            gpu=0,
            terminal=True,
        )
    ]

    @model_validator(mode="after")
    def apply_default_draft(self) -> Self:
        model_repository = self.model_path.partition("@")[0]
        draft = next(
            (
                candidate
                for candidate in DRAFT_REGISTRY
                if candidate.model_path == model_repository
                and candidate.stage == "asr"
                and candidate.is_default_on
            ),
            None,
        )
        stage = self.stage_named("asr")
        engine = stage.engine
        if (
            draft is None
            or engine is None
            or engine.disable_speculative_decoding
            or not current_platform.is_cuda()
        ):
            pass
        elif {
            "speculative_algorithm",
            "speculative_draft_model_path",
        } & engine.model_fields_set:
            pass
        else:
            stage.engine = EngineArgs(
                **(
                    {
                        "speculative_algorithm": draft.algorithm,
                        "speculative_draft_model_path": draft.draft_model_path,
                        **draft.engine_settings,
                    }
                    | engine.model_dump(exclude_unset=True)
                )
            )
        return self


EntryClass = ArkasrPipelineConfig
