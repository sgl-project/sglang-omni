# SPDX-License-Identifier: Apache-2.0
"""Text-only Cosmos3-Nano pipeline configuration."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.cosmos3"

PREPROCESSING_STAGE = "preprocessing"
THINKER_STAGE = "thinker"
DECODE_STAGE = "decode"


def _text_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"thinker_max_seq_len": 8192},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            next=THINKER_STAGE,
        ),
        StageConfig(
            name=THINKER_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_text_executor_from_config",
            factory_args={
                "thinker_max_seq_len": 8192,
                "enable_async_decode": False,
            },
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            gpu=0,
            tp_size=1,
            next=DECODE_STAGE,
            stream_to=[DECODE_STAGE],
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


class Cosmos3TextPipelineConfig(PipelineConfig):
    """Three-stage MVP: preprocessing → text AR → detokenize."""

    architecture: ClassVar[str] = "Cosmos3ForConditionalGeneration"

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {THINKER_STAGE: THINKER_STAGE}

    model_path: str
    stages: list[StageConfig] = Field(default_factory=_text_stages)


EntryClass = Cosmos3TextPipelineConfig

Variants = {"text": Cosmos3TextPipelineConfig}

__all__ = ["Cosmos3TextPipelineConfig", "EntryClass", "Variants"]
