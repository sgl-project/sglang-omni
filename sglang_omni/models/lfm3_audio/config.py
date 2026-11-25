# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration of LFM3-Audio and LFM3-Omni: preprocessing -> AR generation -> decode."""

from __future__ import annotations

import base64
from typing import ClassVar

from sglang_omni.config import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage
from sglang_omni.serve.server_warmup import warmup_tone_wav_bytes

PACKAGE = "sglang_omni.models.lfm3_audio"
PREPROCESSING_STAGE = "preprocessing"
GENERATION_STAGE = "generation"
DECODE_STAGE = "decode"
# Enough for one interleaved block of text and speech frames, so the codec runs too.
WARMUP_MAX_TOKENS = 32


class Lfm3AudioPipelineConfig(PipelineConfig):
    """Text and 16 kHz audio in (LFM3-Omni: also images and videos); text, 24 kHz speech or both out."""

    architecture: ClassVar[str] = "Lfm3AudioForConditionalGeneration"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "Lfm3OmniForConditionalGeneration",
    )
    requires_model_capabilities: ClassVar[bool] = True
    # note (ruilin): the first prefill compiles the extend attention kernels, which held the first request about 2 s.
    server_warmup_request_factory: ClassVar[str] = (
        f"{PACKAGE}.config.build_server_warmup_request"
    )
    supported_speech_fields: ClassVar[frozenset[str] | None] = frozenset(
        {
            "model",
            "input",
            "voice",
            "language",
            "instructions",
            "response_format",
            "sample_rate",
            "max_new_tokens",
            "seed",
            "temperature",
            "top_k",
            "repetition_penalty",
        }
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        GENERATION_STAGE: EngineStageConfig,
    }

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        return frozenset({(PREPROCESSING_STAGE, GENERATION_STAGE)})

    entry_stage: str = PREPROCESSING_STAGE
    stages: list[StageConfig] = [
        StageConfig(
            name=PREPROCESSING_STAGE,
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            next=GENERATION_STAGE,
        ),
        EngineStageConfig(
            name=GENERATION_STAGE,
            process="pipeline",
            factory_path=f"{PACKAGE}.engine_builder.create_generation_executor",
            factory=FactoryArgs(
                speech_sampler="seeded",
                enable_async_decode=True,
                # note (ruilin): each prefill drains the pending decode step, so new requests wait briefly to prefill together.
                prefill_coalesce_requests=32,
                prefill_coalesce_wait_ms=30.0,
            ),
            gpu_memory_fraction=0.70,
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            # note (ruilin): its own process keeps the codec off the engine's scheduler loop.
            process="decode",
            factory_path=f"{PACKAGE}.stages.create_decode_executor",
            gpu_memory_fraction=0.15,
            gpu=0,
            terminal=True,
        ),
    ]


def build_server_warmup_request(
    pipeline_config: Lfm3AudioPipelineConfig,
) -> ChatCompletionRequest:
    """A second of audio answered in text and speech: the audio encoder, prefill, decode and the codec run once before the server reports ready."""
    del pipeline_config
    tone_base64 = base64.b64encode(warmup_tone_wav_bytes()).decode("ascii")
    return ChatCompletionRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "input_audio",
                        "input_audio": {"data": tone_base64, "format": "wav"},
                    },
                    {"type": "text", "text": "Describe this sound."},
                ],
            )
        ],
        modalities=["text", "audio"],
        max_tokens=WARMUP_MAX_TOKENS,
    )


EntryClass = Lfm3AudioPipelineConfig
