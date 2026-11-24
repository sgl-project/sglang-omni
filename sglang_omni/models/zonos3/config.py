# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for Zonos3 TTS: preprocessing -> tts_engine -> vocoder."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config.schema import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.zonos3.hf_config import ARCHITECTURE

PACKAGE = "sglang_omni.models.zonos3"
TTS_ENGINE_GPU_MEMORY_FRACTION = 0.3
# note (zhaochenyang20): Searched on SeedTTS EN at concurrency 16 on one H200; 16 running requests exhaust the linear-attention state slots and stall admission.
MAX_RUNNING_REQUESTS = 64
CUDA_GRAPH_MAX_BATCH_SIZE = 64


class Zonos3PreprocessingFactoryArgs(FactoryArgs):
    max_reference_seconds: float = Field(default=60.0, gt=0)
    # Codec-silence frames that open every audio stream (0.2 s at 15 frames per second) and are not emitted.
    silence_prefill_frames: int = Field(default=3, ge=0)
    # note (zhaochenyang20): Reference loading and encoding of four requests overlap; one serial worker capped SeedTTS EN throughput on one H200. With the 20 s encoder lanes all four were busy 5% of the time and a request waited under 0.1 ms for a worker (streaming SeedTTS EN, one H100).
    max_concurrency: int = Field(default=4, ge=1)


class Zonos3PreprocessingStageConfig(StageConfig):
    factory: Zonos3PreprocessingFactoryArgs = Field(
        default_factory=Zonos3PreprocessingFactoryArgs
    )


class Zonos3EngineFactoryArgs(FactoryArgs):
    """max_new_tokens is the frame budget of a request that sets none."""

    context_length: int = Field(default=8192, gt=0)
    max_running_requests: int = Field(default=MAX_RUNNING_REQUESTS, ge=1)
    max_new_tokens: int = Field(default=2048, gt=0)
    cuda_graph: bool = True
    # note (zhaochenyang20): Decoding one step ahead hides the device step behind the host bookkeeping; SeedTTS EN on one H100 ran 13% more requests per second both non-streaming (3 ABBA pairs) and streaming (4 ABBA pairs), with first audio 16 ms earlier.
    async_decode: bool = True


class Zonos3EngineStageConfig(EngineStageConfig):
    factory: Zonos3EngineFactoryArgs = Field(default_factory=Zonos3EngineFactoryArgs)


class Zonos3VocoderFactoryArgs(FactoryArgs):
    # note (zhaochenyang20): Batched due-only vocoder steps keep a 12-16 frame first chunk in time (first audio 0.77 s vs 1.10 s for 6 frames); once the engine sends codes per vocoder chunk, 16 frames delayed first audio to 0.85 s and 12 frames brought it to 0.77 s at the same C50 (SeedTTS EN on one H100).
    initial_chunk_frames: int = Field(default=12, ge=1)
    steady_chunk_frames: int = Field(default=16, ge=1)
    # note (zhaochenyang20): SeedTTS EN at concurrency 16 averaged about two due streams per step on one H100, and a cap of 8 was no faster than 4.
    max_step_streams: int = Field(default=4, ge=1)
    # note (zhaochenyang20): A slot keeps a stream's codec state (241 MiB of key/value rings) so its decodes run only its new frames; SeedTTS EN at concurrency 16 on one H100 then streamed 7% more requests per second with first audio at 0.26 s instead of 0.29 s (6 ABBA pairs). A stream without a free slot decodes its whole prefix as before: at concurrency 48, 32 slots left 20% of streams slotless and streamed 29.8 requests per second, against 34.7 with one slot per running request (2 runs each) and 29.4 with none. With the default 64 running requests those 65 slots, scratch included, take 15.3 GiB after the engine sizes its cache; concurrency 64 held at most 59 of them, and the H100 peaked at 45 of its 80 GiB. Slots run only on sm90 GPUs, the only ones measured. None takes one slot per running request of the engine; a number overrides it.
    stream_slots: int | None = Field(default=None, ge=0)
    # note (zhaochenyang20): A stream's first decode carries 13 frames and its last one the rest, about 60 frames on SeedTTS EN; one H100 pass costs about 8.5 ms plus 0.04 ms per frame of each row.
    pass_frame_buckets: tuple[int, ...] = (16, 32, 48, 64, 96, 128)


class Zonos3VocoderStageConfig(StageConfig):
    factory: Zonos3VocoderFactoryArgs = Field(default_factory=Zonos3VocoderFactoryArgs)


class Zonos3PipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = ARCHITECTURE
    requires_model_capabilities: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "preprocessing": Zonos3PreprocessingStageConfig,
        "tts_engine": Zonos3EngineStageConfig,
        "vocoder": Zonos3VocoderStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        Zonos3PreprocessingStageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            gpu=0,
            next="tts_engine",
        ),
        Zonos3EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_tts_engine_executor",
            gpu=0,
            gpu_memory_fraction=TTS_ENGINE_GPU_MEMORY_FRACTION,
            engine=EngineArgs(cuda_graph_max_bs=CUDA_GRAPH_MAX_BATCH_SIZE),
            next="vocoder",
            stream_to=["vocoder"],
        ),
        Zonos3VocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, int]:
        if stage_name == "preprocessing":
            engine_factory = self.stage_named("tts_engine").factory
            assert isinstance(engine_factory, Zonos3EngineFactoryArgs)
            return {"max_new_tokens": engine_factory.max_new_tokens}
        elif stage_name == "tts_engine":
            vocoder_factory = self.stage_named("vocoder").factory
            assert isinstance(vocoder_factory, Zonos3VocoderFactoryArgs)
            return {
                "vocoder_initial_chunk_frames": vocoder_factory.initial_chunk_frames,
                "vocoder_steady_chunk_frames": vocoder_factory.steady_chunk_frames,
            }
        elif stage_name == "vocoder":
            engine_factory = self.stage_named("tts_engine").factory
            assert isinstance(engine_factory, Zonos3EngineFactoryArgs)
            # A stream never has more frames than its context holds.
            return {
                "max_stream_frames": engine_factory.context_length,
                "max_running_requests": engine_factory.max_running_requests,
            }
        else:
            return {}

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = Zonos3PipelineConfig
