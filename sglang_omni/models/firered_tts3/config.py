# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for FireRedTTS3 (Base / Instruct / speaker)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar, TypedDict

from pydantic import Field

from sglang_omni.config.schema import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)
from sglang_omni.models.firered_tts3.flow_graph import FlowMode
from sglang_omni.models.firered_tts3.hf_config import (
    FireRedTTS3Config,
    FireRedTTS3Variant,
    detect_fireredtts3_variant,
)

PACKAGE_PATH = "sglang_omni.models.firered_tts3"

DEFAULT_N_TIMESTEPS = 10
DEFAULT_MAX_PATCHES = 400
SPEAKER_MAX_PATCHES = 600
DEFAULT_MAX_RUNNING_REQUESTS = 16


class FireRedTTS3StageFactoryKwargs(TypedDict, total=False):
    n_timesteps: int
    max_patches: int
    max_concurrency: int
    max_batch_size: int
    max_batch_wait_ms: float
    flow_cuda_graph: bool
    flow_cuda_graph_capture_sizes: list[int] | None
    flow_cuda_graph_max_graphs: int
    flow_batching: bool
    control_token_fastpath: bool
    skip_min_patch_stop: bool
    flow_mode: FlowMode
    warmup_on_init: bool
    enable_async_decode: bool
    async_decode_min_batch_size: int
    reference_cache: bool
    stream_slots: int
    prompt_predecode: bool
    stream_batching: bool
    separate_decode_stream: bool
    incremental_cuda_graph: bool
    cuda_graph_latent_sizes: list[int] | None
    cuda_graph_batch_sizes: list[int] | None
    cold_cuda_graph: bool
    cold_cuda_graph_latent_sizes: list[int] | None
    cold_cuda_graph_batch_sizes: list[int] | None


TALKER_FACTORY_DEFAULTS: FireRedTTS3StageFactoryKwargs = {
    "flow_cuda_graph": True,
    "flow_cuda_graph_capture_sizes": None,
    "flow_cuda_graph_max_graphs": 8,
    "flow_batching": True,
    "control_token_fastpath": True,
    "skip_min_patch_stop": True,
    "flow_mode": "compile_bf16",
    "warmup_on_init": True,
    "enable_async_decode": False,
    "async_decode_min_batch_size": 2,
}
VOCODER_FACTORY_DEFAULTS: FireRedTTS3StageFactoryKwargs = {
    "max_batch_size": 4,
    "max_batch_wait_ms": 2.0,
    "prompt_predecode": False,
    "stream_batching": True,
    "separate_decode_stream": True,
    "incremental_cuda_graph": True,
    "cuda_graph_latent_sizes": None,
    "cuda_graph_batch_sizes": None,
    "cold_cuda_graph": True,
    "cold_cuda_graph_latent_sizes": None,
    "cold_cuda_graph_batch_sizes": None,
    "warmup_on_init": True,
}


class FireRedTTS3VocoderFactoryArgs(FactoryArgs):
    """Validate explicit vocoder settings without pinning derived values."""

    stream_slots: int | None = Field(default=None, ge=1)
    prompt_predecode: bool | None = None
    stream_batching: bool | None = None
    separate_decode_stream: bool | None = None
    incremental_cuda_graph: bool | None = None
    cuda_graph_latent_sizes: list[int] | None = None
    cuda_graph_batch_sizes: list[int] | None = None
    cold_cuda_graph: bool | None = None
    cold_cuda_graph_latent_sizes: list[int] | None = None
    cold_cuda_graph_batch_sizes: list[int] | None = None
    warmup_on_init: bool | None = None


class FireRedTTS3VocoderStageConfig(StageConfig):
    factory: FireRedTTS3VocoderFactoryArgs = Field(
        default_factory=FireRedTTS3VocoderFactoryArgs
    )


def load_checkpoint_hf_config(model_path: str) -> FireRedTTS3Config | None:
    """Read the checkpoint config.json for local paths; None when unreadable."""
    config_path = Path(model_path).expanduser() / "config.json"
    if not config_path.is_file():
        return None
    else:
        pass
    with config_path.open(encoding="utf-8") as handle:
        return FireRedTTS3Config(**json.load(handle))


class FireRedTTS3PipelineConfig(PipelineConfig):
    """Build checkpoint-specific stages and derive their shared limits."""

    architecture: ClassVar[str] = "FireRedTTS3ForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "fireredtts3",
        "firered_tts3",
    )
    speech_empty_input_task_types: ClassVar[frozenset[str]] = frozenset(
        {"SemanticEdit", "AcousticEdit"}
    )
    speech_native_speed_control: ClassVar[bool] = True

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "talker": EngineStageConfig,
        "vocoder": FireRedTTS3VocoderStageConfig,
    }

    model_path: str
    variant: FireRedTTS3Variant | None = None
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{PACKAGE_PATH}.stages.create_preprocessing_executor",
            next="reference_encode",
        ),
        StageConfig(
            name="reference_encode",
            process="pipeline",
            factory_path=f"{PACKAGE_PATH}.stages.create_reference_encode_executor",
            gpu=0,
            next="talker",
            stream_to=["vocoder"],
        ),
        EngineStageConfig(
            name="talker",
            process="pipeline",
            factory_path=f"{PACKAGE_PATH}.stages.create_talker_executor",
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        FireRedTTS3VocoderStageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{PACKAGE_PATH}.stages.create_vocoder_executor",
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    @property
    def required_speech_reference_count(self) -> int | None:
        if self.variant == "speaker":
            return 0
        else:
            pass
        return None if self.variant == "instruct" else 1

    @property
    def speech_dual_stream_supported(self) -> bool:
        return self.variant == "speaker"

    @property
    def speech_reference_text_required(self) -> bool:
        return self.variant not in ("instruct", "speaker")

    def model_post_init(self, __context: dict[str, str] | None = None) -> None:
        super().model_post_init(__context)
        if any(stage.tp_size != 1 for stage in self.stages):
            raise ValueError("FireRedTTS3 currently supports tp_size=1 only")
        else:
            pass
        checkpoint_config = load_checkpoint_hf_config(self.model_path)
        if checkpoint_config is not None:
            detected = detect_fireredtts3_variant(checkpoint_config)
            if self.variant is not None and self.variant != detected:
                raise ValueError(
                    f"pinned variant {self.variant!r} disagrees with checkpoint "
                    f"{self.model_path!r} (detected {detected!r})"
                )
            else:
                pass
            self.variant = detected
        else:
            pass
        if self.variant == "speaker":
            self.drop_reference_encode_stage()
        else:
            pass
        n_timesteps, max_patches = self.talker_generation_limits()
        preprocessing_extra = (
            self.stage_named("preprocessing").factory.model_extra or {}
        )
        for key, talker_value in (
            ("n_timesteps", n_timesteps),
            ("max_patches", max_patches),
        ):
            pinned = preprocessing_extra.get(key)
            if pinned is not None and int(pinned) != talker_value:
                raise ValueError(
                    f"FireRedTTS3 preprocessing {key!r} ({int(pinned)}) "
                    f"disagrees with talker {key!r} ({talker_value}); "
                    f"configure only talker.factory.{key} and let "
                    "preprocessing derive it, or set both to the same value"
                )
            else:
                pass
        # note (kermit): Derive slots at launch so config merges do not mistake them for explicit settings.
        stream_slots = self.stage_named("vocoder").factory.stream_slots
        if stream_slots is not None:
            derived = self.talker_max_running_requests(self.stage_named("talker"))
            if int(stream_slots) != derived:
                raise ValueError(
                    f"FireRedTTS3 vocoder stream_slots ({int(stream_slots)}) "
                    f"must equal talker max_running_requests ({derived}); "
                    "omit stream_slots to derive it from the talker"
                )
            else:
                pass
        else:
            pass

    def drop_reference_encode_stage(self) -> None:
        """Remove the reference_encode stage and rewire preprocessing."""
        if all(stage.name != "reference_encode" for stage in self.stages):
            return
        else:
            pass
        self.stages = [
            stage for stage in self.stages if stage.name != "reference_encode"
        ]
        preprocessing = self.stage_named("preprocessing")
        if preprocessing.next == "reference_encode":
            preprocessing.next = "talker"
        else:
            pass
        self.validate_general()

    def talker_generation_limits(self) -> tuple[int, int]:
        """Return (n_timesteps, max_patches) owned by the talker stage."""
        extra = self.stage_named("talker").factory.model_extra or {}
        default_max_patches = (
            SPEAKER_MAX_PATCHES if self.variant == "speaker" else DEFAULT_MAX_PATCHES
        )
        return (
            int(extra.get("n_timesteps", DEFAULT_N_TIMESTEPS)),
            int(extra.get("max_patches", default_max_patches)),
        )

    def stage_factory_kwargs(self, stage_name: str) -> FireRedTTS3StageFactoryKwargs:
        if stage_name in ("preprocessing", "talker"):
            n_timesteps, max_patches = self.talker_generation_limits()
            generation_limits: FireRedTTS3StageFactoryKwargs = {
                "n_timesteps": n_timesteps,
                "max_patches": max_patches,
            }
            if stage_name == "preprocessing":
                return {**generation_limits, "max_concurrency": 8}
            else:
                return {**TALKER_FACTORY_DEFAULTS, **generation_limits}
        elif stage_name == "reference_encode":
            return {
                "max_concurrency": 8,
                "max_batch_size": 1,
                "max_batch_wait_ms": 4.0,
                "warmup_on_init": True,
                "reference_cache": True,
            }
        elif stage_name == "vocoder":
            return {
                **VOCODER_FACTORY_DEFAULTS,
                "stream_slots": self.talker_max_running_requests(
                    self.stage_named("talker")
                ),
            }
        else:
            return {}

    @staticmethod
    def talker_max_running_requests(talker: StageConfig) -> int:
        engine = talker.engine
        value = engine.max_running_requests if engine is not None else None
        if value is None:
            return DEFAULT_MAX_RUNNING_REQUESTS
        else:
            pass
        return int(value)

    @classmethod
    def generation_admission_defaults(cls) -> dict[str, int]:
        return {
            "max_running_requests": DEFAULT_MAX_RUNNING_REQUESTS,
            "max_queued_requests": DEFAULT_MAX_RUNNING_REQUESTS,
        }

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = FireRedTTS3PipelineConfig
