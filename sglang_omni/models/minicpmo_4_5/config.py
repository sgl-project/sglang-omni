# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for the native MiniCPM-o 4.5 duplex stage."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.minicpmo_4_5"
MINICPMO_DUPLEX_STAGE = "minicpmo_duplex"
_NATIVE_DUPLEX_FACTORY = f"{_PKG}.stages.create_minicpmo_duplex_scheduler"


class MiniCPMODuplexSamplingConfig(BaseModel):
    """Sampling policy used for every one-second duplex unit."""

    model_config = ConfigDict(extra="forbid")

    generate_audio: bool = True
    ls_mode: Literal["explicit"] = "explicit"
    force_listen_count: int = Field(default=3, ge=0)
    max_new_speak_tokens_per_chunk: int = Field(default=20, ge=1)
    decode_mode: Literal["sampling", "greedy"] = "sampling"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=20, ge=0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    listen_prob_scale: float = Field(default=1.0, ge=0.0)
    listen_top_k: int | None = Field(default=None, ge=1)
    text_repetition_penalty: float = Field(default=1.05, ge=1.0)
    text_repetition_window_size: int = Field(default=512, ge=1)
    length_penalty: float = Field(default=1.1, ge=0.1, le=5.0)
    tts_temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    tts_repetition_penalty: float = Field(default=1.05, gt=0.0)


def _duplex_stage() -> StageConfig:
    # Perception, main-LLM AR, TTS and token2wav deliberately share this one
    # stage process.  The main LLM is still an ordinary SGLang ModelWorker and
    # therefore uses the framework scheduler, paged KV cache and memory pools.
    return StageConfig(
        name=MINICPMO_DUPLEX_STAGE,
        process=MINICPMO_DUPLEX_STAGE,
        factory=_NATIVE_DUPLEX_FACTORY,
        gpu=0,
        tp_size=1,
        runtime_arg_map={"max_seq_len": "context_length"},
        terminal=True,
    )


class MiniCPMO45PipelineConfig(PipelineConfig):
    """Single-GPU, single-session native SGLang MiniCPM-o 4.5 pipeline."""

    architecture: ClassVar[str] = "MiniCPMO"
    architecture_aliases: ClassVar[tuple[str, ...]] = (
        "MiniCPMOForCausalLM",
        "MiniCPMOForConditionalGeneration",
    )
    requires_model_capabilities: ClassVar[bool] = True

    model_path: str
    revision: str | None = None
    ref_audio_path: str | None = None
    prompt_wav_path: str | None = None
    dtype: Literal["auto", "float16", "bfloat16"] = "bfloat16"
    context_length: int = Field(default=40960, ge=2)
    max_pending_units: int = Field(default=4, ge=1)
    max_pending_commands: int = Field(default=16, ge=1)
    session_ttl_s: float = Field(default=300.0, gt=0.0)
    max_sessions: Literal[1] = 1
    server_args_overrides: dict[str, Any] = Field(default_factory=dict)
    duplex_sampling: MiniCPMODuplexSamplingConfig = Field(
        default_factory=MiniCPMODuplexSamplingConfig
    )
    entry_stage: str = MINICPMO_DUPLEX_STAGE
    stages: list[StageConfig] = Field(default_factory=lambda: [_duplex_stage()])

    @field_validator("model_path")
    @classmethod
    def _validate_model_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_path must be non-empty")
        return value.strip()

    @field_validator("revision")
    @classmethod
    def _validate_revision(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            raise ValueError("revision must be non-empty when provided")
        return value.strip()

    @classmethod
    def mem_fraction_role_to_stage(cls) -> dict[str, str]:
        return {"thinker": MINICPMO_DUPLEX_STAGE}

    @classmethod
    def generation_sglang_role_to_stage(cls) -> dict[str, str]:
        return {"generation": MINICPMO_DUPLEX_STAGE}

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if self.max_pending_commands < self.max_pending_units:
            raise ValueError(
                "max_pending_commands must be greater than or equal to "
                "max_pending_units"
            )
        if len(self.stages) != 1:
            raise ValueError("MiniCPM-o duplex requires exactly one stage")

        stage = self.stages[0]
        if stage.name != MINICPMO_DUPLEX_STAGE:
            raise ValueError(
                f"MiniCPM-o duplex stage must be named {MINICPMO_DUPLEX_STAGE!r}"
            )
        if stage.factory != _NATIVE_DUPLEX_FACTORY:
            raise ValueError(
                "MiniCPM-o duplex stage must use the native SGLang factory "
                f"{_NATIVE_DUPLEX_FACTORY!r}"
            )
        if not stage.terminal or stage.tp_size != 1 or not isinstance(stage.gpu, int):
            raise ValueError(
                "MiniCPM-o duplex stage must be terminal, GPU-backed, and TP=1"
            )

        args = dict(stage.factory_args)
        args.update(
            {
                "model_path": self.model_path,
                "revision": self.revision,
                "ref_audio_path": self.ref_audio_path,
                "prompt_wav_path": self.prompt_wav_path,
                "dtype": self.dtype,
                "context_length": self.context_length,
                "max_pending_units": self.max_pending_units,
                "max_pending_commands": self.max_pending_commands,
                "session_ttl_s": self.session_ttl_s,
                "max_sessions": self.max_sessions,
                "server_args_overrides": dict(self.server_args_overrides),
                # Preserve the Demo's explicit ``listen_top_k=None`` default;
                # dropping it would select a different runtime default.
                "duplex_sampling": self.duplex_sampling.model_dump(),
            }
        )
        stage.factory_args = args


EntryClass = MiniCPMO45PipelineConfig


__all__ = [
    "EntryClass",
    "MINICPMO_DUPLEX_STAGE",
    "MiniCPMO45PipelineConfig",
    "MiniCPMODuplexSamplingConfig",
]
