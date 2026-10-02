# SPDX-License-Identifier: Apache-2.0
"""Request lowering for MOSS-TTS-Nano."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from sglang_omni.proto.request import StagePayload


class NanoReference(BaseModel):
    """One optional voice reference accepted by the speech API."""

    model_config = ConfigDict(extra="ignore")

    audio_path: str | None = None
    ref_audio: str | None = None
    audio: str | None = None
    data: str | None = None
    media_type: str = "audio/wav"
    text: str | None = None


class NanoInputs(BaseModel):
    """Validated text and references from an Omni request."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(validation_alias=AliasChoices("text", "input"))
    references: list[NanoReference] = Field(default_factory=list)


class NanoRequestParams(BaseModel):
    """Generic generation parameters carried by an Omni request."""

    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")

    stream: bool = False
    max_new_tokens: int | None = Field(default=None, gt=0, strict=True)
    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=-1, strict=True)
    repetition_penalty: float | None = Field(default=None, gt=0)
    text_temperature: float | None = Field(default=None, ge=0)
    text_top_p: float | None = Field(default=None, gt=0, le=1)
    text_top_k: int | None = Field(default=None, ge=-1, strict=True)
    audio_temperature: float | None = Field(default=None, ge=0)
    audio_top_p: float | None = Field(default=None, gt=0, le=1)
    audio_top_k: int | None = Field(default=None, ge=-1, strict=True)
    audio_repetition_penalty: float | None = Field(default=None, gt=0)
    num_quantizers: int | None = Field(default=None, alias="nq", gt=0, strict=True)
    seed: int | None = Field(default=None, ge=0, strict=True)
    instructions: str | None = None


class NanoTTSParams(BaseModel):
    """Nano-specific metadata added by the speech API."""

    model_config = ConfigDict(allow_inf_nan=False, extra="ignore")

    explicit_generation_params: list[str] = Field(default_factory=list)
    ref_audio: str | bytes | Path | None = None
    ref_text: str | None = None
    text_temperature: float | None = Field(default=None, ge=0)
    text_top_p: float | None = Field(default=None, gt=0, le=1)
    text_top_k: int | None = Field(default=None, ge=-1, strict=True)
    audio_temperature: float | None = Field(default=None, ge=0)
    audio_top_p: float | None = Field(default=None, gt=0, le=1)
    audio_top_k: int | None = Field(default=None, ge=-1, strict=True)
    audio_repetition_penalty: float | None = Field(default=None, gt=0)
    num_quantizers: int | None = Field(default=None, alias="nq", gt=0, strict=True)
    seed: int | None = Field(default=None, ge=0, strict=True)
    instructions: str | None = None


class NanoSamplingParams(BaseModel):
    """Validated generation options for the checkpoint's inference API."""

    model_config = ConfigDict(allow_inf_nan=False)

    max_new_frames: int = Field(default=375, gt=0, strict=True)
    do_sample: bool = True
    text_temperature: float = Field(default=1.0, ge=0)
    text_top_p: float = Field(default=1.0, gt=0, le=1)
    text_top_k: int = Field(default=50, ge=-1, strict=True)
    audio_temperature: float = Field(default=0.8, ge=0)
    audio_top_p: float = Field(default=0.95, gt=0, le=1)
    audio_top_k: int = Field(default=25, ge=-1, strict=True)
    audio_repetition_penalty: float = Field(default=1.2, gt=0)
    num_quantizers: int | None = Field(
        default=None, validation_alias="nq", serialization_alias="nq", gt=0, strict=True
    )
    seed: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_temperature(self) -> NanoSamplingParams:
        if self.do_sample and (
            self.text_temperature == 0 or self.audio_temperature == 0
        ):
            raise ValueError(
                "MOSS-TTS-Nano temperatures must be positive when sampling"
            )
        else:
            return self


@dataclass(frozen=True, kw_only=True)
class MossTTSNanoRequest:
    text: str
    ref_audio: str | bytes | Path | None
    ref_text: str | None
    generation_kwargs: dict[str, int | float | bool]


def build_moss_tts_nano_request(payload: StagePayload) -> MossTTSNanoRequest:
    raw_inputs = payload.request.inputs
    if isinstance(raw_inputs, str):
        inputs = NanoInputs(text=raw_inputs)
    elif isinstance(raw_inputs, dict):
        inputs = NanoInputs.model_validate(raw_inputs)
    else:
        raise TypeError("MOSS-TTS-Nano inputs must be text or an input object")

    params = NanoRequestParams.model_validate(payload.request.params or {})
    metadata = payload.request.metadata or {}
    raw_tts_params = metadata.get("tts_params")
    tts_params = NanoTTSParams.model_validate(
        raw_tts_params if isinstance(raw_tts_params, dict) else {}
    )
    if params.stream:
        raise ValueError("MOSS-TTS-Nano does not support streaming")

    if not inputs.text.strip():
        raise ValueError("MOSS-TTS-Nano requires non-empty text")
    if len(inputs.references) > 1:
        raise ValueError("MOSS-TTS-Nano accepts at most one reference")

    reference = inputs.references[0] if inputs.references else None
    if reference is None:
        ref_audio = tts_params.ref_audio
        ref_text = tts_params.ref_text
    else:
        ref_audio = reference.audio_path or reference.ref_audio or reference.audio
        if ref_audio is None and reference.data is not None:
            ref_audio = f"data:{reference.media_type};base64,{reference.data}"
        else:
            ref_audio = ref_audio or tts_params.ref_audio
        ref_text = reference.text or tts_params.ref_text

    instructions = tts_params.instructions or params.instructions
    if instructions is not None and instructions.strip():
        raise ValueError("MOSS-TTS-Nano does not support instructions")

    return MossTTSNanoRequest(
        text=inputs.text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        generation_kwargs=build_generation_kwargs(params, tts_params=tts_params),
    )


def build_generation_kwargs(
    params: NanoRequestParams, *, tts_params: NanoTTSParams
) -> dict[str, int | float | bool]:
    explicit_fields = set(tts_params.explicit_generation_params)
    generic_values = {
        "max_new_tokens": params.max_new_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "repetition_penalty": params.repetition_penalty,
    }
    aliases = {
        "max_new_tokens": ("max_new_frames",),
        "temperature": ("text_temperature", "audio_temperature"),
        "top_p": ("text_top_p", "audio_top_p"),
        "top_k": ("text_top_k", "audio_top_k"),
        "repetition_penalty": ("audio_repetition_penalty",),
    }
    generation: dict[str, int | float | bool] = {}
    for source, targets in aliases.items():
        value = generic_values[source]
        if value is None or (
            source != "max_new_tokens" and source not in explicit_fields
        ):
            continue
        else:
            generation.update(dict.fromkeys(targets, value))

    request_overrides = params.model_dump(
        include={
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
            "num_quantizers",
        },
        exclude_none=True,
        by_alias=True,
    )
    tts_overrides = tts_params.model_dump(
        include={
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
            "audio_repetition_penalty",
            "num_quantizers",
        },
        exclude_none=True,
        by_alias=True,
    )
    generation.update(tts_overrides)
    generation.update(request_overrides)
    generation["seed"] = tts_params.seed if tts_params.seed is not None else params.seed
    generation["do_sample"] = not (
        "temperature" in explicit_fields and params.temperature == 0
    )
    return NanoSamplingParams.model_validate(generation).model_dump(
        exclude_none=True, by_alias=True
    )
