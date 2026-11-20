# SPDX-License-Identifier: Apache-2.0
"""MOSS-Transcribe-Diarize state for a Prefill to Decode handoff."""

from dataclasses import fields, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sglang.srt.managers.schedule_batch import Req

from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    MossTranscribeDiarizeRequestData,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

RESUME_SCHEMA = "moss-td-v1"


class MossTDResumeState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    resume_schema: Literal["moss-td-v1"] = Field(alias="schema")
    audio_duration_s: float = Field(ge=0)
    language: str
    engine_start_s: float = Field(ge=0)


def build_prefill_state(
    request: Req,
) -> tuple[dict[str, object], dict[str, object], list[int]]:
    request_data = request.omni_data
    if not isinstance(request_data, MossTranscribeDiarizeRequestData):
        raise TypeError("MOSS-TD PD requires MossTranscribeDiarizeRequestData")
    else:
        pass
    payload = request_data.stage_payload
    metadata = {
        key: value
        for key, value in payload.request.metadata.items()
        if key not in ("audios", "audio_data", "audio")
    }
    projected = replace(
        payload,
        request=replace(payload.request, inputs=None, metadata=metadata),
        data=None,
    )
    state = MossTDResumeState(
        schema=RESUME_SCHEMA,
        audio_duration_s=request_data.audio_duration_s,
        language=request_data.language,
        engine_start_s=request_data.engine_start_s,
    )
    return (
        projected.to_dict(),
        state.model_dump(by_alias=True),
        list(request.origin_input_ids),
    )


def restore_decode_state(
    request: Req,
    request_data: SGLangARRequestData,
    resume: dict[str, object] | None,
) -> None:
    state = MossTDResumeState.model_validate(resume)
    request.omni_data = MossTranscribeDiarizeRequestData(
        **{
            field.name: getattr(request_data, field.name)
            for field in fields(SGLangARRequestData)
        },
        prompt_token_ids=list(request.origin_input_ids),
        audio_duration_s=state.audio_duration_s,
        language=state.language,
        engine_start_s=state.engine_start_s,
    )
