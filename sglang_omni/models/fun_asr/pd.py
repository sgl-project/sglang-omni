# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR request state transferred from Prefill to Decode."""

from dataclasses import dataclass, fields, replace
from typing import ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
from sglang.srt.utils.hf_transformers import attach_additional_stop_token_ids
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.fun_asr.request_builders import FunASRRequestData
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class FunASRResumeState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    resume_schema: Literal["fun-asr-v1"] = Field(alias="schema")
    num_audio_tokens: int = Field(ge=0)
    audio_duration_s: float = Field(ge=0)
    language: str | None
    streaming_prefix_text: str
    engine_start_s: float = Field(ge=0)


@dataclass(kw_only=True)
class FunASRPDAdapter:
    tokenizer: PreTrainedTokenizerBase
    schema: ClassVar[str] = "fun-asr-v1"

    def build_state(
        self, request: Req
    ) -> tuple[dict[str, object], dict[str, object], list[int]]:
        request_data = request.omni_data
        if not isinstance(request_data, FunASRRequestData):
            raise TypeError("Fun-ASR PD requires FunASRRequestData")
        else:
            pass
        multimodal_inputs = request.multimodal_inputs
        if multimodal_inputs is not None and (
            multimodal_inputs.mrope_position_delta is None
            or torch.count_nonzero(multimodal_inputs.mrope_position_delta)
        ):
            raise ValueError("Fun-ASR PD requires text-equivalent mRoPE positions")
        else:
            pass
        payload = request_data.stage_payload
        projected = replace(
            payload, request=replace(payload.request, inputs=None), data=None
        )
        state = FunASRResumeState(
            schema=self.schema,
            num_audio_tokens=request_data.num_audio_tokens,
            audio_duration_s=request_data.audio_duration_s,
            language=request_data.language,
            streaming_prefix_text=request_data.streaming_prefix_text,
            engine_start_s=request_data.engine_start_s,
        )
        return (
            projected.to_dict(),
            state.model_dump(by_alias=True),
            list(request.origin_input_ids),
        )

    def restore_state(
        self,
        request: Req,
        request_data: SGLangARRequestData,
        resume: dict[str, object] | None,
    ) -> None:
        state = FunASRResumeState.model_validate(resume)
        request.omni_data = FunASRRequestData(
            **{
                field.name: getattr(request_data, field.name)
                for field in fields(SGLangARRequestData)
            },
            prompt_token_ids=list(request.origin_input_ids),
            num_audio_tokens=state.num_audio_tokens,
            audio_duration_s=state.audio_duration_s,
            language=state.language,
            streaming_prefix_text=state.streaming_prefix_text,
            engine_start_s=state.engine_start_s,
        )
        if request.sampling_params.stop_strs or request.sampling_params.stop_regex_strs:
            attach_additional_stop_token_ids(self.tokenizer)
            request.tokenizer = self.tokenizer
        else:
            pass
        request.multimodal_inputs = MultimodalInputs(mm_items=[])
        request.multimodal_inputs.mrope_position_delta = torch.zeros(
            1, dtype=torch.long
        )
        request._codec_suppress_tokens = (
            None  # noqa: leading-underscore -- model runner contract
        )
