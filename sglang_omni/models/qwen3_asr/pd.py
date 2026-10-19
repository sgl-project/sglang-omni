# SPDX-License-Identifier: Apache-2.0
"""Chunk-local Qwen3-ASR state across an explicit PD handoff."""

from dataclasses import dataclass, fields
from typing import ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.qwen3_asr.mrope_fast_path import DEGENERATE_MROPE_FLAG
from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRRequestData
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


class ASRResumeState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    resume_schema: Literal["qwen3-asr-v1"] = Field(alias="schema")
    audio_duration_s: float = Field(ge=0)
    language: str | None
    streaming_prefix_text: str
    engine_start_s: float = Field(ge=0)


@dataclass(kw_only=True)
class Qwen3ASRPDAdapter:
    tokenizer: PreTrainedTokenizerBase
    schema: ClassVar[str] = "qwen3-asr-v1"

    def build_state(
        self, request: Req
    ) -> tuple[dict[str, object], dict[str, object], list[int]]:
        request_data = request.omni_data
        if not isinstance(request_data, Qwen3ASRRequestData):
            raise TypeError("Qwen3-ASR PD requires Qwen3ASRRequestData")
        else:
            pass
        multimodal_inputs = request.multimodal_inputs
        if multimodal_inputs is not None and (
            multimodal_inputs.mrope_position_delta is None
            or not torch.equal(
                multimodal_inputs.mrope_position_delta,
                torch.zeros_like(multimodal_inputs.mrope_position_delta),
            )
        ):
            raise ValueError("Qwen3-ASR PD requires text-equivalent mRoPE positions")
        else:
            pass
        payload = request_data.stage_payload
        projected = StagePayload(
            request_id=payload.request_id,
            request=OmniRequest(
                inputs=None,
                params=dict(payload.request.params),
                metadata=dict(payload.request.metadata),
            ),
            data=None,
        )
        resume = ASRResumeState(
            schema=self.schema,
            audio_duration_s=request_data.audio_duration_s,
            language=request_data.language,
            streaming_prefix_text=request_data.streaming_prefix_text,
            engine_start_s=request_data.engine_start_s,
        )
        return (
            projected.to_dict(),
            resume.model_dump(by_alias=True),
            list(request.origin_input_ids),
        )

    def restore_state(
        self,
        request: Req,
        request_data: SGLangARRequestData,
        resume: dict[str, object] | None,
    ) -> None:
        state = ASRResumeState.model_validate(resume)
        restored = Qwen3ASRRequestData(
            **{
                field.name: getattr(request_data, field.name)
                for field in fields(SGLangARRequestData)
            },
            prompt_token_ids=list(request.origin_input_ids),
            audio_duration_s=state.audio_duration_s,
            language=state.language,
            streaming_prefix_text=state.streaming_prefix_text,
            engine_start_s=state.engine_start_s,
        )
        request.omni_data = restored
        request.tokenizer = self.tokenizer
        request.multimodal_inputs = MultimodalInputs(mm_items=[])
        request.multimodal_inputs.mrope_position_delta = torch.tensor(
            [0], dtype=torch.long
        )
        setattr(request.multimodal_inputs, DEGENERATE_MROPE_FLAG, True)
        request._codec_suppress_tokens = (
            None  # noqa: leading-underscore -- model runner contract
        )
