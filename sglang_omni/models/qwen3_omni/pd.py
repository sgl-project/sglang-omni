# SPDX-License-Identifier: Apache-2.0
"""Model state required to continue a qwen3_omni thinker after Prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from sglang.srt.managers.schedule_batch import MultimodalInputs, Req
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(kw_only=True)
class QwenThinkerPDAdapter:
    tokenizer: PreTrainedTokenizerBase
    schema = "qwen3_omni-thinker-v1"

    def build_state(
        self, request: Req
    ) -> tuple[dict[str, object], dict[str, object], list[int]]:
        payload = request.omni_data.stage_payload
        state = Qwen3OmniPipelineState.from_dict(payload.data)
        prompt_tokens = state.prompt["input_ids"]
        if isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = prompt_tokens.flatten().tolist()
        else:
            prompt_tokens = list(prompt_tokens)
        projected = Qwen3OmniPipelineState(
            prompt={"input_ids": prompt_tokens},
            stream_state=dict(state.stream_state),
        )
        projected_payload = StagePayload(
            request_id=payload.request_id,
            request=OmniRequest(
                inputs=None,
                params=dict(payload.request.params),
                metadata=dict(payload.request.metadata),
            ),
            data=projected.to_dict(),
        )
        resume: dict[str, object] = {"schema": self.schema}
        multimodal_inputs = request.multimodal_inputs
        if multimodal_inputs is not None:
            position_delta = multimodal_inputs.mrope_position_delta
            if (
                position_delta is None
                or position_delta.shape != (1, 1)
                or position_delta.dtype not in (torch.int64, torch.float32)
                or not torch.isfinite(position_delta).all().item()
            ):
                raise ValueError(
                    "Qwen PD requires one int64 or float32 mRoPE delta per request"
                )
            else:
                resume["mrope_position_delta"] = position_delta.tolist()
        else:
            resume["mrope_position_delta"] = None
        return projected_payload.to_dict(), resume, prompt_tokens

    def restore_state(
        self,
        request: Req,
        request_data: SGLangARRequestData,
        resume: dict[str, object] | None,
    ) -> None:
        if (
            resume is None
            or set(resume) != {"schema", "mrope_position_delta"}
            or resume["schema"] != self.schema
        ):
            raise ValueError("Invalid Qwen thinker PD resume state")
        else:
            pass
        position_delta = resume["mrope_position_delta"]
        if position_delta is not None:
            if (
                not isinstance(position_delta, list)
                or len(position_delta) != 1
                or not isinstance(position_delta[0], list)
                or len(position_delta[0]) != 1
                or type(position_delta[0][0]) not in (int, float)
            ):
                raise ValueError("Qwen PD mRoPE delta must contain one number")
            else:
                pass
            value = position_delta[0][0]
            if type(value) is int:
                if not -(2**63) <= value < 2**63:
                    raise ValueError("Qwen PD mRoPE delta exceeds int64 range")
                else:
                    delta_dtype = torch.int64
            else:
                if (
                    not math.isfinite(value)
                    or abs(value) > torch.finfo(torch.float32).max
                ):
                    raise ValueError("Qwen PD mRoPE delta must be finite float32")
                else:
                    delta_dtype = torch.float32
            multimodal_inputs = MultimodalInputs(mm_items=[])
            multimodal_inputs.mrope_position_delta = torch.tensor(
                position_delta, dtype=delta_dtype
            )
            request.multimodal_inputs = multimodal_inputs
        else:
            request.multimodal_inputs = None
        request._omni_mm_positions = (
            None  # noqa: leading-underscore -- model runner contract
        )
        request.tokenizer = self.tokenizer
        request._codec_suppress_tokens = (
            None  # noqa: leading-underscore -- model runner contract
        )
        request.omni_model_inputs = None
        request._omni_consumed = (
            None  # noqa: leading-underscore -- model runner contract
        )
        request_data.req = request
