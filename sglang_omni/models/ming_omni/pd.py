# SPDX-License-Identifier: Apache-2.0
"""Model state required to continue a ming_omni thinker after Prefill."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.managers.schedule_batch import Req
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.ming_omni.io import MingOmniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(kw_only=True)
class MingThinkerPDAdapter:
    tokenizer: PreTrainedTokenizerBase
    schema = "ming_omni-thinker-v1"

    def build_state(
        self, request: Req
    ) -> tuple[dict[str, object], dict[str, object], list[int]]:
        payload = request.omni_data.stage_payload
        modalities = payload.request.metadata.get("output_modalities", ["text"])
        if modalities not in (["text"], ("text",), "text", None):
            raise ValueError("Ming PD currently supports thinker text output only")
        else:
            pass
        state = MingOmniPipelineState.from_dict(payload.data)
        prompt_tokens = state.prompt["input_ids"]
        if isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = prompt_tokens.flatten().tolist()
        else:
            prompt_tokens = list(prompt_tokens)
        projected = MingOmniPipelineState(
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
        return projected_payload.to_dict(), resume, prompt_tokens

    def restore_state(
        self,
        request: Req,
        request_data: SGLangARRequestData,
        resume: dict[str, object] | None,
    ) -> None:
        if resume != {"schema": self.schema}:
            raise ValueError("Invalid Ming thinker PD resume state")
        else:
            pass
        request._codec_suppress_tokens = (
            None  # noqa: leading-underscore -- model runner contract
        )
        request.tokenizer = self.tokenizer
        request.omni_model_inputs = None
        request._omni_consumed = (
            None  # noqa: leading-underscore -- model runner contract
        )
        request_data.req = request
