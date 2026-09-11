# SPDX-License-Identifier: Apache-2.0
"""MOSS-TD request state that crosses the Prefill/Decode boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    make_moss_transcribe_diarize_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

MOSS_TD_PD_RESUME_SCHEMA = "moss-td-pd-v1"


def make_scheduler_adapters(**kwargs: Any) -> tuple[Callable, Callable]:
    """Reuse the model adapters while rejecting the unsupported stream path."""

    request_builder, result_adapter = make_moss_transcribe_diarize_scheduler_adapters(
        **kwargs
    )

    def non_streaming_request_builder(payload: StagePayload) -> Any:
        if (payload.request.params or {}).get("stream"):
            raise ValueError(
                "MOSS-Transcribe-Diarize PD currently requires stream=false"
            )
        return request_builder(payload)

    return non_streaming_request_builder, result_adapter


def make_state_adapters() -> tuple[
    Callable[[Any], tuple[dict[str, Any], dict[str, Any], list[int]]],
    Callable[[Any, SGLangARRequestData, dict[str, Any] | None], None],
]:
    """Build the tensor-free MOSS-TD state carried beside transferred KV."""

    def state_builder(
        req: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], list[int]]:
        data = req._omni_data
        payload = data.stage_payload
        continuation_payload = StagePayload(
            request_id=payload.request_id,
            # The raw audio and request metadata are Prefill-only. Decode only
            # needs params retained for the terminal payload.
            request=OmniRequest(
                inputs=None,
                params=dict(payload.request.params or {}),
                metadata={},
            ),
            data=None,
        )
        resume = {
            "schema": MOSS_TD_PD_RESUME_SCHEMA,
            "prompt_token_ids": list(data.prompt_token_ids or req.origin_input_ids),
            "audio_duration_s": float(data.audio_duration_s),
            "language": str(data.language),
            "engine_start_s": float(data.engine_start_s),
            "enforce_request_limits": bool(data.enforce_request_limits),
        }
        return continuation_payload.to_dict(), resume, list(req.origin_input_ids)

    def state_restorer(
        req: Any,
        data: SGLangARRequestData,
        resume: dict[str, Any] | None,
    ) -> None:
        if resume is None or resume.get("schema") != MOSS_TD_PD_RESUME_SCHEMA:
            raise ValueError("invalid MOSS-TD PD resume state")
        data.prompt_token_ids = list(resume["prompt_token_ids"])
        data.audio_duration_s = float(resume["audio_duration_s"])
        data.language = str(resume["language"])
        data.engine_start_s = float(resume["engine_start_s"])
        data.enforce_request_limits = bool(resume["enforce_request_limits"])
        req.multimodal_inputs = None
        req._codec_suppress_tokens = None
        # MOSS-TD only configures token-id stops. Leaving this unset keeps
        # SGLang out of its tokenizer/string-stop finish path.
        req.tokenizer = None

    return state_builder, state_restorer


__all__ = [
    "MOSS_TD_PD_RESUME_SCHEMA",
    "make_scheduler_adapters",
    "make_state_adapters",
]
