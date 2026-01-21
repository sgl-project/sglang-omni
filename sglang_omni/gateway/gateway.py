# SPDX-License-Identifier: Apache-2.0
"""Gateway wrapper for coordinator-based pipelines."""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, Callable

from sglang_omni.gateway.types import (
    AbortLevel,
    AbortResult,
    GenerateChunk,
    GenerateRequest,
    UsageInfo,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, RequestState, StreamMessage


class Gateway:
    """Internal gateway used by API adapters."""

    def __init__(
        self,
        coordinator: Coordinator,
        result_builder: Callable[[str, Any], GenerateChunk] | None = None,
        stream_builder: Callable[[str, StreamMessage], GenerateChunk] | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._result_builder = result_builder or self._default_result_builder
        self._stream_builder = stream_builder or self._default_stream_builder

    async def generate(
        self,
        request: GenerateRequest,
        request_id: str | None = None,
    ) -> AsyncIterator[GenerateChunk]:
        req_id = request_id or str(uuid.uuid4())
        omni_request = self._build_omni_request(request)
        if request.stream:
            async for msg in self._coordinator.stream(req_id, omni_request):
                if isinstance(msg, StreamMessage):
                    yield self._stream_builder(req_id, msg)
                else:
                    yield self._result_builder(req_id, msg.result)
            return

        result = await self._coordinator.submit(req_id, omni_request)
        yield self._result_builder(req_id, result)

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        success = await self._coordinator.abort(request_id)
        return AbortResult(success=success, level_applied=AbortLevel.SOFT)

    async def get_status(self, request_id: str) -> RequestState | None:
        info = self._coordinator.get_request_info(request_id)
        if info is None:
            return None
        return info.state

    def health(self) -> dict[str, Any]:
        return self._coordinator.health()

    @staticmethod
    def _build_omni_request(request: GenerateRequest) -> OmniRequest:
        inputs = _extract_inputs(request)
        params = _build_params(request)
        metadata = dict(request.metadata)
        if request.model:
            metadata.setdefault("model", request.model)
        return OmniRequest(inputs=inputs, params=params, metadata=metadata)

    @staticmethod
    def _default_result_builder(request_id: str, result: Any) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id, finish_reason="stop")
        if isinstance(result, GenerateChunk):
            result.request_id = request_id
            return result
        if isinstance(result, dict):
            text = result.get("text")
            if isinstance(text, str):
                chunk.text = text
            token_ids = result.get("token_ids")
            if token_ids is not None:
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                chunk.token_ids = list(token_ids)
            logprobs = result.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            finish_reason = result.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            chunk.stage_id = result.get("stage_id")
            chunk.stage_name = result.get("stage_name")
            modality = result.get("modality")
            if modality is not None:
                chunk.modality = modality
            chunk.usage = UsageInfo.from_dict(result.get("usage"))
            return chunk
        if isinstance(result, str):
            chunk.text = result
            return chunk
        chunk.text = str(result)
        return chunk

    @staticmethod
    def _default_stream_builder(request_id: str, msg: StreamMessage) -> GenerateChunk:
        chunk = GenerateChunk(request_id=request_id)
        chunk.stage_name = msg.stage_name or msg.from_stage
        chunk.stage_id = msg.stage_id
        if msg.modality:
            chunk.modality = msg.modality

        data = msg.chunk
        if isinstance(data, GenerateChunk):
            return data
        if isinstance(data, dict):
            text = data.get("text")
            if isinstance(text, str):
                chunk.text = text
            token_ids = data.get("token_ids")
            if token_ids is not None:
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                chunk.token_ids = list(token_ids)
            logprobs = data.get("logprobs")
            if logprobs is not None:
                chunk.logprobs = logprobs
            finish_reason = data.get("finish_reason")
            if finish_reason is not None:
                chunk.finish_reason = finish_reason
            usage = data.get("usage")
            if usage is not None:
                chunk.usage = UsageInfo.from_dict(usage)
            stage_name = data.get("stage_name")
            if stage_name is not None:
                chunk.stage_name = stage_name
            stage_id = data.get("stage_id")
            if stage_id is not None:
                chunk.stage_id = stage_id
            modality = data.get("modality")
            if modality is not None:
                chunk.modality = modality
            return chunk
        if isinstance(data, str):
            chunk.text = data
            return chunk
        if isinstance(data, int):
            chunk.token_ids = [data]
            return chunk
        chunk.text = str(data)
        return chunk


def _extract_inputs(request: GenerateRequest) -> Any:
    choices = [
        request.prompt is not None,
        request.prompt_token_ids is not None,
        request.messages is not None,
    ]
    if sum(choices) != 1:
        raise ValueError(
            "GenerateRequest requires exactly one input: "
            "prompt, prompt_token_ids, or messages."
        )
    if request.prompt is not None:
        return request.prompt
    if request.prompt_token_ids is not None:
        return list(request.prompt_token_ids)
    return [msg.to_dict() for msg in request.messages or []]


def _build_params(request: GenerateRequest) -> dict[str, Any]:
    params = request.sampling.to_dict()
    max_new_tokens = request.sampling.max_new_tokens
    if request.max_tokens is not None:
        max_new_tokens = request.max_tokens
    if max_new_tokens is None:
        params.pop("max_new_tokens", None)
    else:
        params["max_new_tokens"] = max_new_tokens
    params["stream"] = request.stream
    if request.stage_sampling:
        params["stage_sampling"] = {
            key: value.to_dict() for key, value in request.stage_sampling.items()
        }
    if request.stage_params:
        params["stage_params"] = request.stage_params
    return params
