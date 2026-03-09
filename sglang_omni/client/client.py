# SPDX-License-Identifier: Apache-2.0
"""Client wrapper for coordinator-based pipelines."""

from __future__ import annotations

import uuid
from typing import Any, AsyncIterator, Callable

import numpy as np

from sglang_omni.client.audio import (
    FORMAT_MIME_TYPES,
    audio_to_base64,
    encode_audio,
    to_numpy,
)
from sglang_omni.client.types import (
    AbortLevel,
    AbortResult,
    ClientError,
    CompletionAudio,
    CompletionResult,
    CompletionStreamChunk,
    GenerateChunk,
    GenerateRequest,
    SpeechResult,
    UsageInfo,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, RequestState, StreamMessage


class Client:
    """Internal client used by API adapters."""

    def __init__(
        self,
        coordinator: Coordinator,
        result_builder: Callable[[str, Any], GenerateChunk] | None = None,
        stream_builder: Callable[[str, StreamMessage], GenerateChunk] | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._result_builder = result_builder or self._default_result_builder
        self._stream_builder = stream_builder or self._default_stream_builder

    # ------------------------------------------------------------------
    # Low-level generate (backward compatible)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # High-level: non-streaming completion
    # ------------------------------------------------------------------

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        """Run a non-streaming completion and return an aggregated result.

        Iterates ``generate()``, accumulates text, concatenates audio chunks,
        and encodes audio to base64.

        Raises:
            ClientError: If the pipeline produces no response at all.
        """
        text_parts: list[str] = []
        audio_chunks: list[Any] = []
        last_chunk: GenerateChunk | None = None
        finish_reason: str | None = None

        async for chunk in self.generate(request, request_id=request_id):
            last_chunk = chunk
            if chunk.modality == "text" and chunk.text:
                text_parts.append(chunk.text)
            if chunk.modality == "audio" and chunk.audio_data is not None:
                audio_chunks.append(chunk.audio_data)
            if chunk.finish_reason is not None:
                finish_reason = chunk.finish_reason

        if last_chunk is None:
            raise ClientError("No response from pipeline")

        full_text = "".join(text_parts)

        audio: CompletionAudio | None = None
        if audio_chunks:
            if len(audio_chunks) == 1:
                combined = audio_chunks[0]
            else:
                combined = np.concatenate([to_numpy(c) for c in audio_chunks])
            audio_b64 = audio_to_base64(combined, output_format=audio_format)
            audio = CompletionAudio(
                id=f"audio-{request_id}",
                data=audio_b64,
                transcript=full_text if full_text else None,
            )

        return CompletionResult(
            request_id=request_id,
            text=full_text,
            audio=audio,
            finish_reason=finish_reason or "stop",
            usage=last_chunk.usage,
        )

    # ------------------------------------------------------------------
    # High-level: streaming completion
    # ------------------------------------------------------------------

    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> AsyncIterator[CompletionStreamChunk]:
        """Iterate ``generate()`` and yield high-level stream chunks.

        Audio data is base64-encoded before yielding so that callers never
        need to touch numpy / raw bytes.
        """
        async for chunk in self.generate(request, request_id=request_id):
            audio_b64: str | None = None
            if chunk.modality == "audio" and chunk.audio_data is not None:
                audio_b64 = audio_to_base64(
                    chunk.audio_data, output_format=audio_format
                )

            yield CompletionStreamChunk(
                request_id=request_id,
                text=chunk.text,
                modality=chunk.modality,
                audio_b64=audio_b64,
                finish_reason=chunk.finish_reason,
                usage=chunk.usage,
                stage_name=chunk.stage_name,
            )

    # ------------------------------------------------------------------
    # High-level: text-to-speech
    # ------------------------------------------------------------------

    async def speech(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        response_format: str = "wav",
        speed: float = 1.0,
    ) -> SpeechResult:
        """Run a TTS request and return encoded audio bytes.

        Raises:
            ClientError: If the pipeline produces no audio output.
        """
        audio_chunks: list[Any] = []
        sample_rate: int | None = None

        async for chunk in self.generate(request, request_id=request_id):
            if chunk.audio_data is not None:
                audio_chunks.append(chunk.audio_data)
            if chunk.sample_rate is not None:
                sample_rate = chunk.sample_rate

        if not audio_chunks:
            raise ClientError("No audio output generated from the pipeline.")

        if len(audio_chunks) == 1:
            audio_data = audio_chunks[0]
        else:
            audio_data = np.concatenate([to_numpy(c) for c in audio_chunks])

        encode_kwargs: dict[str, Any] = {
            "response_format": response_format,
            "speed": speed,
        }
        if sample_rate is not None:
            encode_kwargs["sample_rate"] = sample_rate

        audio_bytes, mime_type = encode_audio(audio_data, **encode_kwargs)

        # Derive actual format from MIME type (encode_audio may fall back
        # to WAV if the requested codec is unavailable).
        actual_format = response_format
        for ext, mt in FORMAT_MIME_TYPES.items():
            if mt == mime_type:
                actual_format = ext
                break

        return SpeechResult(
            audio_bytes=audio_bytes,
            mime_type=mime_type,
            format=actual_format,
        )

    # ------------------------------------------------------------------
    # Other operations
    # ------------------------------------------------------------------

    async def abort(
        self,
        request_id: str,
        level: AbortLevel = AbortLevel.SOFT,
    ) -> AbortResult:
        success = await self._coordinator.abort(request_id)
        return AbortResult(success=success, level_applied=level)

    async def get_status(self, request_id: str) -> RequestState | None:
        info = self._coordinator.get_request_info(request_id)
        if info is None:
            return None
        return info.state

    def health(self) -> dict[str, Any]:
        return self._coordinator.health()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _set_audio_data(chunk: GenerateChunk, data: dict[str, Any]) -> None:
        audio_data = data.get("audio_data") or data.get("audio")
        if audio_data is not None:
            chunk.audio_data = audio_data
            if chunk.modality == "text":
                chunk.modality = "audio"
        sample_rate = data.get("sample_rate")
        if sample_rate is not None:
            chunk.sample_rate = sample_rate

    @staticmethod
    def _build_omni_request(request: GenerateRequest) -> OmniRequest:
        inputs = _extract_inputs(request)
        params = _build_params(request)
        metadata = dict(request.metadata)
        if request.model:
            metadata.setdefault("model", request.model)
        if request.output_modalities:
            metadata["output_modalities"] = request.output_modalities
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
            Client._set_audio_data(chunk, result)
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
            data.request_id = request_id
            if data.stage_name is None:
                data.stage_name = chunk.stage_name
            if data.stage_id is None:
                data.stage_id = chunk.stage_id
            if not data.modality and chunk.modality:
                data.modality = chunk.modality
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
            Client._set_audio_data(chunk, data)
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

    # Build messages list
    messages = [msg.to_dict() for msg in request.messages or []]

    # Check if we have audios, images, or videos in metadata
    audios = request.metadata.get("audios")
    images = request.metadata.get("images")
    videos = request.metadata.get("videos")

    # If we have any media, return a dict with messages and media
    # Otherwise, return just the messages list (for backward compatibility)
    if audios or images or videos:
        result = {"messages": messages}
        if images:
            result["images"] = images
        if audios:
            result["audios"] = audios
        if videos:
            result["videos"] = videos
        return result
    return messages


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
