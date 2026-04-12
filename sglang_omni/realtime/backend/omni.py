# SPDX-License-Identifier: Apache-2.0
"""Response backend for omni chat models that emit text and audio directly."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from sglang_omni.client import Client, GenerateRequest, Message, SamplingParams
from sglang_omni.client.audio import DEFAULT_SAMPLE_RATE
from sglang_omni.realtime.backend.base import (
    BackendCapabilities,
    ResponseBackend,
    ResponseEvent,
    TurnContext,
)


class OmniResponseBackend(ResponseBackend):
    """Realtime backend backed by the existing request-oriented omni client."""

    def __init__(
        self,
        *,
        client: Client,
        model: str,
        max_new_tokens: int = 256,
        output_modalities: tuple[str, ...] = ("text", "audio"),
    ) -> None:
        self._client = client
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._output_modalities = output_modalities
        self._capabilities = BackendCapabilities(
            accepts_audio_input=True,
            accepts_video_input=True,
            returns_text="text" in output_modalities,
            returns_audio="audio" in output_modalities,
            supports_cancel=True,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    async def stream_response(
        self,
        turn: TurnContext,
    ) -> AsyncIterator[ResponseEvent]:
        response_id = uuid.uuid4().hex
        yield ResponseEvent(type="response_started", response_id=response_id)

        finish_reason = "stop"
        request = self._build_request(turn)
        try:
            async for chunk in self._client.generate(request, request_id=response_id):
                if chunk.finish_reason is not None:
                    finish_reason = chunk.finish_reason
                    continue

                if chunk.text:
                    yield ResponseEvent(
                        type="text_delta",
                        response_id=response_id,
                        text=chunk.text,
                    )

                if chunk.audio_data is not None:
                    yield ResponseEvent(
                        type="audio_chunk",
                        response_id=response_id,
                        audio=np.asarray(chunk.audio_data),
                        sample_rate=chunk.sample_rate or DEFAULT_SAMPLE_RATE,
                    )
        except Exception as exc:
            yield ResponseEvent(
                type="error",
                response_id=response_id,
                error=str(exc),
            )
            return

        yield ResponseEvent(
            type="done",
            response_id=response_id,
            finish_reason=finish_reason,
        )

    async def cancel(self, response_id: str) -> None:
        await self._client.abort(response_id)

    def _build_request(self, turn: TurnContext) -> GenerateRequest:
        messages: list[Message] = []
        if turn.instructions:
            messages.append(Message(role="system", content=turn.instructions))
        messages.extend(
            Message(role=item["role"], content=item["content"]) for item in turn.history
        )
        messages.append(Message(role="user", content=turn.user_text or " "))

        metadata: dict[str, Any] = {}
        if turn.user_audio is not None:
            metadata["audios"] = [turn.user_audio]
            metadata["audio_target_sr"] = turn.user_audio_sample_rate
        if turn.recent_video is not None:
            metadata["videos"] = [turn.recent_video]
            metadata["video_fps"] = turn.recent_video_fps

        return GenerateRequest(
            model=self._model,
            messages=messages,
            sampling=SamplingParams(max_new_tokens=self._max_new_tokens),
            stream=True,
            output_modalities=list(self._output_modalities),
            metadata=metadata,
        )
