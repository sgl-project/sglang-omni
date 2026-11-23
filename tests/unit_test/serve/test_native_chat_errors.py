# SPDX-License-Identifier: Apache-2.0
"""Native client-input errors survive shared serving and stage transport."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest as NativeChatRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client
from sglang_omni.client.types import CompletionStreamChunk
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.native_chat_scheduler import NativeChatScheduler
from sglang_omni.serve import create_app


class NativeValidationClient:
    def __init__(self):
        self.closed = 0
        self.scheduler = object.__new__(NativeChatScheduler)
        self.scheduler.stage_name = "reasoner"
        self.scheduler.engine = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                served_model_name="native-model", abort_request=lambda rid: None
            )
        )
        self.scheduler.request_type = NativeChatRequest
        self.scheduler.max_media_bytes = None
        self.scheduler.serving_chat = SimpleNamespace(handle_request=AsyncMock())

    def health(self):
        return {"running": True}

    async def completion(self, request, *, request_id, audio_format):
        payload = StagePayload(request_id, Client.build_omni_request(request), None)
        try:
            return await self.scheduler.complete(payload)
        except ValueError as exc:
            raise QueueFullError.from_message(str(exc)) from exc

    async def completion_stream(self, request, *, request_id, audio_format):
        try:
            await self.completion(
                request, request_id=request_id, audio_format=audio_format
            )
            yield CompletionStreamChunk(
                request_id=request_id, text="unexpected", modality="text"
            )
        finally:
            self.closed += 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "option",
    [
        {"stream_options": {"include_usage": "invalid"}},
        {"n": 2},
        {"return_sampling_mask": True},
    ],
)
def test_native_chat_rejections_survive_http_and_stage_transport(stream, option):
    backend = NativeValidationClient()
    native_validation = "return_sampling_mask" in option
    if native_validation:
        native = object.__new__(OpenAIServingChat)
        backend.scheduler.serving_chat.handle_request = AsyncMock(
            wraps=native.handle_request
        )
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
        "stage_params": {"reasoner": option},
    }
    with TestClient(create_app(backend, model_name="native-model")) as client:
        response = client.post("/v1/chat/completions", json=body)
    if stream:
        assert response.status_code == 200
        frames = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert len(frames) == 2 and frames[-1] == "[DONE]"
        error = json.loads(frames[0])["error"]
        assert error["code"] == 400 and error["type"] == "invalid_request_error"
        assert backend.closed == 1
    else:
        assert response.status_code == 400
    if native_validation:
        backend.scheduler.serving_chat.handle_request.assert_awaited_once()
    else:
        backend.scheduler.serving_chat.handle_request.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("structured", [False, True])
def test_native_internal_errors_remain_server_errors(stream, structured):
    backend = NativeValidationClient()
    if structured:
        native = object.__new__(OpenAIServingChat)
        backend.scheduler.serving_chat.handle_request.return_value = (
            native.create_error_response("internal shape mismatch", status_code=500)
        )
    else:
        backend.scheduler.serving_chat.handle_request.side_effect = ValueError(
            "internal shape mismatch"
        )
    body = {"messages": [{"role": "user", "content": "hello"}], "stream": stream}
    with TestClient(create_app(backend, model_name="native-model")) as client:
        response = client.post("/v1/chat/completions", json=body)
    if stream:
        assert response.status_code == 200
        frames = [
            line.removeprefix("data: ")
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert len(frames) == 2 and frames[-1] == "[DONE]"
        error = json.loads(frames[0])["error"]
        assert error["code"] == 500 and error["type"] == "server_error"
        assert backend.closed == 1
    else:
        assert response.status_code == 500
    backend.scheduler.serving_chat.handle_request.assert_awaited_once()
