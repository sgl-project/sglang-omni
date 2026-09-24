# SPDX-License-Identifier: Apache-2.0
"""Backend chat options must survive the shared HTTP request boundary."""

import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.client import Client
from sglang_omni.client.types import CompletionStreamChunk, UsageInfo
from sglang_omni.models.cosmos3.reasoner import build_chat_fields
from sglang_omni.proto import StagePayload
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest


def test_backend_options_reach_the_native_request_validator():
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest as NativeChatRequest,
    )

    options = {
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,
        "min_tokens": 4,
        "rid": "untrusted-id",
        "stop_token_ids": [7],
    }
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        max_tokens=12,
        temperature=0.0,
        **options,
    )
    generated = build_chat_generate_request(request)
    assert generated.extra_params == options
    payload = StagePayload("owned-request", Client.build_omni_request(generated), None)
    fields = build_chat_fields(
        payload, "native-model", set(NativeChatRequest.model_fields)
    )
    native = NativeChatRequest(**fields)
    assert native.stream_options.include_usage is True
    assert native.chat_template_kwargs == {"enable_thinking": False}
    assert native.ignore_eos is True
    assert native.min_tokens == 4
    assert native.stop_token_ids == [7]
    assert native.rid == "owned-request"
    assert native.max_tokens == 12 and native.temperature == 0.0
    assert request.model_extra == options


def test_backend_options_cannot_replace_declared_params():
    for length in ({"max_tokens": 16}, {}):
        with pytest.raises(ValidationError, match="max_completion_tokens"):
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hello"}],
                max_new_tokens=10**9,
                **length,
            )


def test_invalid_backend_options_reach_native_validation():
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest as NativeChatRequest,
    )

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        stream_options={"include_usage": "invalid"},
    )
    payload = StagePayload(
        "owned-request",
        Client.build_omni_request(build_chat_generate_request(request)),
        None,
    )
    fields = build_chat_fields(
        payload, "native-model", set(NativeChatRequest.model_fields)
    )
    with pytest.raises(ValidationError, match="include_usage"):
        NativeChatRequest(**fields)


@pytest.mark.parametrize("include_usage", [True, False, None])
def test_http_stream_preserves_backend_usage_option(include_usage):
    class UsageClient:
        def __init__(self):
            self.requests = []

        def health(self):
            return {"running": True}

        async def completion_stream(self, request, *, request_id, audio_format):
            self.requests.append(request)
            yield CompletionStreamChunk(
                request_id=request_id, text="hello", modality="text"
            )
            enabled = request.extra_params.get("stream_options", {}).get(
                "include_usage"
            )
            yield CompletionStreamChunk(
                request_id=request_id,
                modality="text",
                finish_reason="length",
                usage=(
                    UsageInfo(prompt_tokens=3, completion_tokens=1, total_tokens=4)
                    if enabled
                    else None
                ),
            )

    backend = UsageClient()
    body = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    if include_usage is not None:
        body["stream_options"] = {"include_usage": include_usage}
    with TestClient(create_app(backend, model_name="native-model")) as client:
        response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    frames = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert frames[-1] == "[DONE]" and frames.count("[DONE]") == 1
    values = [json.loads(frame) for frame in frames[:-1]]
    usage = [value["usage"] for value in values if value.get("usage") is not None]
    assert usage == (
        [{"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}]
        if include_usage
        else []
    )
    assert values[-1]["choices"][0]["finish_reason"] == "length"
    assert (
        "".join(
            choice.get("delta", {}).get("content", "")
            for value in values
            for choice in value["choices"]
        )
        == "hello"
    )
    assert len(backend.requests) == 1
    assert backend.requests[0].extra_params == (
        {"stream_options": {"include_usage": include_usage}}
        if include_usage is not None
        else {}
    )
