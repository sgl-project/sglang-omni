# SPDX-License-Identifier: Apache-2.0
"""Client-side image generation result handling tests."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sglang_omni.client.types import (
    CompletionImage,
    CompletionResult,
    CompletionStreamChunk,
    GenerateChunk,
    GenerateRequest,
    Message,
)
from sglang_omni.proto import StreamMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage


class _NoopCoordinator:
    async def stream(self, request_id: str, omni_request: Any):
        del request_id, omni_request
        if False:
            yield None

    async def submit(self, request_id: str, omni_request: Any):
        del request_id, omni_request
        return None


def _image_payload() -> dict[str, Any]:
    return {
        "b64_json": "iVBORw0KGgo=",
        "format": "png",
        "width": 64,
        "height": 32,
    }


def _chat_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "model": "ming-image",
        "messages": [{"role": "user", "content": "paint a moon"}],
    }
    body.update(overrides)
    return body


def _sse_payloads(text: str) -> list[dict[str, Any]]:
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        payloads.append(json.loads(payload))
    return payloads


class _FakeOpenAIClient:
    def __init__(
        self,
        result: CompletionResult | None = None,
        stream_chunks: list[CompletionStreamChunk] | None = None,
    ) -> None:
        self.result = result
        self.stream_chunks = stream_chunks or []
        self.requests: list[GenerateRequest] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        del request_id, audio_format
        self.requests.append(request)
        assert self.result is not None
        return self.result

    async def completion_stream(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ):
        del request_id, audio_format
        self.requests.append(request)
        for chunk in self.stream_chunks:
            yield chunk


def test_default_result_builder_merges_decode_and_image_result() -> None:
    from sglang_omni.client.client import Client

    chunk = Client._default_result_builder(
        "req-img",
        {
            "decode": {
                "text": "paint a moon",
                "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            },
            "image_gen": {"modality": "image", "images": [_image_payload()]},
        },
    )

    assert chunk.text == "paint a moon"
    assert chunk.modality == "image"
    assert chunk.images == [_image_payload()]
    assert chunk.usage is not None
    assert chunk.usage.total_tokens == 7


def test_default_result_builder_prefers_image_finish_reason_for_image_outputs() -> None:
    from sglang_omni.client.client import Client

    chunk = Client._default_result_builder(
        "req-img-error",
        {
            "decode": {
                "text": "paint a moon",
                "finish_reason": "stop",
            },
            "image_gen": {
                "modality": "image",
                "images": [],
                "finish_reason": "error",
                "error": "diffusion failed",
            },
        },
    )

    assert chunk.text == "paint a moon"
    assert chunk.modality == "image"
    assert chunk.images == []
    assert chunk.finish_reason == "error"


def test_default_result_builder_preserves_audio_and_image_result() -> None:
    from sglang_omni.client.client import Client

    waveform = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    chunk = Client._default_result_builder(
        "req-mm",
        {
            "decode": {
                "text": "describe and render",
                "usage": {"prompt_tokens": 2, "completion_tokens": 5},
            },
            "talker": {
                "audio_waveform": waveform.tobytes(),
                "audio_waveform_dtype": str(waveform.dtype),
                "audio_waveform_shape": list(waveform.shape),
                "sample_rate": 24000,
            },
            "image_gen": {"modality": "image", "images": [_image_payload()]},
        },
    )

    assert chunk.text == "describe and render"
    assert chunk.modality == "multimodal"
    assert chunk.sample_rate == 24000
    np.testing.assert_array_equal(chunk.audio_data, waveform)
    assert chunk.images == [_image_payload()]
    assert chunk.usage is not None
    assert chunk.usage.total_tokens == 7


def test_default_result_builder_sets_images_for_single_stage_dict() -> None:
    from sglang_omni.client.client import Client

    chunk = Client._default_result_builder(
        "req-single",
        {"modality": "image", "images": [_image_payload()]},
    )

    assert chunk.modality == "image"
    assert chunk.images == [_image_payload()]


def test_default_stream_builder_sets_images_for_dict_chunk() -> None:
    from sglang_omni.client.client import Client

    msg = StreamMessage(
        request_id="req-stream",
        from_stage="image_gen",
        stage_name="image_gen",
        modality="image",
        chunk={"images": [_image_payload()]},
    )

    chunk = Client._default_stream_builder("req-stream", msg)

    assert chunk.modality == "image"
    assert chunk.stage_name == "image_gen"
    assert chunk.images == [_image_payload()]


def test_default_stream_builder_sets_multimodal_for_audio_and_image_dict_chunk() -> (
    None
):
    from sglang_omni.client.client import Client

    waveform = np.array([0.5, -0.25], dtype=np.float32)
    msg = StreamMessage(
        request_id="req-stream-mm",
        from_stage="image_gen",
        stage_name="image_gen",
        chunk={
            "audio_waveform": waveform.tobytes(),
            "audio_waveform_dtype": str(waveform.dtype),
            "audio_waveform_shape": list(waveform.shape),
            "images": [_image_payload()],
        },
    )

    chunk = Client._default_stream_builder("req-stream-mm", msg)

    np.testing.assert_array_equal(chunk.audio_data, waveform)
    assert chunk.images == [_image_payload()]
    assert chunk.modality == "multimodal"


def test_default_stream_builder_carries_text_with_images_in_one_chunk() -> None:
    # Streaming text+image: a single delta carries both text and images so a
    # client can render a multimodal stream chunk without a second message.
    from sglang_omni.client.client import Client

    msg = StreamMessage(
        request_id="req-stream-ti",
        from_stage="image_gen",
        stage_name="image_gen",
        chunk={"text": "rendering", "images": [_image_payload()]},
    )

    chunk = Client._default_stream_builder("req-stream-ti", msg)

    assert chunk.text == "rendering"
    assert chunk.images == [_image_payload()]
    assert chunk.stage_name == "image_gen"


def test_generate_chunk_to_dict_includes_images() -> None:
    chunk = GenerateChunk(
        request_id="req-dict",
        images=[_image_payload()],
    )

    assert chunk.to_dict()["images"] == [_image_payload()]


def test_build_omni_request_preserves_image_generation_metadata_in_inputs() -> None:
    from sglang_omni.client.client import Client

    image_generation = {"size": "1024x1024", "response_format": "b64_json"}
    request = GenerateRequest(
        messages=[Message(role="user", content="make an icon")],
        output_modalities=["image"],
        metadata={"image_generation": image_generation},
    )

    omni_request = Client._build_omni_request(request)

    assert omni_request.metadata["image_generation"] == image_generation
    assert omni_request.metadata["output_modalities"] == ["image"]
    assert omni_request.inputs["messages"] == [
        {"role": "user", "content": "make an icon"}
    ]
    assert omni_request.inputs["image_generation"] == image_generation


def test_completion_collects_images_and_aggregates_audio_from_generate(
    monkeypatch,
) -> None:
    from sglang_omni.client import client as client_module
    from sglang_omni.client.client import Client

    monkeypatch.setattr(
        client_module,
        "audio_to_base64",
        lambda audio_data, sample_rate=None, output_format="wav": f"{output_format}:{len(audio_data)}",
    )

    client = Client(coordinator=_NoopCoordinator())

    async def fake_generate(request, request_id=None):
        del request
        yield GenerateChunk(
            request_id=request_id or "req-complete",
            text="cap",
            audio_data=np.array([0.0, 0.1], dtype=np.float32),
        )
        yield GenerateChunk(
            request_id=request_id or "req-complete",
            text="tion",
            audio_data=np.array([0.2, 0.3], dtype=np.float32),
            images=[
                {
                    "b64_json": "aaa",
                    "format": "jpeg",
                    "width": 8,
                    "height": 9,
                },
                {"data": "bbb"},
            ],
            finish_reason="stop",
        )

    monkeypatch.setattr(client, "generate", fake_generate)

    result = asyncio.run(
        client.completion(
            GenerateRequest(prompt="ignored", stream=False),
            request_id="req-complete",
        )
    )

    assert result.text == "caption"
    assert result.audio is not None
    assert result.audio.id == "audio-req-complete"
    assert result.audio.data == "wav:4"
    assert result.audio.transcript == "caption"
    assert [image.id for image in result.images] == [
        "image-req-complete-0",
        "image-req-complete-1",
    ]
    assert result.images[0].data == "aaa"
    assert result.images[0].format == "jpeg"
    assert result.images[0].width == 8
    assert result.images[0].height == 9
    assert result.images[1].data == "bbb"
    assert result.images[1].format == "png"


def test_image_completion_result_collects_images_from_generate() -> None:
    from sglang_omni.client.client import Client

    class FakeCoordinator:
        def __init__(self) -> None:
            self.requests: list[tuple[str, Any]] = []

        async def submit(self, request_id: str, request: Any):
            self.requests.append((request_id, request))
            return {
                "decode": {"text": "prompt text"},
                "image_gen": {
                    "modality": "image",
                    "images": [
                        {
                            "b64_json": "YWJj",
                            "format": "png",
                            "width": 2,
                            "height": 3,
                        }
                    ],
                    "finish_reason": "stop",
                },
            }

        async def stream(self, request_id: str, request: Any):
            del request_id, request
            if False:
                yield None

    coordinator = FakeCoordinator()
    client = Client(coordinator)

    result = asyncio.run(
        client.completion(
            GenerateRequest(
                messages=[Message(role="user", content="draw")],
                output_modalities=["image"],
                stream=False,
            ),
            request_id="req-img",
        )
    )

    assert coordinator.requests[0][0] == "req-img"
    assert result.text == "prompt text"
    assert len(result.images) == 1
    assert result.images[0].data == "YWJj"
    assert result.images[0].format == "png"
    assert result.images[0].width == 2
    assert result.images[0].height == 3


def test_completion_stream_passes_images_and_encodes_multimodal_audio(
    monkeypatch,
) -> None:
    from sglang_omni.client import client as client_module
    from sglang_omni.client.client import Client

    monkeypatch.setattr(
        client_module,
        "audio_to_base64",
        lambda audio_data, sample_rate=None, output_format="wav": f"{output_format}:{len(audio_data)}",
    )

    client = Client(coordinator=_NoopCoordinator())

    async def fake_generate(request, request_id=None):
        del request
        yield GenerateChunk(
            request_id=request_id or "req-stream",
            text="delta",
            modality="multimodal",
            audio_data=np.array([0.0, 0.25], dtype=np.float32),
            images=[_image_payload()],
            stage_name="image_gen",
        )

    monkeypatch.setattr(client, "generate", fake_generate)

    async def collect():
        chunks = []
        async for chunk in client.completion_stream(
            GenerateRequest(prompt="ignored"),
            request_id="req-stream",
            audio_format="wav",
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    assert len(chunks) == 1
    assert chunks[0].text == "delta"
    assert chunks[0].modality == "multimodal"
    assert chunks[0].audio_b64 == "wav:2"
    assert chunks[0].images == [_image_payload()]


def test_build_chat_generate_request_propagates_image_generation() -> None:
    image_generation = {"size": "1024x1024", "response_format": "b64_json"}
    request = ChatCompletionRequest(
        model="ming-image",
        messages=[ChatMessage(role="user", content="make an icon")],
        modalities=["image"],
        image_generation=image_generation,
    )

    generate_request = _build_chat_generate_request(request)

    assert generate_request.metadata["image_generation"] == image_generation
    assert generate_request.output_modalities == ["image"]


@pytest.mark.parametrize(
    "body",
    [
        _chat_body(image_generation={"size": "64x64"}),
        _chat_body(modalities=["text"], image_generation={"size": "64x64"}),
    ],
)
def test_chat_completion_rejects_image_generation_without_image_modality(body) -> None:
    fake_client = _FakeOpenAIClient()
    client = TestClient(create_app(fake_client, model_name="ming-image"))

    response = client.post(
        "/v1/chat/completions",
        json=body,
    )

    assert response.status_code == 400
    assert (
        'image_generation requires modalities to include "image"'
        in response.json()["detail"]
    )
    assert fake_client.requests == []


def test_chat_completion_non_streaming_returns_images() -> None:
    fake_client = _FakeOpenAIClient(
        result=CompletionResult(
            request_id="req-image",
            text="",
            images=[
                CompletionImage(
                    id="image-req-image-0",
                    data="abc123",
                    format="jpeg",
                    width=128,
                    height=96,
                )
            ],
        )
    )
    client = TestClient(create_app(fake_client, model_name="ming-image"))

    response = client.post(
        "/v1/chat/completions",
        json=_chat_body(
            modalities=["image"],
            image_generation={"size": "128x96", "format": "jpeg"},
        ),
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["images"] == [
        {
            "id": "image-req-image-0",
            "data": "abc123",
            "format": "jpeg",
            "width": 128,
            "height": 96,
        }
    ]
    assert "content" not in message
    assert fake_client.requests[0].metadata["image_generation"] == {
        "size": "128x96",
        "format": "jpeg",
    }


def test_chat_completion_streaming_returns_images() -> None:
    fake_client = _FakeOpenAIClient(
        stream_chunks=[
            CompletionStreamChunk(
                request_id="req-stream-image",
                modality="image",
                images=[
                    {
                        "b64_json": "stream-image",
                        "format": "png",
                        "width": 64,
                        "height": 32,
                    }
                ],
                finish_reason="stop",
            )
        ]
    )
    client = TestClient(create_app(fake_client, model_name="ming-image"))

    response = client.post(
        "/v1/chat/completions",
        json=_chat_body(
            request_id="req-stream-image",
            stream=True,
            modalities=["image"],
            image_generation={"size": "64x32"},
        ),
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    image_chunk = next(
        payload for payload in payloads if payload["choices"][0]["delta"].get("images")
    )
    assert image_chunk["choices"][0]["delta"]["images"] == [
        {
            "id": "image-req-stream-image-0",
            "data": "stream-image",
            "format": "png",
            "width": 64,
            "height": 32,
        }
    ]
    assert image_chunk["choices"][0]["finish_reason"] is None
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completion_streaming_multimodal_chunk_emits_audio_and_images() -> None:
    fake_client = _FakeOpenAIClient(
        stream_chunks=[
            CompletionStreamChunk(
                request_id="req-stream-mm",
                modality="multimodal",
                audio_b64="audio-data",
                images=[{"data": "image-data", "format": "webp"}],
            )
        ]
    )
    client = TestClient(create_app(fake_client, model_name="ming-image"))

    response = client.post(
        "/v1/chat/completions",
        json=_chat_body(
            request_id="req-stream-mm",
            stream=True,
            modalities=["audio", "image"],
        ),
    )

    assert response.status_code == 200
    payloads = _sse_payloads(response.text)
    multimodal_chunk = next(
        payload
        for payload in payloads
        if payload["choices"][0]["delta"].get("audio")
        and payload["choices"][0]["delta"].get("images")
    )
    delta = multimodal_chunk["choices"][0]["delta"]
    assert delta["audio"] == {"id": "audio-req-stream-mm", "data": "audio-data"}
    assert delta["images"] == [
        {
            "id": "image-req-stream-mm-0",
            "data": "image-data",
            "format": "webp",
        }
    ]
