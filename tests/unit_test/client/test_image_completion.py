# SPDX-License-Identifier: Apache-2.0
"""Non-streaming terminal image results and merged audio compatibility."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from typing import Literal, TypedDict

import pytest

from sglang_omni.client import Client
from sglang_omni.client.client import umm_segment_chunk
from sglang_omni.client.image_media import InlineImage
from sglang_omni.client.types import (
    GenerateChunk,
    GenerateRequest,
    ImageUsageDict,
    UMMGenerateChunk,
    UMMSegment,
)
from sglang_omni.proto import CompleteMessage, StreamMessage


class DecodePayload(TypedDict, total=False):
    text: str
    token_ids: list[int]
    logprobs: list[float]
    output_token_logprobs: list[list[float | int]]
    omni_rollout: dict[str, int | list[str]]
    weight_version: str
    finish_reason: str
    language: str
    stage_id: int
    stage_name: str
    usage: dict[str, int | float]
    engine_time_s: float
    image: str


class ImageDecodePayload(TypedDict, total=False):
    image: str | None


class MergedResult(TypedDict, total=False):
    decode: DecodePayload
    image_decode: ImageDecodePayload | None


CoordinatorResult = GenerateChunk | DecodePayload | MergedResult | None


class RecordingCoordinator:
    def __init__(self, result: CoordinatorResult) -> None:
        self.result = result
        self.requests: list[GenerateRequest] = []

    async def submit(
        self, request_id: str, request: GenerateRequest
    ) -> CoordinatorResult:
        self.requests.append(request)
        return self.result

    async def stream(
        self, request_id: str, request: GenerateRequest
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        raise AssertionError("Image requests must not reach coordinator.stream")
        yield


@pytest.mark.parametrize("merged", [False, True])
def test_image_result_preserves_decode_fields(merged: bool) -> None:
    decode = {
        "text": "A red square",
        "token_ids": [11, 12],
        "logprobs": [-0.1, -0.2],
        "output_token_logprobs": [[-0.1, 11], [-0.2, 12]],
        "omni_rollout": {"version": 1, "action_streams": []},
        "weight_version": "v3",
        "finish_reason": "length",
        "language": "English",
        "stage_id": 2,
        "stage_name": "decode",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "engine_time_s": 0.25,
    }
    result = (
        {"decode": decode, "image_decode": {"image": "cG5n"}}
        if merged
        else {**decode, "image": "cG5n"}
    )
    chunk = Client.default_result_builder("r1", result)
    assert chunk.image == "cG5n"
    assert chunk.modality == "image"
    assert chunk.to_dict()["image"] == "cG5n"
    assert chunk.token_ids == decode["token_ids"]
    assert chunk.logprobs == decode["logprobs"]
    assert chunk.stage_id == 2
    assert chunk.stage_name == "decode"

    client = Client(RecordingCoordinator(result))
    completion = asyncio.run(
        client.completion(
            GenerateRequest(prompt="draw a square", stream=False), request_id="r1"
        )
    )
    assert completion.image == "cG5n"
    assert completion.audio is None
    assert completion.text == decode["text"]
    assert completion.finish_reason == "length"
    assert completion.output_token_logprobs == decode["output_token_logprobs"]
    assert completion.omni_rollout == decode["omni_rollout"]
    assert completion.weight_version == "v3"
    assert completion.language == "English"
    assert completion.usage.total_tokens == 5
    assert completion.usage.engine_time_s == 0.25


@pytest.mark.parametrize("image_result", [None, {}, {"image": None}])
def test_empty_image_terminal_preserves_text(
    image_result: ImageDecodePayload | None,
) -> None:
    chunk = Client.default_result_builder(
        "r1", {"decode": {"text": "No image"}, "image_decode": image_result}
    )
    assert chunk.text == "No image"
    assert chunk.image is None
    assert chunk.modality == "text"


def test_custom_image_chunk_completes() -> None:
    chunk = GenerateChunk(request_id="original", image="cG5n", modality="image")
    client = Client(RecordingCoordinator(chunk))
    result = asyncio.run(
        client.completion(GenerateRequest(prompt="draw", stream=False), request_id="r1")
    )
    assert result.image == "cG5n"
    assert result.request_id == chunk.request_id == "r1"


@pytest.mark.parametrize("audio_stage", ["code2wav", "talker", "talker_stream"])
@pytest.mark.parametrize("with_decode_usage", [False, True])
def test_merged_audio_keeps_usage_fallback(
    audio_stage: str, with_decode_usage: bool
) -> None:
    decode = {"text": "hello", "language": "English"}
    if with_decode_usage:
        decode["usage"] = {"prompt_tokens": 3, "completion_tokens": 2}
    chunk = Client.default_result_builder(
        "r1",
        {
            "decode": decode,
            audio_stage: {
                "audio_data": [0.1, -0.1],
                "sample_rate": 24000,
                "usage": {"completion_tokens": 10},
            },
        },
    )
    assert chunk.text == "hello"
    assert chunk.language == "English"
    assert chunk.audio_data == [0.1, -0.1]
    assert chunk.sample_rate == 24000
    assert chunk.modality == "audio"
    assert chunk.image is None
    assert chunk.usage.total_tokens == (5 if with_decode_usage else 10)


IMAGE_USAGE: ImageUsageDict = {
    "input_tokens": 3,
    "total_tokens": 3,
    "input_tokens_details": {"text_tokens": 3, "image_tokens": 4},
}


def result_with_image_usage(usage_data: object, streamed: bool) -> dict[str, object]:
    segments = [
        UMMSegment("execution", 0, "text", "hello").to_dict(),
        UMMSegment("execution", 1, "usage", usage_data).to_dict(),
    ]
    if streamed:
        for segment in segments:
            segment.pop("data", None)
    else:
        pass
    return {
        "type": "umm_result",
        "execution_id": "execution",
        "segments": segments,
    }


@pytest.mark.parametrize("streamed", [False, True])
def test_final_result_reads_image_usage_from_its_usage_segment(streamed: bool) -> None:
    usage_data = {"image_usage": IMAGE_USAGE, "stage_name": "generate"}
    chunk = Client.default_result_builder(
        "request", result_with_image_usage(usage_data, streamed)
    )
    assert isinstance(chunk, UMMGenerateChunk)
    if streamed:
        assert chunk.image_usage is None and chunk.stage_name is None
    else:
        assert chunk.image_usage == IMAGE_USAGE and chunk.stage_name == "generate"
    streamed_chunk = umm_segment_chunk(
        GenerateChunk("request", stage_name="output"),
        UMMSegment("execution", 1, "usage", usage_data).to_dict(),
    )
    assert streamed_chunk.stage_name == "generate"
    assert isinstance(streamed_chunk, UMMGenerateChunk)
    assert streamed_chunk.segment is not None
    assert streamed_chunk.segment.data == usage_data


@pytest.mark.parametrize(
    "usage_data",
    [
        "not a mapping",
        {"image_usage": {"input_tokens": -1, "total_tokens": 0}},
        {"image_usage": {"input_tokens": True}},
        {"image_usage": {"input_tokens": "3"}},
        {
            "image_usage": {
                "input_tokens_details": {"text_tokens": 3, "image_tokens": -1}
            }
        },
    ],
)
def test_invalid_usage_segment_is_rejected(usage_data: object) -> None:
    with pytest.raises(ValueError):
        Client.default_result_builder(
            "request", result_with_image_usage(usage_data, False)
        )


def test_usage_without_token_statistics_remains_unavailable() -> None:
    chunk = Client.default_result_builder(
        "request", result_with_image_usage({"stage_name": "generate"}, False)
    )
    assert chunk.image_usage is None
    assert chunk.stage_name == "generate"


def test_result_builder_keeps_only_string_image_lists() -> None:
    chunk = Client.default_result_builder(
        "request", {"image_b64": "a", "images_b64": ["a", "b"]}
    )
    assert chunk.image_b64 == "a" and chunk.images_b64 == ["a", "b"]
    serialized = chunk.to_dict()
    assert serialized["image_b64"] == "a"
    assert serialized["images_b64"] == ["a", "b"]
    invalid_chunk = Client.default_result_builder(
        "request", {"image_b64": 1, "images_b64": ["a", 1]}
    )
    assert invalid_chunk.image_b64 is None
    assert invalid_chunk.images_b64 is None


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "url"])
def test_inline_image_rejects_tampered_content(
    field: Literal["sha256", "size_bytes", "url"],
) -> None:
    encoded_image = base64.b64encode(b"png bytes").decode("ascii")
    media = InlineImage.from_png(encoded_image)
    assert media.encoded_image() == encoded_image
    changes = {
        "sha256": "0" * 64,
        "size_bytes": media.size_bytes + 1,
        "url": "https://example.invalid/image.png",
    }
    with pytest.raises(ValueError):
        media.model_copy(update={field: changes[field]}).encoded_image()
