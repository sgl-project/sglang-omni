# SPDX-License-Identifier: Apache-2.0
"""Non-streaming terminal image results and merged audio compatibility."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypedDict

import pytest

from sglang_omni.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
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
