# SPDX-License-Identifier: Apache-2.0
"""Client.completion() surfaces RL-rollout artifacts (logprobs, weight_version)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from sglang_omni.client import Client
from sglang_omni.client.client import _extract_inputs
from sglang_omni.client.types import GenerateRequest


class _SubmitStubCoordinator:
    """Non-streaming coordinator stub: completion() only needs submit()."""

    def __init__(self, result: Any) -> None:
        self._result = result

    async def submit(self, request_id: str, omni_request: Any) -> Any:
        del request_id, omni_request
        return self._result


class _StreamStubCoordinator:
    """Streaming coordinator stub: yields the given StreamMessages in order."""

    def __init__(self, messages: list[Any]) -> None:
        self._messages = messages

    async def stream(self, request_id: str, omni_request: Any):
        del request_id, omni_request
        for message in self._messages:
            yield message


def test_completion_surfaces_logprobs_and_weight_version() -> None:
    result = {
        "text": "hello",
        "finish_reason": "stop",
        "output_token_logprobs": [[-0.1, 11], [-0.2, 22], [-0.3, 33]],
        "weight_version": "v7",
        "completion_tokens": 3,
    }
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.output_token_logprobs == [[-0.1, 11], [-0.2, 22], [-0.3, 33]]
    assert out.weight_version == "v7"


def test_completion_surfaces_omni_rollout() -> None:
    rollout = {
        "version": 1,
        "model_family": "qwen3_omni",
        "stages": ["talker"],
        "total_action_count": 1,
        "action_streams": [],
    }
    result = {
        "text": "hello",
        "finish_reason": "stop",
        "omni_rollout": rollout,
    }
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.omni_rollout == rollout


def test_completion_without_logprobs_leaves_fields_none() -> None:
    result = {"text": "hello", "finish_reason": "stop"}
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.output_token_logprobs is None
    assert out.weight_version is None
    assert out.omni_rollout is None


def test_completion_preserves_empty_logprob_list() -> None:
    result = {
        "text": "",
        "finish_reason": "stop",
        "output_token_logprobs": [],
    }
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.output_token_logprobs == []


def test_completion_surfaces_rollout_from_multiterminal_decode() -> None:
    result = {
        "decode": {
            "text": "hi",
            "finish_reason": "stop",
            "output_token_logprobs": [[-0.5, 9]],
            "weight_version": "v9",
            "omni_rollout": {"version": 1, "action_streams": []},
        },
        "code2wav": {"audio_data": [0.0, 0.1, -0.1], "sample_rate": 24000},
    }
    client = Client(_SubmitStubCoordinator(result))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=False), request_id="r1")
    )

    assert out.text == "hi"
    assert out.audio is not None
    assert out.output_token_logprobs == [[-0.5, 9]]
    assert out.weight_version == "v9"
    assert out.omni_rollout == {"version": 1, "action_streams": []}


def test_completion_concatenates_streamed_logprobs() -> None:
    from sglang_omni.proto import StreamMessage

    messages = [
        StreamMessage(
            request_id="r1",
            from_stage="decode",
            chunk={
                "text": "he",
                "output_token_logprobs": [[-0.1, 11], [-0.2, 22]],
                "weight_version": "v7",
            },
            stage_name="decode",
            modality="text",
        ),
        StreamMessage(
            request_id="r1",
            from_stage="decode",
            chunk={
                "text": "llo",
                "output_token_logprobs": [[-0.3, 33]],
                "finish_reason": "stop",
                "weight_version": "v7",
                "omni_rollout": {"version": 1, "action_streams": []},
            },
            stage_name="decode",
            modality="text",
        ),
    ]
    client = Client(_StreamStubCoordinator(messages))

    out = asyncio.run(
        client.completion(GenerateRequest(prompt="hi", stream=True), request_id="r1")
    )

    assert out.text == "hello"
    assert out.output_token_logprobs == [[-0.1, 11], [-0.2, 22], [-0.3, 33]]
    assert out.weight_version == "v7"
    assert out.omni_rollout == {"version": 1, "action_streams": []}


def test_extract_inputs_rejects_prompt_with_multimodal_train_inputs() -> None:
    request = GenerateRequest(
        prompt="hi",
        multimodal_train_inputs={"version": 1, "tensors": {}},
    )

    with pytest.raises(ValueError, match="requires prompt_token_ids"):
        _extract_inputs(request)


def test_extract_inputs_passes_pretokenized_multimodal_train_inputs() -> None:
    bundle = {"version": 1, "tensors": {}}
    request = GenerateRequest(
        prompt_token_ids=[1, 2, 3],
        multimodal_train_inputs=bundle,
    )

    assert _extract_inputs(request) == {
        "input_ids": [1, 2, 3],
        "multimodal_train_inputs": bundle,
    }
