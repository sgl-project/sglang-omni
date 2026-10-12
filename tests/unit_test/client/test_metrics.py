# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import TypedDict

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.replicas import ReplicaTopology
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.proto.request import OmniRequest
from sglang_omni.serve.metrics import RequestMetrics
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class DecodeMetricsResult(TypedDict):
    text: str
    finish_reason: str
    stage_name: str
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: dict[str, int]


class CacheFields(TypedDict, total=False):
    cached_tokens: int
    prompt_tokens_details: dict[str, int]
    usage: dict[str, dict[str, int]]


@pytest.mark.parametrize("audio_stage", ["code2wav", "talker", "talker_stream"])
@pytest.mark.parametrize("native_tokens", [False, True])
@pytest.mark.parametrize("decode_usage", [False, True])
def test_merged_audio_preserves_only_decode_token_metrics(
    audio_stage: str, native_tokens: bool, decode_usage: bool
) -> None:
    decode_result = (
        {
            "text": "hello",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        if decode_usage
        else {"text": "hello"}
    )
    chunk = Client.default_result_builder(
        "merged",
        {
            "decode": decode_result,
            audio_stage: {
                "audio_data": [0.0] * 160,
                "sample_rate": 16000,
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        },
    )
    assert chunk.modality == "audio"
    metrics = RequestMetrics("test")

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        if native_tokens:
            metrics.observe_tokens("merged", "decode", 10, 1)
            metrics.observe_tokens("merged", "decode", 10, 2)
        else:
            pass
        yield chunk

    async def run() -> None:
        async for observed in metrics.generate(source(), False, "merged"):
            assert observed is chunk

    asyncio.run(run())
    labels = {"model_name": "test", "is_streaming": "false"}
    expected_prompt = 10 if decode_usage or native_tokens else None
    expected_generation = 2 if decode_usage or native_tokens else None
    assert (
        metrics.registry.get_sample_value("sglang:prompt_tokens_total", labels)
        == expected_prompt
    )
    assert (
        metrics.registry.get_sample_value("sglang:generation_tokens_total", labels)
        == expected_generation
    )
    assert metrics.registry.get_sample_value(
        "sglang:cached_tokens_total", {"model_name": "test", "cache_source": "total"}
    ) == (4 if decode_usage else None)
    assert metrics.active_requests == {}
    assert "text_usage_stage" not in chunk.to_dict()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("terminal", ["generate", "generate@r0"])
def test_nonstream_image_metrics_keep_logical_stage(
    enabled: bool, terminal: str
) -> None:
    async def run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["generate"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")
        coordinator.replica_topology = ReplicaTopology(
            replicas={"generate": ("generate@r0",)}
        )
        metrics = RequestMetrics("test")
        if enabled:
            coordinator.request_metrics = metrics
        else:
            pass
        await coordinator.submit_request("image-request", {"text": "hello"})
        original = {"image_b64": "encoded-image"}
        await coordinator.handle_completion(
            CompleteMessage("image-request", terminal, True, result=original)
        )
        result = coordinator.completion_futures["image-request"].result()
        assert "stage_name" not in original
        if enabled:
            assert result["stage_name"] == "generate"
            chunk = Client.default_result_builder("image-request", result)

            async def source() -> AsyncGenerator[GenerateChunk, None]:
                yield chunk

            async for observed in metrics.generate(source(), False):
                assert observed.stage_name == "generate"
            assert (
                metrics.registry.get_sample_value(
                    "sglang:e2e_request_latency_seconds_count",
                    {"model_name": "test", "is_streaming": "false"},
                )
                == 1
            )
        else:
            assert result is original

    asyncio.run(run())


@pytest.mark.parametrize(
    "cache_fields",
    [
        {"cached_tokens": 4},
        {"prompt_tokens_details": {"cached_tokens": 4}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": 4}}},
    ],
)
def test_cache_metrics_do_not_require_chat_usage_extensions(
    cache_fields: CacheFields,
) -> None:
    result = {
        "text": "hello",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        **cache_fields,
    }
    chunk = Client.default_result_builder("cache-request", result)
    assert chunk.usage.prompt_tokens_details == {"cached_tokens": 4}
    assert "prompt_tokens_details" not in chunk.usage.to_dict()

    metrics = RequestMetrics("test")

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        yield chunk

    async def run() -> None:
        async for _ in metrics.generate(source(), False):
            pass

    asyncio.run(run())
    assert (
        metrics.registry.get_sample_value(
            "sglang:cached_tokens_total",
            {"model_name": "test", "cache_source": "total"},
        )
        == 4
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("token_stage", ["decode", "decode@r0"])
def test_client_coordinator_token_telemetry_does_not_leak(
    stream: bool, token_stage: str
) -> None:
    metrics = RequestMetrics("test")

    class FakeCoordinator:
        request_metrics: RequestMetrics = metrics
        replica_topology: ReplicaTopology = ReplicaTopology(
            replicas={"decode": ("decode@r0",)}
        )

        async def report(self, request_id: str, request: OmniRequest) -> None:
            assert request.metadata["omni_metrics_enabled"] is True
            for count in [1, 3]:
                message = StreamMessage(
                    request_id=request_id,
                    from_stage=token_stage,
                    modality="metrics",
                    chunk={"prompt_tokens": 10, "completion_tokens": count},
                )
                await Coordinator.handle_stream(
                    self, deserialize_message(serialize_message(message))
                )

        async def submit(
            self, request_id: str, request: OmniRequest
        ) -> DecodeMetricsResult:
            await self.report(request_id, request)
            return {
                "text": "hello",
                "finish_reason": "stop",
                "stage_name": "decode",
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 4},
            }

        async def stream(
            self, request_id: str, request: OmniRequest
        ) -> AsyncGenerator[SimpleNamespace, None]:
            yield SimpleNamespace(result=await self.submit(request_id, request))

    client = Client(FakeCoordinator())
    client.request_metrics = metrics

    async def run() -> list[GenerateChunk]:
        return [
            chunk
            async for chunk in client.generate(
                GenerateRequest(prompt="test", stream=stream), "r"
            )
        ]

    chunks = asyncio.run(run())
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert (
        metrics.registry.get_sample_value(
            "sglang:cached_tokens_total",
            {"model_name": "test", "cache_source": "total"},
        )
        == 4
    )
    assert metrics.active_requests == {}
    labels = {"model_name": "test", "is_streaming": str(stream).lower()}
    assert (
        metrics.registry.get_sample_value(
            "sglang:time_to_first_token_seconds_count", labels
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value("sglang:generation_tokens_total", labels) == 3
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"model_name": "test"}
        )
        == 2
    )


def test_disabled_client_does_not_request_telemetry() -> None:
    class FakeCoordinator:
        async def submit(self, request_id: str, request: OmniRequest) -> str:
            assert "omni_metrics_enabled" not in request.metadata
            return "hello"

    async def run() -> list[GenerateChunk]:
        return [
            chunk
            async for chunk in Client(FakeCoordinator()).generate(
                GenerateRequest(prompt="test", stream=False)
            )
        ]

    assert asyncio.run(run())[0].text == "hello"
