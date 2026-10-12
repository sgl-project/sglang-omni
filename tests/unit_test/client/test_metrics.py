# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import StreamMessage
from sglang_omni.serve.metrics import RequestMetrics


@pytest.mark.parametrize(
    "cache_fields",
    [
        {"cached_tokens": 4},
        {"prompt_tokens_details": {"cached_tokens": 4}},
        {"usage": {"prompt_tokens_details": {"cached_tokens": 4}}},
    ],
)
def test_cache_metrics_do_not_require_chat_usage_extensions(cache_fields):
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

    async def source():
        yield chunk

    async def run():
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
def test_client_coordinator_token_telemetry_does_not_leak(stream):
    metrics = RequestMetrics("test")

    class FakeCoordinator:
        request_metrics = metrics

        async def report(self, request_id, request):
            assert request.metadata["omni_metrics_enabled"] is True
            for count in [1, 3]:
                message = StreamMessage(
                    request_id=request_id,
                    from_stage="decode",
                    modality="metrics",
                    chunk={"prompt_tokens": 10, "completion_tokens": count},
                )
                await Coordinator.handle_stream(
                    self, deserialize_message(serialize_message(message))
                )

        async def submit(self, request_id, request):
            await self.report(request_id, request)
            return {"text": "hello", "finish_reason": "stop"}

        async def stream(self, request_id, request):
            await self.report(request_id, request)
            yield SimpleNamespace(result={"text": "hello", "finish_reason": "stop"})

    client = Client(FakeCoordinator())
    client.request_metrics = metrics

    async def run():
        return [
            chunk
            async for chunk in client.generate(
                GenerateRequest(prompt="test", stream=stream), "r"
            )
        ]

    chunks = asyncio.run(run())
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
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


def test_disabled_client_does_not_request_telemetry():
    class FakeCoordinator:
        async def submit(self, request_id, request):
            assert "omni_metrics_enabled" not in request.metadata
            return "hello"

    async def run():
        return [
            chunk
            async for chunk in Client(FakeCoordinator()).generate(
                GenerateRequest(prompt="test", stream=False)
            )
        ]

    assert asyncio.run(run())[0].text == "hello"
