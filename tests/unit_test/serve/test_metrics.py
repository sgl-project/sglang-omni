# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import AsyncGenerator

import pytest
from starlette.types import Message, Receive, Scope, Send

from sglang_omni.client.types import GenerateChunk, UsageInfo
from sglang_omni.scheduling.metrics_snapshot import MetricsAdminResponse
from sglang_omni.serve.metrics import MetricsMiddleware, RequestMetrics


def test_http_stream_records_only_after_final_body() -> None:
    metrics = RequestMetrics("test")
    labels = {
        "method": "POST",
        "route": "unmatched",
        "status_code": "200",
        "outcome": "completed",
    }

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        assert (
            metrics.registry.get_sample_value("sglang_omni:http_requests_inflight") == 1
        )
        assert (
            metrics.registry.get_sample_value("sglang_omni:http_requests_total", labels)
            is None
        )
        await send({"type": "http.response.body", "body": b"last"})

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    asyncio.run(
        MetricsMiddleware(app, metrics)(
            {"type": "http", "path": "/generate", "method": "POST"}, receive, send
        )
    )
    assert metrics.registry.get_sample_value("sglang_omni:http_requests_inflight") == 0
    assert (
        metrics.registry.get_sample_value("sglang_omni:http_requests_total", labels)
        == 1
    )


def test_scheduler_events_export_queue_batch_and_outcome() -> None:
    metrics = RequestMetrics("test")
    metrics.observe_scheduler_event("generate", {"event": "queue", "duration": 2})
    metrics.observe_scheduler_event("generate", {"event": "batch", "size": 3})
    metrics.observe_scheduler_event(
        "generate", {"event": "execution", "duration": 5, "outcome": "error"}
    )
    labels = {"model_name": "test", "stage": "generate"}
    assert (
        metrics.registry.get_sample_value("sglang:queue_time_seconds_sum", labels) == 2
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang_omni:scheduler_batch_size_sum", labels
        )
        == 3
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang_omni:stage_compute_duration_seconds_sum",
            {**labels, "outcome": "error"},
        )
        == 5
    )


@pytest.mark.parametrize(
    "error,outcome", [(ValueError, "error"), (asyncio.CancelledError, "cancelled")]
)
def test_http_error_and_cancel_return_inflight_to_zero(
    error: type[ValueError] | type[asyncio.CancelledError], outcome: str
) -> None:
    metrics = RequestMetrics("test")

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise error()

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        pass

    with pytest.raises(error):
        asyncio.run(
            MetricsMiddleware(app, metrics)(
                {"type": "http", "path": "/generate", "method": "POST"}, receive, send
            )
        )
    assert metrics.registry.get_sample_value("sglang_omni:http_requests_inflight") == 0
    assert (
        metrics.registry.get_sample_value(
            "sglang_omni:http_requests_total",
            {
                "method": "POST",
                "route": "unmatched",
                "status_code": "500",
                "outcome": outcome,
            },
        )
        == 1
    )


def chunk(
    count: int | None = None,
    *,
    stage: str = "decode",
    modality: str = "text",
    finish: str | None = None,
) -> GenerateChunk:
    return GenerateChunk(
        request_id="r",
        modality=modality,
        stage_name=stage,
        finish_reason=finish,
        usage=(
            UsageInfo(prompt_tokens=10, completion_tokens=count)
            if count is not None
            else None
        ),
    )


@pytest.mark.parametrize("cached", [None, 0, 4])
def test_finished_token_histograms_and_cache(cached: int | None) -> None:
    metrics = RequestMetrics("test")

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        for count in (1, 4):
            item = chunk(count)
            item.usage.prompt_tokens_details = (
                {"cached_tokens": cached} if cached is not None else None
            )
            yield item

    async def run() -> list[GenerateChunk]:
        return [item async for item in metrics.generate(source(), False)]

    asyncio.run(run())
    labels = {"model_name": "test"}
    for name, value in (("prompt_tokens", 10), ("generation_tokens", 4)):
        assert (
            metrics.registry.get_sample_value(f"sglang:{name}_histogram_count", labels)
            == 1
        )
        assert (
            metrics.registry.get_sample_value(f"sglang:{name}_histogram_sum", labels)
            == value
        )
    assert metrics.registry.get_sample_value(
        "sglang:uncached_prompt_tokens_histogram_sum", labels
    ) == (None if cached is None else 10 - cached)
    assert metrics.registry.get_sample_value(
        "sglang:cached_tokens_total", {**labels, "cache_source": "total"}
    ) == (cached if cached else None)


def test_invalid_cache_count_is_not_published() -> None:
    metrics = RequestMetrics("test")
    metrics.observe_finished(
        stream=False,
        latency=1,
        prompt_tokens=10,
        generation_tokens=4,
        cached_tokens=11,
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang:uncached_prompt_tokens_histogram_count", {"model_name": "test"}
        )
        is None
    )


def test_images_do_not_publish_token_histograms() -> None:
    metrics = RequestMetrics("test")
    metrics.observe_finished(
        stream=False, latency=1, prompt_tokens=None, generation_tokens=None
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang:prompt_tokens_histogram_count", {"model_name": "test"}
        )
        is None
    )


def test_cache_from_another_stage_is_not_combined_with_native_tokens() -> None:
    metrics = RequestMetrics("test")

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        metrics.observe_tokens("r", "decode", 10, 4)
        item = chunk(4, stage="other")
        item.usage.prompt_tokens_details = {"cached_tokens": 4}
        yield item

    async def run() -> list[GenerateChunk]:
        return [item async for item in metrics.generate(source(), False, "r")]

    asyncio.run(run())
    assert (
        metrics.registry.get_sample_value(
            "sglang:cached_tokens_total",
            {"model_name": "test", "cache_source": "total"},
        )
        is None
    )


def test_generate_keeps_latest_usage_without_summing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = RequestMetrics("test")
    times = iter([0, 1, 1.06, 1.07, 1.08, 1.09])
    monkeypatch.setattr(
        "sglang_omni.serve.metrics.time.perf_counter", lambda: next(times)
    )

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        yield chunk(1)
        yield chunk(4)
        yield chunk(100, modality="audio", stage="vocoder")
        yield chunk(finish="stop")

    async def run() -> list[GenerateChunk]:
        return [item async for item in metrics.generate(source(), True)]

    assert len(asyncio.run(run())) == 4
    labels = {"model_name": "test", "is_streaming": "true"}
    assert (
        metrics.registry.get_sample_value("sglang:generation_tokens_total", labels) == 4
    )
    assert metrics.registry.get_sample_value("sglang:prompt_tokens_total", labels) == 10
    assert (
        metrics.registry.get_sample_value(
            "sglang:time_to_first_token_seconds_sum", labels
        )
        is None
    )


def test_deferred_request_features_are_not_registered() -> None:
    metrics = RequestMetrics("test")
    families = {family.name for family in metrics.registry.collect()}
    assert "sglang:spec_verify_calls" not in families
    assert "sglang:num_so_requests" not in families


@pytest.mark.parametrize("stream", [True, False])
def test_backend_events_provide_ttft_for_both_request_modes(stream: bool) -> None:
    metrics = RequestMetrics("test")

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        metrics.observe_tokens("r", "decode", 10, 1)
        metrics.observe_tokens("r", "decode", 10, 4)
        metrics.observe_tokens("r", "decode", 10, 4)
        metrics.observe_tokens("r", "talker", 99, 100)
        yield chunk(4, finish="stop")

    async def run() -> list[GenerateChunk]:
        return [item async for item in metrics.generate(source(), stream, "r")]

    asyncio.run(run())
    labels = {"model_name": "test", "is_streaming": "true" if stream else "false"}
    assert (
        metrics.registry.get_sample_value(
            "sglang:time_to_first_token_seconds_count", labels
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"model_name": "test"}
        )
        == 3
    )
    assert (
        metrics.registry.get_sample_value("sglang:generation_tokens_total", labels) == 4
    )
    assert metrics.active_requests == {}
    metrics.observe_tokens("r", "decode", 10, 5)
    assert (
        metrics.registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"model_name": "test"}
        )
        == 3
    )


@pytest.mark.parametrize("error", [ValueError, asyncio.CancelledError])
def test_failure_does_not_publish_success_usage(
    error: type[ValueError] | type[asyncio.CancelledError],
) -> None:
    metrics = RequestMetrics("test")

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        yield chunk(1)
        raise error()

    async def run() -> None:
        async for _ in metrics.generate(source(), True):
            pass

    with pytest.raises(error):
        asyncio.run(run())
    labels = {"model_name": "test", "is_streaming": "true"}
    assert (
        metrics.registry.get_sample_value("sglang:generation_tokens_total", labels)
        is None
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang:generation_tokens_histogram_count", {"model_name": "test"}
        )
        is None
    )
    name = (
        "sglang:num_aborted_requests_total"
        if error is asyncio.CancelledError
        else "sglang_omni:failed_requests_total"
    )
    assert metrics.registry.get_sample_value(name, {"model_name": "test"}) == 1


def test_closing_stream_cleans_up_source() -> None:
    metrics = RequestMetrics("test")
    closed = []

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        try:
            yield chunk(1)
            yield chunk(2)
        finally:
            closed.append(True)

    async def run() -> None:
        stream = metrics.generate(source(), True)
        await anext(stream)
        await stream.aclose()

    asyncio.run(run())
    assert closed == [True]
    assert (
        metrics.registry.get_sample_value(
            "sglang:num_aborted_requests_total", {"model_name": "test"}
        )
        == 1
    )


def test_scheduler_leader_only_and_no_stale_series() -> None:
    metrics = RequestMetrics("test")

    class Client:
        async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
            assert action == "metrics"
            return {
                "results": [
                    {
                        "success": True,
                        "stage": "decode",
                        "data": {
                            "num_running_reqs": 2,
                            "num_queue_reqs": 3,
                            "rank_results": [{"num_running_reqs": 2}],
                        },
                    }
                ]
            }

    asyncio.run(metrics.refresh_scheduler(Client()))
    labels = {"model_name": "test", "stage": "decode"}
    assert metrics.registry.get_sample_value("sglang:num_running_reqs", labels) == 2
    assert metrics.registry.get_sample_value("sglang:num_queue_reqs", labels) == 3


def test_interval_is_weighted_by_new_tokens() -> None:
    metrics = RequestMetrics("test")
    metrics.observe_interval(0.06, 3)
    labels = {"model_name": "test"}
    assert (
        metrics.registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", labels
        )
        == 3
    )
    assert metrics.registry.get_sample_value(
        "sglang:inter_token_latency_seconds_sum", labels
    ) == pytest.approx(0.06)
    assert (
        metrics.registry.get_sample_value(
            "sglang:inter_token_latency_seconds_bucket", {**labels, "le": "0.02"}
        )
        == 3
    )


def test_engine_gauges_are_leader_snapshots_and_removed_when_missing() -> None:
    metrics = RequestMetrics("test")

    class Client:
        async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
            return {
                "results": [
                    {
                        "stage": "decode",
                        "success": True,
                        "data": {
                            "num_running_reqs": 1,
                            "num_queue_reqs": 2,
                            "engine_gauges": {"token_usage": 0.5},
                            "rank_results": [{"engine_gauges": {"token_usage": 0.9}}],
                        },
                    }
                ]
            }

    asyncio.run(metrics.refresh_scheduler(Client()))
    labels = {"model_name": "test", "stage": "decode"}
    assert metrics.registry.get_sample_value("sglang:token_usage", labels) == 0.5

    class EmptyClient:
        async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
            return {"results": []}

    asyncio.run(metrics.refresh_scheduler(EmptyClient()))
    assert metrics.registry.get_sample_value("sglang:token_usage", labels) is None


@pytest.mark.parametrize("aborted", [False, True])
def test_modality_counts_use_waveform_and_final_image_list(aborted: bool) -> None:
    metrics = RequestMetrics("test")

    async def source() -> AsyncGenerator[GenerateChunk, None]:
        audio = chunk(stage="vocoder", modality="audio")
        audio.audio_data = [0.0] * 20
        audio.sample_rate = 10
        yield audio
        image = chunk(stage="generate", modality="image")
        image.images_b64 = ["image1", "image2"]
        yield image
        yield chunk(finish="abort" if aborted else "stop")

    async def run() -> list[GenerateChunk]:
        return [item async for item in metrics.generate(source(), False)]

    asyncio.run(run())
    assert metrics.registry.get_sample_value(
        "sglang_omni:audio_output_seconds_total",
        {"model_name": "test", "stage": "vocoder"},
    ) == (None if aborted else 2)
    assert metrics.registry.get_sample_value(
        "sglang_omni:generated_images_total",
        {"model_name": "test", "stage": "generate"},
    ) == (None if aborted else 2)


def test_backend_export_preserves_types_and_excludes_duplicate_owned_names() -> None:
    metrics = RequestMetrics("test")
    metrics.backend.snapshots = [
        (
            "decode",
            [
                {
                    "name": "sglang:num_retracted_reqs",
                    "documentation": "Verify",
                    "type": "counter",
                    "samples": [
                        {
                            "name": "sglang:num_retracted_reqs_total",
                            "labels": {"mode": "decode"},
                            "value": 3,
                        }
                    ],
                },
                {
                    "name": "sglang:prompt_tokens",
                    "documentation": "Duplicate",
                    "type": "counter",
                    "samples": [
                        {
                            "name": "sglang:prompt_tokens_total",
                            "labels": {},
                            "value": 999,
                        }
                    ],
                },
            ],
        )
    ]
    assert (
        metrics.registry.get_sample_value(
            "sglang:num_retracted_reqs_total",
            {
                "model_name": "test",
                "omni_stage": "decode",
                "mode": "decode",
            },
        )
        == 3
    )
    assert (
        metrics.registry.get_sample_value(
            "sglang:prompt_tokens_total",
            {
                "model_name": "test",
                "stage": "decode",
            },
        )
        is None
    )


def test_nonpositive_token_delta_is_ignored() -> None:
    metrics = RequestMetrics("test")
    metrics.observe_interval(1, 0)
    metrics.observe_interval(1, -1)
    assert (
        metrics.registry.get_sample_value(
            "sglang:inter_token_latency_seconds_count", {"model_name": "test"}
        )
        is None
    )


def test_finished_usage_and_registry_isolation() -> None:
    metrics = RequestMetrics("test")
    other = RequestMetrics("test")
    metrics.observe_finished(
        stream=True, latency=2, prompt_tokens=10, generation_tokens=4
    )
    labels = {"model_name": "test", "is_streaming": "true"}
    assert metrics.registry.get_sample_value("sglang:prompt_tokens_total", labels) == 10
    assert (
        metrics.registry.get_sample_value("sglang:generation_tokens_total", labels) == 4
    )
    assert other.registry.get_sample_value("sglang:prompt_tokens_total", labels) is None


def test_multiprocess_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "/unused")
    with pytest.raises(ValueError, match="Unset"):
        RequestMetrics("test")


def test_failed_stage_does_not_serve_stale_gauges() -> None:
    metrics = RequestMetrics("test")

    class FailedClient:
        async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
            return {"success": False, "results": []}

    with pytest.raises(RuntimeError):
        asyncio.run(metrics.refresh_scheduler(FailedClient()))


def test_missing_usage_is_not_zero_usage() -> None:
    metrics = RequestMetrics("test")
    metrics.observe_finished(
        stream=False, latency=1, prompt_tokens=None, generation_tokens=None
    )
    labels = {"model_name": "test", "is_streaming": "false"}
    assert (
        metrics.registry.get_sample_value(
            "sglang:e2e_request_latency_seconds_count", labels
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value("sglang:generation_tokens_total", labels)
        is None
    )
