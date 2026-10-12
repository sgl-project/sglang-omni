# SPDX-License-Identifier: Apache-2.0
"""Request metrics using SGLang's Prometheus names and token weighting.

Callers supply lifecycle timestamps; HTTP chunks are not token observations.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing
from dataclasses import dataclass

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, UsageInfo
from sglang_omni.scheduling.metrics_snapshot import ENGINE_GAUGES, BackendMetricFamily

METRICS_DEPENDENCY_ERROR: ModuleNotFoundError | None = None
try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
    from prometheus_client.core import Metric
except ModuleNotFoundError as error:
    if error.name != "prometheus_client":
        raise
    else:
        METRICS_DEPENDENCY_ERROR = error
else:
    pass


def validate_metrics_environment() -> None:
    if METRICS_DEPENDENCY_ERROR is not None:
        raise RuntimeError(
            "Metrics require prometheus-client; install sglang-omni[metrics]."
        ) from METRICS_DEPENDENCY_ERROR
    else:
        pass
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        raise ValueError(
            "Omni metrics use a single HTTP-process registry. Unset "
            "PROMETHEUS_MULTIPROC_DIR before launching the service."
        )
    else:
        pass


class BackendMetricsCollector:
    """Export leader snapshots, excluding HTTP-owned metric families."""

    def __init__(self, model_name: str, owned_names: set[str]) -> None:
        self.model_name: str = model_name
        self.owned_names: set[str] = owned_names
        self.snapshots: list[tuple[str, list[BackendMetricFamily]]] = []

    def describe(self) -> list[Metric]:
        return []

    def collect(self) -> Iterator[Metric]:
        families: dict[str, Metric] = {}
        for stage, snapshot in self.snapshots:
            for item in snapshot:
                name = item["name"]
                if name in self.owned_names:
                    continue
                else:
                    pass
                family = families.setdefault(
                    name, Metric(name, item["documentation"], item["type"])
                )
                if family.type != item["type"]:
                    raise ValueError(f"Conflicting backend metric type: {name}")
                else:
                    pass
                for sample in item["samples"]:
                    labels = {
                        **sample["labels"],
                        "model_name": self.model_name,
                        "omni_stage": stage,
                    }
                    family.add_sample(sample["name"], labels, sample["value"])
        yield from families.values()


@dataclass
class TokenObservation:
    started: float
    stream: bool
    stage: str | None = None
    last_time: float | None = None
    completion_tokens: int = 0
    prompt_tokens: int | None = None


class RequestMetrics:
    def __init__(self, model_name: str) -> None:
        validate_metrics_environment()
        self.registry: CollectorRegistry = CollectorRegistry()
        self.model_name: str = model_name
        labels = ("model_name", "is_streaming")
        latency_buckets = (
            0.1,
            0.2,
            0.4,
            0.6,
            0.8,
            1,
            2,
            4,
            6,
            8,
            10,
            20,
            40,
            60,
            80,
            100,
            200,
            400,
        )
        self.e2e: Histogram = Histogram(
            "sglang:e2e_request_latency_seconds",
            "Histogram of End-to-end request latency in seconds",
            labels,
            buckets=(*latency_buckets, 600, 1200, 1800, 2400),
            registry=self.registry,
        )
        self.ttft: Histogram = Histogram(
            "sglang:time_to_first_token_seconds",
            "Histogram of time to first token in seconds.",
            labels,
            buckets=latency_buckets,
            registry=self.registry,
        )
        self.itl: Histogram = Histogram(
            "sglang:inter_token_latency_seconds",
            "Histogram of inter-token latency in seconds.",
            ("model_name",),
            buckets=(
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.060,
                0.080,
                0.100,
                0.200,
                0.400,
                0.600,
                0.800,
                1,
                2,
                4,
                6,
                8,
            ),
            registry=self.registry,
        )
        self.prompt_tokens: Counter = Counter(
            "sglang:prompt_tokens_total",
            "Number of prompt tokens processed.",
            labels,
            registry=self.registry,
        )
        self.generation_tokens: Counter = Counter(
            "sglang:generation_tokens_total",
            "Number of tokens generated.",
            labels,
            registry=self.registry,
        )
        token_buckets = (
            100,
            300,
            500,
            700,
            1000,
            1500,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            12500,
            15000,
            17500,
            20000,
            22500,
            25000,
            27500,
            30000,
            35000,
            40000,
            60000,
            80000,
            100000,
            200000,
            300000,
            400000,
            600000,
            800000,
            1000000,
            1100000,
        )
        self.token_lengths: dict[str, Histogram] = {
            name: Histogram(
                f"sglang:{name}_histogram",
                f"Histogram of {name.replace('_', ' ')} length.",
                ("model_name",),
                buckets=token_buckets,
                registry=self.registry,
            )
            for name in ("prompt_tokens", "uncached_prompt_tokens", "generation_tokens")
        }
        self.cached_tokens: Counter = Counter(
            "sglang:cached_tokens_total",
            "Number of cached prompt tokens by source.",
            ("model_name", "cache_source"),
            registry=self.registry,
        )
        self.requests: Counter = Counter(
            "sglang:num_requests_total",
            "Number of requests processed.",
            labels,
            registry=self.registry,
        )
        self.aborted: Counter = Counter(
            "sglang:num_aborted_requests_total",
            "Number of requests aborted.",
            ("model_name",),
            registry=self.registry,
        )
        self.failed: Counter = Counter(
            "sglang_omni:failed_requests_total",
            "Number of failed pipeline requests.",
            ("model_name",),
            registry=self.registry,
        )
        self.running: Gauge = Gauge(
            "sglang:num_running_reqs",
            "Number of requests running in a stage.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.waiting: Gauge = Gauge(
            "sglang:num_queue_reqs",
            "Number of requests waiting in a stage.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.scrape_lock: asyncio.Lock = asyncio.Lock()
        self.engine_gauges: dict[str, Gauge] = {
            name: Gauge(
                f"sglang:{name}",
                f"Backend scheduler {name}.",
                ("model_name", "stage"),
                registry=self.registry,
            )
            for name in ENGINE_GAUGES
        }
        self.queue_latency: Histogram = Histogram(
            "sglang:queue_time_seconds",
            "Scheduler queue duration in seconds.",
            ("model_name", "stage"),
            registry=self.registry,
            buckets=(
                0,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.2,
                0.5,
                1,
                2,
                3,
                4,
                5,
                10,
                15,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
                1200,
                1400,
                1600,
                1800,
                2000,
                2500,
                3000,
            ),
        )
        self.batch_size: Histogram = Histogram(
            "sglang_omni:scheduler_batch_size",
            "Requests dispatched per compute call.",
            ("model_name", "stage"),
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
            registry=self.registry,
        )
        self.execution: Histogram = Histogram(
            "sglang_omni:stage_compute_duration_seconds",
            "Host compute duration, without device synchronization.",
            ("model_name", "stage", "outcome"),
            registry=self.registry,
        )
        self.stage_duration: Histogram = Histogram(
            "sglang_omni:stage_request_duration_seconds",
            "Stage dispatch through routing/cleanup, including scheduler waiting.",
            ("model_name", "stage", "outcome"),
            registry=self.registry,
        )
        self.audio_seconds: Counter = Counter(
            "sglang_omni:audio_output_seconds_total",
            "Completed output waveform duration before encoding/speed conversion.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.images: Counter = Counter(
            "sglang_omni:generated_images_total",
            "Images in the completed output list, not diffusion steps.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.active_requests: dict[str, TokenObservation] = {}
        self.http_requests: Counter = Counter(
            "sglang_omni:http_requests_total",
            "HTTP requests including errors.",
            ("method", "route", "status_code", "outcome"),
            registry=self.registry,
        )
        self.http_latency: Histogram = Histogram(
            "sglang_omni:http_request_duration_seconds",
            "HTTP duration through the final response body or disconnect.",
            ("method", "route", "outcome"),
            registry=self.registry,
            buckets=(*latency_buckets, 600, 1200, 1800, 2400),
        )
        self.http_inflight: Gauge = Gauge(
            "sglang_omni:http_requests_inflight",
            "Active HTTP requests.",
            registry=self.registry,
        )

        self.backend: BackendMetricsCollector = BackendMetricsCollector(
            self.model_name, {family.name for family in self.registry.collect()}
        )
        self.registry.register(self.backend)

    def observe_scheduler_event(
        self, stage: str, data: dict[str, str | int | float]
    ) -> None:
        labels = (self.model_name, stage)
        if data["event"] == "queue":
            self.queue_latency.labels(*labels).observe(data["duration"])
        elif data["event"] == "batch":
            self.batch_size.labels(*labels).observe(data["size"])
        elif data["event"] == "execution":
            self.execution.labels(*labels, data["outcome"]).observe(data["duration"])
        elif data["event"] == "stage":
            self.stage_duration.labels(*labels, data["outcome"]).observe(
                data["duration"]
            )
        else:
            pass

    def observe_tokens(
        self,
        request_id: str,
        stage: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        state = self.active_requests.get(request_id)
        if state is None or completion_tokens <= 0:
            return
        else:
            pass
        if state.stage is None:
            state.stage = stage
        else:
            pass
        if state.stage != stage:
            return
        else:
            pass
        now = time.perf_counter()
        if state.last_time is None:
            self.ttft.labels(
                self.model_name, "true" if state.stream else "false"
            ).observe(now - state.started)
        else:
            self.observe_interval(
                now - state.last_time, completion_tokens - state.completion_tokens
            )
        if completion_tokens != state.completion_tokens:
            state.last_time = now
            state.completion_tokens = completion_tokens
        else:
            pass
        state.prompt_tokens = prompt_tokens

    async def generate(
        self,
        source: AsyncGenerator[GenerateChunk, None],
        stream: bool,
        request_id: str | None = None,
    ) -> AsyncGenerator[GenerateChunk, None]:
        started = time.perf_counter()
        state = TokenObservation(started, stream)
        if request_id is not None:
            self.active_requests[request_id] = state
        else:
            pass
        text_stage: str | None = None
        selected_text_stage = False
        finished_at: float | None = None
        usage: UsageInfo | None = None
        aborted = False
        audio_seconds: dict[str, float] = {}
        image_counts: dict[str, int] = {}
        try:
            async with aclosing(source):
                async for chunk in source:
                    now = time.perf_counter()
                    finished_at = now
                    # Audio stages may carry their own token counts. Do not add
                    # their cumulative snapshots to the text engine's usage.
                    usage_stage = chunk.text_usage_stage or chunk.stage_name
                    if chunk.modality == "text" or chunk.text_usage_stage is not None:
                        if not selected_text_stage and chunk.usage is not None:
                            text_stage = usage_stage
                            selected_text_stage = True
                        else:
                            pass
                        if selected_text_stage and usage_stage == text_stage:
                            if chunk.usage is not None:
                                usage = chunk.usage
                            else:
                                pass
                        else:
                            pass
                    else:
                        pass
                    aborted = aborted or chunk.finish_reason == "abort"
                    stage = chunk.stage_name or "unknown"
                    audio = chunk.audio_data
                    sample_rate = chunk.sample_rate
                    if (
                        audio is not None
                        and sample_rate is not None
                        and sample_rate > 0
                    ):
                        shape = getattr(audio, "shape", None)
                        if shape is not None and len(shape) in (1, 2):
                            samples = shape[-1]
                        elif isinstance(audio, (list, tuple)) and (
                            not audio or isinstance(audio[0], (int, float))
                        ):
                            samples = len(audio)
                        else:
                            samples = None
                        if samples is not None:
                            audio_seconds[stage] = (
                                audio_seconds.get(stage, 0) + samples / sample_rate
                            )
                        else:
                            pass
                    else:
                        pass
                    images = getattr(chunk, "images_b64", None)
                    if images is not None:
                        image_counts[stage] = len(images)
                    elif getattr(chunk, "image_b64", None):
                        image_counts[stage] = 1
                    else:
                        pass
                    yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            self.aborted.labels(self.model_name).inc()
            raise
        except Exception:
            self.failed.labels(self.model_name).inc()
            raise
        else:
            if finished_at is None:
                return
            else:
                pass
            if aborted:
                self.aborted.labels(self.model_name).inc()
            else:
                for stage, seconds in audio_seconds.items():
                    self.audio_seconds.labels(self.model_name, stage).inc(seconds)
                for stage, count in image_counts.items():
                    self.images.labels(self.model_name, stage).inc(count)
            # Cache details belong to the same text stage as the token counts.
            # Never combine another stage's usage with native AR observations.
            details = (
                usage.prompt_tokens_details
                if usage is not None
                and (state.stage is None or state.stage == text_stage)
                else None
            )
            self.observe_finished(
                stream=stream,
                latency=finished_at - started,
                prompt_tokens=(
                    state.prompt_tokens
                    if state.last_time is not None
                    else (usage.prompt_tokens if usage else None)
                ),
                generation_tokens=(
                    state.completion_tokens
                    if state.last_time is not None
                    else (usage.completion_tokens if usage else None)
                ),
                cached_tokens=details.get("cached_tokens") if details else None,
            )
        finally:
            if request_id is not None:
                self.active_requests.pop(request_id, None)
            else:
                pass

    async def refresh_scheduler(self, client: Client) -> None:
        # Reuse the control plane: no extra ports, shared files or per-TP
        # counter aggregation. Each stage reports its leader snapshot.
        async with self.scrape_lock:
            response = await asyncio.wait_for(
                client.admin("metrics", timeout_s=5.0), timeout=5.0
            )
            if not response.get("success", True):
                raise RuntimeError("A stage failed to report scheduler metrics")
            else:
                pass
            self.running.clear()
            self.waiting.clear()
            snapshots = []
            for gauge in self.engine_gauges.values():
                gauge.clear()
            for result in response["results"]:
                if not result["success"]:
                    continue
                else:
                    pass
                data = result["data"]
                if "num_running_reqs" not in data:
                    continue
                else:
                    pass
                stage = result["stage"]
                snapshots.append((stage, data.get("backend_metrics", [])))
                self.running.labels(self.model_name, stage).set(
                    data["num_running_reqs"]
                )
                self.waiting.labels(self.model_name, stage).set(data["num_queue_reqs"])
                for name, value in data.get("engine_gauges", {}).items():
                    if name in self.engine_gauges:
                        self.engine_gauges[name].labels(self.model_name, stage).set(
                            value
                        )
                    else:
                        pass

            self.backend.snapshots = snapshots

    def observe_interval(self, interval: float, num_new_tokens: int) -> None:
        if num_new_tokens <= 0:
            return
        else:
            pass
        histogram = self.itl.labels(self.model_name)
        # Match SGLang's weighted Histogram update without one lock per token.
        histogram._sum.inc(
            interval
        )  # noqa: leading-underscore - Prometheus weighted histogram update
        for bound, bucket in zip(
            histogram._upper_bounds,
            histogram._buckets,  # noqa: leading-underscore - Prometheus bucket storage
        ):
            if interval / num_new_tokens <= bound:
                bucket.inc(num_new_tokens)
                break
            else:
                pass

    def observe_finished(
        self,
        *,
        stream: bool,
        latency: float,
        prompt_tokens: int | None,
        generation_tokens: int | None,
        cached_tokens: int | None = None,
    ) -> None:
        labels = (self.model_name, "true" if stream else "false")
        self.requests.labels(*labels).inc()
        self.e2e.labels(*labels).observe(latency)
        if prompt_tokens is not None:
            self.prompt_tokens.labels(*labels).inc(prompt_tokens)
            self.token_lengths["prompt_tokens"].labels(self.model_name).observe(
                prompt_tokens
            )
        else:
            pass
        if generation_tokens is not None:
            self.generation_tokens.labels(*labels).inc(generation_tokens)
            self.token_lengths["generation_tokens"].labels(self.model_name).observe(
                generation_tokens
            )
        else:
            pass
        if (
            cached_tokens is not None
            and prompt_tokens is not None
            and 0 <= cached_tokens <= prompt_tokens
        ):
            self.token_lengths["uncached_prompt_tokens"].labels(
                self.model_name
            ).observe(prompt_tokens - cached_tokens)
            if cached_tokens > 0:
                self.cached_tokens.labels(self.model_name, "total").inc(cached_tokens)
            else:
                pass
        else:
            pass


class MetricsMiddleware:
    """Pure ASGI instrumentation: do not buffer or parse streaming bodies."""

    def __init__(self, app: ASGIApp, metrics: RequestMetrics) -> None:
        self.app: ASGIApp = app
        self.metrics: RequestMetrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in {
            "/metrics",
            "/health",
            "/health_generate",
            "/favicon.ico",
        }:
            return await self.app(scope, receive, send)
        else:
            pass
        started = time.perf_counter()
        finished = None
        status = 500
        outcome = "incomplete"
        disconnected = False
        self.metrics.http_inflight.inc()

        async def wrapped_receive() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            else:
                pass
            return message

        async def wrapped_send(message: Message) -> None:
            nonlocal status, finished, outcome
            if message["type"] == "http.response.start":
                status = message["status"]
            else:
                pass
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                finished = time.perf_counter()
                outcome = "completed"
            else:
                pass

        try:
            await self.app(scope, wrapped_receive, wrapped_send)
            if disconnected and finished is None:
                outcome = "disconnected"
            else:
                pass
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            self.metrics.http_inflight.dec()
            method = scope["method"]
            if method not in {
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
                "HEAD",
                "OPTIONS",
            }:
                method = "OTHER"
            else:
                pass
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.http_requests.labels(method, route, str(status), outcome).inc()
            self.metrics.http_latency.labels(method, route, outcome).observe(
                (finished if finished is not None else time.perf_counter()) - started
            )
