# SPDX-License-Identifier: Apache-2.0
"""Request metrics using SGLang's Prometheus names and token weighting.

Callers supply lifecycle timestamps; HTTP chunks are not token observations.
"""

import asyncio
import os
import time
from contextlib import aclosing
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.core import Metric

from sglang_omni.scheduling.metrics_snapshot import ENGINE_GAUGES


def validate_metrics_environment():
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        raise ValueError(
            "Omni metrics use a single HTTP-process registry. Unset "
            "PROMETHEUS_MULTIPROC_DIR before launching the service."
        )


class BackendMetricsCollector:
    """Export leader snapshots, excluding HTTP-owned metric families."""

    def __init__(self, model_name, owned_names):
        self.model_name = model_name
        self.owned_names = owned_names
        self.snapshots = []

    def describe(self):
        return []

    def collect(self):
        families = {}
        for stage, snapshot in self.snapshots:
            for item in snapshot:
                name = item["name"]
                if name in self.owned_names:
                    continue
                family = families.setdefault(
                    name, Metric(name, item["documentation"], item["type"])
                )
                if family.type != item["type"]:
                    raise ValueError(f"Conflicting backend metric type: {name}")
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
    def __init__(self, model_name: str):
        validate_metrics_environment()
        self.registry = CollectorRegistry()
        self.model_name = model_name
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
        self.e2e = Histogram(
            "sglang:e2e_request_latency_seconds",
            "Histogram of End-to-end request latency in seconds",
            labels,
            buckets=(*latency_buckets, 600, 1200, 1800, 2400),
            registry=self.registry,
        )
        self.ttft = Histogram(
            "sglang:time_to_first_token_seconds",
            "Histogram of time to first token in seconds.",
            labels,
            buckets=latency_buckets,
            registry=self.registry,
        )
        self.itl = Histogram(
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
        self.prompt_tokens = Counter(
            "sglang:prompt_tokens_total",
            "Number of prompt tokens processed.",
            labels,
            registry=self.registry,
        )
        self.generation_tokens = Counter(
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
        self.token_lengths = {
            name: Histogram(
                f"sglang:{name}_histogram",
                f"Histogram of {name.replace('_', ' ')} length.",
                ("model_name",),
                buckets=token_buckets,
                registry=self.registry,
            )
            for name in ("prompt_tokens", "uncached_prompt_tokens", "generation_tokens")
        }
        self.cached_tokens = Counter(
            "sglang:cached_tokens_total",
            "Number of cached prompt tokens by source.",
            ("model_name", "cache_source"),
            registry=self.registry,
        )
        self.requests = Counter(
            "sglang:num_requests_total",
            "Number of requests processed.",
            labels,
            registry=self.registry,
        )
        self.aborted = Counter(
            "sglang:num_aborted_requests_total",
            "Number of requests aborted.",
            ("model_name",),
            registry=self.registry,
        )
        self.failed = Counter(
            "sglang_omni:failed_requests_total",
            "Number of failed pipeline requests.",
            ("model_name",),
            registry=self.registry,
        )
        self.running = Gauge(
            "sglang:num_running_reqs",
            "Number of requests running in a stage.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.waiting = Gauge(
            "sglang:num_queue_reqs",
            "Number of requests waiting in a stage.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.scrape_lock = asyncio.Lock()
        self.engine_gauges = {
            name: Gauge(
                f"sglang:{name}",
                f"Backend scheduler {name}.",
                ("model_name", "stage"),
                registry=self.registry,
            )
            for name in ENGINE_GAUGES
        }
        self.queue_latency = Histogram(
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
        self.batch_size = Histogram(
            "sglang_omni:scheduler_batch_size",
            "Requests dispatched per compute call.",
            ("model_name", "stage"),
            buckets=(1, 2, 4, 8, 16, 32, 64, 128),
            registry=self.registry,
        )
        self.execution = Histogram(
            "sglang_omni:stage_compute_duration_seconds",
            "Host compute duration, without device synchronization.",
            ("model_name", "stage", "outcome"),
            registry=self.registry,
        )
        self.stage_duration = Histogram(
            "sglang_omni:stage_request_duration_seconds",
            "Stage dispatch through routing/cleanup, including scheduler waiting.",
            ("model_name", "stage", "outcome"),
            registry=self.registry,
        )
        self.audio_seconds = Counter(
            "sglang_omni:audio_output_seconds_total",
            "Completed output waveform duration before encoding/speed conversion.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.images = Counter(
            "sglang_omni:generated_images_total",
            "Images in the completed output list, not diffusion steps.",
            ("model_name", "stage"),
            registry=self.registry,
        )
        self.active_requests: dict[str, TokenObservation] = {}
        self.http_requests = Counter(
            "sglang_omni:http_requests_total",
            "HTTP requests including errors.",
            ("method", "route", "status_code", "outcome"),
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "sglang_omni:http_request_duration_seconds",
            "HTTP duration through the final response body or disconnect.",
            ("method", "route", "outcome"),
            registry=self.registry,
            buckets=(*latency_buckets, 600, 1200, 1800, 2400),
        )
        self.http_inflight = Gauge(
            "sglang_omni:http_requests_inflight",
            "Active HTTP requests.",
            registry=self.registry,
        )

        self.backend = BackendMetricsCollector(
            self.model_name, {family.name for family in self.registry.collect()}
        )
        self.registry.register(self.backend)

    def observe_scheduler_event(self, stage, data):
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

    def observe_tokens(self, request_id, stage, prompt_tokens, completion_tokens):
        state = self.active_requests.get(request_id)
        if state is None or completion_tokens <= 0:
            return
        if state.stage is None:
            state.stage = stage
        if state.stage != stage:
            return
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
        state.prompt_tokens = prompt_tokens

    async def generate(self, source, stream, request_id=None):
        started = time.perf_counter()
        state = TokenObservation(started, stream)
        if request_id is not None:
            self.active_requests[request_id] = state
        text_stage = None
        selected_text_stage = False
        finished_at = None
        usage = None
        aborted = False
        audio_seconds = {}
        image_counts = {}
        try:
            async with aclosing(source):
                async for chunk in source:
                    now = time.perf_counter()
                    finished_at = now
                    # Audio stages may carry their own token counts. Do not add
                    # their cumulative snapshots to the text engine's usage.
                    if chunk.modality == "text":
                        if not selected_text_stage and chunk.usage is not None:
                            text_stage = chunk.stage_name
                            selected_text_stage = True
                        if selected_text_stage and chunk.stage_name == text_stage:
                            if chunk.usage is not None:
                                usage = chunk.usage
                    aborted = aborted or chunk.finish_reason == "abort"
                    stage = chunk.stage_name or "unknown"
                    audio = getattr(chunk, "audio_data", None)
                    sample_rate = getattr(chunk, "sample_rate", None)
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
                    images = getattr(chunk, "images_b64", None)
                    if images is not None:
                        image_counts[stage] = len(images)
                    elif getattr(chunk, "image_b64", None):
                        image_counts[stage] = 1
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
                getattr(usage, "prompt_tokens_details", None)
                if state.stage is None or state.stage == text_stage
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

    async def refresh_scheduler(self, client):
        # Reuse the control plane: no extra ports, shared files or per-TP
        # counter aggregation. Each stage reports its leader snapshot.
        async with self.scrape_lock:
            response = await asyncio.wait_for(
                client.admin("metrics", timeout_s=5.0), timeout=5.0
            )
            if not response.get("success", True):
                raise RuntimeError("A stage failed to report scheduler metrics")
            self.running.clear()
            self.waiting.clear()
            snapshots = []
            for gauge in self.engine_gauges.values():
                gauge.clear()
            for result in response["results"]:
                if not result["success"]:
                    continue
                data = result["data"]
                if "num_running_reqs" not in data:
                    continue
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

            self.backend.snapshots = snapshots

    def observe_interval(self, interval: float, num_new_tokens: int) -> None:
        if num_new_tokens <= 0:
            return
        histogram = self.itl.labels(self.model_name)
        # Match SGLang's weighted Histogram update without one lock per token.
        histogram._sum.inc(interval)
        for bound, bucket in zip(histogram._upper_bounds, histogram._buckets):
            if interval / num_new_tokens <= bound:
                bucket.inc(num_new_tokens)
                break

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
        if generation_tokens is not None:
            self.generation_tokens.labels(*labels).inc(generation_tokens)
            self.token_lengths["generation_tokens"].labels(self.model_name).observe(
                generation_tokens
            )
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


class MetricsMiddleware:
    """Pure ASGI instrumentation: do not buffer or parse streaming bodies."""

    def __init__(self, app, metrics: RequestMetrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {
            "/metrics",
            "/health",
            "/health_generate",
            "/favicon.ico",
        }:
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        finished = None
        status = 500
        outcome = "incomplete"
        disconnected = False
        self.metrics.http_inflight.inc()

        async def wrapped_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def wrapped_send(message):
            nonlocal status, finished, outcome
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)
            if message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                finished = time.perf_counter()
                outcome = "completed"

        try:
            await self.app(scope, wrapped_receive, wrapped_send)
            if disconnected and finished is None:
                outcome = "disconnected"
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
            route = getattr(scope.get("route"), "path", "unmatched")
            self.metrics.http_requests.labels(method, route, str(status), outcome).inc()
            self.metrics.http_latency.labels(method, route, outcome).observe(
                (finished if finished is not None else time.perf_counter()) - started
            )
