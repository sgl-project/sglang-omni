# SPDX-License-Identifier: Apache-2.0
"""Track request lifecycle, scheduler observations, and output usage."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass

import sglang_omni.serve.metrics_registry as metrics_registry
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, UMMGenerateChunk, UsageInfo
from sglang_omni.metrics.audio import AudioMetrics
from sglang_omni.scheduling.metrics_snapshot import SchedulerMetricEvent

SCHEDULER_METRICS_TIMEOUT_S = 5.0


@dataclass
class TokenObservation:
    started_at_s: float
    stream: bool
    stage: str | None = None
    last_token_at_s: float | None = None
    completion_tokens: int = 0
    prompt_tokens: int | None = None


class RequestMetrics:
    def __init__(self, model_name: str) -> None:
        collectors = metrics_registry.create_request_collectors(model_name)
        self.model_name: str = model_name
        self.registry: metrics_registry.CollectorRegistry = collectors.registry
        self.audio_metrics: AudioMetrics = AudioMetrics(self.registry)
        self.e2e: metrics_registry.Histogram = collectors.e2e
        self.ttft: metrics_registry.Histogram = collectors.ttft
        self.itl: metrics_registry.Histogram = collectors.itl
        self.prompt_tokens: metrics_registry.Counter = collectors.prompt_tokens
        self.generation_tokens: metrics_registry.Counter = collectors.generation_tokens
        self.token_lengths: dict[str, metrics_registry.Histogram] = (
            collectors.token_lengths
        )
        self.cached_tokens: metrics_registry.Counter = collectors.cached_tokens
        self.requests: metrics_registry.Counter = collectors.requests
        self.aborted: metrics_registry.Counter = collectors.aborted
        self.failed: metrics_registry.Counter = collectors.failed
        self.running: metrics_registry.Gauge = collectors.running
        self.waiting: metrics_registry.Gauge = collectors.waiting
        self.engine_gauges: dict[str, metrics_registry.Gauge] = collectors.engine_gauges
        self.queue_latency: metrics_registry.Histogram = collectors.queue_latency
        self.batch_size: metrics_registry.Histogram = collectors.batch_size
        self.execution: metrics_registry.Histogram = collectors.execution
        self.stage_duration: metrics_registry.Histogram = collectors.stage_duration
        self.audio_seconds: metrics_registry.Counter = collectors.audio_seconds
        self.images: metrics_registry.Counter = collectors.images
        self.http_requests: metrics_registry.Counter = collectors.http_requests
        self.http_latency: metrics_registry.Histogram = collectors.http_latency
        self.http_inflight: metrics_registry.Gauge = collectors.http_inflight
        self.backend: metrics_registry.BackendMetricsCollector = collectors.backend
        self.scrape_lock: asyncio.Lock = asyncio.Lock()
        self.active_requests: dict[str, TokenObservation] = {}

    def observe_scheduler_event(
        self, stage: str, scheduler_event: SchedulerMetricEvent
    ) -> None:
        labels = (self.model_name, stage)
        if scheduler_event["event"] == "queue":
            self.queue_latency.labels(*labels).observe(scheduler_event["duration"])
        elif scheduler_event["event"] == "batch":
            self.batch_size.labels(*labels).observe(scheduler_event["size"])
        elif scheduler_event["event"] == "execution":
            self.execution.labels(*labels, scheduler_event["outcome"]).observe(
                scheduler_event["duration"]
            )
        elif scheduler_event["event"] == "stage":
            self.stage_duration.labels(*labels, scheduler_event["outcome"]).observe(
                scheduler_event["duration"]
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
        observed_at_s = time.perf_counter()
        if state.last_token_at_s is None:
            self.ttft.labels(
                self.model_name, "true" if state.stream else "false"
            ).observe(observed_at_s - state.started_at_s)
        else:
            self.observe_interval(
                observed_at_s - state.last_token_at_s,
                completion_tokens - state.completion_tokens,
            )
        if completion_tokens != state.completion_tokens:
            state.last_token_at_s = observed_at_s
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
        started_at_s = time.perf_counter()
        state = TokenObservation(started_at_s, stream)
        if request_id is not None:
            self.active_requests[request_id] = state
        else:
            pass
        text_stage: str | None = None
        selected_text_stage = False
        finished_at_s: float | None = None
        usage: UsageInfo | None = None
        aborted = False
        audio_seconds: dict[str, float] = {}
        image_counts: dict[str, int] = {}
        try:
            async with aclosing(source):
                async for chunk in source:
                    observed_at_s = time.perf_counter()
                    finished_at_s = observed_at_s
                    # note (syd520zy): Audio token counts must not inflate text usage.
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
                            samples = max(shape)
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
                    if isinstance(chunk, UMMGenerateChunk) and chunk.segments:
                        image_count = sum(
                            1 for segment in chunk.segments if segment.kind == "image"
                        )
                        if image_count:
                            image_counts[stage] = image_count
                        else:
                            pass
                    elif chunk.image is not None:
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
            if finished_at_s is None:
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
            # note (syd520zy): Cache details and token counts must share a text stage.
            details = (
                usage.prompt_tokens_details
                if usage is not None
                and (state.stage is None or state.stage == text_stage)
                else None
            )
            self.observe_finished(
                stream=stream,
                latency_s=finished_at_s - started_at_s,
                prompt_tokens=(
                    state.prompt_tokens
                    if state.last_token_at_s is not None
                    else (usage.prompt_tokens if usage else None)
                ),
                generation_tokens=(
                    state.completion_tokens
                    if state.last_token_at_s is not None
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
        # note (syd520zy): Leader snapshots avoid double-counting TP worker metrics.
        async with self.scrape_lock:
            response = await asyncio.wait_for(
                client.admin("metrics", timeout_s=SCHEDULER_METRICS_TIMEOUT_S),
                timeout=SCHEDULER_METRICS_TIMEOUT_S,
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
                scheduler_snapshot = result["data"]
                if "num_running_reqs" not in scheduler_snapshot:
                    continue
                else:
                    pass
                stage = result["stage"]
                snapshots.append((stage, scheduler_snapshot.get("backend_metrics", [])))
                self.running.labels(self.model_name, stage).set(
                    scheduler_snapshot["num_running_reqs"]
                )
                self.waiting.labels(self.model_name, stage).set(
                    scheduler_snapshot["num_queue_reqs"]
                )
                for name, value in scheduler_snapshot.get("engine_gauges", {}).items():
                    if name in self.engine_gauges:
                        self.engine_gauges[name].labels(self.model_name, stage).set(
                            value
                        )
                    else:
                        pass

            self.backend.snapshots = snapshots

    def observe_interval(self, interval_s: float, num_new_tokens: int) -> None:
        if num_new_tokens <= 0:
            return
        else:
            pass
        histogram = self.itl.labels(self.model_name)
        # note (syd520zy): Weighted updates avoid acquiring one lock per token.
        histogram._sum.inc(
            interval_s
        )  # noqa: leading-underscore - Prometheus weighted histogram update
        for bound, bucket in zip(
            histogram._upper_bounds,
            histogram._buckets,  # noqa: leading-underscore - Prometheus bucket storage
        ):
            if interval_s / num_new_tokens <= bound:
                bucket.inc(num_new_tokens)
                break
            else:
                pass

    def observe_finished(
        self,
        *,
        stream: bool,
        latency_s: float,
        prompt_tokens: int | None,
        generation_tokens: int | None,
        cached_tokens: int | None = None,
    ) -> None:
        labels = (self.model_name, "true" if stream else "false")
        self.requests.labels(*labels).inc()
        self.e2e.labels(*labels).observe(latency_s)
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
