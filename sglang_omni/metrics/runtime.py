# SPDX-License-Identifier: Apache-2.0
"""Prometheus exposition for SGLang and generated audio metrics."""

from __future__ import annotations

import os
import time

from prometheus_client import (
    CollectorRegistry,
    GCCollector,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
    multiprocess,
    values,
)

from sglang_omni.metrics.audio import AudioMetrics
from sglang_omni.metrics.types import MetricDetails, RuntimeMetricEvent


class RuntimeMetrics:
    """Expose upstream SGLang metrics and generated audio measurements."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        ProcessCollector(registry=self.registry)
        PlatformCollector(registry=self.registry)
        GCCollector(registry=self.registry)
        self.audio = AudioMetrics(self.registry)

    def record(
        self,
        event_name: RuntimeMetricEvent,
        request_id: str,
        stage: str,
        timestamp_ns: int,
        metadata: MetricDetails | None = None,
    ) -> None:
        del stage
        self.audio.record(event_name, request_id, timestamp_ns, metadata or {})

    def render(self) -> bytes:
        self.audio.prune(time.perf_counter_ns())
        metrics_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
        if metrics_dir is None:
            return generate_latest(self.registry)

        upstream_registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(upstream_registry, path=metrics_dir)
        upstream_metrics = generate_latest(upstream_registry)
        if values.ValueClass.__name__ == "MmapedValue":
            return upstream_metrics
        return upstream_metrics + generate_latest(self.registry)
