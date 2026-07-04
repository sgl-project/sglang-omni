# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request, Response

PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
DEFAULT_DURATION_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)


@dataclass(frozen=True)
class MetricFamily:
    name: str
    metric_type: str
    help_text: str


GAUGE_FAMILIES = (
    MetricFamily(
        "sglang_omni_pipeline_running",
        "gauge",
        "Whether the Omni coordinator reports the pipeline as running.",
    ),
    MetricFamily(
        "sglang_omni_pipeline_tracked_requests",
        "gauge",
        "Number of requests tracked by the Omni coordinator health snapshot.",
    ),
    MetricFamily(
        "sglang_omni_pipeline_pending_completions",
        "gauge",
        "Number of pending completions reported by the Omni coordinator.",
    ),
    MetricFamily(
        "sglang_omni_request_states",
        "gauge",
        "Number of Omni requests by coordinator state.",
    ),
    MetricFamily(
        "sglang_omni_stage_present",
        "gauge",
        "Whether a stage is present in the Omni coordinator health snapshot.",
    ),
)
HTTP_REQUESTS_FAMILY = MetricFamily(
    "sglang_omni_http_requests_total",
    "counter",
    "Total number of HTTP requests handled by the Omni API server.",
)
HTTP_DURATION_FAMILY = MetricFamily(
    "sglang_omni_http_request_duration_seconds",
    "histogram",
    "HTTP request handler duration in seconds for the Omni API server.",
)
METRIC_FAMILIES = (*GAUGE_FAMILIES, HTTP_REQUESTS_FAMILY, HTTP_DURATION_FAMILY)
GAUGE_FAMILY_NAMES = frozenset(family.name for family in GAUGE_FAMILIES)


class OmniPrometheusMetrics:
    """Small app-local Prometheus text renderer for the Omni API server."""

    def __init__(
        self,
        model_name: str = "unknown",
        *,
        duration_buckets: tuple[float, ...] = DEFAULT_DURATION_BUCKETS,
    ) -> None:
        self.model_name = model_name or "unknown"
        self.duration_buckets = tuple(
            sorted(float(bucket) for bucket in duration_buckets)
        )
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._known_states: set[str] = set()
        self._known_stages: set[str] = set()
        self._http_requests_total: defaultdict[tuple[str, str, str], float] = defaultdict(
            float
        )
        self._http_duration_count: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._http_duration_sum: defaultdict[tuple[str, str], float] = defaultdict(float)
        self._http_duration_buckets: defaultdict[tuple[str, str, float], float] = (
            defaultdict(float)
        )

    def update_from_health(self, info: Mapping[str, Any]) -> None:
        model_labels = {"model_name": self.model_name}
        self._set_gauge(
            "sglang_omni_pipeline_running",
            model_labels,
            1.0 if info.get("running") else 0.0,
        )
        self._set_gauge(
            "sglang_omni_pipeline_pending_completions",
            model_labels,
            _float_or_zero(info.get("pending_completions")),
        )

        request_states = info.get("request_states") or {}
        if isinstance(request_states, Mapping):
            tracked_requests = 0.0
            for state in self._known_states:
                self._set_gauge(
                    "sglang_omni_request_states",
                    {"model_name": self.model_name, "state": state},
                    0.0,
                )
            for state, value in request_states.items():
                state_label = str(state)
                state_value = _float_or_zero(value)
                tracked_requests += state_value
                self._known_states.add(state_label)
                self._set_gauge(
                    "sglang_omni_request_states",
                    {"model_name": self.model_name, "state": state_label},
                    state_value,
                )
        else:
            tracked_requests = _float_or_zero(info.get("total_requests"))

        self._set_gauge(
            "sglang_omni_pipeline_tracked_requests",
            model_labels,
            tracked_requests,
        )

        stages = info.get("stages") or []
        if isinstance(stages, list):
            for stage in self._known_stages:
                self._set_gauge(
                    "sglang_omni_stage_present",
                    {"model_name": self.model_name, "stage": stage},
                    0.0,
                )
            for stage in stages:
                stage_label = str(stage)
                self._known_stages.add(stage_label)
                self._set_gauge(
                    "sglang_omni_stage_present",
                    {"model_name": self.model_name, "stage": stage_label},
                    1.0,
                )

    def observe_http_request(
        self,
        *,
        method: str,
        route: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        duration_seconds = max(float(duration_seconds), 0.0)
        self._http_requests_total[(method, route, status)] += 1.0
        self._http_duration_count[(method, route)] += 1.0
        self._http_duration_sum[(method, route)] += duration_seconds
        for bucket in self.duration_buckets:
            if duration_seconds <= bucket:
                self._http_duration_buckets[(method, route, bucket)] += 1.0

    def render(self) -> Response:
        return Response(
            content=self.render_text(),
            media_type=PROMETHEUS_CONTENT_TYPE,
        )

    def render_text(self) -> str:
        lines: list[str] = []
        for family in METRIC_FAMILIES:
            lines.extend(_render_family_header(family))
            lines.extend(self._render_family_samples(family.name))
        return "\n".join(lines) + "\n"

    def _set_gauge(self, name: str, labels: Mapping[str, str], value: float) -> None:
        self._gauges[(name, _label_tuple(labels))] = float(value)

    def _render_gauge_samples(self, name: str) -> list[str]:
        samples = [
            (labels, value)
            for (sample_name, labels), value in self._gauges.items()
            if sample_name == name
        ]
        return [
            _render_sample(name, dict(labels), value)
            for labels, value in sorted(samples, key=lambda item: item[0])
        ]

    def _render_family_samples(self, name: str) -> list[str]:
        if name in GAUGE_FAMILY_NAMES:
            return self._render_gauge_samples(name)
        if name == HTTP_REQUESTS_FAMILY.name:
            return self._render_http_request_samples()
        if name == HTTP_DURATION_FAMILY.name:
            return self._render_http_duration_samples()
        raise AssertionError(f"unknown metric family: {name}")

    def _render_http_request_samples(self) -> list[str]:
        lines = []
        for method, route, status in sorted(self._http_requests_total):
            lines.append(
                _render_sample(
                    "sglang_omni_http_requests_total",
                    {"method": method, "route": route, "status": status},
                    self._http_requests_total[(method, route, status)],
                )
            )
        return lines

    def _render_http_duration_samples(self) -> list[str]:
        lines = []
        for method, route in sorted(self._http_duration_count):
            base_labels = {"method": method, "route": route}
            for bucket in self.duration_buckets:
                lines.append(
                    _render_sample(
                        "sglang_omni_http_request_duration_seconds_bucket",
                        {**base_labels, "le": _format_bucket(bucket)},
                        self._http_duration_buckets[(method, route, bucket)],
                    )
                )
            lines.append(
                _render_sample(
                    "sglang_omni_http_request_duration_seconds_bucket",
                    {**base_labels, "le": "+Inf"},
                    self._http_duration_count[(method, route)],
                )
            )
            lines.append(
                _render_sample(
                    "sglang_omni_http_request_duration_seconds_count",
                    base_labels,
                    self._http_duration_count[(method, route)],
                )
            )
            lines.append(
                _render_sample(
                    "sglang_omni_http_request_duration_seconds_sum",
                    base_labels,
                    self._http_duration_sum[(method, route)],
                )
            )
        return lines


def install_metrics_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def prometheus_metrics_middleware(request: Request, call_next):
        metrics: OmniPrometheusMetrics | None = getattr(app.state, "omni_metrics", None)
        if metrics is None or request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        status = "500"
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            route = request.scope.get("route")
            route_path = getattr(route, "path", None) or "__unmatched__"
            metrics.observe_http_request(
                method=request.method,
                route=route_path,
                status=status,
                duration_seconds=time.perf_counter() - start,
            )


def _render_family_header(family: MetricFamily) -> list[str]:
    return [
        f"# HELP {family.name} {family.help_text}",
        f"# TYPE {family.name} {family.metric_type}",
    ]


def _render_sample(name: str, labels: Mapping[str, str], value: float) -> str:
    rendered_labels = ",".join(
        f'{key}="{_escape_label_value(value)}"'
        for key, value in sorted(labels.items())
    )
    suffix = f"{{{rendered_labels}}}" if rendered_labels else ""
    return f"{name}{suffix} {_format_value(value)}"


def _label_tuple(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, str(value)) for key, value in labels.items()))


def _escape_label_value(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _float_or_zero(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isfinite(result):
        return result
    return 0.0


def _format_bucket(value: float) -> str:
    return f"{value:g}"


def _format_value(value: float) -> str:
    value = float(value)
    if value == 0:
        return "0.0"
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:.17g}"
