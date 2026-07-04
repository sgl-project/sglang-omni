# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, Response
from fastapi.testclient import TestClient

from sglang_omni.serve.metrics import (
    OmniPrometheusMetrics,
    _render_sample,
    install_metrics_middleware,
)


class MetricsClient:
    def __init__(self, snapshots: list[dict[str, Any]] | None = None) -> None:
        self.snapshots = snapshots or [
            {
                "running": True,
                "stages": ["preprocess", "tts_engine"],
                "entry_stage": "preprocess",
                "total_requests": 3,
                "pending_completions": 2,
                "request_states": {"running": 2, "queued": 1},
            }
        ]
        self.health_calls = 0

    def health(self) -> dict[str, Any]:
        index = min(self.health_calls, len(self.snapshots) - 1)
        self.health_calls += 1
        return self.snapshots[index]


def create_metrics_app(*args: Any, **kwargs: Any):
    pytest.importorskip("torch")
    from sglang_omni.serve.openai_api import create_app

    return create_app(*args, **kwargs)


def test_metrics_endpoint_disabled_by_default_when_api_importable() -> None:
    client = TestClient(
        create_metrics_app(MetricsClient(), model_name="moss-tts-local-v15")
    )

    resp = client.get("/metrics")

    assert resp.status_code == 404


def test_metrics_endpoint_renders_snapshot_when_enabled_and_api_importable() -> None:
    client_impl = MetricsClient()
    client = TestClient(
        create_metrics_app(
            client_impl,
            model_name="moss-tts-local-v15",
            enable_metrics=True,
        )
    )

    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert (
        'sglang_omni_pipeline_running{model_name="moss-tts-local-v15"} 1.0'
        in resp.text
    )


def test_coordinator_snapshot_gauges_reset_missing_state_and_stage() -> None:
    metrics = OmniPrometheusMetrics(model_name="moss-tts-local-v15")

    metrics.update_from_health(
        {
            "running": True,
            "stages": ["preprocess", "tts_engine"],
            "pending_completions": 1,
            "request_states": {"running": 1, "queued": 2},
        }
    )
    metrics.update_from_health(
        {
            "running": True,
            "stages": ["preprocess"],
            "pending_completions": 0,
            "request_states": {"running": 1},
        }
    )
    text = metrics.render_text()

    assert (
        'sglang_omni_pipeline_running{model_name="moss-tts-local-v15"} 1.0'
        in text
    )
    assert (
        'sglang_omni_pipeline_tracked_requests{model_name="moss-tts-local-v15"} 1.0'
        in text
    )
    assert (
        'sglang_omni_pipeline_pending_completions{model_name="moss-tts-local-v15"} 0.0'
        in text
    )
    assert (
        'sglang_omni_request_states{model_name="moss-tts-local-v15",state="queued"} 0.0'
        in text
    )
    assert (
        'sglang_omni_stage_present{model_name="moss-tts-local-v15",stage="tts_engine"} 0.0'
        in text
    )


def test_label_keys_are_sorted_and_values_are_escaped() -> None:
    sample = _render_sample(
        "sample_metric",
        {"z_label": 'quote"slash\\newline\n', "a_label": "first"},
        1,
    )

    assert (
        sample
        == 'sample_metric{a_label="first",z_label="quote\\"slash\\\\newline\\n"} 1.0'
    )


def test_histogram_buckets_are_cumulative() -> None:
    metrics = OmniPrometheusMetrics(model_name="moss-tts-local-v15")

    metrics.observe_http_request(
        method="POST",
        route="/v1/audio/speech",
        status="200",
        duration_seconds=0.02,
    )
    text = metrics.render_text()

    assert _http_duration_bucket(le="0.01") + " 0.0" in text
    assert _http_duration_bucket(le="0.025") + " 1.0" in text
    assert _http_duration_bucket(le="+Inf") + " 1.0" in text
    assert (
        'sglang_omni_http_request_duration_seconds_count{method="POST",route="/v1/audio/speech"} 1.0'
        in text
    )


def test_http_middleware_records_statuses_routes_and_skips_metrics_scrape() -> None:
    app = FastAPI()
    app.state.omni_metrics = OmniPrometheusMetrics(model_name="moss-tts-local-v15")
    install_metrics_middleware(app)

    @app.get("/ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/bad")
    async def bad() -> Response:
        return Response(status_code=400)

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("boom")

    @app.get("/metrics")
    async def metrics() -> Response:
        return app.state.omni_metrics.render()

    client = TestClient(app, raise_server_exceptions=False)

    assert client.get("/ok").status_code == 200
    assert client.get("/bad").status_code == 400
    assert client.get("/boom").status_code == 500
    assert client.get("/missing").status_code == 404
    metrics_resp = client.get("/metrics")

    assert metrics_resp.status_code == 200
    text = metrics_resp.text
    assert _http_requests_total(route="/ok", status="200") + " 1.0" in text
    assert _http_requests_total(route="/bad", status="400") + " 1.0" in text
    assert _http_requests_total(route="/boom", status="500") + " 1.0" in text
    assert _http_requests_total(route="__unmatched__", status="404") + " 1.0" in text
    assert 'route="/metrics"' not in text


def _http_duration_bucket(*, le: str) -> str:
    return (
        'sglang_omni_http_request_duration_seconds_bucket{'
        f'le="{le}",method="POST",route="/v1/audio/speech"'
        "}"
    )


def _http_requests_total(*, route: str, status: str) -> str:
    return (
        'sglang_omni_http_requests_total{'
        f'method="GET",route="{route}",status="{status}"'
        "}"
    )


def test_metrics_instances_do_not_share_state() -> None:
    first = OmniPrometheusMetrics(model_name="first-model")
    second = OmniPrometheusMetrics(model_name="second-model")

    first.update_from_health(MetricsClient().health())
    second.update_from_health(MetricsClient().health())
    first_metrics = first.render_text()
    second_metrics = second.render_text()

    assert 'model_name="first-model"' in first_metrics
    assert 'model_name="second-model"' not in first_metrics
    assert 'model_name="second-model"' in second_metrics
    assert 'model_name="first-model"' not in second_metrics


def test_rendered_metrics_are_parseable_when_prometheus_client_is_available() -> None:
    prometheus_parser = pytest.importorskip("prometheus_client.parser")
    metrics = OmniPrometheusMetrics(model_name="moss-tts-local-v15")

    metrics.update_from_health(MetricsClient().health())
    metrics.observe_http_request(
        method="GET",
        route="/health",
        status="200",
        duration_seconds=0.01,
    )

    families = list(
        prometheus_parser.text_string_to_metric_families(metrics.render_text())
    )
    family_names = {family.name for family in families}
    sample_names = {
        sample.name
        for family in families
        for sample in getattr(family, "samples", ())
    }

    assert "sglang_omni_pipeline_running" in family_names
    assert "sglang_omni_http_requests_total" in sample_names
    assert "sglang_omni_http_request_duration_seconds_bucket" in sample_names
