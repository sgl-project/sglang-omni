# SPDX-License-Identifier: Apache-2.0
import importlib
import sys

import pytest
from fastapi.testclient import TestClient

import sglang_omni.serve.metrics_registry as metrics_module
import sglang_omni.serve.openai_api as api_module
from sglang_omni.metrics.types import ORPHANED_STATE_TTL_NS
from sglang_omni.scheduling.metrics_snapshot import MetricsAdminResponse
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.request_metrics import RequestMetrics


class MetricsClient:
    request_metrics: RequestMetrics
    coordinator: None = None

    async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
        return {
            "success": True,
            "results": [
                {
                    "success": True,
                    "stage": "generate",
                    "data": {
                        "num_running_reqs": 1,
                        "num_queue_reqs": 2,
                    },
                }
            ],
        }


def test_metrics_are_opt_in() -> None:
    app = create_app(MetricsClient(), model_name="test")
    assert TestClient(app).get("/metrics").status_code == 404


def test_scrape_and_registry_isolation() -> None:
    for _ in range(2):
        client = MetricsClient()
        app = create_app(client, model_name="test", enable_metrics=True)
        response = TestClient(app).get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert (
            'sglang:num_running_reqs{model_name="test",stage="generate"} 1.0'
            in response.text
        )
        assert (
            'sglang:num_queue_reqs{model_name="test",stage="generate"} 2.0'
            in response.text
        )
        assert 'route="/metrics"' not in response.text
        assert client.request_metrics is app.state.request_metrics


def test_failed_scheduler_scrape_is_not_successful() -> None:
    class FailedClient:
        coordinator: None = None

        async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
            raise TimeoutError()

    app = create_app(FailedClient(), enable_metrics=True)
    assert TestClient(app).get("/metrics").status_code == 503


def test_audio_metrics_share_the_existing_scrape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio observations must survive backend refresh in the same registry."""
    client = MetricsClient()
    app = create_app(client, model_name="test", enable_metrics=True)
    audio_metrics = client.request_metrics.audio_metrics
    audio_metrics.record("audio_response_start", "completed", 1)
    audio_metrics.record(
        "audio_response_done", "completed", 2, {"audio_duration_s": 0.5}
    )
    audio_metrics.record("audio_response_start", "orphaned", 1)
    audio_metrics.record("audio_chunk", "orphaned", 2, {"duration_s": 0.1})
    monkeypatch.setattr(
        api_module.time, "perf_counter_ns", lambda: ORPHANED_STATE_TTL_NS + 3
    )

    response = TestClient(app).get("/metrics")

    assert response.status_code == 200
    assert "sglang_omni:audio_duration_s_sum 0.5" in response.text
    assert (
        'sglang:num_running_reqs{model_name="test",stage="generate"} 1.0'
        in response.text
    )
    audio_metrics.record("audio_response_done", "orphaned", ORPHANED_STATE_TTL_NS + 4)
    assert (
        client.request_metrics.registry.get_sample_value(
            "sglang_omni:audio_duration_s_count"
        )
        == 1
    )


def test_backend_histogram_is_visible_on_http_scrape() -> None:
    class BackendClient:
        coordinator: None = None

        async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
            return {
                "success": True,
                "results": [
                    {
                        "success": True,
                        "stage": "decode",
                        "data": {
                            "num_running_reqs": 0,
                            "num_queue_reqs": 0,
                            "backend_metrics": [
                                {
                                    "name": "sglang:per_stage_req_latency_seconds",
                                    "documentation": "Backend latency",
                                    "type": "histogram",
                                    "samples": [
                                        {
                                            "name": "sglang:per_stage_req_latency_seconds_bucket",
                                            "labels": {
                                                "stage": "prefill_waiting",
                                                "le": "+Inf",
                                            },
                                            "value": 2,
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ],
            }

    app = create_app(BackendClient(), model_name="test", enable_metrics=True)
    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert 'omni_stage="decode"' in response.text
    assert 'stage="prefill_waiting"' in response.text
    assert "sglang:per_stage_req_latency_seconds_bucket{" in response.text


def test_unmatched_paths_do_not_become_labels() -> None:
    app = create_app(MetricsClient(), enable_metrics=True)
    client = TestClient(app)
    assert client.get("/unknown/private-id").status_code == 404
    text = client.get("/metrics").text
    assert 'route="unmatched"' in text
    assert "private-id" not in text


@pytest.mark.parametrize("enabled", [False, True])
def test_metrics_dependency_is_required_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
) -> None:
    try:
        with monkeypatch.context() as isolated:
            isolated.setitem(sys.modules, "prometheus_client", None)
            importlib.reload(metrics_module)
            importlib.reload(api_module)
            if enabled:
                with pytest.raises(RuntimeError, match=r"sglang-omni\[metrics\]"):
                    api_module.create_app(MetricsClient(), enable_metrics=True)
            else:
                app = api_module.create_app(MetricsClient())
                assert TestClient(app).get("/metrics").status_code == 404
    finally:
        importlib.reload(metrics_module)
        importlib.reload(api_module)


def test_metrics_import_does_not_hide_broken_transitive_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        with monkeypatch.context() as isolated:
            isolated.setitem(sys.modules, "prometheus_client.core", None)
            with pytest.raises(ModuleNotFoundError) as raised:
                importlib.reload(metrics_module)
            assert raised.value.name == "prometheus_client.core"
    finally:
        importlib.reload(metrics_module)
