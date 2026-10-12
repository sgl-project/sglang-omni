# SPDX-License-Identifier: Apache-2.0
import base64
import importlib
import io
import sys
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import sglang_omni.serve.metrics as metrics_module
import sglang_omni.serve.openai_api as api_module
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.scheduling.metrics_snapshot import MetricsAdminResponse
from sglang_omni.serve.metrics import RequestMetrics
from sglang_omni.serve.openai_api import create_app


class MetricsClient:
    request_metrics: RequestMetrics

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
        async def admin(self, action: str, timeout_s: float) -> MetricsAdminResponse:
            raise TimeoutError()

    app = create_app(FailedClient(), enable_metrics=True)
    assert TestClient(app).get("/metrics").status_code == 503


def test_backend_histogram_is_visible_on_http_scrape() -> None:
    class BackendClient:
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


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("fail_after_image", [False, True])
def test_images_finish_iteration_before_recording_success(
    editing: bool, fail_after_image: bool
) -> None:
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    encoded = base64.b64encode(image.getvalue()).decode("ascii")

    class ImageClient(Client, MetricsClient):
        def __init__(self) -> None:
            super().__init__(None)
            self.exhausted = False

        async def _generate_uninstrumented(
            self, request: GenerateRequest, request_id: str
        ) -> AsyncGenerator[GenerateChunk, None]:
            yield GenerateChunk(
                request_id=request_id,
                stage_name="generate",
                modality="image",
                image_b64=encoded,
                image_usage={
                    "input_tokens": 3,
                    "total_tokens": 3,
                    "input_tokens_details": {
                        "text_tokens": 3,
                        "image_tokens": 4 if editing else 0,
                    },
                },
                finish_reason="stop",
            )
            if fail_after_image:
                raise RuntimeError("Image backend failed during completion")
            self.exhausted = True

    backend = ImageClient()
    app = create_app(
        backend, model_name="test", architectures=["NEOChatModel"], enable_metrics=True
    )
    with TestClient(app) as client:
        if editing:
            image = io.BytesIO()
            Image.new("RGB", (32, 32)).save(image, format="PNG")
            response = client.post(
                "/v1/images/edits",
                data={"model": "test", "prompt": "Make the background blue"},
                files={"image": ("reference.png", image.getvalue(), "image/png")},
            )
        else:
            response = client.post(
                "/v1/images/generations",
                json={"model": "test", "prompt": "A red mug", "size": "512x512"},
            )
        assert response.status_code == (500 if fail_after_image else 200)
        metrics = app.state.request_metrics
        labels = {"model_name": "test", "is_streaming": "false"}
        assert (
            metrics.registry.get_sample_value(
                "sglang:num_aborted_requests_total", {"model_name": "test"}
            )
            or 0
        ) == 0
        if fail_after_image:
            assert (
                metrics.registry.get_sample_value(
                    "sglang_omni:failed_requests_total", {"model_name": "test"}
                )
                == 1
            )
            assert (
                metrics.registry.get_sample_value(
                    "sglang:e2e_request_latency_seconds_count", labels
                )
                is None
            )
        else:
            assert backend.exhausted
            assert response.json()["data"] == [
                {"b64_json": encoded, "output_format": "png"}
            ]
            assert response.json()["usage"]["input_tokens"] == 3
            assert response.json()["usage"]["input_tokens_details"]["image_tokens"] == (
                4 if editing else 0
            )
            assert (
                metrics.registry.get_sample_value(
                    "sglang:e2e_request_latency_seconds_count", labels
                )
                == 1
            )
            assert (
                metrics.registry.get_sample_value("sglang:num_requests_total", labels)
                == 1
            )
            assert (
                metrics.registry.get_sample_value(
                    "sglang_omni:generated_images_total",
                    {"model_name": "test", "stage": "generate"},
                )
                == 1
            )
            assert (
                metrics.registry.get_sample_value("sglang:prompt_tokens_total", labels)
                is None
            )
            assert (
                metrics.registry.get_sample_value(
                    "sglang:generation_tokens_total", labels
                )
                is None
            )


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
