# SPDX-License-Identifier: Apache-2.0
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.serve.openai_api import create_app


class MetricsClient:
    async def admin(self, action, timeout_s):
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


def test_metrics_are_opt_in():
    app = create_app(MetricsClient(), model_name="test")
    assert TestClient(app).get("/metrics").status_code == 404


def test_scrape_and_registry_isolation():
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


def test_failed_scheduler_scrape_is_not_successful():
    class FailedClient:
        async def admin(self, *args, **kwargs):
            raise TimeoutError()

    app = create_app(FailedClient(), enable_metrics=True)
    assert TestClient(app).get("/metrics").status_code == 503


def test_backend_histogram_is_visible_on_http_scrape():
    class BackendClient:
        async def admin(self, *args, **kwargs):
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


def test_unmatched_paths_do_not_become_labels():
    app = create_app(MetricsClient(), enable_metrics=True)
    client = TestClient(app)
    assert client.get("/unknown/private-id").status_code == 404
    text = client.get("/metrics").text
    assert 'route="unmatched"' in text
    assert "private-id" not in text


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("fail_after_image", [False, True])
def test_images_finish_iteration_before_recording_success(editing, fail_after_image):
    class ImageClient(Client, MetricsClient):
        def __init__(self):
            super().__init__(None)
            self.exhausted = False

        async def _generate_uninstrumented(self, request, request_id):
            yield GenerateChunk(
                request_id=request_id,
                stage_name="generate",
                modality="image",
                image_b64="aW1hZ2U=",
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
            assert metrics.registry.get_sample_value(
                "sglang_omni:failed_requests_total", {"model_name": "test"}
            ) == 1
            assert metrics.registry.get_sample_value(
                "sglang:e2e_request_latency_seconds_count", labels
            ) is None
        else:
            assert backend.exhausted
            assert response.json()["data"] == [{"b64_json": "aW1hZ2U="}]
            assert metrics.registry.get_sample_value(
                "sglang:e2e_request_latency_seconds_count", labels
            ) == 1
            assert metrics.registry.get_sample_value(
                "sglang:num_requests_total", labels
            ) == 1
            assert metrics.registry.get_sample_value(
                "sglang_omni:generated_images_total",
                {"model_name": "test", "stage": "generate"},
            ) == 1
