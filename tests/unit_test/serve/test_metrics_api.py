# SPDX-License-Identifier: Apache-2.0
from fastapi.testclient import TestClient

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
