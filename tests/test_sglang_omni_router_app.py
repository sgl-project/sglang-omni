from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from sglang_omni_router.app import create_app
from sglang_omni_router.config import RouterConfig, WorkerConfig


def _request_netloc(request: httpx.Request) -> str:
    return f"{request.url.host}:{request.url.port}"


def _router_config(
    route_log_path: Path | None = None,
    *,
    policy: str = "round_robin",
    max_payload_size: int = 512 * 1024 * 1024,
    worker_configs: list[WorkerConfig] | None = None,
) -> RouterConfig:
    return RouterConfig(
        worker_urls=worker_configs
        or [
            WorkerConfig(url="http://worker-a:8101"),
            WorkerConfig(url="http://worker-b:8102"),
        ],
        policy=policy,
        max_payload_size=max_payload_size,
        health_success_threshold=1,
        health_failure_threshold=1,
        route_log_path=str(route_log_path) if route_log_path else None,
    )


def test_health_surfaces_distinguish_router_readiness_from_pool_health() -> None:
    health_status = {
        "worker-a:8101": 500,
        "worker-b:8102": 200,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                health_status[_request_netloc(request)],
                json={"status": "worker"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        assert client.get("/live").status_code == 200
        assert client.get("/ready").status_code == 200

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["healthy_workers"] == 1
        assert health.json()["dead_workers"] == 1
        assert health.json()["routable_workers"] == 1

        workers = client.get("/workers").json()["workers"]
        assert [worker["health_state"] for worker in workers] == ["dead", "healthy"]

    health_status["worker-b:8102"] = 500
    app = create_app(_router_config(), client=async_client)
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/health").status_code == 503


def test_worker_crud_updates_runtime_pool_and_validates_payloads() -> None:
    health_status = {
        "worker-a:8101": 200,
        "worker-b:8102": 200,
        "worker-c:8103": 200,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                health_status[_request_netloc(request)],
                json={"status": "worker"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        created = client.post(
            "/workers",
            json={
                "url": "http://worker-c:8103",
                "model": "qwen3-omni",
                "capabilities": ["chat", "streaming"],
            },
        )
        assert created.status_code == 200
        worker = created.json()["worker"]
        assert worker["health_state"] == "healthy"
        assert worker["capabilities"] == ["chat", "streaming"]

        duplicate = client.post("/workers", json={"url": "http://worker-c:8103"})
        assert duplicate.status_code == 409

        worker_id = worker["worker_id"]
        disabled = client.put(f"/workers/{worker_id}", json={"disabled": True})
        assert disabled.status_code == 200
        assert disabled.json()["worker"]["disabled"] is True
        assert disabled.json()["worker"]["routable"] is False

        marked_dead = client.put(f"/workers/{worker_id}", json={"is_dead": True})
        assert marked_dead.status_code == 200
        assert marked_dead.json()["worker"]["health_state"] == "dead"

        recovered = client.put(f"/workers/{worker_id}", json={"is_dead": False})
        assert recovered.status_code == 200
        assert recovered.json()["worker"]["health_state"] == "healthy"
        assert recovered.json()["worker"]["disabled"] is True

        unsupported = client.put(f"/workers/{worker_id}", json={"sleeping": True})
        assert unsupported.status_code == 400

        deleted = client.delete(f"/workers/{worker_id}")
        assert deleted.status_code == 200
        assert client.get(f"/workers/{worker_id}").status_code == 404


def test_models_merge_queries_only_healthy_workers_and_deduplicates() -> None:
    model_requests: list[str] = []
    model_queries: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            status = 500 if _request_netloc(request) == "worker-a:8101" else 200
            return httpx.Response(status, json={"status": "worker"}, request=request)
        if request.url.path == "/v1/models":
            model_requests.append(_request_netloc(request))
            model_queries.append(request.url.query)
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "qwen3-omni", "object": "model", "created": 0},
                    ],
                },
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.get("/v1/models?detail=1")

    assert response.status_code == 200
    assert model_requests == ["worker-b:8102"]
    assert model_queries == [b"detail=1"]
    assert response.json()["data"] == [
        {"id": "qwen3-omni", "object": "model", "created": 0}
    ]


def test_models_merge_reports_per_worker_failures_when_all_routable_workers_fail() -> (
    None
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/v1/models":
            if _request_netloc(request) == "worker-a:8101":
                return httpx.Response(500, json={"error": "boom"}, request=request)
            return httpx.Response(
                200,
                json={"object": "list", "data": {"not": "a list"}},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["message"] == "failed to fetch models from workers"
    assert error["details"] == {
        "http://worker-a:8101": "status=500",
        "http://worker-b:8102": "invalid models payload",
    }


def test_round_robin_proxies_raw_bytes_logs_route_and_alternates_workers(
    tmp_path: Path,
) -> None:
    seen_bodies: list[bytes] = []
    seen_workers: list[str] = []
    route_log_path = tmp_path / "routes.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            assert request.url.query == b"trace=abc"
            seen_bodies.append(request.content)
            seen_workers.append(_request_netloc(request))
            return httpx.Response(
                200,
                content=b'{"ok": true}',
                headers={"content-type": "application/json"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(route_log_path), client=async_client)
    body = {
        "model": "qwen3-omni",
        "request_id": "req-1",
        "messages": [{"role": "user", "content": "hi"}],
        "stage_params": {"kept": True},
    }

    with TestClient(app) as client:
        first = client.post("/v1/chat/completions?trace=abc", json=body)
        second = client.post(
            "/v1/chat/completions?trace=abc", json=body | {"request_id": "req-2"}
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["x-sglang-omni-request-id"] == "req-1"
    assert second.headers["x-sglang-omni-request-id"] == "req-2"
    assert json.loads(seen_bodies[0]) == body
    assert seen_workers == ["worker-a:8101", "worker-b:8102"]

    routes = [json.loads(line) for line in route_log_path.read_text().splitlines()]
    assert [route["worker_url"] for route in routes] == [
        "http://worker-a:8101",
        "http://worker-b:8102",
    ]
    assert {route["policy"] for route in routes} == {"round_robin"}
    assert {tuple(route["required_capabilities"]) for route in routes} == {("chat",)}
    assert {route["worker_health_state"] for route in routes} == {"healthy"}
    assert {route["worker_disabled"] for route in routes} == {False}
    assert {route["worker_routable"] for route in routes} == {True}
    assert all(route["completed"] is True for route in routes)


def test_route_log_write_failure_does_not_fail_proxy(tmp_path: Path) -> None:
    route_log_path = tmp_path / "routes" / "routes.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(route_log_path), client=async_client)
    route_log_path.parent.rmdir()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "messages": []},
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert all(worker.active_requests == 0 for worker in app.state.workers)


def test_upstream_request_failure_logs_and_cleans_active_count(
    tmp_path: Path,
) -> None:
    route_log_path = tmp_path / "routes.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            raise httpx.ConnectError("worker down", request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(route_log_path), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "messages": []},
        )

    assert response.status_code == 502
    assert all(worker.active_requests == 0 for worker in app.state.workers)
    route = json.loads(route_log_path.read_text().splitlines()[0])
    assert route["event"] == "request_complete"
    assert route["completed"] is False
    assert route["error_type"] == "ConnectError"


def test_streaming_upstream_error_logs_and_cleans_active_count(
    tmp_path: Path,
) -> None:
    route_log_path = tmp_path / "routes.jsonl"

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: start\n\n"
            raise RuntimeError("stream boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                stream=BrokenStream(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(route_log_path), client=async_client)

    with TestClient(app) as client:
        with pytest.raises(RuntimeError, match="stream boom"):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "qwen3-omni", "stream": True},
            ) as response:
                b"".join(response.iter_bytes())

    assert all(worker.active_requests == 0 for worker in app.state.workers)
    route = json.loads(route_log_path.read_text().splitlines()[0])
    assert route["event"] == "stream_error"
    assert route["completed"] is False
    assert route["error_type"] == "RuntimeError"


def test_least_request_avoids_worker_with_active_stream_load() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                content=b"audio",
                headers={"content-type": "audio/wav"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(policy="least_request"), client=async_client)

    with TestClient(app) as client:
        workers = app.state.workers
        workers[0].active_requests = 10
        response = client.post("/v1/audio/speech", json={"model": "qwen3-omni"})

    assert response.status_code == 200
    assert response.headers["x-sglang-omni-worker"].endswith("worker-b%3A8102")


def test_chat_modality_capabilities_filter_mixed_worker_pool() -> None:
    seen_bodies: list[bytes] = []
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_bodies.append(request.content)
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(
            url="http://worker-a:8101",
            capabilities={"chat", "streaming", "image_input"},
        ),
        WorkerConfig(url="http://worker-b:8102"),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )
    body = {
        "model": "qwen3-omni",
        "request_id": "req-mm",
        "messages": [{"role": "user", "content": "describe"}],
        "audios": ["audio.wav"],
        "videos": ["clip.mp4"],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "stage_sampling": {"thinker": {"temperature": 0.7}},
        "stage_params": {"preprocessor": {"video_fps": 1.0}},
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]
    assert json.loads(seen_bodies[0]) == body


def test_speech_stream_requires_speech_and_streaming_capabilities() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/speech":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(
                200,
                content=b"data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat", "streaming"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"speech", "streaming"}),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/audio/speech",
            json={"model": "qwen3-omni", "input": "hello", "stream": True},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]
    assert body == b"data: [DONE]\n\n"


def test_streaming_chat_relays_exact_sse_bytes() -> None:
    chunks = b"data: one\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                content=chunks,
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "stream": True},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert body == chunks


def test_payload_too_large_is_rejected_before_worker_selection() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(max_payload_size=4), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=b"too-large",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert seen_paths == ["/health", "/health"]
