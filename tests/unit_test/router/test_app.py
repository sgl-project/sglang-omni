from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from sglang_omni_router import proxy as proxy_module
from sglang_omni_router.app import _broadcast_admin_request, create_app
from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.selector import WorkerSelector
from sglang_omni_router.worker import build_workers


def _request_netloc(request: httpx.Request) -> str:
    return f"{request.url.host}:{request.url.port}"


def _router_config(
    policy: str = "round_robin",
    max_payload_size: int = 512 * 1024 * 1024,
    max_connections: int = 100,
    health_failure_threshold: int = 1,
    health_check_timeout_secs: int = 5,
    worker_configs: list[WorkerConfig] | None = None,
) -> RouterConfig:
    return RouterConfig(
        workers=worker_configs
        or [
            WorkerConfig(url="http://worker-a:8101"),
            WorkerConfig(url="http://worker-b:8102"),
        ],
        policy=policy,
        max_payload_size=max_payload_size,
        max_connections=max_connections,
        health_success_threshold=1,
        health_failure_threshold=health_failure_threshold,
        health_check_timeout_secs=health_check_timeout_secs,
    )


def _large_json_body(payload: dict[str, object]) -> bytes:
    return json.dumps(payload | {"padding": "x" * (1024 * 1024 + 128)}).encode()


def _request_without_content_length(chunks: list[bytes]) -> Request:
    messages = [
        {"type": "http.request", "body": chunk, "more_body": True}
        for chunk in chunks[:-1]
    ]
    messages.append(
        {
            "type": "http.request",
            "body": chunks[-1] if chunks else b"",
            "more_body": False,
        }
    )

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        },
        receive,
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
        assert health.json()["dead_workers"] == 0
        assert health.json()["unhealthy_workers"] == 1
        assert health.json()["routable_workers"] == 1

        workers = client.get("/workers").json()["workers"]
        assert [worker["health_state"] for worker in workers] == [
            "unhealthy",
            "healthy",
        ]
        assert "state" not in workers[0]

    health_status["worker-b:8102"] = 500
    app = create_app(_router_config(), client=async_client)
    with TestClient(app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["status"] == "not_ready"
        assert client.get("/health").status_code == 503


def test_health_checks_use_separate_client_from_data_plane_client() -> None:
    health_paths: list[str] = []
    data_paths: list[str] = []

    def data_handler(request: httpx.Request) -> httpx.Response:
        data_paths.append(request.url.path)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"data-plane client should not call {request.url.path}")

    def health_handler(request: httpx.Request) -> httpx.Response:
        health_paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        raise AssertionError(f"health client should not call {request.url.path}")

    data_client = httpx.AsyncClient(transport=httpx.MockTransport(data_handler))
    health_client = httpx.AsyncClient(transport=httpx.MockTransport(health_handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=data_client,
        health_client=health_client,
    )

    with TestClient(app) as client:
        ready = client.get("/ready")
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "messages": [{"role": "user"}]},
        )

    assert ready.status_code == 200
    assert response.status_code == 200
    assert health_paths == ["/health"]
    assert data_paths == ["/v1/chat/completions"]


def test_router_liveness_does_not_wait_for_worker_health_probe() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            await asyncio.sleep(60)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=async_client,
    )

    with TestClient(app) as client:
        live = client.get("/live")
        ready = client.get("/ready")

    assert live.status_code == 200
    assert ready.status_code == 503


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

        misspelled = client.post(
            "/workers",
            json={
                "url": "http://worker-d:8104",
                "capabilites": ["chat"],
            },
        )
        assert misspelled.status_code == 400
        assert "capabilites" in misspelled.json()["error"]["message"]
        assert client.get("/workers").json()["total_workers"] == 3

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


def test_worker_update_validation_failure_is_atomic() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=async_client,
    )

    with TestClient(app) as client:
        worker = app.state.workers[0]
        worker_id = worker.worker_id
        worker.consecutive_failures = 2
        worker.consecutive_successes = 3
        before = worker.to_dict()

        response = client.put(
            f"/workers/{worker_id}",
            json={
                "disabled": True,
                "is_dead": True,
                "model": "changed-model",
                "capabilities": [],
            },
        )

        assert response.status_code == 400
        assert worker.to_dict() == before


def test_manual_dead_worker_is_not_recovered_by_health_check() -> None:
    health_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal health_calls
        if request.url.path == "/health":
            health_calls += 1
            return httpx.Response(200, json={"status": "worker"}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(
            worker_configs=[WorkerConfig(url="http://worker-a:8101")],
        ),
        client=async_client,
    )

    with TestClient(app) as client:
        worker = app.state.workers[0]
        worker_id = worker.worker_id
        assert client.get("/ready").status_code == 200

        marked_dead = client.put(f"/workers/{worker_id}", json={"is_dead": True})
        assert marked_dead.status_code == 200
        assert marked_dead.json()["worker"]["health_state"] == "dead"
        assert client.get("/ready").status_code == 503

        calls_before_check = health_calls
        asyncio.run(app.state.health_checker.check_worker_health(worker))

        assert health_calls == calls_before_check
        assert worker.state == "dead"
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["dead_workers"] == 1
        assert health.json()["unhealthy_workers"] == 0


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
        for worker in app.state.workers:
            worker.active_requests = 7
        response = client.get("/v1/models?detail=1")

    assert response.status_code == 200
    assert model_requests == ["worker-b:8102"]
    assert model_queries == [b"detail=1"]
    assert [worker.active_requests for worker in app.state.workers] == [7, 7]
    assert response.json()["data"] == [
        {"id": "qwen3-omni", "object": "model", "created": 0}
    ]


def test_admin_routes_broadcast_to_live_workers_and_preserve_query() -> None:
    seen: list[tuple[str, bytes, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/weights_checker":
            seen.append((_request_netloc(request), request.url.query, request.content))
            return httpx.Response(
                200,
                json={"success": True, "worker": _request_netloc(request)},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.get("/weights_checker?action=checksum")

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert {item[0] for item in seen} == {"worker-a:8101", "worker-b:8102"}
    assert [item[1] for item in seen] == [b"action=checksum", b"action=checksum"]
    assert [item[2] for item in seen] == [b"", b""]


def test_model_info_broadcast_exposes_sglang_compatible_weight_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/model_info":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "weight_version": "v7",
                    "model_path": "/tmp/model-v7",
                    "load_format": "safetensors",
                    "stages": [
                        {
                            "stage": "decode",
                            "success": True,
                            "data": {
                                "weight_version": "v7",
                                "model_path": "/tmp/model-v7",
                                "load_format": "safetensors",
                            },
                        }
                    ],
                },
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.get("/model_info")

    body = response.json()
    assert response.status_code == 200
    assert body["weight_version"] == "v7"
    assert body["model_path"] == "/tmp/model-v7"
    assert body["load_format"] == "safetensors"
    assert len(body["workers"]) == 2


def test_model_info_broadcast_rejects_mixed_worker_weight_versions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/model_info":
            version = "v1" if _request_netloc(request) == "worker-a:8101" else "v2"
            return httpx.Response(
                200,
                json={"success": True, "weight_version": version},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.get("/model_info")

    body = response.json()["detail"]
    assert response.status_code == 409
    assert body["success"] is False
    assert set(body["mixed_state"]["weight_version"]) == {"v1", "v2"}


def test_admin_update_temporarily_disables_workers_and_restores_state() -> None:
    app_holder: dict[str, Any] = {}
    disabled_snapshots: list[tuple[bool, bool]] = []
    seen_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/pause_generation":
            workers = app_holder["app"].state.workers
            disabled_snapshots.append(tuple(worker.disabled for worker in workers))
            seen_bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"success": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)
    app_holder["app"] = app

    with TestClient(app) as client:
        app.state.workers[1].set_disabled(True)
        response = client.post("/pause_generation", json={"mode": "in_place"})

    assert response.status_code == 200
    assert disabled_snapshots == [(True, True), (True, True)]
    assert seen_bodies == [{"mode": "in_place"}, {"mode": "in_place"}]
    assert [worker.disabled for worker in app.state.workers] == [False, True]


def test_models_merge_queries_workers_concurrently_with_control_timeout() -> None:
    started_workers: list[str] = []
    model_timeouts: list[dict[str, float]] = []
    release_models: asyncio.Event | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal release_models
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/v1/models":
            if release_models is None:
                release_models = asyncio.Event()
            started_workers.append(_request_netloc(request))
            model_timeouts.append(request.extensions["timeout"])
            if len(started_workers) == 2:
                release_models.set()
            await asyncio.wait_for(release_models.wait(), timeout=1)
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": _request_netloc(request)}]},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(health_check_timeout_secs=2),
        client=async_client,
    )

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert set(started_workers) == {"worker-a:8101", "worker-b:8102"}
    assert {card["id"] for card in response.json()["data"]} == {
        "worker-a:8101",
        "worker-b:8102",
    }
    assert model_timeouts == [
        {"connect": 2, "read": 2, "write": 2, "pool": 2},
        {"connect": 2, "read": 2, "write": 2, "pool": 2},
    ]


def test_requested_model_routes_only_to_matching_model_worker() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/models":
            model_id = (
                "model-a" if _request_netloc(request) == "worker-a:8101" else "model-b"
            )
            return httpx.Response(
                200,
                json={"object": "list", "data": [{"id": model_id}]},
                request=request,
            )
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", model="model-a"),
        WorkerConfig(url="http://worker-b:8102", model="model-b"),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(worker_configs=worker_configs), client=async_client)

    with TestClient(app) as client:
        models = client.get("/v1/models")
        first = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user"}]},
        )
        second = client.post(
            "/v1/chat/completions",
            json={"model": "model-a", "messages": [{"role": "user"}]},
        )
        missing = client.post(
            "/v1/chat/completions",
            json={"model": "missing-model", "messages": [{"role": "user"}]},
        )

    assert models.status_code == 200
    assert {card["id"] for card in models.json()["data"]} == {"model-a", "model-b"}
    assert first.status_code == 200
    assert second.status_code == 200
    assert missing.status_code == 503
    assert seen_workers == ["worker-a:8101", "worker-a:8101"]


def test_large_body_uses_scanned_model_for_mixed_model_pool() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", model="model-a"),
        WorkerConfig(url="http://worker-b:8102", model="model-b"),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(worker_configs=worker_configs), client=async_client)
    body = _large_json_body(
        {
            "padding_first": "x" * (1024 * 1024 + 128),
            "model": "model-b",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


def test_large_body_model_hint_narrows_before_capability_ambiguity() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(
            url="http://worker-a:8101",
            model="model-a",
            capabilities={"chat"},
        ),
        WorkerConfig(
            url="http://worker-b:8102",
            model="model-b",
            capabilities={"chat", "video_input"},
        ),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(worker_configs=worker_configs), client=async_client)
    body = _large_json_body(
        {
            "model": "model-a",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={
                "content-type": "application/json",
                "x-sglang-omni-route-model": "model-a",
            },
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-a:8101"]


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


def test_round_robin_proxies_raw_bytes_and_alternates_workers() -> None:
    seen_bodies: list[bytes] = []
    seen_workers: list[str] = []

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
                headers={
                    "content-encoding": "identity",
                    "content-type": "application/json",
                    "date": "Sat, 16 May 2026 10:00:00 GMT",
                    "server": "upstream-server",
                },
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)
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
    assert "content-encoding" not in first.headers
    assert "Sat, 16 May 2026" not in first.headers.get("date", "")
    assert "upstream-server" not in first.headers.get("server", "")
    assert first.headers["x-sglang-omni-request-id"] == "req-1"
    assert second.headers["x-sglang-omni-request-id"] == "req-2"
    assert json.loads(seen_bodies[0]) == body
    assert seen_workers == ["worker-a:8101", "worker-b:8102"]


@pytest.mark.parametrize(
    ("headers", "body", "error_fragment"),
    [
        (
            {"x-sglang-omni-route-model": "model-b"},
            {"model": "model-a", "messages": [{"role": "user"}]},
            "x-sglang-omni-route-model",
        ),
        (
            {"x-sglang-omni-route-stream": "true"},
            {"model": "model-a", "messages": [{"role": "user"}]},
            "x-sglang-omni-route-stream",
        ),
        (
            {"x-sglang-omni-route-capabilities": "video_input"},
            {"model": "model-a", "messages": [{"role": "user"}]},
            "x-sglang-omni-route-capabilities",
        ),
    ],
)
def test_small_json_body_rejects_conflicting_route_headers(
    headers: dict[str, str],
    body: dict[str, object],
    error_fragment: str,
) -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", model="model-a"),
        WorkerConfig(
            url="http://worker-b:8102",
            model="model-b",
            capabilities={"chat", "streaming", "video_input"},
        ),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(worker_configs=worker_configs), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=body,
            headers=headers,
        )

    assert response.status_code == 400
    assert error_fragment in response.json()["error"]["message"]
    assert seen_paths == ["/health", "/health"]


def test_buffered_route_completion_log_includes_selection_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with caplog.at_level(logging.INFO, logger="sglang_omni_router.proxy"):
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"x-request-id": "buffered-log-1"},
                json={"model": "qwen3-omni", "messages": [{"role": "user"}]},
            )

    assert response.status_code == 200
    worker = app.state.workers[0]
    assert worker.routed_requests == 1
    assert worker.successful_requests == 1
    assert worker.failed_requests == 0
    route_logs = [
        record.getMessage()
        for record in caplog.records
        if "route_completed" in record.getMessage()
    ]
    assert len(route_logs) == 1
    assert "request_id=buffered-log-1" in route_logs[0]
    assert "worker=worker-a:8101" in route_logs[0]
    assert "stream=False" in route_logs[0]
    assert "capabilities=chat" in route_logs[0]
    assert "status_code=200" in route_logs[0]
    assert "outcome=completed" in route_logs[0]


def test_upstream_request_failure_returns_502_and_cleans_active_count() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            raise httpx.ConnectError("worker down", request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "messages": []},
        )

    assert response.status_code == 502
    assert all(worker.active_requests == 0 for worker in app.state.workers)
    worker = app.state.workers[0]
    assert worker.routed_requests == 1
    assert worker.successful_requests == 0
    assert worker.failed_requests == 1
    assert worker.state == "unhealthy"
    assert worker.last_error == "ConnectError"


def test_router_response_errors_do_not_refresh_worker_routability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    def fail_response_header_filter(
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, str]:
        raise RuntimeError("router response bug")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)
    monkeypatch.setattr(
        proxy_module,
        "filter_response_headers",
        fail_response_header_filter,
    )

    with TestClient(app) as client:
        with pytest.raises(RuntimeError, match="router response bug"):
            client.post(
                "/v1/chat/completions",
                json={"model": "qwen3-omni", "messages": []},
            )

    worker = app.state.workers[0]
    assert worker.state == "healthy"
    assert worker.consecutive_failures == 0
    assert worker.active_requests == 0


def test_retryable_upstream_status_refreshes_worker_routability() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            if _request_netloc(request) == "worker-a:8101":
                return httpx.Response(
                    502,
                    content=b"",
                    request=request,
                )
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "messages": []},
        )
        second = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "messages": []},
        )

    assert first.status_code == 502
    assert second.status_code == 200
    assert seen_workers == ["worker-a:8101", "worker-b:8102"]
    assert app.state.workers[0].state == "unhealthy"
    assert app.state.workers[0].last_error == "status=502"


def test_worker_validation_error_does_not_refresh_worker_routability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                422,
                json={"detail": [{"type": "missing", "loc": ["body", "messages"]}]},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=async_client,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "messages": []},
        )

    assert response.status_code == 422
    worker = app.state.workers[0]
    assert worker.state == "healthy"
    assert worker.consecutive_failures == 0


def test_streaming_upstream_error_cleans_active_count() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: start\n\n"
            raise httpx.ReadError("stream boom")

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
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        with pytest.raises(httpx.ReadError, match="stream boom"):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "qwen3-omni", "stream": True},
            ) as response:
                b"".join(response.iter_bytes())

    assert all(worker.active_requests == 0 for worker in app.state.workers)


def test_streaming_failure_records_single_worker_failure() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: start\n\n"
            raise httpx.ReadError("stream boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                502,
                stream=BrokenStream(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(
            health_failure_threshold=2,
            worker_configs=[WorkerConfig(url="http://worker-a:8101")],
        ),
        client=async_client,
    )

    with TestClient(app) as client:
        with pytest.raises(httpx.ReadError, match="stream boom"):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "qwen3-omni", "stream": True},
            ) as response:
                b"".join(response.iter_bytes())

    worker = app.state.workers[0]
    assert worker.routed_requests == 1
    assert worker.successful_requests == 0
    assert worker.failed_requests == 1
    assert worker.consecutive_failures == 1
    assert worker.state == "healthy"


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


def test_chat_message_part_capabilities_filter_mixed_worker_pool() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"chat", "image_input"}),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "file.jpg"}},
                        ],
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


def test_large_chat_body_uses_unique_capability_superset_without_route_header() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"chat", "video_input"}),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )
    body = _large_json_body(
        {
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "describe"}],
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


def test_large_chat_body_requires_capability_header_for_ambiguous_worker_pool() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat", "video_input"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"chat", "audio_input"}),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )
    body = _large_json_body(
        {
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "describe"}],
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert "x-sglang-omni-route-capabilities" in response.json()["error"]["message"]
    assert seen_workers == []


def test_large_chat_body_routes_homogeneous_pool_without_route_headers() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101"),
        WorkerConfig(url="http://worker-b:8102"),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )
    body = _large_json_body(
        {
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "describe"}],
            "videos": ["sample"],
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-a:8101"]


@pytest.mark.parametrize(
    ("payload_field", "capability"),
    [
        ("videos", "video_input"),
        ("audios", "audio_input"),
    ],
)
def test_large_chat_body_preserves_modality_capability_routing(
    payload_field: str,
    capability: str,
) -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat"}),
        WorkerConfig(
            url="http://worker-b:8102",
            capabilities={"chat", capability},
        ),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )
    body = _large_json_body(
        {
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "describe"}],
            payload_field: ["sample"],
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={
                "content-type": "application/json",
                "x-sglang-omni-route-capabilities": capability,
            },
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


def test_large_route_capability_hint_is_not_forwarded_to_worker() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            assert "x-sglang-omni-route-capabilities" not in request.headers
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"chat", "video_input"}),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )
    body = _large_json_body(
        {
            "padding_first": "x" * (1024 * 1024 + 128),
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "describe"}],
            "videos": ["sample"],
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={
                "content-type": "application/json",
                "x-sglang-omni-route-capabilities": "video_input",
            },
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


def test_large_streaming_chat_body_preserves_sse_routing() -> None:
    seen_workers: list[str] = []
    chunks = b"data: one\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            seen_workers.append(_request_netloc(request))
            if _request_netloc(request) == "worker-a:8101":
                return httpx.Response(200, json={"wrong": True}, request=request)
            return httpx.Response(
                200,
                content=chunks,
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"chat", "streaming"}),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )
    body = _large_json_body(
        {
            "model": "qwen3-omni",
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        }
    )

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        ) as response:
            stream_body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert seen_workers == ["worker-b:8102"]
    assert stream_body == chunks


@pytest.mark.parametrize(
    "body",
    [
        b'{"model":"qwen3-omni","padding":"\\x"}',
        b'{"model":"qwen3-omni","bad":01}',
        b'["not", "an", "object"]',
    ],
)
def test_route_metadata_rejects_invalid_json(body: bytes) -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert seen_paths == ["/health", "/health"]


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


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "qwen3-omni", "input": "hello", "ref_audio": "voice.wav"},
        {
            "model": "qwen3-omni",
            "input": "hello",
            "references": [{"audio_path": "voice.wav", "text": "hello"}],
        },
    ],
)
def test_speech_reference_audio_requires_audio_input_capability(
    payload: dict[str, object],
) -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/speech":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(
                200,
                content=b"audio",
                headers={"content-type": "audio/wav"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"speech"}),
        WorkerConfig(
            url="http://worker-b:8102",
            capabilities={"speech", "audio_input"},
        ),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=async_client,
    )

    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


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


def test_streaming_route_completion_log_includes_stream_lifetime(
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    with caplog.at_level(logging.INFO, logger="sglang_omni_router.proxy"):
        with TestClient(app) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"x-request-id": "stream-log-1"},
                json={"model": "qwen3-omni", "stream": True},
            ) as response:
                body = b"".join(response.iter_bytes())

    assert response.status_code == 200
    assert body == chunks
    route_logs = [
        record.getMessage()
        for record in caplog.records
        if "route_completed" in record.getMessage()
    ]
    assert len(route_logs) == 1
    assert "request_id=stream-log-1" in route_logs[0]
    assert "stream=True" in route_logs[0]
    assert "capabilities=chat,streaming" in route_logs[0]
    assert "status_code=200" in route_logs[0]
    assert "outcome=completed" in route_logs[0]


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


def test_payload_without_content_length_is_rejected_while_streaming_body() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        raise AssertionError("over-limit request should not reach a worker")

    config = _router_config(max_payload_size=8)
    workers = build_workers(config.workers)
    proxy = proxy_module.ProxyHandler(
        config=config,
        workers=workers,
        selector=WorkerSelector(config.policy),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    request = _request_without_content_length([b'{"model"', b':"qwen3-omni"}'])

    response = asyncio.run(proxy.forward_model_request(request, "/v1/chat/completions"))

    assert response.status_code == 413
    assert seen_paths == []


# ---------------------------------------------------------------------------
# Admin auth tests - router
# ---------------------------------------------------------------------------

_ROUTER_ADMIN_PATHS = [
    ("GET", "/model_info"),
    ("POST", "/model_info"),
    ("POST", "/pause_generation"),
    ("POST", "/continue_generation"),
    ("POST", "/update_weights_from_disk"),
    ("POST", "/update_weights_from_tensor"),
    ("POST", "/update_weights_from_distributed"),
    ("GET", "/weights_checker"),
    ("POST", "/weights_checker"),
]

_ROUTER_ADMIN_API_KEY = "router-secret"


def _admin_headers(
    key: str = _ROUTER_ADMIN_API_KEY,
    *,
    scheme: str = "Bearer",
) -> dict[str, str]:
    return {"Authorization": f"{scheme} {key}"}


def _admin_router_app(admin_api_key: str | None = None) -> FastAPI:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path in ("/model_info", "/weights_checker"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "message": "ok",
                    "results": [],
                    "weight_version": "v1",
                    "model_path": "/tmp/m",
                    "load_format": "safetensors",
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={"success": True, "message": "ok", "results": []},
            request=request,
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return create_app(
        _router_config(),
        client=async_client,
        admin_api_key=admin_api_key,
    )


def test_router_admin_routes_open_without_key() -> None:
    """Admin routes are accessible with no auth header when no key is configured."""
    app = _admin_router_app(admin_api_key=None)
    with TestClient(app) as client:
        resp = client.get("/model_info")
        assert resp.status_code == 200


def test_router_admin_routes_require_bearer_when_key_set() -> None:
    app = _admin_router_app(admin_api_key=_ROUTER_ADMIN_API_KEY)
    with TestClient(app) as client:
        for method, path in _ROUTER_ADMIN_PATHS:
            resp = client.request(method, path, json={})
            assert (
                resp.status_code == 401
            ), f"{method} {path} expected 401, got {resp.status_code}"
            assert "WWW-Authenticate" in resp.headers


def test_router_admin_routes_reject_wrong_token() -> None:
    app = _admin_router_app(admin_api_key=_ROUTER_ADMIN_API_KEY)
    with TestClient(app) as client:
        resp = client.get("/model_info", headers=_admin_headers("wrong"))
        assert resp.status_code == 403


def test_router_admin_routes_accept_correct_token() -> None:
    app = _admin_router_app(admin_api_key=_ROUTER_ADMIN_API_KEY)
    with TestClient(app) as client:
        resp = client.get("/model_info", headers=_admin_headers(scheme="bearer"))
        assert resp.status_code == 200


def test_router_admin_env_key(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_ADMIN_KEY", "env-router-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "message": "ok",
                "results": [],
                "weight_version": None,
                "model_path": None,
                "load_format": None,
            },
            request=request,
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        assert client.get("/model_info").status_code == 401
        resp = client.get("/model_info", headers=_admin_headers("env-router-key"))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Router stub endpoint 501 tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/update_weights_from_tensor", {}),
        (
            "/update_weights_from_distributed",
            {"names": [], "dtypes": [], "shapes": []},
        ),
    ],
)
def test_router_unimplemented_weight_update_endpoints_return_501(
    path: str,
    payload: dict[str, Any],
) -> None:
    app = _admin_router_app()
    with TestClient(app) as client:
        resp = client.post(path, json=payload)
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "not_implemented"


# ---------------------------------------------------------------------------
# Router admin_update_lock timeout test
# ---------------------------------------------------------------------------


def test_router_admin_update_lock_timeout_returns_503(monkeypatch) -> None:
    """If the lock is held beyond timeout, the request returns 503."""

    async def _run():
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/health":
                return httpx.Response(200, json={"status": "healthy"}, request=request)
            return httpx.Response(
                200,
                json={"success": True, "message": "ok", "results": []},
                request=request,
            )

        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app = create_app(_router_config(), client=async_client)

        # Simulate a held lock by acquiring it before the request
        async with app.router.lifespan_context(app):
            lock = app.state.admin_update_lock
            await lock.acquire()
            monkeypatch.setattr(
                "sglang_omni_router.app._ADMIN_UPDATE_LOCK_TIMEOUT_S",
                0.05,
            )
            try:
                scope = {
                    "type": "http",
                    "method": "POST",
                    "path": "/update_weights_from_disk",
                    "headers": [(b"content-type", b"application/json")],
                    "query_string": b"",
                    "scheme": "http",
                    "server": ("testserver", 80),
                    "client": ("testclient", 50000),
                }

                async def receive():
                    return {"type": "http.request", "body": b"{}", "more_body": False}

                fake_request = Request(scope, receive)
                result = await _broadcast_admin_request(
                    app, fake_request, "/update_weights_from_disk"
                )
                return result
            finally:
                lock.release()

    result = asyncio.run(_run())
    assert result.status_code == 503
    body = json.loads(result.body)
    assert "lock" in body["error"]["message"].lower()
