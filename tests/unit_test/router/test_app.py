from __future__ import annotations

import asyncio
import gc
import json
import logging
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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
    max_connections: int | None = None,
    max_inflight: int | None = None,
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
        max_inflight=max_inflight,
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


def test_generate_is_forwarded_opaquely_to_a_worker() -> None:
    data_paths: list[str] = []

    def data_handler(request: httpx.Request) -> httpx.Response:
        data_paths.append(request.url.path)
        if request.url.path == "/generate":
            return httpx.Response(
                200, json={"text": "hi", "meta_info": {}}, request=request
            )
        raise AssertionError(f"data-plane client should not call {request.url.path}")

    def health_handler(request: httpx.Request) -> httpx.Response:
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
            "/generate",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "sampling_params": {},
            },
        )

    assert ready.status_code == 200
    assert response.status_code == 200
    assert response.json()["text"] == "hi"
    assert data_paths == ["/generate"]


def test_generate_audio_output_routes_to_audio_worker() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/generate":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(
                200, json={"text": "hi", "meta_info": {}}, request=request
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"chat", "audio_output"}),
    ]
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(worker_configs=worker_configs), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={
                "messages": [{"role": "user", "content": "say hi"}],
                "sampling_params": {},
                "output_modalities": ["audio"],
            },
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


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
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "stream": True},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert b"upstream_error" in body
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
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "stream": True},
        ) as response:
            body = b"".join(response.iter_bytes())

    assert b"upstream_error" in body
    worker = app.state.workers[0]
    assert worker.routed_requests == 1
    assert worker.successful_requests == 0
    assert worker.failed_requests == 1
    assert worker.consecutive_failures == 1
    assert worker.state == "healthy"


def test_streaming_inflight_count_decrements_even_if_aclose_raises() -> None:
    class AcloseRaisingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: chunk\n\n"
            raise httpx.ReadError("stream boom")

        async def aclose(self) -> None:
            raise RuntimeError("aclose boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                stream=AcloseRaisingStream(),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app, raise_server_exceptions=False) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "stream": True},
        ) as response:
            b"".join(response.iter_bytes())

    # Note (Jiaxin Deng): the in-flight count must decrement even though aclose()
    # raised, otherwise it leaks and least_request drifts permanently.
    assert all(worker.active_requests == 0 for worker in app.state.workers)
    # Note (Jiaxin Deng): record_routed_request() runs in the same finally, so the
    # broken stream is still booked as a routed failure rather than silently
    # dropped; guards against a future change skipping the completion accounting.
    assert sum(worker.routed_requests for worker in app.state.workers) == 1
    assert sum(worker.failed_requests for worker in app.state.workers) == 1


def _admission_proxy(
    handler,
    *,
    max_inflight: int,
    max_payload_size: int = 512 * 1024 * 1024,
    routable: bool = True,
) -> proxy_module.ProxyHandler:
    config = RouterConfig(
        workers=[WorkerConfig(url="http://worker-a:8101")],
        max_inflight=max_inflight,
        max_payload_size=max_payload_size,
    )
    workers = build_workers(config.workers)
    if routable:
        workers[0].state = "healthy"
    return proxy_module.ProxyHandler(
        config=config,
        workers=workers,
        selector=WorkerSelector("round_robin"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _chat_request(body: bytes = b'{"model": "qwen3-omni"}') -> Request:
    return _request_without_content_length([body])


@pytest.mark.asyncio
async def test_admission_bounds_inflight_and_fast_rejects_the_rest() -> None:
    gate = asyncio.Event()
    upstream_started = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_started
        if request.url.path == "/v1/chat/completions":
            upstream_started += 1
            await gate.wait()
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    proxy = _admission_proxy(handler, max_inflight=4)

    tasks = [
        asyncio.create_task(
            proxy.forward_model_request(_chat_request(), "/v1/chat/completions")
        )
        for _ in range(20)
    ]
    for _ in range(100):
        if sum(task.done() for task in tasks) == 16 and upstream_started == 4:
            break
        await asyncio.sleep(0.01)
    gate.set()
    responses = await asyncio.gather(*tasks)

    admitted = [r for r in responses if isinstance(r, StreamingResponse)]
    rejected = [r for r in responses if not isinstance(r, StreamingResponse)]
    assert len(admitted) == 4
    assert len(rejected) == 16
    # The upstream never saw more than the bound.
    assert upstream_started == 4
    for response in rejected:
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
        payload = json.loads(response.body)
        assert payload["error"]["type"] == "overloaded_error"
    for response in admitted:
        assert response.status_code == 200
        async for _ in response.body_iterator:
            pass
    assert proxy.admission.inflight == 0


@pytest.mark.asyncio
async def test_admission_slot_released_on_early_and_error_paths() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            raise httpx.ConnectError("worker down", request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    proxy = _admission_proxy(handler, max_inflight=2, max_payload_size=64)

    upstream_error = await proxy.forward_model_request(
        _chat_request(), "/v1/chat/completions"
    )
    assert upstream_error.status_code == 502
    assert proxy.admission.inflight == 0

    too_large = await proxy.forward_model_request(
        _chat_request(b"x" * 128), "/v1/chat/completions"
    )
    assert too_large.status_code == 413
    assert proxy.admission.inflight == 0

    no_worker_proxy = _admission_proxy(handler, max_inflight=2, routable=False)
    no_eligible = await no_worker_proxy.forward_model_request(
        _chat_request(), "/v1/chat/completions"
    )
    assert no_eligible.status_code == 503
    assert json.loads(no_eligible.body) == {
        "error": {"message": "no eligible upstream"}
    }
    assert "retry-after" not in no_eligible.headers
    assert no_worker_proxy.admission.inflight == 0


@pytest.mark.asyncio
async def test_admission_slot_released_when_stream_never_starts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    proxy = _admission_proxy(handler, max_inflight=2)

    response = await proxy.forward_model_request(
        _chat_request(), "/v1/chat/completions"
    )
    assert isinstance(response, StreamingResponse)
    assert proxy.admission.inflight == 1

    # The ASGI stack dropping the response without ever starting its iterator
    # (e.g. the client vanished between handler return and response send).
    del response
    gc.collect()
    assert proxy.admission.inflight == 0


@pytest.mark.asyncio
async def test_admission_slot_released_when_upstream_close_raises() -> None:
    class BrokenCloseStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"first": true}'
            yield b'{"second": true}'

        async def aclose(self) -> None:
            raise RuntimeError("close boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                stream=BrokenCloseStream(),
                headers={"content-type": "application/json"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    proxy = _admission_proxy(handler, max_inflight=2)

    response = await proxy.forward_model_request(
        _chat_request(), "/v1/chat/completions"
    )
    iterator = response.body_iterator
    assert await iterator.__anext__() == b'{"first": true}'

    # Client vanishes mid-stream: the relay generator is closed while the
    # upstream stream is unconsumed, and closing that stream itself raises.
    with pytest.raises(RuntimeError, match="close boom"):
        await iterator.aclose()

    assert proxy.admission.inflight == 0


@pytest.mark.asyncio
async def test_streaming_response_send_failure_releases_all_resources_once() -> None:
    # Note (Jiaxin Deng): Starlette sends http.response.start before iterating the
    # body, so a send failure there never enters the iterator's finally. Failing
    # before: only the admission slot had a GC backstop, so the upstream stream
    # and the worker in-flight count leaked when the response send failed.
    aclose_calls = 0

    class TrackedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"chunk": true}'

        async def aclose(self) -> None:
            nonlocal aclose_calls
            aclose_calls += 1

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                stream=TrackedStream(),
                headers={"content-type": "application/json"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    proxy = _admission_proxy(handler, max_inflight=2)
    worker = proxy._workers[0]

    response = await proxy.forward_model_request(
        _chat_request(), "/v1/chat/completions"
    )
    assert isinstance(response, StreamingResponse)
    assert worker.active_requests == 1
    assert proxy.admission.inflight == 1

    start_sends = 0

    async def send(message: dict) -> None:
        nonlocal start_sends
        if message["type"] == "http.response.start":
            start_sends += 1
            raise RuntimeError("client vanished during response.start")

    never = asyncio.Event()

    async def receive() -> dict:
        await never.wait()
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }

    raised = False
    try:
        await response(scope, receive, send)
    except BaseException:
        raised = True

    # The failure struck on http.response.start, before any body was iterated.
    assert start_sends == 1
    assert raised
    # The response __call__ is the sole cleanup owner on this path: upstream
    # close, worker in-flight count, and admission slot each release exactly once.
    assert aclose_calls == 1
    assert worker.active_requests == 0
    assert proxy.admission.inflight == 0
    # The routed request is still booked once, as a client-side failure.
    assert worker.routed_requests == 1
    assert worker.failed_requests == 1


def test_admission_slot_released_after_midstream_failure() -> None:
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
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "stream": True},
        ) as response:
            body = b"".join(response.iter_bytes())
        assert app.state.admission_controller.inflight == 0

    assert b"upstream stream failed" in body


def test_app_fast_rejects_when_admission_bound_reached() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(max_inflight=1), client=async_client)

    with TestClient(app) as client:
        assert app.state.admission_controller.try_acquire()
        try:
            overloaded = client.post(
                "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
            )
        finally:
            app.state.admission_controller.release()
        recovered = client.post(
            "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
        )

    assert overloaded.status_code == 503
    assert overloaded.headers["retry-after"] == "1"
    assert overloaded.json()["error"]["type"] == "overloaded_error"
    assert recovered.status_code == 200
    assert app.state.admission_controller.inflight == 0


def test_admission_exempts_management_endpoints_and_reports_stats() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/model_info":
            return httpx.Response(200, json={"model": "qwen3-omni"}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(max_inflight=1), client=async_client)

    with TestClient(app) as client:
        assert app.state.admission_controller.try_acquire()
        try:
            for _ in range(2):
                rejected = client.post(
                    "/v1/chat/completions",
                    json={"model": "qwen3-omni", "messages": []},
                )
                assert rejected.status_code == 503

            # Health, readiness, and management endpoints are not gated by
            # admission, and none of them touch its counters.
            assert client.get("/live").status_code == 200
            assert client.get("/ready").status_code == 200
            assert client.get("/workers").status_code == 200
            assert client.get("/model_info").status_code == 200
            health = client.get("/health")
            assert health.status_code == 200
        finally:
            app.state.admission_controller.release()

    admission = health.json()["admission"]
    assert admission["inflight"] == 1
    assert admission["max_inflight"] == 1
    assert admission["peak_inflight"] == 1
    assert admission["rejected_total"] == 2


# Streaming-relay semantics for the non-streaming path (Phase 0, #920): the relay
# no longer buffers the full body, so failures after the status commits truncate,
# and the response is chunked with a status-code-based worker error string.


def test_non_streaming_midstream_failure_truncates_instead_of_502() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"partial": true'
            raise httpx.ReadError("body boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                stream=BrokenStream(),
                headers={"content-type": "application/json"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        with pytest.raises(httpx.ReadError, match="body boom"):
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "qwen3-omni", "messages": []},
            ) as response:
                assert response.status_code == 200
                b"".join(response.iter_bytes())

    assert all(worker.active_requests == 0 for worker in app.state.workers)


def test_non_streaming_error_status_relays_full_body_not_truncated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                502,
                content=b'{"error": "upstream declined"}',
                headers={"content-type": "application/json"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
        )

    # Note (Jiaxin Deng): a complete error-status body relays cleanly (only a
    # failure after the body starts truncates); the connect-time 502 boundary is
    # pinned by test_upstream_request_failure_returns_502_and_cleans_active_count.
    assert response.status_code == 502
    assert response.json() == {"error": "upstream declined"}
    assert all(worker.active_requests == 0 for worker in app.state.workers)


def test_worker_failure_last_error_uses_status_code_not_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                502,
                content=b'{"detail": "bad gateway reaching the model backend"}',
                headers={"content-type": "application/json"},
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
        response = client.post(
            "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
        )

    assert response.status_code == 502
    worker = app.state.workers[0]
    # Note (Jiaxin Deng): an evicting status (502) is recorded as the status code,
    # not a snippet of the (non-empty) body, which the streaming relay cannot read
    # without consuming.
    assert worker.last_error == "status=502"


def test_bad_input_500s_do_not_evict_worker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                500,
                json={"error": {"message": "deterministic bad input"}},
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
        responses = [
            client.post(
                "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
            )
            for _ in range(3)
        ]

    # Note (Jiaxin Deng): before the fix, two relayed 500s evicted the only
    # worker and the third request failed with 503 no_eligible_upstream.
    assert [response.status_code for response in responses] == [500, 500, 500]
    assert responses[-1].json() == {"error": {"message": "deterministic bad input"}}
    worker = app.state.workers[0]
    assert worker.state == "healthy"
    assert worker.consecutive_failures == 0
    assert worker.failed_requests == 3


def test_transport_errors_still_evict_worker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            raise httpx.ConnectError("worker down", request=request)
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
        first = client.post(
            "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
        )
        second = client.post(
            "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
        )
        third = client.post(
            "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
        )

    assert first.status_code == 502
    assert second.status_code == 502
    assert third.status_code == 503
    assert third.json() == {"error": {"message": "no eligible upstream"}}
    assert app.state.workers[0].state == "unhealthy"


@pytest.mark.parametrize(
    ("status_code", "evicts"),
    [
        (429, False),  # capacity backpressure
        (503, False),  # scheduler overloaded, retry later
        (408, False),  # request timeout
        (500, False),  # deterministic bad input
        (502, True),  # bad gateway
        (504, True),  # gateway timeout
    ],
)
def test_relayed_status_evicts_only_on_gateway_failure(
    status_code: int, evicts: bool
) -> None:
    # Note (Jiaxin Deng): failing before for 429/503/408, which used to feed the
    # liveness circuit. A reachable worker returning backpressure or an
    # application status stays routable (liveness is owned by /health probes);
    # only gateway failures (502/504) evict.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                status_code,
                json={"detail": "worker response"},
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
        responses = [
            client.post(
                "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
            )
            for _ in range(2)
        ]

    worker = app.state.workers[0]
    # A healthy worker is never short-circuited into a router-generated 503
    # no_eligible_upstream; both requests relay the worker's own status.
    assert [response.status_code for response in responses] == [
        status_code,
        status_code,
    ]
    # Every relayed response is counted in request statistics whether or not it
    # touches liveness.
    assert worker.routed_requests == 2
    assert worker.failed_requests == 2
    if evicts:
        assert worker.state == "unhealthy"
        assert worker.consecutive_failures == 2
    else:
        assert worker.state == "healthy"
        assert worker.consecutive_failures == 0


def test_relayed_500_outcome_labeled_upstream_5xx(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                500, json={"error": {"message": "bad input"}}, request=request
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=async_client,
    )

    with caplog.at_level(logging.INFO, logger="sglang_omni_router.proxy"):
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={"model": "qwen3-omni", "messages": []},
            )

    assert response.status_code == 500
    route_logs = [
        record.getMessage()
        for record in caplog.records
        if "route_completed" in record.getMessage()
    ]
    assert len(route_logs) == 1
    assert "outcome=upstream_5xx" in route_logs[0]


def test_relayed_response_has_no_content_length_header() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                content=b'{"ok": true}',
                headers={
                    "content-type": "application/json",
                    "content-length": "12",
                },
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json={"model": "qwen3-omni", "messages": []}
        )

    assert response.status_code == 200
    assert response.content == b'{"ok": true}'
    assert "content-length" not in {key.lower() for key in response.headers}


def test_sse_terminal_error_event_appended_after_prior_events() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"i": 1}\n\n'
            yield b'data: {"i": 2}\n\n'
            yield b'data: {"i": 3}\n\n'
            raise httpx.ReadError("sse boom")

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
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "qwen3-omni", "stream": True},
        ) as response:
            assert response.status_code == 200
            body = b"".join(response.iter_bytes())

    assert (
        body.index(b'{"i": 1}')
        < body.index(b'{"i": 2}')
        < body.index(b'{"i": 3}')
        < body.index(b"upstream_error")
    )
    assert body.count(b"data: ") == 4  # 3 upstream events + 1 terminal error event
    assert all(worker.active_requests == 0 for worker in app.state.workers)


def test_non_sse_midstream_failure_is_not_injected_with_error_event() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"RIFF....partial-wav"
            raise httpx.ReadError("audio boom")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                stream=BrokenStream(),
                headers={"content-type": "audio/wav"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)

    with TestClient(app) as client:
        # Note (Jiaxin Deng): a non-SSE body truncates (ReadError propagates); if
        # the SSE error event were wrongly injected, it would end cleanly instead.
        with pytest.raises(httpx.ReadError, match="audio boom"):
            with client.stream(
                "POST",
                "/v1/audio/speech",
                json={"model": "qwen3-omni"},
            ) as response:
                assert response.status_code == 200
                b"".join(response.iter_bytes())

    assert all(worker.active_requests == 0 for worker in app.state.workers)


def test_active_requests_held_across_body_relay_then_released() -> None:
    # Note (Jiaxin Deng): recording active_requests at each yield proves the
    # count is held while the body is still relaying (0 if released early), 0 after.
    app_holder: list[FastAPI] = []
    observed_during_relay: list[int] = []

    class ObservingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            worker = app_holder[0].state.workers[0]
            observed_during_relay.append(worker.active_requests)
            yield b"chunk-1"
            observed_during_relay.append(worker.active_requests)
            yield b"chunk-2"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200,
                stream=ObservingStream(),
                headers={"content-type": "audio/wav"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=async_client,
    )
    app_holder.append(app)

    with TestClient(app) as client:
        response = client.post("/v1/audio/speech", json={"model": "qwen3-omni"})

    assert response.status_code == 200
    assert response.content == b"chunk-1chunk-2"
    assert observed_during_relay == [1, 1]
    assert all(worker.active_requests == 0 for worker in app.state.workers)


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
    ("POST", "/init_weights_update_group"),
    ("POST", "/destroy_weights_update_group"),
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


def test_router_unimplemented_tensor_weight_update_returns_501() -> None:
    app = _admin_router_app()
    with TestClient(app) as client:
        resp = client.post("/update_weights_from_tensor", json={})
    assert resp.status_code == 501
    assert resp.json()["error"]["code"] == "not_implemented"


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/update_weights_from_distributed",
            {"names": ["w.0"], "dtypes": ["bfloat16"], "shapes": [[2, 2]]},
        ),
        ("/destroy_weights_update_group", {"group_name": "weight_update_group"}),
    ],
)
def test_router_distributed_weight_update_routes_broadcast(
    path: str,
    payload: dict[str, Any],
) -> None:
    app = _admin_router_app()
    with TestClient(app) as client:
        resp = client.post(path, json=payload)
    assert resp.status_code == 200
    assert resp.json()["success"] is True


_INIT_GROUP_PAYLOAD = {
    "master_address": "localhost",
    "master_port": 12355,
    "world_size": 2,
    "rank_offset": 1,
    "group_name": "weight_update_group",
    "backend": "nccl",
}


def test_router_init_weights_update_group_single_replica_broadcasts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        return httpx.Response(
            200, json={"success": True, "message": "ok", "results": []}, request=request
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=async_client,
    )
    with TestClient(app) as client:
        resp = client.post("/init_weights_update_group", json=_INIT_GROUP_PAYLOAD)
    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert app.state.workers[0].disabled is False


def test_router_init_weights_update_group_failure_keeps_worker_disabled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/init_weights_update_group":
            return httpx.Response(
                504,
                json={"success": False, "message": "rendezvous timed out"},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(worker_configs=[WorkerConfig(url="http://worker-a:8101")]),
        client=async_client,
    )
    with TestClient(app) as client:
        resp = client.post("/init_weights_update_group", json=_INIT_GROUP_PAYLOAD)
    assert resp.status_code == 502
    assert resp.json()["success"] is False
    assert app.state.workers[0].disabled is True


def test_router_init_weights_update_group_rejects_multiple_replicas() -> None:
    app = _admin_router_app()
    with TestClient(app) as client:
        resp = client.post("/init_weights_update_group", json=_INIT_GROUP_PAYLOAD)
    assert resp.status_code == 422
    assert "single-replica" in resp.json()["error"]["message"]


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


def test_max_connections_auto_sizes_to_worker_count() -> None:
    config = RouterConfig(
        workers=[
            WorkerConfig(url="http://worker-a:8101"),
            WorkerConfig(url="http://worker-b:8102"),
            WorkerConfig(url="http://worker-c:8103"),
        ],
    )
    # Note: (Jiaxin Deng) 128 per worker: pool-wide cap must exceed in-flight capacity.
    assert config.max_connections == 384


def test_max_connections_auto_caps_at_4096() -> None:
    workers = [WorkerConfig(url=f"http://worker-{i}:8101") for i in range(40)]
    config = RouterConfig(workers=workers)
    assert config.max_connections == 4096


def test_max_connections_explicit_value_is_preserved() -> None:
    config = _router_config(max_connections=512)
    assert config.max_connections == 512


def test_max_connections_explicit_below_worker_budget_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="sglang_omni_router.config"):
        config = _router_config(max_connections=100)
    assert config.max_connections == 100
    assert any("under-feed" in record.getMessage() for record in caplog.records)


def test_max_connections_auto_at_cap_still_warns_when_pool_outgrows_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = [WorkerConfig(url=f"http://worker-{i}:8101") for i in range(70)]
    with caplog.at_level(logging.WARNING, logger="sglang_omni_router.config"):
        config = RouterConfig(workers=workers)
    assert config.max_connections == 4096
    assert any("under-feed" in record.getMessage() for record in caplog.records)
