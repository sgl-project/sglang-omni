from __future__ import annotations

import asyncio
import gc
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from sglang_omni_router.python import proxy as proxy_module
from sglang_omni_router.python import websocket_proxy as websocket_proxy_module
from sglang_omni_router.python.app import _broadcast_admin_request, create_app
from sglang_omni_router.python.config import (
    DEFAULT_CAPABILITIES,
    Capability,
    RouterConfig,
    WorkerConfig,
)
from sglang_omni_router.python.health import HealthChecker
from sglang_omni_router.python.route_metadata import RouteKind
from sglang_omni_router.python.selector import WorkerSelector
from sglang_omni_router.python.update_journal import (
    JournalUnwritableError,
    UpdateJournal,
)
from sglang_omni_router.python.voice_routing import VoiceRoutingState
from sglang_omni_router.python.worker import build_workers, worker_id_from_url


def _request_netloc(request: httpx.Request) -> str:
    return f"{request.url.host}:{request.url.port}"


def _router_config(
    policy: str = "round_robin",
    max_payload_size: int = 512 * 1024 * 1024,
    max_connections: int | None = None,
    max_inflight: int | None = None,
    health_failure_threshold: int = 1,
    health_check_timeout_secs: int = 5,
    health_check_interval_secs: int = 10,
    worker_configs: list[WorkerConfig] | None = None,
    voice_owner_worker_url: str | None = None,
    router_state_dir: str | None = None,
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
        health_check_interval_secs=health_check_interval_secs,
        voice_owner_worker_url=voice_owner_worker_url,
        router_state_dir=router_state_dir,
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


def test_worker_crud_rejects_voice_owner_deletion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(
                200,
                json={"uploaded_voices": []},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    app = create_app(
        _router_config(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        owner = client.get("/workers").json()["workers"][0]
        response = client.delete(f"/workers/{owner['worker_id']}")

    assert response.status_code == 409
    assert response.json()["error"]["message"] == (
        "voice owner worker cannot be deleted"
    )


def test_worker_crud_rejects_removing_voice_owner_capabilities() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(
                200,
                json={"uploaded_voices": []},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    app = create_app(
        _router_config(),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        owner = client.get("/workers").json()["workers"][0]
        response = client.put(
            f"/workers/{owner['worker_id']}",
            json={"capabilities": ["chat", "speech"]},
        )

    assert response.status_code == 409
    assert response.json()["error"]["message"] == (
        "voice owner worker must retain speech and audio_input capabilities"
    )
    assert app.state.workers[0].capabilities == set(DEFAULT_CAPABILITIES)


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


def test_single_process_recovers_an_unresolved_weight_update_fail_closed(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): a single-process router that died mid weight-update must fail
    # closed: recover the journaled target disabled and 409 a retry.
    journal_path = str(tmp_path / "update_journal.json")
    worker_id = worker_id_from_url("http://worker-a:8101")
    UpdateJournal(journal_path).begin("/update_weights_from_disk", [worker_id])

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/update_weights_from_disk":
            return httpx.Response(
                200,
                json={"success": True, "worker": _request_netloc(request)},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client, journal_path=journal_path)
    with TestClient(app) as client:
        workers = {w["url"]: w for w in client.get("/workers").json()["workers"]}
        assert workers["http://worker-a:8101"]["disabled"] is True
        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 409
        assert "did not complete" in response.json()["error"]["message"]


def test_partial_weight_update_stays_disabled_and_journaled(tmp_path: Path) -> None:
    journal_path = str(tmp_path / "update_journal.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/update_weights_from_disk":
            success = _request_netloc(request) == "worker-a:8101"
            return httpx.Response(
                200 if success else 500,
                json={"success": success},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client, journal_path=journal_path)
    with TestClient(app) as client:
        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 502
        assert all(worker.disabled for worker in app.state.workers)
        assert UpdateJournal(journal_path).has_pending() is True


def test_weight_update_is_refused_when_the_journal_is_not_durable(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): a journal write that never reached the disk must not be
    # reported as success: refuse before any target is disabled or sent, so a host crash
    # cannot re-enable a mixed-weight pool
    journal_path = str(tmp_path / "update_journal.json")
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/update_weights_from_disk":
            sent.append(_request_netloc(request))
            return httpx.Response(200, json={"success": True}, request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client, journal_path=journal_path)

    def _unwritable(path: str, worker_ids: list[str]) -> None:
        raise JournalUnwritableError("no space left on device")

    with TestClient(app) as client:
        app.state.update_journal.begin = _unwritable
        response = client.post("/update_weights_from_disk", json={"path": "/m"})

    assert response.status_code == 503
    assert "could not be durably written" in response.json()["error"]["message"]
    assert sent == []
    assert not any(worker.disabled for worker in app.state.workers)


def _journal_app(tmp_path: Path, journaled_ids: list[str]):
    journal_path = str(tmp_path / "update_journal.json")
    if journaled_ids:
        UpdateJournal(journal_path).begin("/update_weights_from_disk", journaled_ids)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "worker"}, request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client, journal_path=journal_path)
    return app, journal_path


def test_absent_journaled_worker_survives_restart_as_a_tombstone(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): a dynamically added worker is absent after a full restart (the
    # registry rebuilds from static config): its journal entry must survive as a
    # tombstone, and re-registering that stable ID must create it disabled
    dynamic_url = "http://worker-dyn:8103"
    dynamic_id = worker_id_from_url(dynamic_url)
    app, journal_path = _journal_app(tmp_path, [dynamic_id])
    with TestClient(app) as client:
        # Note (Jiaxin Deng): recovery kept the tombstone and the 409 gate stays closed
        assert UpdateJournal(journal_path).pending() == [dynamic_id]
        response = client.post("/update_weights_from_disk", json={"path": "/m"})
        assert response.status_code == 409

        # Note (Jiaxin Deng): re-registering the journaled stable ID creates the worker
        # disabled
        created = client.post("/workers", json={"url": dynamic_url})
        assert created.status_code == 200
        assert created.json()["worker"]["disabled"] is True

        # Note (Jiaxin Deng): an authenticated re-enable resolves the tombstone and
        # unblocks updates
        assert (
            client.put(f"/workers/{dynamic_id}", json={"disabled": False}).status_code
            == 200
        )
        assert UpdateJournal(journal_path).pending() == []
        assert (
            client.post("/update_weights_from_disk", json={"path": "/m"}).status_code
            == 200
        )


def test_deleting_a_journaled_worker_keeps_the_tombstone_for_readd(
    tmp_path: Path,
) -> None:
    worker_url = "http://worker-b:8102"
    worker_id = worker_id_from_url(worker_url)
    app, journal_path = _journal_app(tmp_path, [worker_id])
    with TestClient(app) as client:
        assert client.delete(f"/workers/{worker_id}").status_code == 200
        # Note (Jiaxin Deng): deletion must not erase the tombstone
        assert UpdateJournal(journal_path).pending() == [worker_id]
        readded = client.post("/workers", json={"url": worker_url})
        assert readded.status_code == 200
        assert readded.json()["worker"]["disabled"] is True


def test_rejected_reenable_commits_no_part_of_the_staged_update(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): every fallible precondition is checked before committing: a
    # 503 on the journal must not leave a half-applied model change for the CP keepalive
    # to publish
    journal_path = str(tmp_path / "update_journal.json")
    Path(journal_path).write_bytes(b"{corrupt")
    worker_id = worker_id_from_url("http://worker-a:8101")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "worker"}, request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client, journal_path=journal_path)
    with TestClient(app) as client:
        response = client.put(
            f"/workers/{worker_id}",
            json={"disabled": False, "model": "new-model"},
        )
        assert response.status_code == 503
        listed = {w["url"]: w for w in client.get("/workers").json()["workers"]}
        assert listed["http://worker-a:8101"]["model"] != "new-model"
        assert listed["http://worker-a:8101"]["disabled"] is True


def test_reenable_fails_when_the_journal_cannot_be_resolved(tmp_path: Path) -> None:
    # Note (Jiaxin Deng): discard() cannot modify an unreadable journal: the re-enable
    # must fail instead of returning 200 while every update stays blocked with 409
    journal_path = str(tmp_path / "update_journal.json")
    Path(journal_path).write_bytes(b"{corrupt")
    worker_id = worker_id_from_url("http://worker-a:8101")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "worker"}, request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client, journal_path=journal_path)
    with TestClient(app) as client:
        response = client.put(f"/workers/{worker_id}", json={"disabled": False})
        assert response.status_code == 503
        assert "could not be durably resolved" in response.json()["error"]["message"]
        listed = {w["url"]: w for w in client.get("/workers").json()["workers"]}
        assert listed["http://worker-a:8101"]["disabled"] is True


def test_journal_survives_a_host_reboot_that_wipes_the_per_run_workdir(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): remote workers outlive the router host, so an unresolved
    # update must still be found after a reboot, not just after a process restart
    state_dir = tmp_path / "persistent"
    workdir = tempfile.mkdtemp(prefix="sglang-omni-router-")
    dynamic_url = "http://worker-dyn:8103"
    dynamic_id = worker_id_from_url(dynamic_url)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "worker"}, request=request)
        if request.url.path == "/update_weights_from_disk":
            success = _request_netloc(request) == "worker-a:8101"
            return httpx.Response(
                200 if success else 500, json={"success": success}, request=request
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = _router_config(router_state_dir=str(state_dir))
    before = create_app(config, client=async_client)
    with TestClient(before) as client:
        assert client.post("/workers", json={"url": dynamic_url}).status_code == 200
        assert (
            client.post("/update_weights_from_disk", json={"path": "/m"}).status_code
            == 502
        )
    journal_path = Path(before.state.update_journal.path)
    assert journal_path.is_relative_to(state_dir)
    assert not journal_path.is_relative_to(workdir)

    # Note (Jiaxin Deng): the supervisor and its per-run workdir are gone; only the
    # state dir is carried across the reboot
    shutil.rmtree(workdir)
    after = create_app(
        _router_config(router_state_dir=str(state_dir)), client=async_client
    )
    with TestClient(after) as client:
        listed = {w["url"]: w for w in client.get("/workers").json()["workers"]}
        assert listed["http://worker-a:8101"]["disabled"] is True
        assert listed["http://worker-b:8102"]["disabled"] is True
        assert dynamic_url not in listed  # absent, kept only as a tombstone
        assert dynamic_id in UpdateJournal(str(journal_path)).pending()
        assert (
            client.post("/update_weights_from_disk", json={"path": "/m"}).status_code
            == 409
        )
        readded = client.post("/workers", json={"url": dynamic_url})
        assert readded.json()["worker"]["disabled"] is True


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

    with caplog.at_level(logging.INFO, logger="sglang_omni_router.python.proxy"):
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

    with caplog.at_level(logging.INFO, logger="sglang_omni_router.python.proxy"):
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
        {
            "model": "qwen3-omni",
            "input": "hello",
            "references": [{"data": "base64-audio", "text": "hello"}],
        },
        {
            "model": "qwen3-omni",
            "input": "hello",
            "references": [{"audio": "base64-audio", "text": "hello"}],
        },
        {
            "model": "qwen3-omni",
            "input": "hello",
            "references": [{"ref_audio": "base64-audio", "text": "hello"}],
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


@pytest.mark.parametrize(
    ("route_path", "payload"),
    [
        ("/v1/audio/speech", {"input": "hello", "voice": "Clone"}),
        (
            "/v1/audio/speech/batch",
            {"voice": "default", "items": [{"input": "hello", "voice": "Clone"}]},
        ),
    ],
)
def test_speech_json_without_content_type_preserves_voice_ownership(
    route_path: str,
    payload: dict[str, object],
) -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(
                200,
                json={"uploaded_voices": [{"name": "Clone"}]},
                request=request,
            )
        if request.url.path == route_path:
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, content=b"audio", request=request)
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101"),
        WorkerConfig(url="http://worker-b:8102"),
    ]
    app = create_app(
        _router_config(
            worker_configs=worker_configs,
            voice_owner_worker_url="http://worker-b:8102",
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        assert client.get("/v1/audio/voices").status_code == 200
        for _ in range(100):
            if (
                client.get("/health").json()["voice_routing"]["registry_state"]
                == "ready"
            ):
                break
            time.sleep(0.01)
        response = client.post(
            route_path,
            content=json.dumps(payload).encode(),
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


@pytest.mark.parametrize(
    "non_owner_capabilities",
    [
        {"speech", "audio_input", "video_input"},
        {"speech", "video_input"},
    ],
)
@pytest.mark.parametrize(
    ("route_path", "request_fields"),
    [
        (
            "/v1/audio/speech",
            {"input": "hello", "voice": "default"},
        ),
        (
            "/v1/audio/speech/batch",
            {
                "voice": "default",
                "items": [{"input": "hello"}],
            },
        ),
    ],
)
def test_large_tts_body_uses_voice_owner_in_heterogeneous_pool(
    non_owner_capabilities: set[Capability],
    route_path: str,
    request_fields: dict[str, object],
) -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(
                200,
                json={"uploaded_voices": []},
                request=request,
            )
        if request.url.path == route_path:
            seen_workers.append(_request_netloc(request))
            return httpx.Response(
                200,
                content=b"ok",
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    worker_configs = [
        WorkerConfig(
            url="http://worker-a:8101",
            capabilities={"speech", "audio_input"},
        ),
        WorkerConfig(
            url="http://worker-b:8102",
            capabilities=non_owner_capabilities,
        ),
    ]
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    body = _large_json_body(
        {
            "model": "qwen3-omni",
            **request_fields,
        }
    )

    with TestClient(app) as client:
        response = client.post(
            route_path,
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-a:8101"]


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

    with caplog.at_level(logging.INFO, logger="sglang_omni_router.python.proxy"):
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


def test_voice_upload_uses_endpoint_specific_body_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(
                200,
                json={"uploaded_voices": []},
                request=request,
            )
        raise AssertionError(f"unexpected request path: {request.url.path}")

    monkeypatch.setattr(proxy_module, "MAX_VOICE_UPLOAD_BODY_BYTES", 4)
    app = create_app(
        _router_config(max_payload_size=128),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        response = client.post("/v1/audio/voices", content=b"too-large")

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "message": "request body must be at most 4 bytes",
            "type": "RequestTooLargeError",
            "param": "audio_sample",
            "code": 413,
        }
    }
    assert ("POST", "/v1/audio/voices") not in seen_requests


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


def test_router_init_weights_update_group_failure_keeps_worker_disabled(
    tmp_path: Path,
) -> None:
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
        journal_path=str(tmp_path / "update_journal.json"),
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
                "sglang_omni_router.python.app._ADMIN_UPDATE_LOCK_TIMEOUT_S",
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
    with caplog.at_level(logging.WARNING, logger="sglang_omni_router.python.config"):
        config = _router_config(max_connections=100)
    assert config.max_connections == 100
    assert any("under-feed" in record.getMessage() for record in caplog.records)


def test_max_connections_auto_at_cap_still_warns_when_pool_outgrows_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    workers = [WorkerConfig(url=f"http://worker-{i}:8101") for i in range(70)]
    with caplog.at_level(logging.WARNING, logger="sglang_omni_router.python.config"):
        config = RouterConfig(workers=workers)
    assert config.max_connections == 4096
    assert any("under-feed" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_lifespan_unwinds_all_resources_when_voice_stop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def health_start(_self) -> None:
        events.append("health_start")

    async def health_stop(_self) -> None:
        events.append("health_stop")

    async def voice_start(_self) -> None:
        events.append("voice_start")

    async def voice_stop(_self) -> None:
        events.append("voice_stop")
        raise RuntimeError("voice stop failed")

    async def client_close(_self) -> None:
        events.append("client_close")

    monkeypatch.setattr(HealthChecker, "start", health_start)
    monkeypatch.setattr(HealthChecker, "stop", health_stop)
    monkeypatch.setattr(VoiceRoutingState, "start", voice_start)
    monkeypatch.setattr(VoiceRoutingState, "stop", voice_stop)
    monkeypatch.setattr(httpx.AsyncClient, "aclose", client_close)

    app = create_app(_router_config())
    with pytest.raises(RuntimeError, match="voice stop failed"):
        async with app.router.lifespan_context(app):
            pass

    assert events == [
        "health_start",
        "voice_start",
        "voice_stop",
        "health_stop",
        "client_close",
        "client_close",
    ]


def test_route_registration_split_exposes_exact_route_sets() -> None:
    from sglang_omni_router.python.app import (
        register_admin_routes,
        register_data_routes,
        register_health_routes,
        register_public_metadata_routes,
        register_tts_routes,
    )

    config = _router_config()
    workers = build_workers(config.workers)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    )
    voice_routing = VoiceRoutingState(
        workers=workers,
        owner_url=config.voice_owner_worker_url,
        client=client,
        timeout_secs=config.health_check_timeout_secs,
        retry_interval_secs=config.health_check_interval_secs,
    )
    proxy = proxy_module.ProxyHandler(
        config=config,
        workers=workers,
        selector=WorkerSelector(config.policy),
        client=client,
        voice_routing=voice_routing,
    )
    websocket_proxy = websocket_proxy_module.TTSWebSocketProxy(
        config=config,
        workers=workers,
        selector=WorkerSelector(config.policy),
        admission=proxy.admission,
        voice_routing=voice_routing,
    )

    def _paths(register) -> set[str]:
        app = FastAPI()
        base = {route.path for route in app.routes}
        register(app)
        return {route.path for route in app.routes} - base

    assert _paths(
        lambda app: register_health_routes(app, workers, proxy, voice_routing)
    ) == {
        "/live",
        "/ready",
        "/health",
    }
    assert _paths(
        lambda app: register_admin_routes(app, workers, config, admin_api_key=None)
    ) == {
        "/workers",
        "/workers/{worker_id:path}",
        "/model_info",
        "/pause_generation",
        "/continue_generation",
        "/update_weights_from_disk",
        "/update_weights_from_tensor",
        "/init_weights_update_group",
        "/destroy_weights_update_group",
        "/update_weights_from_distributed",
        "/weights_checker",
        "/weight_update_journal/resolve",
    }
    assert _paths(
        lambda app: register_public_metadata_routes(app, workers, config)
    ) == {"/v1/models"}
    assert _paths(lambda app: register_data_routes(app, proxy)) == {
        "/generate",
        "/v1/chat/completions",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    }
    assert _paths(lambda app: register_tts_routes(app, proxy, websocket_proxy)) == {
        "/v1/audio/speech/batch",
        "/v1/audio/speech/stream",
        "/v1/audio/voices",
        "/v1/audio/voices/{name}",
    }


# Note (Jeffro): Worker /v1/ routes that the router does not forward on purpose. Every other
# /v1/ route on the worker must also exist on the router; if you add a worker
# endpoint and forget the router, test_router_exposes_every_worker_v1_route
# fails and points you here.
_WORKER_ROUTES_NOT_PROXIED = {
    "/v1/realtime",  # the router has no websocket proxy for it yet
}


def test_router_exposes_every_worker_v1_route() -> None:
    from sglang_omni.serve import create_app as create_worker_app

    class _NoopClient:
        pass

    worker_app = create_worker_app(
        _NoopClient(),
        model_name="worker",
        enable_realtime=True,
        supports_audio_translation=True,
    )
    router_app = create_app(_router_config())

    def _v1_routes(app: FastAPI) -> dict[str, frozenset[str]]:
        routes: dict[str, set[str]] = {}
        for route in app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/v1/"):
                continue
            methods = getattr(route, "methods", None) or {"WEBSOCKET"}
            routes.setdefault(path, set()).update(methods - {"HEAD"})
        return {path: frozenset(methods) for path, methods in routes.items()}

    worker_routes = _v1_routes(worker_app)
    router_routes = _v1_routes(router_app)

    assert _WORKER_ROUTES_NOT_PROXIED <= set(
        worker_routes
    ), "stale entry in _WORKER_ROUTES_NOT_PROXIED"
    missing = {
        path: methods
        for path, methods in worker_routes.items()
        if path not in _WORKER_ROUTES_NOT_PROXIED
        and not methods <= router_routes.get(path, frozenset())
    }
    assert not missing, (
        f"worker routes missing from the router: {missing}; add a forward in "
        "sglang_omni_router/python/app.py and a classify_route branch in "
        "sglang_omni_router/python/route_metadata.py"
    )


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/v1/audio/translations"])
def test_speech_to_text_routes_select_audio_input_workers(path: str) -> None:
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        seen.append((_request_netloc(request), request.url.path))
        return httpx.Response(200, json={"text": "hi"}, request=request)

    worker_configs = [
        WorkerConfig(url="http://worker-a:8101", capabilities={"chat", "speech"}),
        WorkerConfig(url="http://worker-b:8102", capabilities={"audio_input"}),
    ]
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        response = client.post(
            path,
            files={"file": ("a.wav", b"RIFF....WAVE", "audio/wav")},
            data={"model": "whisper"},
        )

    assert response.status_code == 200, response.text
    assert seen == [("worker-b:8102", path)]
    workers = client.get("/workers").json()["workers"]
    by_url = {worker["url"]: worker for worker in workers}
    assert by_url["http://worker-b:8102"]["routed_requests_by_class"] == {
        "transcription": 1
    }


_ASR_BOUNDARY = "omni-test-boundary"


def _speech_to_text_multipart(
    model: str | None,
    *,
    model_first: bool = True,
    stream: str | None = None,
) -> tuple[bytes, dict[str, str]]:
    """Build a raw multipart body so tests control the field order."""

    def _field(name: str, value: str) -> bytes:
        return (
            f"--{_ASR_BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    file_part = (
        f"--{_ASR_BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + b"RIFF....WAVE\r\n"
    parts = [file_part]
    if model is not None:
        model_part = _field("model", model)
        parts = [model_part, file_part] if model_first else [file_part, model_part]
    if stream is not None:
        parts.append(_field("stream", stream))
    body = b"".join(parts) + f"--{_ASR_BOUNDARY}--\r\n".encode()
    headers = {"content-type": f"multipart/form-data; boundary={_ASR_BOUNDARY}"}
    return body, headers


def _mixed_asr_pool_app(seen_workers: list[str]) -> FastAPI:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        seen_workers.append(_request_netloc(request))
        return httpx.Response(200, json={"text": "hi"}, request=request)

    worker_configs = [
        WorkerConfig(
            url="http://qwen3-asr:8101", model="qwen3-asr", capabilities={"audio_input"}
        ),
        WorkerConfig(
            url="http://whisper:8102", model="whisper", capabilities={"audio_input"}
        ),
    ]
    return create_app(
        _router_config(worker_configs=worker_configs),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/v1/audio/translations"])
@pytest.mark.parametrize("model_first", [True, False])
def test_multipart_form_model_selects_matching_worker(
    path: str, model_first: bool
) -> None:
    # The model field must route correctly whether it comes before or after
    # the file part — after means the scan has to skip the file bytes.
    seen_workers: list[str] = []
    app = _mixed_asr_pool_app(seen_workers)
    body, headers = _speech_to_text_multipart("whisper", model_first=model_first)

    with TestClient(app) as client:
        for _ in range(4):
            response = client.post(path, content=body, headers=headers)
            assert response.status_code == 200, response.text

    assert seen_workers == ["whisper:8102"] * 4


def test_multipart_form_model_conflicting_route_header_is_rejected() -> None:
    seen_workers: list[str] = []
    app = _mixed_asr_pool_app(seen_workers)
    body, headers = _speech_to_text_multipart("whisper")
    headers["x-sglang-omni-route-model"] = "qwen3-asr"

    with TestClient(app) as client:
        response = client.post("/v1/audio/translations", content=body, headers=headers)

    assert response.status_code == 400, response.text
    assert "conflicts with the multipart form model" in response.text
    assert seen_workers == []


def test_multipart_body_router_cannot_parse_falls_back_to_route_header() -> None:
    # The worker's form parser is authoritative: a body our scan cannot read
    # must still be forwarded (here pinned by the header), never rejected.
    seen_workers: list[str] = []
    app = _mixed_asr_pool_app(seen_workers)
    headers = {
        "content-type": f"multipart/form-data; boundary={_ASR_BOUNDARY}",
        "x-sglang-omni-route-model": "whisper",
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/translations",
            content=b"not really multipart",
            headers=headers,
        )

    assert response.status_code == 200, response.text
    assert seen_workers == ["whisper:8102"]


def test_multipart_form_stream_requires_streaming_capability() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        seen_workers.append(_request_netloc(request))
        return httpx.Response(200, json={"text": "hi"}, request=request)

    worker_configs = [
        WorkerConfig(url="http://batch-asr:8101", capabilities={"audio_input"}),
        WorkerConfig(
            url="http://sse-asr:8102", capabilities={"audio_input", "streaming"}
        ),
    ]
    app = create_app(
        _router_config(worker_configs=worker_configs),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    body, headers = _speech_to_text_multipart("whisper", stream="true")

    with TestClient(app) as client:
        for _ in range(4):
            response = client.post(
                "/v1/audio/translations", content=body, headers=headers
            )
            assert response.status_code == 200, response.text

    assert seen_workers == ["sse-asr:8102"] * 4


def test_multipart_form_stream_conflicting_route_header_is_rejected() -> None:
    seen_workers: list[str] = []
    app = _mixed_asr_pool_app(seen_workers)
    body, headers = _speech_to_text_multipart("whisper", stream="true")
    headers["x-sglang-omni-route-stream"] = "false"

    with TestClient(app) as client:
        response = client.post("/v1/audio/translations", content=body, headers=headers)

    assert response.status_code == 400, response.text
    assert "conflicts with the multipart form stream" in response.text
    assert seen_workers == []


def test_worker_crud_stays_unauthenticated_even_with_admin_key() -> None:
    # Note (Jiaxin Deng): current behavior, frozen: worker CRUD carries no admin auth
    # while the weight-update/broadcast routes do; the route split must not change this.
    app = _admin_router_app(admin_api_key=_ROUTER_ADMIN_API_KEY)
    with TestClient(app) as client:
        created = client.post("/workers", json={"url": "http://127.0.0.1:8199"})
        assert created.status_code == 200
        assert client.get("/workers").status_code == 200
        worker_id = created.json()["worker"]["worker_id"]
        assert (
            client.put(f"/workers/{worker_id}", json={"disabled": True}).status_code
            == 200
        )
        assert client.delete(f"/workers/{worker_id}").status_code == 200


def test_pool_timeout_is_router_local_not_a_worker_failure() -> None:
    # Note (Jiaxin Deng): PoolTimeout means THIS router's pool is exhausted; it must
    # shed with a retryable 503 and must not feed the worker-eviction signal
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        raise httpx.PoolTimeout("pool exhausted", request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)
    with TestClient(app) as client:
        response = client.post("/generate", json={"prompt": "x"})
        assert response.status_code == 503
        assert response.json()["error"]["type"] == "overloaded_error"
        assert "Retry-After" in response.headers
        workers = client.get("/workers").json()["workers"]
        assert all(w["consecutive_failures"] == 0 for w in workers)
        assert all(w["health_state"] != "unhealthy" for w in workers)
        # Note (Jiaxin Deng): router-local exhaustion must not inflate the worker's
        # failure count
        assert all(w["failed_requests"] == 0 for w in workers)


def test_worker_registration_probes_outside_the_update_lock() -> None:
    # Note (Jiaxin Deng): the lock that CRUD shares with weight updates must cover the
    # authoritative mutation, not an arbitrary worker's /health: a blackholed candidate
    # would otherwise 409 every other CRUD call and stall an RL update for the full
    # health timeout
    import threading

    health_started = threading.Event()
    release_health = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health" and "8199" in str(request.url):
            health_started.set()
            release_health.wait(timeout=5.0)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"})
        return httpx.Response(200, json={"success": True})

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client)
    with TestClient(app) as client:
        result: list[int] = []
        creator = threading.Thread(
            target=lambda: result.append(
                client.post(
                    "/workers", json={"url": "http://127.0.0.1:8199"}
                ).status_code
            )
        )
        creator.start()
        assert health_started.wait(timeout=5.0)
        # Note (Jiaxin Deng): the staged worker is being probed: the lock is free, so a
        # weight update or another CRUD call is not blocked behind this network call
        assert app.state.admin_update_lock.locked() is False
        release_health.set()
        creator.join(timeout=5.0)
        assert result == [200]
        assert app.state.admin_update_lock.locked() is False
        assert any(
            worker.url == "http://127.0.0.1:8199" for worker in app.state.workers
        )


def _resolve_journal_app(tmp_path: Path, journal_bytes: bytes | None):
    journal_path = str(tmp_path / "update_journal.json")
    if journal_bytes is not None:
        Path(journal_path).write_bytes(journal_bytes)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "status": "worker"})

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(),
        client=async_client,
        journal_path=journal_path,
        admin_api_key="secret-key",
    )
    return app, journal_path


_ADMIN = {"Authorization": "Bearer secret-key"}


def test_resolving_the_journal_requires_an_explicit_acknowledgement(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): the record is what keeps workers with uncertain
    # weight versions disabled, so it must not be droppable by a bare POST.
    worker_id = worker_id_from_url("http://worker-a:8101")
    app, journal_path = _resolve_journal_app(tmp_path, None)
    UpdateJournal(journal_path).begin("/update_weights_from_disk", [worker_id])
    with TestClient(app) as client:
        for body in ({}, {"acknowledge": False}, {"acknowledge": "yes"}):
            response = client.post(
                "/weight_update_journal/resolve", json=body, headers=_ADMIN
            )
            assert response.status_code == 422
        assert UpdateJournal(journal_path).pending() == [worker_id]


def test_resolving_the_journal_needs_admin_auth(tmp_path: Path) -> None:
    app, journal_path = _resolve_journal_app(tmp_path, None)
    UpdateJournal(journal_path).begin("/x", ["w0"])
    with TestClient(app) as client:
        response = client.post(
            "/weight_update_journal/resolve", json={"acknowledge": True}
        )
        assert response.status_code == 401
        assert UpdateJournal(journal_path).pending() == ["w0"]


def test_resolving_an_unreadable_journal_unblocks_weight_updates(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): discard() cannot edit a corrupt file, so without this
    # operation the 409 gate can only be cleared from a shell on the host.
    worker_id = worker_id_from_url("http://worker-a:8101")
    app, journal_path = _resolve_journal_app(tmp_path, b"{corrupt")
    with TestClient(app) as client:
        update = {"path": "/m"}
        assert (
            client.post(
                "/update_weights_from_disk", json=update, headers=_ADMIN
            ).status_code
            == 409
        )
        # re-enabling is the documented way out, and it cannot resolve a
        # journal it cannot read
        assert (
            client.put(
                f"/workers/{worker_id}", json={"disabled": False}, headers=_ADMIN
            ).status_code
            == 503
        )

        response = client.post(
            "/weight_update_journal/resolve", json={"acknowledge": True}, headers=_ADMIN
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "journal_readable": False,
            "resolved_worker_ids": [],
        }
        assert not Path(journal_path).exists()

        assert (
            client.put(
                f"/workers/{worker_id}", json={"disabled": False}, headers=_ADMIN
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/update_weights_from_disk", json=update, headers=_ADMIN
            ).status_code
            == 200
        )


def test_resolving_a_readable_journal_reports_what_it_dropped(tmp_path: Path) -> None:
    worker_id = worker_id_from_url("http://worker-a:8101")
    app, journal_path = _resolve_journal_app(tmp_path, None)
    UpdateJournal(journal_path).begin("/update_weights_from_disk", [worker_id])
    with TestClient(app) as client:
        response = client.post(
            "/weight_update_journal/resolve", json={"acknowledge": True}, headers=_ADMIN
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["journal_readable"] is True
        assert payload["resolved_worker_ids"] == [worker_id]
        assert UpdateJournal(journal_path).pending() == []


def test_resolving_the_journal_is_rejected_while_an_update_holds_the_lock(
    tmp_path: Path,
) -> None:
    app, journal_path = _resolve_journal_app(tmp_path, None)
    UpdateJournal(journal_path).begin("/x", ["w0"])
    with TestClient(app) as client:
        app.state.admin_update_lock = asyncio.Lock()

        async def _hold() -> None:
            await app.state.admin_update_lock.acquire()

        asyncio.run(_hold())
        response = client.post(
            "/weight_update_journal/resolve", json={"acknowledge": True}, headers=_ADMIN
        )
        assert response.status_code == 409
        assert UpdateJournal(journal_path).pending() == ["w0"]


def test_resolving_the_journal_fails_closed_when_it_cannot_be_removed(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): reporting success on a file that survives would leave
    # every later update blocked behind a gate the operator believes is gone.
    app, journal_path = _resolve_journal_app(tmp_path, None)
    UpdateJournal(journal_path).begin("/x", ["w0"])
    with TestClient(app) as client:

        def _unwritable() -> None:
            raise JournalUnwritableError("read-only file system")

        app.state.update_journal.clear = _unwritable
        response = client.post(
            "/weight_update_journal/resolve", json={"acknowledge": True}, headers=_ADMIN
        )
        assert response.status_code == 503
        assert journal_path in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_registry_lock_rejects_while_an_update_is_queued() -> None:
    # Note (Jiaxin Deng): release() wakes the first waiter without marking the
    # lock held, so locked() alone would admit a caller that then runs after an
    # update it never saw.
    from fastapi import FastAPI as _FastAPI

    from sglang_omni_router.python.app import _registry_lock_or_reject

    app = _FastAPI()
    app.state.admin_update_lock = asyncio.Lock()
    lock = app.state.admin_update_lock
    await lock.acquire()
    queued = asyncio.create_task(lock.acquire())
    await asyncio.sleep(0)
    lock.release()

    assert lock.locked() is False  # the window locked() alone misses
    _, rejected = _registry_lock_or_reject(app)
    assert rejected is not None and rejected.status_code == 409

    queued.cancel()


def test_a_journal_that_cannot_be_cleared_is_reported_on_the_success(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): the broadcast really did succeed, but a surviving
    # entry blocks every later update, which a bare 200 hides.
    journal_path = str(tmp_path / "update_journal.json")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "status": "worker"})

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(), client=async_client, journal_path=journal_path)
    with TestClient(app) as client:

        def _unwritable() -> None:
            raise JournalUnwritableError("read-only file system")

        app.state.update_journal.clear = _unwritable
        response = client.post("/update_weights_from_disk", json={"path": "/m"})

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert journal_path in response.json()["journal_error"]


def test_weight_update_is_refused_without_a_durable_state_directory(
    tmp_path: Path,
) -> None:
    # Note (Jiaxin Deng): startup stays up for a plain relay, but an update
    # with nowhere to record its target set must fail closed.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        sent.append(request.url.path)
        return httpx.Response(200, json={"success": True}, request=request)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    config = _router_config(router_state_dir=str(blocker / "state"))
    app = create_app(config, client=async_client)
    with TestClient(app) as client:
        response = client.post("/update_weights_from_disk", json={"path": "/m"})

    assert response.status_code == 503
    assert sent == []
    assert not any(worker.disabled for worker in app.state.workers)


@pytest.mark.asyncio
async def test_a_cancelled_upstream_send_returns_the_active_gauge() -> None:
    # Note (Jiaxin Deng): a client disconnect cancels the send await, and
    # neither httpx handler sees it; a leaked gauge keeps the worker looking
    # busy and stops a retiring incarnation from ever draining.
    from sglang_omni_router.python.route_metadata import RouteMetadata

    config = _router_config()
    workers = build_workers(config.workers)
    worker = workers[0]

    class _CancellingClient:
        def build_request(self, *args, **kwargs):
            return object()

        async def send(self, request, **kwargs):
            raise asyncio.CancelledError()

    proxy = proxy_module.ProxyHandler(
        config=config,
        workers=workers,
        selector=WorkerSelector(config.policy),
        client=_CancellingClient(),
    )
    metadata = RouteMetadata(
        request_id="r1",
        model=None,
        stream=False,
        required_capabilities=set(),
        is_body_over_metadata_limit=False,
        has_route_model_header=False,
        has_route_capabilities_header=False,
        route_kind=RouteKind.GENERATION,
        service_class="generation",
        voice_names_requiring_registry=set(),
    )
    release = proxy_module._ReleaseOnce(proxy.admission)

    with pytest.raises(asyncio.CancelledError):
        await proxy._forward_relay(
            _request_without_content_length([b"{}"]),
            "/generate",
            b"{}",
            metadata,
            worker,
            release,
        )

    assert worker.active_requests == 0
