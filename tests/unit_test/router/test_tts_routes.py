from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

import httpx
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.datastructures import URL

from sglang_omni_router.python import proxy as proxy_module
from sglang_omni_router.python import websocket_proxy as websocket_proxy_module
from sglang_omni_router.python.app import create_app
from sglang_omni_router.python.config import RouterConfig, WorkerConfig
from sglang_omni_router.python.route_metadata import ROUTE_HEADER_NAMES
from sglang_omni_router.python.selector import WorkerSelector
from sglang_omni_router.python.voice_routing import (
    MAX_VOICE_REGISTRY_RESPONSE_BYTES,
    VoiceMutation,
    VoiceRoutingState,
)
from sglang_omni_router.python.worker import Worker, build_workers, worker_id_from_url


def _request_netloc(request: httpx.Request) -> str:
    return f"{request.url.host}:{request.url.port}"


def _wait_for_router_ready(client: TestClient) -> None:
    for _ in range(100):
        if client.get("/ready").status_code == 200:
            return
        time.sleep(0.01)
    raise AssertionError("router did not become ready")


def _wait_for_voice_registry_ready(client: TestClient) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for _ in range(100):
        state = client.get("/health").json()["voice_routing"]
        if state["registry_state"] == "ready":
            return state
        time.sleep(0.01)
    raise AssertionError(f"voice registry did not become ready: {state}")


def _router_config(
    *,
    max_payload_size: int = 512 * 1024 * 1024,
    health_failure_threshold: int = 1,
    health_check_interval_secs: int = 10,
    worker_configs: list[WorkerConfig] | None = None,
    voice_owner_worker_url: str | None = None,
    max_inflight: int | None = None,
) -> RouterConfig:
    return RouterConfig(
        workers=worker_configs
        or [
            WorkerConfig(url="http://worker-a:8101"),
            WorkerConfig(url="http://worker-b:8102"),
        ],
        max_payload_size=max_payload_size,
        health_success_threshold=1,
        health_failure_threshold=health_failure_threshold,
        health_check_interval_secs=health_check_interval_secs,
        voice_owner_worker_url=voice_owner_worker_url,
        max_inflight=max_inflight,
    )


def _voice_routing(
    config: RouterConfig,
    workers: list[Worker],
    client: httpx.AsyncClient,
) -> VoiceRoutingState:
    return VoiceRoutingState(
        workers=workers,
        owner_url=config.voice_owner_worker_url,
        client=client,
        timeout_secs=config.health_check_timeout_secs,
        retry_interval_secs=config.health_check_interval_secs,
    )


def test_voice_owner_config_must_identify_a_capable_worker() -> None:
    with pytest.raises(
        ValueError,
        match="voice_owner_worker_url must identify a configured worker",
    ):
        _router_config(voice_owner_worker_url="http://worker-c:8103")

    with pytest.raises(
        ValueError,
        match="must identify a worker with capabilities: audio_input, speech",
    ):
        _router_config(
            worker_configs=[
                WorkerConfig(
                    url="http://worker-a:8101",
                    capabilities={"speech"},
                )
            ],
            voice_owner_worker_url="http://worker-a:8101",
        )


@pytest.mark.asyncio
async def test_automatic_voice_owner_tracks_dynamic_worker_registration() -> None:
    config = _router_config(
        worker_configs=[WorkerConfig(url="http://worker-a:8101", capabilities={"chat"})]
    )
    workers = build_workers(config.workers)
    async with httpx.AsyncClient() as client:
        state = VoiceRoutingState(
            workers=workers,
            owner_url=None,
            client=client,
            timeout_secs=5,
            retry_interval_secs=1,
        )
        assert state.owner() is None
        assert state.to_dict()["registry_state"] == "unassigned"
        assert not state.requires_owner({"preset"})

        workers.append(
            build_workers(
                [
                    WorkerConfig(
                        url="http://worker-b:8102",
                        capabilities={"speech", "audio_input"},
                    )
                ]
            )[0]
        )
        workers[1].state = "healthy"
        assert state.owner() is None
        assert state.ensure_owner() is workers[1]
        assert state.owner() is workers[1]
        assert state.requires_owner({"preset"})
        workers[0].replace_config(
            WorkerConfig(
                url=workers[0].url,
                capabilities={"speech", "audio_input"},
            )
        )
        assert state.owner() is workers[1]


def test_automatic_voice_owner_skips_an_unroutable_candidate() -> None:
    workers = build_workers(_router_config().workers)
    workers[0].state = "healthy"
    workers[0].set_disabled(True)
    workers[1].state = "healthy"
    config = _router_config()

    async def run() -> None:
        async with httpx.AsyncClient() as client:
            state = _voice_routing(config, workers, client)
            assert state.ensure_owner() is workers[1]
            assert state.to_dict() == {
                "owner_worker_id": workers[1].worker_id,
                "owner_url": workers[1].url,
                "owner_routable": True,
                "registry_state": "inactive",
                "uploaded_voice_count": 0,
                "last_refresh_error": None,
            }
            workers[0].set_disabled(False)
            assert state.owner() is workers[1]

    asyncio.run(run())


def test_voice_registry_mutation_before_hydration_preserves_owner_state() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/audio/voices"
            return httpx.Response(
                200,
                json={
                    "uploaded_voices": [
                        {"name": "Existing"},
                        {"name": "New"},
                    ]
                },
                request=request,
            )

        workers = build_workers(_router_config().workers)
        workers[0].state = "healthy"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            state = VoiceRoutingState(
                workers=workers,
                owner_url=workers[0].url,
                client=client,
                timeout_secs=5,
                retry_interval_secs=1,
            )
            state.begin_mutation()
            state.mark_mutation_dispatched()
            mutation = VoiceMutation.create("upload", "New")
            assert mutation is not None
            state.apply(mutation)
            state.end_mutation()
            await state.start()
            try:
                for _ in range(100):
                    if not state.requires_owner({"preset"}):
                        break
                    await asyncio.sleep(0.01)
                assert not state.requires_owner({"preset"})
                assert state.requires_owner({"existing"})
                assert state.requires_owner({"new"})
            finally:
                await state.stop()

    asyncio.run(run())


def test_voice_registry_mutations_during_hydration_are_not_overwritten() -> None:
    async def run() -> None:
        hydration_started = asyncio.Event()
        release_hydration = asyncio.Event()
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            assert request.url.path == "/v1/audio/voices"
            request_count += 1
            if request_count > 1:
                return httpx.Response(
                    200,
                    json={
                        "uploaded_voices": [
                            {"name": "Existing"},
                            {"name": "New"},
                        ]
                    },
                    request=request,
                )
            hydration_started.set()
            await release_hydration.wait()
            return httpx.Response(
                200,
                json={
                    "uploaded_voices": [
                        {"name": "Deleted"},
                        {"name": "Existing"},
                    ]
                },
                request=request,
            )

        workers = build_workers(_router_config().workers)
        workers[0].state = "healthy"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            state = VoiceRoutingState(
                workers=workers,
                owner_url=workers[0].url,
                client=client,
                timeout_secs=5,
                retry_interval_secs=1,
            )
            await state.start()
            state.activate()
            await hydration_started.wait()
            state.begin_mutation()
            state.mark_mutation_dispatched()
            upload = VoiceMutation.create("upload", "New")
            delete = VoiceMutation.create("delete", "Deleted")
            assert upload is not None and delete is not None
            state.apply(upload)
            state.apply(delete)
            state.end_mutation()
            release_hydration.set()
            try:
                for _ in range(100):
                    if not state.requires_owner({"preset"}):
                        break
                    await asyncio.sleep(0.01)
                assert not state.requires_owner({"preset"})
                assert state.requires_owner({"existing"})
                assert state.requires_owner({"new"})
                assert not state.requires_owner({"deleted"})
            finally:
                await state.stop()

    asyncio.run(run())


def test_voice_registry_hydration_retries_without_request_path_io() -> None:
    async def run() -> None:
        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            assert request.url.path == "/v1/audio/voices"
            request_count += 1
            if request_count == 1:
                return httpx.Response(503, request=request)
            return httpx.Response(
                200,
                json={"uploaded_voices": []},
                request=request,
            )

        workers = build_workers(_router_config().workers)
        workers[0].state = "healthy"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            state = VoiceRoutingState(
                workers=workers,
                owner_url=workers[0].url,
                client=client,
                timeout_secs=5,
                retry_interval_secs=0.01,
            )
            await state.start()
            try:
                assert state.requires_owner({"preset"})
                for _ in range(100):
                    if request_count >= 2 and not state.requires_owner({"preset"}):
                        break
                    await asyncio.sleep(0.01)
                assert request_count == 2
                assert not state.requires_owner({"preset"})
                await asyncio.sleep(0.03)
                assert request_count == 2
            finally:
                await state.stop()

    asyncio.run(run())


def test_missing_voice_registry_keeps_unknown_names_on_the_owner() -> None:
    async def run() -> None:
        request_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            assert request.url.path == "/v1/audio/voices"
            request_count += 1
            return httpx.Response(404, request=request)

        workers = build_workers(_router_config().workers)
        workers[0].state = "healthy"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            state = VoiceRoutingState(
                workers=workers,
                owner_url=workers[0].url,
                client=client,
                timeout_secs=5,
                retry_interval_secs=0.01,
            )
            await state.start()
            try:
                for _ in range(100):
                    state.requires_owner({"Vivian"})
                    if request_count >= 2:
                        break
                    await asyncio.sleep(0.01)
                assert request_count >= 2
                assert state.to_dict()["registry_state"] == "degraded"
                assert (
                    state.to_dict()["last_refresh_error"]
                    == "HTTPStatusError:status=404"
                )
                assert state.requires_owner({"Vivian"})
            finally:
                await state.stop()

    asyncio.run(run())


def test_voice_registry_response_size_is_bounded() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/audio/voices"
            return httpx.Response(
                200,
                content=b"x" * (MAX_VOICE_REGISTRY_RESPONSE_BYTES + 1),
                request=request,
            )

        workers = build_workers(_router_config().workers)
        workers[0].state = "healthy"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            state = VoiceRoutingState(
                workers=workers,
                owner_url=workers[0].url,
                client=client,
                timeout_secs=5,
                retry_interval_secs=1,
            )
            await state.start()
            try:
                state.requires_owner({"Vivian"})
                for _ in range(100):
                    if state.to_dict()["last_refresh_error"] is not None:
                        break
                    await asyncio.sleep(0.01)
                assert (
                    state.to_dict()["last_refresh_error"]
                    == "VoiceRegistryResponseTooLargeError"
                )
                assert state.requires_owner({"Vivian"})
            finally:
                await state.stop()

    asyncio.run(run())


def test_undispatched_mutation_preserves_the_confirmed_registry() -> None:
    async def run() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/audio/voices"
            return httpx.Response(
                200,
                json={"uploaded_voices": []},
                request=request,
            )

        workers = build_workers(_router_config().workers)
        workers[0].state = "healthy"
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            state = VoiceRoutingState(
                workers=workers,
                owner_url=workers[0].url,
                client=client,
                timeout_secs=5,
                retry_interval_secs=0.01,
            )
            await state.start()
            try:
                for _ in range(100):
                    if not state.requires_owner({"Vivian"}):
                        break
                    await asyncio.sleep(0.01)
                assert state.to_dict()["registry_state"] == "ready"

                workers[0].set_disabled(True)
                state.begin_mutation()
                state.end_mutation()
                await asyncio.sleep(0.02)

                assert state.to_dict()["registry_state"] == "ready"
                assert not state.requires_owner({"Vivian"})
            finally:
                await state.stop()

    asyncio.run(run())


def test_rejected_voice_mutation_preserves_builtin_routing() -> None:
    seen_speech_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices" and request.method == "GET":
            return httpx.Response(
                200,
                json={"uploaded_voices": []},
                request=request,
            )
        if request.url.path == "/v1/audio/speech":
            seen_speech_workers.append(_request_netloc(request))
            return httpx.Response(200, content=b"audio", request=request)
        raise AssertionError(
            f"unexpected upstream request: {request.method} {request.url.path}"
        )

    app = create_app(
        _router_config(voice_owner_worker_url="http://worker-b:8102"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        _wait_for_router_ready(client)
        assert client.get("/v1/audio/voices").status_code == 200
        _wait_for_voice_registry_ready(client)

        disabled = client.put(
            f"/workers/{worker_id_from_url('http://worker-b:8102')}",
            json={"disabled": True},
        )
        assert disabled.status_code == 200
        rejected = client.post(
            "/v1/audio/voices",
            data={"name": "Clone", "consent": "true"},
            files={"audio_sample": ("clone.wav", b"RIFF", "audio/wav")},
        )
        registry_after_rejection = client.get("/health").json()["voice_routing"]
        speech = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "Vivian"},
        )

    assert rejected.status_code == 503
    assert registry_after_rejection["registry_state"] == "ready"
    assert speech.status_code == 200
    assert seen_speech_workers == ["worker-a:8101"]


def test_voice_registry_uses_the_control_http_client() -> None:
    data_plane_requests: list[str] = []
    control_requests: list[str] = []

    def data_handler(request: httpx.Request) -> httpx.Response:
        data_plane_requests.append(request.url.path)
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(200, content=b"audio", request=request)
        raise AssertionError(
            f"control request used data-plane client: {request.url.path}"
        )

    def control_handler(request: httpx.Request) -> httpx.Response:
        control_requests.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            assert request.url.params.get("names_only") == "true"
            return httpx.Response(
                200,
                json={"uploaded_voice_names": []},
                request=request,
            )
        raise AssertionError(f"unexpected control request: {request.url.path}")

    app = create_app(
        _router_config(voice_owner_worker_url="http://worker-a:8101"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(data_handler)),
        health_client=httpx.AsyncClient(transport=httpx.MockTransport(control_handler)),
    )
    with TestClient(app) as client:
        _wait_for_router_ready(client)
        response = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "Vivian"},
        )
        _wait_for_voice_registry_ready(client)

    assert response.status_code == 200
    assert "/v1/audio/voices" in control_requests
    assert data_plane_requests == ["/v1/audio/speech"]


def test_tts_http_routes_preserve_batch_identity_and_uploaded_voice_owner() -> None:
    seen: list[tuple[str, str, str]] = []
    uploaded = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal uploaded
        worker = _request_netloc(request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        seen.append((request.method, worker, path))
        if path == "/v1/audio/voices" and request.method == "POST":
            uploaded = True
            return httpx.Response(200, json={"name": "Clone"}, request=request)
        if path == "/v1/audio/voices" and request.method == "GET":
            return httpx.Response(
                200,
                json={"uploaded_voices": [{"name": "Clone"}] if uploaded else []},
                request=request,
            )
        if path == "/v1/audio/voices/Clone" and request.method == "DELETE":
            uploaded = False
            return httpx.Response(200, json={"success": True}, request=request)
        if path == "/v1/audio/speech/batch":
            return httpx.Response(
                200,
                json={
                    "id": "batch-1",
                    "results": [],
                    "total": 0,
                    "succeeded": 0,
                    "failed": 0,
                },
                request=request,
            )
        if path == "/v1/audio/speech":
            status = 400 if worker == "worker-a:8101" else 200
            return httpx.Response(
                status,
                content=b"audio" if status == 200 else b'{"error": {}}',
                request=request,
            )
        raise AssertionError(f"unexpected request: {request.method} {path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(
        _router_config(
            voice_owner_worker_url="http://worker-b:8102",
            health_check_interval_secs=1,
        ),
        client=async_client,
    )

    with TestClient(app) as client:
        _wait_for_router_ready(client)
        assert client.get("/v1/audio/voices").status_code == 200
        _wait_for_voice_registry_ready(client)
        seen.clear()
        upload = client.post(
            "/v1/audio/voices",
            data={"name": "Clone", "consent": "true"},
            files={"audio_sample": ("clone.wav", b"RIFF", "audio/wav")},
        )
        speech = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "Clone"},
        )
        batch = client.post(
            "/v1/audio/speech/batch",
            json={
                "voice": "default",
                "items": [
                    {"input": "one"},
                    {"input": "two", "voice": "Clone"},
                ],
            },
        )
        deleted = client.delete("/v1/audio/voices/Clone")
        _wait_for_voice_registry_ready(client)
        deleted_voice = client.post(
            "/v1/audio/speech",
            json={"input": "hello", "voice": "Clone"},
        )

    assert upload.status_code == 200
    assert speech.status_code == 200
    assert batch.status_code == 200
    assert deleted.status_code == 200
    assert deleted_voice.status_code == 400
    assert ("GET", "worker-b:8102", "/v1/audio/voices") in seen
    routed_requests = [request for request in seen if request[0] != "GET"]
    assert routed_requests == [
        ("POST", "worker-b:8102", "/v1/audio/voices"),
        ("POST", "worker-b:8102", "/v1/audio/speech"),
        ("POST", "worker-b:8102", "/v1/audio/speech/batch"),
        ("DELETE", "worker-b:8102", "/v1/audio/voices/Clone"),
        ("POST", "worker-a:8101", "/v1/audio/speech"),
    ]
    owner = app.state.workers[1].to_dict()
    assert owner["routed_requests_by_class"] == {
        "voice_control": 3,
        "speech_http": 1,
        "speech_batch": 1,
    }


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/audio/speech",
            {"input": "hello", "voice": "Clone", "ref_audio": "data:audio/wav"},
        ),
        (
            "/v1/audio/speech/batch",
            {
                "voice": "Clone",
                "ref_audio": "data:audio/wav",
                "items": [{"input": "hello"}],
            },
        ),
    ],
)
def test_explicit_reference_does_not_require_the_uploaded_voice_owner(
    path: str,
    payload: dict[str, Any],
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
        if request.url.path == path:
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"ok": True}, request=request)
        raise AssertionError(f"unexpected upstream request: {request.url.path}")

    app = create_app(
        _router_config(voice_owner_worker_url="http://worker-b:8102"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        _wait_for_router_ready(client)
        assert client.get("/v1/audio/voices").status_code == 200
        registry = _wait_for_voice_registry_ready(client)
        assert registry["uploaded_voice_count"] == 1
        disabled = client.put(
            f"/workers/{worker_id_from_url('http://worker-b:8102')}",
            json={"disabled": True},
        )
        assert disabled.status_code == 200
        response = client.post(path, json=payload)

    assert response.status_code == 200
    assert seen_workers == ["worker-a:8101"]


def test_batch_item_reference_keeps_uploaded_default_voice_on_owner() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(
                200,
                json={"uploaded_voice_names": ["Clone"]},
                request=request,
            )
        if request.url.path == "/v1/audio/speech/batch":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"results": []}, request=request)
        raise AssertionError(f"unexpected upstream request: {request.url.path}")

    app = create_app(
        _router_config(voice_owner_worker_url="http://worker-b:8102"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech/batch",
            json={
                "voice": "Clone",
                "items": [{"input": "hello", "ref_audio": "data:audio/wav"}],
            },
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


def test_batch_item_model_override_selects_its_effective_model() -> None:
    seen_workers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/speech/batch":
            seen_workers.append(_request_netloc(request))
            return httpx.Response(200, json={"results": []}, request=request)
        raise AssertionError(f"unexpected upstream request: {request.url.path}")

    app = create_app(
        _router_config(
            worker_configs=[
                WorkerConfig(url="http://worker-a:8101", model="model-a"),
                WorkerConfig(url="http://worker-b:8102", model="model-b"),
            ]
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech/batch",
            json={"model": "model-a", "items": [{"input": "x", "model": "model-b"}]},
        )

    assert response.status_code == 200
    assert seen_workers == ["worker-b:8102"]


def test_mixed_model_batch_is_rejected_before_forwarding() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={}, request=request)

    app = create_app(
        _router_config(
            worker_configs=[
                WorkerConfig(url="http://worker-a:8101", model="model-a"),
                WorkerConfig(url="http://worker-b:8102", model="model-b"),
            ]
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech/batch",
            json={
                "model": "model-a",
                "items": [{"input": "one"}, {"input": "two", "model": "model-b"}],
            },
        )

    assert response.status_code == 400
    assert "one model" in response.json()["error"]["message"]
    assert seen_paths == []


def test_batch_streaming_is_rejected_before_worker_selection() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={}, request=request)

    app = create_app(
        _router_config(
            worker_configs=[
                WorkerConfig(
                    url="http://worker-a:8101",
                    capabilities={"speech"},
                )
            ]
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech/batch",
            json={"stream": True, "items": [{"input": "hello"}]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == (
        "stream is not supported for batch speech requests"
    )
    assert seen_paths == []


def test_large_batch_item_model_requires_a_route_hint_with_a_voice_owner() -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={}, request=request)

    app = create_app(
        _router_config(
            worker_configs=[
                WorkerConfig(url="http://worker-a:8101", model="model-a"),
                WorkerConfig(url="http://worker-b:8102", model="model-b"),
            ],
            voice_owner_worker_url="http://worker-a:8101",
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    body = json.dumps(
        {
            "items": [
                {"input": "x" * (1024 * 1024), "model": "model-b"},
            ]
        }
    ).encode()
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech/batch",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert "x-sglang-omni-route-model" in response.json()["error"]["message"]
    assert seen_paths == []


def test_large_speech_body_without_voice_owner_requires_audio_input() -> None:
    forwarded: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        forwarded.append(request.url.path)
        return httpx.Response(200, json={}, request=request)

    app = create_app(
        _router_config(
            worker_configs=[
                WorkerConfig(
                    url="http://worker-a:8101",
                    capabilities={"speech"},
                )
            ]
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    body = json.dumps(
        {
            "input": "hello",
            "voice": "default",
            "ref_audio": "data:audio/wav;base64," + "A" * (1024 * 1024),
        }
    ).encode()
    with TestClient(app) as client:
        response = client.post(
            "/v1/audio/speech",
            content=body,
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 503
    assert forwarded == []


@pytest.mark.asyncio
async def test_voice_mutations_do_not_serialize_upstream_requests() -> None:
    upload_started = asyncio.Event()
    delete_started = asyncio.Event()
    release_upload = asyncio.Event()
    dispatched: list[str] = []
    uploaded = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal uploaded
        if request.method == "GET":
            return httpx.Response(
                200,
                json={"uploaded_voices": [{"name": "Clone"}] if uploaded else []},
                request=request,
            )
        dispatched.append(request.method)
        if request.method == "POST":
            upload_started.set()
            await release_upload.wait()
            uploaded = True
        else:
            delete_started.set()
            uploaded = False
        payload = {"name": "Clone"} if request.method == "POST" else {"ok": True}
        return httpx.Response(200, json=payload, request=request)

    def request(method: str, path: str) -> Request:
        boundary = "voice-boundary"
        body = (
            (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="name"\r\n\r\n'
                "Clone\r\n"
                f"--{boundary}--\r\n"
            ).encode()
            if method == "POST"
            else b""
        )
        messages = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive() -> dict[str, Any]:
            if messages:
                return messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        headers = (
            [
                (
                    b"content-type",
                    f"multipart/form-data; boundary={boundary}".encode(),
                )
            ]
            if method == "POST"
            else []
        )
        return Request(
            {
                "type": "http",
                "method": method,
                "path": path,
                "headers": headers,
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 50000),
            },
            receive,
        )

    config = _router_config()
    workers = build_workers(config.workers)
    workers[0].state = "healthy"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as async_client:
        voice_routing = _voice_routing(config, workers, async_client)
        proxy = proxy_module.ProxyHandler(
            config=config,
            workers=workers,
            selector=WorkerSelector(config.policy),
            client=async_client,
            voice_routing=voice_routing,
        )
        upload_task = asyncio.create_task(
            proxy.forward_model_request(
                request("POST", "/v1/audio/voices"),
                "/v1/audio/voices",
            )
        )
        await upload_started.wait()
        delete_task = asyncio.create_task(
            proxy.forward_model_request(
                request("DELETE", "/v1/audio/voices/Clone"),
                "/v1/audio/voices/Clone",
            )
        )
        await asyncio.wait_for(delete_started.wait(), timeout=1)

        release_upload.set()
        upload_response, delete_response = await asyncio.gather(
            upload_task,
            delete_task,
        )
        assert dispatched == ["POST", "DELETE"]
        assert upload_response.status_code == 200
        assert upload_response.body == b'{"name":"Clone"}'
        assert delete_response.status_code == 200
        assert delete_response.body == b'{"ok":true}'

        await voice_routing.start()
        try:
            for _ in range(100):
                if not voice_routing.requires_owner({"preset"}):
                    break
                await asyncio.sleep(0.01)
            assert voice_routing.requires_owner({"clone"})
        finally:
            await voice_routing.stop()


@pytest.mark.parametrize("upload_status", [200, 400])
@pytest.mark.asyncio
async def test_voice_upload_uses_the_stored_name_from_the_success_response(
    upload_status: int,
) -> None:
    boundary = "quoted-voice-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"\r\n\r\n'
        "Wrong\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; filename="sample.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
        "audio-bytes\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"\r\n\r\n'
        "Clone\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> dict[str, Any]:
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/voices",
            "headers": [
                (
                    b"content-type",
                    f'multipart/form-data; boundary="{boundary}"'.encode(),
                )
            ],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        },
        receive,
    )

    uploaded = False

    def handler(upstream_request: httpx.Request) -> httpx.Response:
        nonlocal uploaded
        if upstream_request.method == "GET":
            return httpx.Response(
                200,
                json={"uploaded_voices": [{"name": "Clone"}] if uploaded else []},
                request=upstream_request,
            )
        uploaded = upload_status == 200
        return httpx.Response(
            upload_status,
            json={"name": "Clone"},
            request=upstream_request,
        )

    config = _router_config()
    workers = build_workers(config.workers)
    workers[0].state = "healthy"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as async_client:
        voice_routing = _voice_routing(config, workers, async_client)
        proxy = proxy_module.ProxyHandler(
            config=config,
            workers=workers,
            selector=WorkerSelector(config.policy),
            client=async_client,
            voice_routing=voice_routing,
        )
        await voice_routing.start()
        try:
            for _ in range(100):
                if not voice_routing.requires_owner({"preset"}):
                    break
                await asyncio.sleep(0.01)
            assert not voice_routing.requires_owner({"clone"})

            response = await proxy.forward_model_request(request, "/v1/audio/voices")
            assert response.status_code == upload_status
            assert response.body == b'{"name":"Clone"}'
            for _ in range(100):
                if voice_routing.to_dict()["registry_state"] == "ready":
                    break
                await asyncio.sleep(0.01)
            assert not voice_routing.requires_owner({"wrong"})
            assert voice_routing.requires_owner({"clone"}) is (upload_status == 200)
        finally:
            await voice_routing.stop()


@pytest.mark.asyncio
async def test_uncertain_voice_upload_is_reconciled_before_balancing() -> None:
    committed = False

    def handler(upstream_request: httpx.Request) -> httpx.Response:
        nonlocal committed
        if upstream_request.method == "GET":
            uploaded = [{"name": "Clone"}] if committed else []
            return httpx.Response(
                200,
                json={"uploaded_voices": uploaded},
                request=upstream_request,
            )
        committed = True
        raise httpx.ReadError("response lost", request=upstream_request)

    messages = [{"type": "http.request", "body": b"upload", "more_body": False}]

    async def receive() -> dict[str, Any]:
        if messages:
            return messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/voices",
            "headers": [(b"content-type", b"multipart/form-data; boundary=x")],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        },
        receive,
    )
    config = _router_config(
        health_failure_threshold=3,
        health_check_interval_secs=1,
    )
    workers = build_workers(config.workers)
    workers[0].state = "healthy"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as async_client:
        voice_routing = _voice_routing(config, workers, async_client)
        proxy = proxy_module.ProxyHandler(
            config=config,
            workers=workers,
            selector=WorkerSelector(config.policy),
            client=async_client,
            voice_routing=voice_routing,
        )
        await voice_routing.start()
        try:
            for _ in range(100):
                if voice_routing.to_dict()["registry_state"] == "ready":
                    break
                await asyncio.sleep(0.01)

            response = await proxy.forward_model_request(request, "/v1/audio/voices")
            assert response.status_code == 502
            assert voice_routing.requires_owner({"Clone"})

            for _ in range(100):
                if voice_routing.to_dict()["registry_state"] == "ready":
                    break
                await asyncio.sleep(0.01)
            assert voice_routing.to_dict()["registry_state"] == "ready"
            assert voice_routing.requires_owner({"Clone"})
        finally:
            await voice_routing.stop()


@pytest.mark.asyncio
async def test_voice_upload_ownership_uses_the_completed_response() -> None:
    release_upload_body = asyncio.Event()
    uploaded = False
    boundary = "voice-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="name"\r\n\r\n'
        "Clone\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    request_messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> dict[str, Any]:
        if request_messages:
            return request_messages.pop(0)
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/voices",
            "headers": [
                (
                    b"content-type",
                    f"multipart/form-data; boundary={boundary}".encode(),
                )
            ],
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        },
        receive,
    )

    class StalledUploadBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            await release_upload_body.wait()
            yield b'{"name":"Clone"}'

    def handler(upstream_request: httpx.Request) -> httpx.Response:
        nonlocal uploaded
        if upstream_request.method == "GET":
            return httpx.Response(
                200,
                json={"uploaded_voices": [{"name": "Clone"}] if uploaded else []},
                request=upstream_request,
            )
        uploaded = True
        return httpx.Response(
            200,
            stream=StalledUploadBody(),
            request=upstream_request,
        )

    config = _router_config(voice_owner_worker_url="http://worker-b:8102")
    workers = build_workers(config.workers)
    workers[1].state = "healthy"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as async_client:
        voice_routing = _voice_routing(config, workers, async_client)
        proxy = proxy_module.ProxyHandler(
            config=config,
            workers=workers,
            selector=WorkerSelector(config.policy),
            client=async_client,
            voice_routing=voice_routing,
        )
        await voice_routing.start()
        try:
            for _ in range(100):
                if not voice_routing.requires_owner({"preset"}):
                    break
                await asyncio.sleep(0.01)
            assert not voice_routing.requires_owner({"preset"})

            response_task = asyncio.create_task(
                proxy.forward_model_request(request, "/v1/audio/voices")
            )
            await asyncio.sleep(0)
            state = voice_routing.to_dict()
            assert state["registry_state"] == "mutating"
            assert state["uploaded_voice_count"] == 0
            assert voice_routing.requires_owner({"Clone"})

            release_upload_body.set()
            response = await response_task
            assert response.status_code == 200
            for _ in range(100):
                if voice_routing.to_dict()["registry_state"] == "ready":
                    break
                await asyncio.sleep(0.01)
            assert voice_routing.requires_owner({"Clone"})
            assert response.body == b'{"name":"Clone"}'
        finally:
            release_upload_body.set()
            await voice_routing.stop()


class _FakeTTSUpstream:
    def __init__(self, messages: list[str | bytes]) -> None:
        self.sent: list[str | bytes] = []
        self.messages = messages
        self.close_code = 1000
        self.close_reason = ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.messages:
            raise StopAsyncIteration
        return self.messages.pop(0)

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def close(self, *, code: int = 1000) -> None:
        self.close_code = code


@pytest.mark.asyncio
async def test_tts_websocket_rejects_oversized_followup_before_upstream_send() -> None:
    class OversizedClient:
        application_state = websocket_proxy_module.WebSocketState.CONNECTED

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.receive", "text": "x" * 5}

    class WaitingUpstream(_FakeTTSUpstream):
        async def __anext__(self):
            await asyncio.Event().wait()

    upstream = WaitingUpstream([])
    outcome = await websocket_proxy_module._relay(
        OversizedClient(),
        upstream,
        max_client_message_bytes=4,
    )

    assert outcome == "client_message_too_large"
    assert upstream.sent == []
    assert upstream.close_code == 1009


@pytest.mark.asyncio
async def test_tts_websocket_relay_propagates_simultaneous_secondary_failure() -> None:
    class ConnectedWebSocket:
        application_state = websocket_proxy_module.WebSocketState.CONNECTED

    class Upstream:
        close_code = 1000
        close_reason = ""

    async def client_result():
        return "client_disconnected"

    async def upstream_failure():
        raise AssertionError("simultaneous relay defect")

    client_task = asyncio.create_task(client_result())
    upstream_task = asyncio.create_task(upstream_failure())
    await asyncio.sleep(0)
    assert client_task.done()
    assert upstream_task.done()

    with pytest.raises(AssertionError, match="simultaneous relay defect"):
        await websocket_proxy_module._coordinate_relay(
            ConnectedWebSocket(),
            Upstream(),
            client_task=client_task,
            upstream_task=upstream_task,
        )


def test_tts_websocket_is_pinned_and_counted_on_one_worker(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream = _FakeTTSUpstream(
        [
            json.dumps({"type": "session.configured"}),
            b"pcm",
            json.dumps({"type": "session.done"}),
        ]
    )

    def fake_connect(*args, **kwargs):
        assert args[0] == "ws://worker-a:8101/v1/audio/speech/stream"
        assert kwargs["compression"] is None
        assert kwargs["max_queue"] == 1
        assert kwargs["max_size"] == 512 * 1024 * 1024
        forwarded_headers = kwargs[websocket_proxy_module._WEBSOCKET_HEADERS_ARGUMENT]
        assert ROUTE_HEADER_NAMES.isdisjoint(forwarded_headers)
        assert forwarded_headers["x-client-header"] == "kept"
        return upstream

    monkeypatch.setattr(websocket_proxy_module, "websocket_connect", fake_connect)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"uploaded_voices": []}, request=request)
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(_router_config(max_payload_size=1024), client=async_client)

    with caplog.at_level(
        logging.INFO, logger="sglang_omni_router.python.websocket_proxy"
    ):
        with TestClient(app) as client:
            with client.websocket_connect(
                "/v1/audio/speech/stream",
                headers={
                    **{name: "ignored" for name in ROUTE_HEADER_NAMES},
                    "x-client-header": "kept",
                },
            ) as websocket:
                websocket.send_json(
                    {
                        "type": "session.config",
                        "model": None,
                        "voice": "default",
                    }
                )
                assert websocket.receive_json()["type"] == "session.configured"
                assert websocket.receive_bytes() == b"pcm"
                assert websocket.receive_json()["type"] == "session.done"

    assert len(upstream.sent) == 1
    assert json.loads(upstream.sent[0])["type"] == "session.config"
    worker = app.state.workers[0]
    assert worker.active_requests == 0
    assert worker.routed_requests_by_class == {"tts_websocket": 1}
    assert worker.successful_requests_by_class == {"tts_websocket": 1}
    terminal_logs = [
        record.getMessage()
        for record in caplog.records
        if "tts_websocket_completed" in record.getMessage()
    ]
    assert len(terminal_logs) == 1
    assert "outcome=completed status_code=200" in terminal_logs[0]


@pytest.mark.asyncio
async def test_tts_websocket_cancellation_emits_one_terminal_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CancellingWebSocket:
        application_state = websocket_proxy_module.WebSocketState.CONNECTED
        client_state = websocket_proxy_module.WebSocketState.CONNECTED
        headers: dict[str, str] = {}
        url = URL("ws://router/v1/audio/speech/stream")

        async def accept(self) -> None:
            pass

        async def receive(self) -> dict[str, Any]:
            return {"type": "websocket.receive", "text": "oversized"}

        async def send_json(self, payload: dict[str, Any]) -> None:
            raise asyncio.CancelledError

    config = _router_config(max_payload_size=1)
    workers = build_workers(config.workers)
    async with httpx.AsyncClient() as client:
        proxy = websocket_proxy_module.TTSWebSocketProxy(
            config=config,
            workers=workers,
            selector=WorkerSelector(config.policy),
            admission=proxy_module.AdmissionController(config.effective_max_inflight),
            voice_routing=_voice_routing(config, workers, client),
        )
        with caplog.at_level(
            logging.INFO,
            logger="sglang_omni_router.python.websocket_proxy",
        ):
            with pytest.raises(asyncio.CancelledError):
                await proxy.forward(CancellingWebSocket())

    terminal_logs = [
        record.getMessage()
        for record in caplog.records
        if "tts_websocket_completed" in record.getMessage()
    ]
    assert len(terminal_logs) == 1
    assert "outcome=client_message_too_large status_code=413" in terminal_logs[0]


def test_tts_websocket_overload_records_a_terminal_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(
        _router_config(max_inflight=1),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"status": "healthy"},
                    request=request,
                )
            )
        ),
    )
    with TestClient(app) as client:
        assert app.state.proxy.admission.try_acquire()
        with caplog.at_level(
            logging.WARNING,
            logger="sglang_omni_router.python.websocket_proxy",
        ):
            with client.websocket_connect("/v1/audio/speech/stream") as websocket:
                error = websocket.receive_json()
                close = websocket.receive()
        app.state.proxy.admission.release()

    assert error["error_type"] == "overloaded_error"
    assert close["code"] == 1013
    terminal_logs = [
        record.getMessage()
        for record in caplog.records
        if "tts_websocket_completed" in record.getMessage()
    ]
    assert len(terminal_logs) == 1
    assert "worker=- outcome=router_overloaded status_code=503" in terminal_logs[0]


def test_tts_websocket_relay_uses_the_installed_websockets_client() -> None:
    from websockets.sync.server import serve

    upstream_messages: list[dict[str, Any]] = []

    def upstream_handler(connection) -> None:
        upstream_messages.append(json.loads(connection.recv()))
        connection.send(json.dumps({"type": "session.done"}))

    with serve(upstream_handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        try:

            def handler(request: httpx.Request) -> httpx.Response:
                if request.url.path == "/health":
                    return httpx.Response(
                        200,
                        json={"status": "healthy"},
                        request=request,
                    )
                raise AssertionError(f"unexpected HTTP request: {request.url.path}")

            app = create_app(
                _router_config(
                    worker_configs=[WorkerConfig(url=f"http://127.0.0.1:{port}")]
                ),
                client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            )
            with TestClient(app) as client:
                with client.websocket_connect("/v1/audio/speech/stream") as websocket:
                    websocket.send_json({"type": "session.config", "voice": "default"})
                    assert websocket.receive_json()["type"] == "session.done"
        finally:
            server.shutdown()
            server_thread.join(timeout=5)

    assert not server_thread.is_alive()
    assert upstream_messages == [{"type": "session.config", "voice": "default"}]


def test_tts_websocket_explicit_reference_bypasses_uploaded_voice_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _FakeTTSUpstream([json.dumps({"type": "session.done"})])

    def fake_connect(url: str, **kwargs):
        assert url == "ws://worker-a:8101/v1/audio/speech/stream"
        return upstream

    monkeypatch.setattr(websocket_proxy_module, "websocket_connect", fake_connect)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(
                200,
                json={"uploaded_voices": [{"name": "Clone"}]},
                request=request,
            )
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    app = create_app(
        _router_config(voice_owner_worker_url="http://worker-b:8102"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with TestClient(app) as client:
        _wait_for_router_ready(client)
        assert client.get("/v1/audio/voices").status_code == 200
        registry = _wait_for_voice_registry_ready(client)
        assert registry["uploaded_voice_count"] == 1
        disabled = client.put(
            f"/workers/{worker_id_from_url('http://worker-b:8102')}",
            json={"disabled": True},
        )
        assert disabled.status_code == 200
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json(
                {
                    "type": "session.config",
                    "voice": "Clone",
                    "ref_audio": "data:audio/wav;base64,AAAA",
                }
            )
            assert websocket.receive_json()["type"] == "session.done"

    assert len(upstream.sent) == 1


def test_tts_websocket_protocol_rejection_does_not_evict_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _FakeTTSUpstream(
        [
            json.dumps(
                {
                    "type": "error",
                    "message": "first WebSocket message must be session.config",
                    "error_type": "invalid_request_error",
                    "param": "type",
                }
            )
        ]
    )
    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: upstream,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"uploaded_voices": []}, request=request)
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    app = create_app(
        _router_config(health_failure_threshold=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json({"type": "not.session.config"})
            error = websocket.receive_json()
            assert error["type"] == "error"
            assert error["error_type"] == "invalid_request_error"
            assert websocket.receive()["type"] == "websocket.close"

    worker = app.state.workers[0]
    assert worker.is_healthy
    assert worker.consecutive_failures == 0
    assert worker.failed_requests_by_class == {"tts_websocket": 1}


def test_tts_websocket_sender_close_drains_terminal_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UpstreamConnectionClosed(Exception):
        pass

    class ClosingUpstream:
        close_code = 1000
        close_reason = ""

        def __init__(self) -> None:
            self.send_count = 0
            self.send_observed_close = asyncio.Event()
            self.messages = [
                json.dumps(
                    {
                        "type": "error",
                        "message": "recoverable input error",
                        "error_type": "invalid_request_error",
                    }
                ),
                json.dumps({"type": "session.done"}),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self.send_observed_close.wait()
            if not self.messages:
                raise StopAsyncIteration
            return self.messages.pop(0)

        async def send(self, message: str | bytes) -> None:
            self.send_count += 1
            if self.send_count > 1:
                self.send_observed_close.set()
                raise UpstreamConnectionClosed

    upstream = ClosingUpstream()
    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: upstream,
    )
    monkeypatch.setattr(
        websocket_proxy_module,
        "ConnectionClosed",
        UpstreamConnectionClosed,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"uploaded_voices": []}, request=request)
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    app = create_app(
        _router_config(health_failure_threshold=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json(
                {
                    "type": "session.config",
                    "voice": "default",
                }
            )
            websocket.send_json({"type": "input.text", "text": "late message"})
            assert websocket.receive_json()["type"] == "error"
            assert websocket.receive_json()["type"] == "session.done"
            assert websocket.receive()["type"] == "websocket.close"

    worker = app.state.workers[0]
    assert worker.is_healthy
    assert worker.consecutive_failures == 0
    assert worker.successful_requests_by_class == {"tts_websocket": 1}


def test_tts_websocket_failed_audio_is_not_counted_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = _FakeTTSUpstream(
        [
            json.dumps(
                {
                    "type": "error",
                    "message": "generation failed",
                    "error_type": "server_error",
                }
            ),
            json.dumps({"type": "audio.done", "error": True}),
            json.dumps({"type": "session.done"}),
        ]
    )
    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: upstream,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"uploaded_voices": []}, request=request)
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    app = create_app(
        _router_config(health_failure_threshold=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json({"type": "session.config", "voice": "default"})
            assert websocket.receive_json()["type"] == "error"
            assert websocket.receive_json()["type"] == "audio.done"
            assert websocket.receive_json()["type"] == "session.done"
            assert websocket.receive()["type"] == "websocket.close"

    worker = app.state.workers[0]
    assert worker.is_healthy
    assert worker.consecutive_failures == 0
    assert worker.failed_requests_by_class == {"tts_websocket": 1}


@pytest.mark.asyncio
async def test_tts_websocket_client_disconnect_is_counted_as_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WaitingUpstream(_FakeTTSUpstream):
        def __init__(self) -> None:
            super().__init__([json.dumps({"type": "session.configured"})])
            self.closed = asyncio.Event()

        async def __anext__(self):
            if self.messages:
                return self.messages.pop(0)
            await self.closed.wait()
            raise StopAsyncIteration

        async def close(self, *, code: int = 1000) -> None:
            self.close_code = code
            self.closed.set()

    class DisconnectingWebSocket:
        def __init__(self) -> None:
            self.application_state = websocket_proxy_module.WebSocketState.CONNECTED
            self.client_state = websocket_proxy_module.WebSocketState.CONNECTED
            self.headers: dict[str, str] = {}
            self.url = URL("ws://router/v1/audio/speech/stream")
            self.first_message = True
            self.downstream_send_started = asyncio.Event()

        async def accept(self) -> None:
            return None

        async def receive(self) -> dict[str, Any]:
            if self.first_message:
                self.first_message = False
                return {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "session.config", "voice": "default"}),
                }
            await self.downstream_send_started.wait()
            return {"type": "websocket.disconnect"}

        async def send_text(self, _message: str) -> None:
            self.downstream_send_started.set()
            raise OSError("client disconnected during send")

        async def close(self, *, code: int, reason: str = "") -> None:
            self.application_state = websocket_proxy_module.WebSocketState.DISCONNECTED

    upstream = WaitingUpstream()
    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: upstream,
    )

    config = _router_config(health_failure_threshold=1)
    workers = build_workers(config.workers)
    workers[0].state = "healthy"
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: None))
    voice_routing = _voice_routing(config, workers, async_client)
    proxy = proxy_module.ProxyHandler(
        config=config,
        workers=workers,
        selector=WorkerSelector(config.policy),
        client=async_client,
        voice_routing=voice_routing,
    )
    websocket_proxy = websocket_proxy_module.TTSWebSocketProxy(
        config=config,
        workers=workers,
        selector=WorkerSelector(config.policy),
        admission=proxy.admission,
        voice_routing=voice_routing,
    )
    try:
        await websocket_proxy.forward(DisconnectingWebSocket())
    finally:
        await async_client.aclose()

    assert workers[0].is_healthy
    assert workers[0].consecutive_failures == 0
    assert workers[0].failed_requests_by_class == {"tts_websocket": 1}


@pytest.mark.parametrize(
    (
        "handshake_status",
        "expected_error_type",
        "expected_close_code",
        "worker_remains_healthy",
    ),
    [
        (401, "upstream_error", 1008, True),
        (429, "overloaded_error", 1013, True),
        (500, "upstream_error", 1011, True),
        (502, "upstream_error", 1011, False),
        (503, "server_error", 1013, True),
    ],
)
def test_tts_websocket_handshake_status_controls_worker_health(
    monkeypatch: pytest.MonkeyPatch,
    handshake_status: int,
    expected_error_type: str,
    expected_close_code: int,
    worker_remains_healthy: bool,
) -> None:
    class RejectedHandshake(websocket_proxy_module.WebSocketException):
        class Response:
            def __init__(self, status_code: int) -> None:
                self.status_code = status_code

        response = Response(handshake_status)

    class RejectedConnection:
        async def __aenter__(self):
            raise RejectedHandshake

        async def __aexit__(self, exc_type, exc, traceback) -> None:
            return None

    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: RejectedConnection(),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"uploaded_voices": []}, request=request)
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    app = create_app(
        _router_config(health_failure_threshold=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json({"type": "session.config", "voice": "default"})
            error = websocket.receive_json()
            close = websocket.receive()

    assert error["type"] == "error"
    assert error["error_type"] == expected_error_type
    assert error["code"] == handshake_status
    assert close["type"] == "websocket.close"
    assert close["code"] == expected_close_code

    worker = app.state.workers[0]
    assert worker.is_healthy is worker_remains_healthy
    assert worker.consecutive_failures == (0 if worker_remains_healthy else 1)
    assert worker.failed_requests_by_class == {"tts_websocket": 1}


@pytest.mark.asyncio
async def test_tts_websocket_unexpected_relay_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingUpstream(_FakeTTSUpstream):
        async def __anext__(self):
            raise AssertionError("unexpected relay defect")

    class WaitingWebSocket:
        application_state = websocket_proxy_module.WebSocketState.CONNECTED
        client_state = websocket_proxy_module.WebSocketState.CONNECTED
        headers: dict[str, str] = {}
        url = URL("ws://router/v1/audio/speech/stream")

        def __init__(self) -> None:
            self._first_message = True

        async def accept(self) -> None:
            return None

        async def receive(self) -> dict[str, Any]:
            if self._first_message:
                self._first_message = False
                return {
                    "type": "websocket.receive",
                    "text": json.dumps({"type": "session.config", "voice": "default"}),
                }
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send_text(self, _message: str) -> None:
            return None

        async def send_bytes(self, _message: bytes) -> None:
            return None

        async def close(self, *, code: int, reason: str = "") -> None:
            self.application_state = websocket_proxy_module.WebSocketState.DISCONNECTED

    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: FailingUpstream([]),
    )
    config = _router_config(health_failure_threshold=1)
    workers = build_workers(config.workers)
    workers[0].state = "healthy"
    async with httpx.AsyncClient() as async_client:
        voice_routing = _voice_routing(config, workers, async_client)
        proxy = proxy_module.ProxyHandler(
            config=config,
            workers=workers,
            selector=WorkerSelector(config.policy),
            client=async_client,
            voice_routing=voice_routing,
        )
        websocket_proxy = websocket_proxy_module.TTSWebSocketProxy(
            config=config,
            workers=workers,
            selector=WorkerSelector(config.policy),
            admission=proxy.admission,
            voice_routing=voice_routing,
        )

        with pytest.raises(AssertionError, match="unexpected relay defect"):
            await websocket_proxy.forward(WaitingWebSocket())

    assert workers[0].is_healthy
    assert workers[0].consecutive_failures == 0
    assert workers[0].active_requests == 0
    assert workers[0].routed_requests == 0
    assert proxy.admission.inflight == 0


def test_tts_websocket_policy_close_does_not_evict_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PolicyConnectionClosed(Exception):
        class ReceivedClose:
            code = 1008

        rcvd = ReceivedClose()

    class PolicyClosingUpstream(_FakeTTSUpstream):
        def __init__(self) -> None:
            super().__init__([])
            self.close_code = 1008

        async def __anext__(self):
            raise PolicyConnectionClosed

    upstream = PolicyClosingUpstream()
    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: upstream,
    )
    monkeypatch.setattr(
        websocket_proxy_module,
        "ConnectionClosed",
        PolicyConnectionClosed,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"uploaded_voices": []}, request=request)
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    app = create_app(
        _router_config(health_failure_threshold=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json({"type": "session.config", "voice": "default"})
            assert websocket.receive()["type"] == "websocket.close"

    worker = app.state.workers[0]
    assert worker.is_healthy
    assert worker.consecutive_failures == 0
    assert worker.failed_requests_by_class == {"tts_websocket": 1}


def test_tts_websocket_policy_close_during_initial_send_is_application_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PolicyConnectionClosed(websocket_proxy_module.WebSocketException):
        class ReceivedClose:
            code = 1008

        rcvd = ReceivedClose()

    class PolicyClosingUpstream(_FakeTTSUpstream):
        async def send(self, message: str | bytes) -> None:
            raise PolicyConnectionClosed

    monkeypatch.setattr(
        websocket_proxy_module,
        "websocket_connect",
        lambda *args, **kwargs: PolicyClosingUpstream([]),
    )
    monkeypatch.setattr(
        websocket_proxy_module,
        "ConnectionClosed",
        PolicyConnectionClosed,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, request=request)
        if request.url.path == "/v1/audio/voices":
            return httpx.Response(200, json={"uploaded_voices": []}, request=request)
        raise AssertionError(f"unexpected HTTP request path: {request.url.path}")

    app = create_app(
        _router_config(health_failure_threshold=1),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/audio/speech/stream") as websocket:
            websocket.send_json({"type": "session.config", "voice": "default"})
            error = websocket.receive_json()
            close = websocket.receive()

    assert error["type"] == "error"
    assert error["error_type"] == "upstream_error"
    assert error["code"] == 400
    assert close["type"] == "websocket.close"
    assert close["code"] == 1008

    worker = app.state.workers[0]
    assert worker.is_healthy
    assert worker.consecutive_failures == 0
    assert worker.active_requests == 0
    assert worker.failed_requests_by_class == {"tts_websocket": 1}
