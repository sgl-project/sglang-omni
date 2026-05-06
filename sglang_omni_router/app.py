# SPDX-License-Identifier: Apache-2.0
"""FastAPI application wiring for the external Omni router."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from sglang_omni_router.config import RouterConfig
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.proxy import ProxyHandler, RouteLogger
from sglang_omni_router.selector import WorkerSelector
from sglang_omni_router.worker import Worker, build_workers


def create_app(
    config: RouterConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> FastAPI:
    workers = build_workers(config.worker_urls)
    timeout = httpx.Timeout(config.request_timeout_secs)
    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=timeout)
    health_checker = HealthChecker(workers=workers, config=config, client=client)
    selector = WorkerSelector(config.policy)
    route_logger = RouteLogger(config.route_log_path)
    proxy = ProxyHandler(
        config=config,
        workers=workers,
        selector=selector,
        client=client,
        route_logger=route_logger,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_config = config
        app.state.workers = workers
        app.state.http_client = client
        app.state.health_checker = health_checker
        app.state.proxy = proxy
        # Initial probes satisfy the configured success threshold before the
        # router starts accepting traffic; later probes run in the background.
        for _ in range(config.health_success_threshold):
            await health_checker.check_once()
        await health_checker.start()
        try:
            yield
        finally:
            await health_checker.stop()
            if owns_client:
                await client.aclose()

    app = FastAPI(title="sglang-omni-router", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app, workers, proxy)
    return app


def register_routes(app: FastAPI, workers: list[Worker], proxy: ProxyHandler) -> None:
    @app.get("/live")
    async def live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return JSONResponse(_pool_summary(workers, status="ready"))

    @app.get("/health")
    async def health() -> JSONResponse:
        healthy = sum(1 for worker in workers if worker.is_healthy)
        status_code = 200 if healthy > 0 else 503
        status = "healthy" if healthy > 0 else "unhealthy"
        return JSONResponse(
            _pool_summary(workers, status=status), status_code=status_code
        )

    @app.get("/workers")
    async def list_workers() -> JSONResponse:
        return JSONResponse(_pool_summary(workers, status="ok", include_workers=True))

    @app.get("/workers/{worker_id:path}")
    async def get_worker(worker_id: str) -> JSONResponse:
        worker = _find_worker(workers, worker_id)
        if worker is None:
            return JSONResponse(
                status_code=404,
                content={"error": {"message": "worker not found"}},
            )
        return JSONResponse(worker.to_dict())

    @app.get("/v1/models")
    async def models() -> JSONResponse:
        return await _merge_models(workers, app.state.http_client)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await proxy.forward_model_request(request, "/v1/chat/completions")

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request) -> Response:
        return await proxy.forward_model_request(request, "/v1/audio/speech")


def _pool_summary(
    workers: list[Worker],
    *,
    status: str,
    include_workers: bool = True,
) -> dict[str, Any]:
    healthy = sum(1 for worker in workers if worker.is_healthy)
    payload: dict[str, Any] = {
        "status": status,
        "healthy_workers": healthy,
        "unhealthy_workers": len(workers) - healthy,
        "total_workers": len(workers),
    }
    if include_workers:
        payload["workers"] = [worker.to_dict() for worker in workers]
    return payload


def _find_worker(workers: list[Worker], worker_id: str) -> Worker | None:
    decoded = unquote(worker_id)
    for worker in workers:
        if worker.worker_id == worker_id or worker.url == decoded:
            return worker
    return None


async def _merge_models(
    workers: list[Worker], client: httpx.AsyncClient
) -> JSONResponse:
    healthy_workers = [worker for worker in workers if worker.is_healthy]
    if not healthy_workers:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "no healthy upstream"}},
        )

    cards_by_id: dict[str, dict[str, Any]] = {}
    for worker in healthy_workers:
        try:
            response = await client.get(f"{worker.url}/v1/models")
        except Exception:
            continue
        if not 200 <= response.status_code < 300:
            continue
        try:
            payload = response.json()
        except Exception:
            continue
        for card in payload.get("data", []):
            if isinstance(card, dict) and isinstance(card.get("id"), str):
                cards_by_id.setdefault(card["id"], card)

    if not cards_by_id:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "no models available"}},
        )

    return JSONResponse({"object": "list", "data": list(cards_by_id.values())})
