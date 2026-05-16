# SPDX-License-Identifier: Apache-2.0
"""FastAPI application wiring for the external Omni router."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from sglang_omni_router.config import RouterConfig, WorkerConfig
from sglang_omni_router.health import HealthChecker
from sglang_omni_router.proxy import ProxyHandler, filter_request_headers
from sglang_omni_router.selector import WorkerSelector
from sglang_omni_router.worker import (
    HEALTH_STATE_UNHEALTHY,
    HEALTH_STATE_UNKNOWN,
    Worker,
    build_workers,
)

logger = logging.getLogger(__name__)


def create_app(
    config: RouterConfig,
    *,
    client: httpx.AsyncClient | None = None,
    health_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    workers = build_workers(config.workers)
    timeout = httpx.Timeout(config.request_timeout_secs)
    owns_client = client is None
    if client is None:
        limits = httpx.Limits(max_connections=config.max_connections)
        client = httpx.AsyncClient(timeout=timeout, limits=limits)
    owns_health_client = health_client is None and owns_client
    if health_client is None:
        if owns_client:
            health_limits = httpx.Limits(
                max_connections=max(1, len(workers)),
                max_keepalive_connections=max(1, len(workers)),
            )
            health_client = httpx.AsyncClient(
                timeout=httpx.Timeout(config.health_check_timeout_secs),
                limits=health_limits,
            )
        else:
            health_client = client
    health_checker = HealthChecker(
        workers=workers,
        config=config,
        client=health_client,
    )
    selector = WorkerSelector(config.policy)
    proxy = ProxyHandler(
        config=config,
        workers=workers,
        selector=selector,
        client=client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.router_config = config
        app.state.workers = workers
        app.state.http_client = client
        app.state.health_http_client = health_client
        app.state.health_checker = health_checker
        app.state.proxy = proxy
        await health_checker.start()
        try:
            yield
        finally:
            await health_checker.stop()
            if owns_health_client:
                await health_client.aclose()
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

    register_routes(app, workers, proxy, config)
    return app


def register_routes(
    app: FastAPI,
    workers: list[Worker],
    proxy: ProxyHandler,
    config: RouterConfig,
) -> None:
    @app.get("/live")
    async def live() -> JSONResponse:
        return JSONResponse({"status": "alive"})

    @app.get("/ready")
    async def ready() -> JSONResponse:
        return _worker_pool_status_response(
            workers,
            available_status="ready",
            unavailable_status="not_ready",
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        return _worker_pool_status_response(
            workers,
            available_status="healthy",
            unavailable_status="unhealthy",
        )

    @app.post("/workers")
    async def create_worker(request: Request) -> JSONResponse:
        payload, error = await _read_json_object(request)
        if error is not None:
            return error
        allowed_fields = {"url", "worker_url", "capabilities", "model"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            return _error_response(
                400, f"unsupported fields: {', '.join(unknown_fields)}"
            )
        worker_url = request.query_params.get("url") or request.query_params.get(
            "worker_url"
        )
        worker_url = worker_url or _string_or_none(
            payload.get("url") or payload.get("worker_url")
        )
        if worker_url is None:
            return _error_response(400, "worker url is required")
        worker_config_kwargs: dict[str, Any] = {
            "url": worker_url,
            "model": payload.get("model"),
        }
        if "capabilities" in payload:
            worker_config_kwargs["capabilities"] = payload["capabilities"]
        try:
            worker_config = WorkerConfig(**worker_config_kwargs)
        except ValidationError as exc:
            return _error_response(400, str(exc))
        if any(worker.url == worker_config.url for worker in workers):
            return _error_response(409, "worker already registered")

        worker = Worker(config=worker_config)
        workers.append(worker)
        await app.state.health_checker.check_worker_health(worker)
        logger.info(
            f"worker_registered worker={worker.display_id} url={worker.url} "
            f"model={worker.model or '-'} "
            f"capabilities={','.join(sorted(worker.capabilities))} "
            f"health_state={worker.state} disabled={worker.disabled}",
        )
        return JSONResponse({"status": "ok", "worker": worker.to_dict()})

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

    @app.put("/workers/{worker_id:path}")
    async def update_worker(worker_id: str, request: Request) -> JSONResponse:
        worker = _find_worker(workers, worker_id)
        if worker is None:
            return _error_response(404, "worker not found")

        payload, error = await _read_json_object(request)
        if error is not None:
            return error
        allowed_fields = {"is_dead", "disabled", "capabilities", "model"}
        unknown_fields = sorted(set(payload) - allowed_fields)
        if unknown_fields:
            return _error_response(
                400, f"unsupported fields: {', '.join(unknown_fields)}"
            )
        if not payload:
            return _error_response(400, "at least one worker field is required")

        requested_is_dead: bool | None = None
        requested_disabled: bool | None = None

        if "is_dead" in payload:
            requested_is_dead = payload["is_dead"]
            if not isinstance(requested_is_dead, bool):
                return _error_response(400, "is_dead must be a boolean")

        if "disabled" in payload:
            requested_disabled = payload["disabled"]
            if not isinstance(requested_disabled, bool):
                return _error_response(400, "disabled must be a boolean")

        next_config = worker.config

        if "capabilities" in payload or "model" in payload:
            try:
                next_config = WorkerConfig(
                    url=worker.url,
                    model=(
                        payload.get("model") if "model" in payload else worker.model
                    ),
                    capabilities=(
                        payload.get("capabilities")
                        if "capabilities" in payload
                        else worker.capabilities
                    ),
                )
            except ValidationError as exc:
                return _error_response(400, str(exc))

        worker.replace_config(next_config)

        if requested_disabled is not None:
            worker.set_disabled(requested_disabled)

        if requested_is_dead is not None:
            if requested_is_dead:
                worker.mark_dead()
            else:
                worker.clear_dead()
                await app.state.health_checker.check_worker_health(worker)

        logger.info(
            f"worker_updated worker={worker.display_id} url={worker.url} "
            f"model={worker.model or '-'} "
            f"capabilities={','.join(sorted(worker.capabilities))} "
            f"health_state={worker.state} disabled={worker.disabled}",
        )
        return JSONResponse({"status": "ok", "worker": worker.to_dict()})

    @app.delete("/workers/{worker_id:path}")
    async def delete_worker(worker_id: str) -> JSONResponse:
        worker = _find_worker(workers, worker_id)
        if worker is None:
            return _error_response(404, "worker not found")
        workers.remove(worker)
        logger.info(
            f"worker_deleted worker={worker.display_id} url={worker.url} "
            f"model={worker.model or '-'}",
        )
        return JSONResponse({"status": "ok", "worker_id": worker.worker_id})

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        return await _merge_models(
            workers,
            app.state.http_client,
            request,
            timeout_secs=config.health_check_timeout_secs,
        )

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
    dead = sum(1 for worker in workers if worker.is_dead)
    unhealthy = sum(1 for worker in workers if worker.state == HEALTH_STATE_UNHEALTHY)
    unknown = sum(1 for worker in workers if worker.state == HEALTH_STATE_UNKNOWN)
    disabled = sum(1 for worker in workers if worker.disabled)
    routable = sum(1 for worker in workers if worker.is_routable)
    payload: dict[str, Any] = {
        "status": status,
        "healthy_workers": healthy,
        "dead_workers": dead,
        "disabled_workers": disabled,
        "routable_workers": routable,
        "unhealthy_workers": unhealthy,
        "unknown_workers": unknown,
        "total_workers": len(workers),
    }
    if include_workers:
        payload["workers"] = [worker.to_dict() for worker in workers]
    return payload


def _worker_pool_status_response(
    workers: list[Worker],
    *,
    available_status: str,
    unavailable_status: str,
) -> JSONResponse:
    routable = sum(1 for worker in workers if worker.is_routable)
    status_code = 200 if routable > 0 else 503
    status = available_status if routable > 0 else unavailable_status
    return JSONResponse(
        _pool_summary(workers, status=status),
        status_code=status_code,
    )


def _find_worker(workers: list[Worker], worker_id: str) -> Worker | None:
    decoded = unquote(worker_id)
    for worker in workers:
        if worker.worker_id == worker_id or worker.url == decoded:
            return worker
    return None


async def _read_json_object(
    request: Request,
) -> tuple[dict[str, Any], JSONResponse | None]:
    body = await request.body()
    if not body:
        return {}, None
    try:
        payload = await request.json()
    except Exception:
        return {}, _error_response(400, "invalid JSON body")
    if not isinstance(payload, dict):
        return {}, _error_response(400, "request body must be a JSON object")
    return payload, None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _error_response(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message}},
    )


async def _merge_models(
    workers: list[Worker],
    client: httpx.AsyncClient,
    request: Request,
    *,
    timeout_secs: int,
) -> JSONResponse:
    routable_workers = [worker for worker in workers if worker.is_routable]
    if not routable_workers:
        return JSONResponse(
            status_code=503,
            content={"error": {"message": "no routable upstream"}},
        )

    request_headers = filter_request_headers(request)
    query = request.url.query
    cards_by_id: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    worker_results = await asyncio.gather(
        *(
            _fetch_worker_models(
                worker,
                client,
                request_headers,
                query,
                timeout_secs=timeout_secs,
            )
            for worker in routable_workers
        )
    )
    for worker, data, error in worker_results:
        if error is not None:
            errors[worker.url] = error
            continue
        if data is None:
            errors[worker.url] = "invalid models payload"
            continue
        for card in data:
            if not isinstance(card, dict):
                continue
            model_id = card.get("id") or card.get("model")
            dedupe_key = (
                model_id
                if isinstance(model_id, str) and model_id
                else json.dumps(card, sort_keys=True)
            )
            cards_by_id.setdefault(dedupe_key, card)

    if not cards_by_id:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "failed to fetch models from workers",
                    "details": errors,
                }
            },
        )

    return JSONResponse({"object": "list", "data": list(cards_by_id.values())})


async def _fetch_worker_models(
    worker: Worker,
    client: httpx.AsyncClient,
    request_headers: dict[str, str],
    query: bytes,
    *,
    timeout_secs: int,
) -> tuple[Worker, list[Any] | None, str | None]:
    url = f"{worker.url}/v1/models" if not query else f"{worker.url}/v1/models?{query}"
    try:
        response = await client.get(
            url,
            headers=request_headers,
            timeout=timeout_secs,
        )
    except Exception as exc:
        return worker, None, type(exc).__name__
    if not 200 <= response.status_code < 300:
        return worker, None, f"status={response.status_code}"
    try:
        payload = response.json()
    except Exception as exc:
        return worker, None, type(exc).__name__
    data = payload.get("data")
    if not isinstance(data, list):
        return worker, None, "invalid models payload"
    return worker, data, None
