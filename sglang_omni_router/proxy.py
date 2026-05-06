from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni_router.config import Capability, RouterConfig
from sglang_omni_router.logging import RouteLogger
from sglang_omni_router.selector import NoEligibleWorkerError, WorkerSelector
from sglang_omni_router.worker import Worker

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
REQUEST_HEADERS_TO_STRIP = HOP_BY_HOP_HEADERS | {
    "host",
    "content-length",
    "accept-encoding",
}
RESPONSE_HEADERS_TO_STRIP = HOP_BY_HOP_HEADERS | {
    "content-length",
}
JSON_METADATA_LIMIT_BYTES = 1024 * 1024


@dataclass
class RouteMetadata:
    request_id: str
    model: str | None
    stream: bool
    idempotency_key_present: bool


def infer_capability(path: str) -> Capability:
    if path == "/v1/audio/speech":
        return "speech"
    return "chat"


def extract_route_metadata(request: Request, body: bytes) -> RouteMetadata:
    request_id = (
        request.headers.get("x-sglang-omni-request-id")
        or request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
    )
    model: str | None = None
    stream = False

    content_type = request.headers.get("content-type", "")
    if "json" in content_type and len(body) <= JSON_METADATA_LIMIT_BYTES:
        try:
            payload = json.loads(body)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            request_id = request_id or _string_or_none(payload.get("request_id"))
            model = _string_or_none(payload.get("model"))
            stream = payload.get("stream") is True

    return RouteMetadata(
        request_id=request_id or str(uuid.uuid4()),
        model=model,
        stream=stream,
        idempotency_key_present=bool(request.headers.get("idempotency-key")),
    )


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def filter_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in REQUEST_HEADERS_TO_STRIP
    }


def filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in RESPONSE_HEADERS_TO_STRIP
    }


class ProxyHandler:
    def __init__(
        self,
        *,
        config: RouterConfig,
        workers: list[Worker],
        selector: WorkerSelector,
        client: httpx.AsyncClient,
        route_logger: RouteLogger,
    ) -> None:
        self._config = config
        self._workers = workers
        self._selector = selector
        self._client = client
        self._route_logger = route_logger

    async def forward_model_request(self, request: Request, path: str) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None and _exceeds_max_size(
            content_length, self._config.max_payload_size
        ):
            return JSONResponse(
                status_code=413,
                content={"error": {"message": "payload too large"}},
            )

        body = await request.body()
        if len(body) > self._config.max_payload_size:
            return JSONResponse(
                status_code=413,
                content={"error": {"message": "payload too large"}},
            )

        metadata = extract_route_metadata(request, body)
        capability = infer_capability(path)
        try:
            worker = self._selector.select(self._workers, capability=capability)
        except NoEligibleWorkerError:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "no healthy upstream"}},
            )

        if metadata.stream:
            return await self._forward_streaming(request, path, body, metadata, worker)
        return await self._forward_non_streaming(request, path, body, metadata, worker)

    async def _forward_non_streaming(
        self,
        request: Request,
        path: str,
        body: bytes,
        metadata: RouteMetadata,
        worker: Worker,
    ) -> Response:
        start = time.monotonic()
        status_code: int | None = None
        response_bytes = 0
        error_type: str | None = None
        worker.increment_active()
        try:
            response = await self._client.request(
                request.method,
                f"{worker.url}{path}",
                content=body,
                headers=filter_request_headers(request),
            )
            status_code = response.status_code
            response_bytes = len(response.content)
            headers = filter_response_headers(response.headers)
            headers.update(self._diagnostic_headers(worker, metadata))
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=headers,
                media_type=response.headers.get("content-type"),
            )
        except Exception as exc:
            error_type = type(exc).__name__
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "upstream request failed"}},
                headers=self._diagnostic_headers(worker, metadata),
            )
        finally:
            worker.decrement_active()
            self._log_route(
                event="request_complete",
                request=request,
                metadata=metadata,
                worker=worker,
                path=path,
                status_code=status_code,
                start=start,
                request_bytes=len(body),
                response_bytes=response_bytes,
                completed=error_type is None,
                client_disconnected=False,
                error_type=error_type,
                ttfb_s=None,
            )

    async def _forward_streaming(
        self,
        request: Request,
        path: str,
        body: bytes,
        metadata: RouteMetadata,
        worker: Worker,
    ) -> StreamingResponse | JSONResponse:
        start = time.monotonic()
        worker.increment_active()
        try:
            upstream_request = self._client.build_request(
                request.method,
                f"{worker.url}{path}",
                content=body,
                headers=filter_request_headers(request),
            )
            upstream = await self._client.send(upstream_request, stream=True)
        except Exception as exc:
            worker.decrement_active()
            self._log_route(
                event="request_complete",
                request=request,
                metadata=metadata,
                worker=worker,
                path=path,
                status_code=None,
                start=start,
                request_bytes=len(body),
                response_bytes=0,
                completed=False,
                client_disconnected=False,
                error_type=type(exc).__name__,
                ttfb_s=None,
            )
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "upstream request failed"}},
                headers=self._diagnostic_headers(worker, metadata),
            )

        response_bytes = 0
        first_byte_at: float | None = None
        completed = False
        client_disconnected = False
        error_type: str | None = None

        async def iter_bytes():
            nonlocal response_bytes, first_byte_at, completed
            nonlocal client_disconnected, error_type
            try:
                async for chunk in upstream.aiter_bytes():
                    if chunk and first_byte_at is None:
                        first_byte_at = time.monotonic()
                    response_bytes += len(chunk)
                    yield chunk
                completed = True
            except asyncio.CancelledError:
                client_disconnected = True
                raise
            except Exception as exc:
                error_type = type(exc).__name__
                raise
            finally:
                await upstream.aclose()
                worker.decrement_active()
                self._log_route(
                    event="stream_complete" if completed else "stream_error",
                    request=request,
                    metadata=metadata,
                    worker=worker,
                    path=path,
                    status_code=upstream.status_code,
                    start=start,
                    request_bytes=len(body),
                    response_bytes=response_bytes,
                    completed=completed,
                    client_disconnected=client_disconnected,
                    error_type=error_type,
                    ttfb_s=(
                        first_byte_at - start if first_byte_at is not None else None
                    ),
                )

        headers = filter_response_headers(upstream.headers)
        headers.update(self._diagnostic_headers(worker, metadata))
        return StreamingResponse(
            iter_bytes(),
            status_code=upstream.status_code,
            headers=headers,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
        )

    def _diagnostic_headers(
        self,
        worker: Worker,
        metadata: RouteMetadata,
    ) -> dict[str, str]:
        return {
            "X-SGLang-Omni-Worker": worker.worker_id,
            "X-SGLang-Omni-Request-ID": metadata.request_id,
            "X-SGLang-Omni-Route-Attempt": "1",
        }

    def _log_route(
        self,
        *,
        event: str,
        request: Request,
        metadata: RouteMetadata,
        worker: Worker,
        path: str,
        status_code: int | None,
        start: float,
        request_bytes: int,
        response_bytes: int,
        completed: bool,
        client_disconnected: bool,
        error_type: str | None,
        ttfb_s: float | None,
    ) -> None:
        self._route_logger.write(
            {
                "event": event,
                "request_id": metadata.request_id,
                "method": request.method,
                "path": path,
                "model": metadata.model,
                "stream": metadata.stream,
                "worker_id": worker.worker_id,
                "worker_url": worker.url,
                "policy": self._config.policy,
                "worker_state": worker.state,
                "attempt": 1,
                "status_code": status_code,
                "duration_s": time.monotonic() - start,
                "ttfb_s": ttfb_s,
                "request_bytes": request_bytes,
                "response_bytes": response_bytes,
                "completed": completed,
                "client_disconnected": client_disconnected,
                "error_type": error_type,
                "idempotency_key_present": metadata.idempotency_key_present,
            }
        )


def _exceeds_max_size(value: str, max_size: int) -> bool:
    try:
        return int(value) > max_size
    except ValueError:
        return True
