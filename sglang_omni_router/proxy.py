# SPDX-License-Identifier: Apache-2.0
"""Proxy request forwarding and response relay."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni_router.config import Capability, RouterConfig
from sglang_omni_router.selector import NoEligibleWorkerError, WorkerSelector
from sglang_omni_router.worker import Worker

logger = logging.getLogger(__name__)

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
BUFFERED_RESPONSE_HEADERS_TO_STRIP = RESPONSE_HEADERS_TO_STRIP | {
    "content-encoding",
}
WORKER_REQUEST_FAILURE_STATUS_CODES = {
    HTTPStatus.REQUEST_TIMEOUT.value,
    HTTPStatus.TOO_MANY_REQUESTS.value,
    HTTPStatus.INTERNAL_SERVER_ERROR.value,
    HTTPStatus.BAD_GATEWAY.value,
    HTTPStatus.SERVICE_UNAVAILABLE.value,
    HTTPStatus.GATEWAY_TIMEOUT.value,
}


class RouteMetadataError(ValueError):
    pass


@dataclass
class RouteMetadata:
    request_id: str
    model: str | None
    stream: bool
    required_capabilities: set[Capability]
    idempotency_key_present: bool


def infer_required_capabilities(
    path: str,
    payload: dict[str, Any] | None,
    *,
    stream: bool,
) -> set[Capability]:
    if path == "/v1/audio/speech":
        capabilities: set[Capability] = {"speech"}
        if stream:
            capabilities.add("streaming")
        return capabilities

    capabilities = {"chat"}
    if stream:
        capabilities.add("streaming")
    if not payload:
        return capabilities

    if _has_non_empty(payload.get("images")) or _has_non_empty(payload.get("image")):
        capabilities.add("image_input")
    if _has_non_empty(payload.get("audios")) or _has_non_empty(
        payload.get("audio_inputs")
    ):
        capabilities.add("audio_input")
    if _has_non_empty(payload.get("videos")) or _has_non_empty(payload.get("video")):
        capabilities.add("video_input")
    if _modalities_include_audio(payload) or _has_non_empty(payload.get("audio")):
        capabilities.add("audio_output")

    message_capabilities = _infer_message_part_capabilities(payload.get("messages"))
    capabilities.update(message_capabilities)
    return capabilities


def extract_route_metadata(request: Request, path: str, body: bytes) -> RouteMetadata:
    request_id = (
        request.headers.get("x-sglang-omni-request-id")
        or request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
    )
    model: str | None = None
    stream = False
    payload: dict[str, Any] | None = None

    content_type = request.headers.get("content-type", "").lower()
    if "json" in content_type and body:
        try:
            parsed_payload = json.loads(body)
        except Exception:
            raise RouteMetadataError("invalid JSON body") from None
        if not isinstance(parsed_payload, dict):
            raise RouteMetadataError("JSON request body must be an object")
        payload = parsed_payload
        request_id = request_id or _string_or_none(payload.get("request_id"))
        model = _string_or_none(payload.get("model"))
        stream = payload.get("stream") is True

    return RouteMetadata(
        request_id=request_id or str(uuid.uuid4()),
        model=model,
        stream=stream,
        required_capabilities=infer_required_capabilities(path, payload, stream=stream),
        idempotency_key_present=bool(request.headers.get("idempotency-key")),
    )


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _has_non_empty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, list, dict)):
        return bool(value)
    return True


def _modalities_include_audio(payload: dict[str, Any]) -> bool:
    modalities = payload.get("modalities")
    if not isinstance(modalities, list):
        return False
    return any(item == "audio" for item in modalities)


def _infer_message_part_capabilities(messages: Any) -> set[Capability]:
    capabilities: set[Capability] = set()
    if not isinstance(messages, list):
        return capabilities
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in {"image", "image_url", "input_image"}:
                capabilities.add("image_input")
            elif part_type in {"audio", "audio_url", "input_audio"}:
                capabilities.add("audio_input")
            elif part_type in {"video", "video_url", "input_video"}:
                capabilities.add("video_input")
    return capabilities


def filter_request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in REQUEST_HEADERS_TO_STRIP
    }


def filter_response_headers(
    headers: httpx.Headers,
    *,
    buffered: bool = False,
) -> dict[str, str]:
    headers_to_strip = (
        BUFFERED_RESPONSE_HEADERS_TO_STRIP if buffered else RESPONSE_HEADERS_TO_STRIP
    )
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in headers_to_strip
    }


def build_upstream_url(worker: Worker, path: str, request: Request) -> str:
    query = request.url.query
    return f"{worker.url}{path}" if not query else f"{worker.url}{path}?{query}"


class ProxyHandler:
    def __init__(
        self,
        *,
        config: RouterConfig,
        workers: list[Worker],
        selector: WorkerSelector,
        client: httpx.AsyncClient,
    ) -> None:
        self._config = config
        self._workers = workers
        self._selector = selector
        self._client = client

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

        try:
            metadata = extract_route_metadata(request, path, body)
        except RouteMetadataError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc)}},
            )
        try:
            worker = self._selector.select(
                self._workers,
                required_capabilities=metadata.required_capabilities,
                requested_model=metadata.model,
            )
        except NoEligibleWorkerError:
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "no eligible upstream"}},
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
        with worker.request_guard():
            start_time = time.perf_counter()
            try:
                response = await self._client.request(
                    request.method,
                    build_upstream_url(worker, path, request),
                    content=body,
                    headers=filter_request_headers(request),
                )
                if response.status_code in WORKER_REQUEST_FAILURE_STATUS_CODES:
                    self._record_worker_request_failure(
                        worker,
                        status_code=response.status_code,
                        error=_response_error(response),
                    )
                outcome = _response_outcome(response.status_code)
                self._log_route_completion(
                    worker=worker,
                    path=path,
                    metadata=metadata,
                    status_code=response.status_code,
                    outcome=outcome,
                    start_time=start_time,
                )
                headers = filter_response_headers(response.headers, buffered=True)
                headers.update(self._diagnostic_headers(worker, metadata))
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.headers.get("content-type"),
                )
            except Exception as exc:
                self._record_worker_request_failure(
                    worker,
                    error=type(exc).__name__,
                )
                self._log_route_completion(
                    worker=worker,
                    path=path,
                    metadata=metadata,
                    status_code=502,
                    outcome="upstream_error",
                    start_time=start_time,
                )
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "upstream request failed"}},
                    headers=self._diagnostic_headers(worker, metadata),
                )

    async def _forward_streaming(
        self,
        request: Request,
        path: str,
        body: bytes,
        metadata: RouteMetadata,
        worker: Worker,
    ) -> StreamingResponse | JSONResponse:
        worker.increment_active()
        start_time = time.perf_counter()
        try:
            upstream_request = self._client.build_request(
                request.method,
                build_upstream_url(worker, path, request),
                content=body,
                headers=filter_request_headers(request),
            )
            upstream = await self._client.send(upstream_request, stream=True)
        except Exception as exc:
            worker.decrement_active()
            self._record_worker_request_failure(
                worker,
                error=type(exc).__name__,
            )
            self._log_route_completion(
                worker=worker,
                path=path,
                metadata=metadata,
                status_code=502,
                outcome="upstream_error",
                start_time=start_time,
            )
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "upstream request failed"}},
                headers=self._diagnostic_headers(worker, metadata),
            )

        worker_failure_recorded = False

        def record_worker_failure_once(
            *,
            status_code: int | None = None,
            error: str | None = None,
        ) -> None:
            nonlocal worker_failure_recorded
            if worker_failure_recorded:
                return
            worker_failure_recorded = True
            self._record_worker_request_failure(
                worker,
                status_code=status_code,
                error=error,
            )

        if upstream.status_code in WORKER_REQUEST_FAILURE_STATUS_CODES:
            record_worker_failure_once(
                status_code=upstream.status_code,
                error=f"status={upstream.status_code}",
            )

        async def iter_bytes():
            outcome = _response_outcome(upstream.status_code)
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            except asyncio.CancelledError:
                outcome = "stream_cancelled"
                raise
            except Exception as exc:
                outcome = "stream_error"
                record_worker_failure_once(error=type(exc).__name__)
                raise
            finally:
                await upstream.aclose()
                worker.decrement_active()
                self._log_route_completion(
                    worker=worker,
                    path=path,
                    metadata=metadata,
                    status_code=upstream.status_code,
                    outcome=outcome,
                    start_time=start_time,
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

    def _record_worker_request_failure(
        self,
        worker: Worker,
        *,
        status_code: int | None = None,
        error: str | None = None,
    ) -> None:
        if worker.is_dead:
            return
        worker.record_request_failure(
            failure_threshold=self._config.health_failure_threshold,
            status_code=status_code,
            error=error,
        )
        logger.warning(
            f"worker={worker.display_id} worker_request_failure "
            f"status_code={status_code} error={error} "
            f"consecutive_failures={worker.consecutive_failures}",
        )

    def _log_route_completion(
        self,
        *,
        worker: Worker,
        path: str,
        metadata: RouteMetadata,
        status_code: int,
        outcome: str,
        start_time: float,
    ) -> None:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"route_completed request_id={metadata.request_id} "
            f"worker={worker.display_id} path={path} stream={metadata.stream} "
            f"capabilities={_format_capabilities(metadata.required_capabilities)} "
            f"status_code={status_code} duration_ms={duration_ms:.2f} "
            f"outcome={outcome}",
        )


def _exceeds_max_size(value: str, max_size: int) -> bool:
    try:
        return int(value) > max_size
    except ValueError:
        return True


def _response_error(response: httpx.Response) -> str:
    content = response.content[:512].decode("utf-8", errors="replace")
    return content or f"status={response.status_code}"


def _response_outcome(status_code: int) -> str:
    if status_code in WORKER_REQUEST_FAILURE_STATUS_CODES:
        return "worker_failure_status"
    return "completed"


def _format_capabilities(capabilities: set[Capability]) -> str:
    if not capabilities:
        return "-"
    return ",".join(sorted(capabilities))
