# SPDX-License-Identifier: Apache-2.0
"""Proxy request forwarding and response relay."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni_router.config import Capability, RouterConfig
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
BUFFERED_RESPONSE_HEADERS_TO_STRIP = RESPONSE_HEADERS_TO_STRIP | {
    "content-encoding",
}
ROUTE_METADATA_JSON_LIMIT_BYTES = 1024 * 1024


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

    content_type = request.headers.get("content-type", "")
    if "json" in content_type and len(body) <= ROUTE_METADATA_JSON_LIMIT_BYTES:
        try:
            parsed_payload = json.loads(body)
        except Exception:
            parsed_payload = None
        if isinstance(parsed_payload, dict):
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

        metadata = extract_route_metadata(request, path, body)
        try:
            worker = self._selector.select(
                self._workers,
                required_capabilities=metadata.required_capabilities,
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
            try:
                response = await self._client.request(
                    request.method,
                    build_upstream_url(worker, path, request),
                    content=body,
                    headers=filter_request_headers(request),
                )
                headers = filter_response_headers(response.headers, buffered=True)
                headers.update(self._diagnostic_headers(worker, metadata))
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=headers,
                    media_type=response.headers.get("content-type"),
                )
            except Exception:
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
        try:
            upstream_request = self._client.build_request(
                request.method,
                build_upstream_url(worker, path, request),
                content=body,
                headers=filter_request_headers(request),
            )
            upstream = await self._client.send(upstream_request, stream=True)
        except Exception:
            worker.decrement_active()
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "upstream request failed"}},
                headers=self._diagnostic_headers(worker, metadata),
            )

        async def iter_bytes():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                worker.decrement_active()

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


def _exceeds_max_size(value: str, max_size: int) -> bool:
    try:
        return int(value) > max_size
    except ValueError:
        return True
