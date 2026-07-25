# SPDX-License-Identifier: Apache-2.0
"""Proxy request forwarding and response relay."""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections.abc import Callable
from http import HTTPStatus

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni_router.config import Capability, RouterConfig
from sglang_omni_router.route_metadata import (
    ROUTE_HEADER_NAMES,
    RouteMetadata,
    RouteMetadataError,
    extract_route_metadata,
)
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
REQUEST_HEADERS_TO_STRIP = (
    HOP_BY_HOP_HEADERS
    | {
        "host",
        "content-length",
        "accept-encoding",
    }
    | ROUTE_HEADER_NAMES
)
RESPONSE_HEADERS_TO_STRIP = HOP_BY_HOP_HEADERS | {
    "content-length",
    "date",
    "server",
}
BUFFERED_RESPONSE_HEADERS_TO_STRIP = RESPONSE_HEADERS_TO_STRIP | {
    "content-encoding",
}
# Note (Jiaxin Deng): liveness is owned by /health probes; a relayed status
# evicts only on gateway failure (502/504). Capacity backpressure (429/503),
# request timeouts (408), and bad-input 500s are per-request failures counted in
# worker stats, so one overloaded or bad-input client cannot cascade the pool
# unhealthy.
WORKER_EVICTION_STATUS_CODES = {
    HTTPStatus.BAD_GATEWAY.value,
    HTTPStatus.GATEWAY_TIMEOUT.value,
}

# Note (Jiaxin Deng): terminal SSE event injected when an event-stream relay
# fails mid-body, so the client gets a valid error frame (fields mirror the
# worker's OpenAI error envelope) instead of a silently truncated stream.
_SSE_UPSTREAM_ERROR_EVENT = (
    b'data: {"error": {"message": "upstream stream failed before completion", '
    b'"type": "upstream_error", "param": null, "code": 502}}\n\n'
)

_OVERLOAD_RETRY_AFTER_SECS = "1"


class PayloadTooLargeError(ValueError):
    pass


class AdmissionController:
    """Bounded global in-flight count for the model data plane.

    Runs on uvicorn's single event loop: check-and-increment happens
    synchronously between awaits, so no locking is needed.
    """

    def __init__(self, max_inflight: int) -> None:
        self._max_inflight = max_inflight
        self._inflight = 0
        self._peak_inflight = 0
        self._rejected_total = 0

    @property
    def inflight(self) -> int:
        return self._inflight

    def try_acquire(self) -> bool:
        if self._inflight >= self._max_inflight:
            self._rejected_total += 1
            return False
        self._inflight += 1
        if self._inflight > self._peak_inflight:
            self._peak_inflight = self._inflight
        return True

    def to_dict(self) -> dict[str, int]:
        return {
            "inflight": self._inflight,
            "max_inflight": self._max_inflight,
            "peak_inflight": self._peak_inflight,
            "rejected_total": self._rejected_total,
        }

    def release(self) -> None:
        assert self._inflight > 0, "in-flight count cannot be negative"
        self._inflight -= 1


class _ReleaseOnce:
    """Idempotent release so every response path can call it safely."""

    def __init__(self, admission: AdmissionController) -> None:
        self._admission = admission
        self._released = False

    def __call__(self) -> None:
        if self._released:
            return
        self._released = True
        self._admission.release()


class _RelayCleanup:
    """Idempotent teardown for one streamed relay.

    Closes the upstream stream, decrements the worker in-flight count, releases
    the admission slot, and records completion exactly once, whichever path
    finishes the response: the body iterator (normal completion, mid-stream
    error, or cancellation) or the response ``__call__`` when ``send`` fails on
    ``http.response.start`` before the iterator ever runs.
    """

    def __init__(
        self,
        *,
        upstream: httpx.Response,
        worker: Worker,
        release: _ReleaseOnce,
        record_completion: Callable[[str], None],
    ) -> None:
        self._upstream = upstream
        self._worker = worker
        self._release = release
        self._record_completion = record_completion
        self._done = False

    async def __call__(self, outcome: str) -> None:
        if self._done:
            return
        self._done = True
        # aclose() raising on a broken mid-stream connection must not skip the
        # decrement, otherwise least_request drifts permanently.
        try:
            await self._upstream.aclose()
        finally:
            self._worker.decrement_active()
            self._release()
            self._record_completion(outcome)


class _RelayResponse(StreamingResponse):
    """Streaming relay response that cleans up even if the body never streams.

    Starlette sends ``http.response.start`` before it iterates the body, so a
    client disconnect (or any ``send`` failure) during that first send skips the
    iterator's ``finally``. This ``__call__`` is the cleanup owner for that path.
    """

    def __init__(self, *args, cleanup: _RelayCleanup, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._cleanup("client_disconnected")


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
        self._admission = AdmissionController(config.effective_max_inflight)

    @property
    def admission(self) -> AdmissionController:
        return self._admission

    async def forward_model_request(self, request: Request, path: str) -> Response:
        # Note (Jiaxin Deng): reject before reading the body; past the bound
        # the relay must shed load, not queue it.
        if not self._admission.try_acquire():
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=503,
                reason="router_overloaded",
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": (
                            "router overloaded: max in-flight requests reached"
                        ),
                        "type": "overloaded_error",
                        "code": 503,
                    }
                },
                headers={"Retry-After": _OVERLOAD_RETRY_AFTER_SECS},
            )
        release = _ReleaseOnce(self._admission)
        try:
            response = await self._forward_admitted(request, path, release)
        except BaseException:
            release()
            raise
        # Note (Jiaxin Deng): a streamed relay releases its slot from the cleanup
        # owner (iterator or __call__ finally); other responses release once built.
        # The finalizer is only a GC backstop if the ASGI stack never calls it.
        if not isinstance(response, StreamingResponse):
            release()
        else:
            weakref.finalize(response, release)
        return response

    async def _forward_admitted(
        self,
        request: Request,
        path: str,
        release: _ReleaseOnce,
    ) -> Response:
        content_length = request.headers.get("content-length")
        if content_length is not None and _exceeds_max_size(
            content_length, self._config.max_payload_size
        ):
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=413,
                reason="payload_too_large",
            )
            return JSONResponse(
                status_code=413,
                content={"error": {"message": "payload too large"}},
            )

        try:
            body = await _read_body_with_limit(request, self._config.max_payload_size)
        except PayloadTooLargeError:
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=413,
                reason="payload_too_large",
            )
            return JSONResponse(
                status_code=413,
                content={"error": {"message": "payload too large"}},
            )

        try:
            metadata = extract_route_metadata(request, path, body)
        except RouteMetadataError as exc:
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=400,
                reason=str(exc).replace(" ", "_"),
            )
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc)}},
            )

        extra_capabilities, large_request_error = (
            _large_request_extra_capabilities_or_error(self._workers, metadata)
        )
        if large_request_error is not None:
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=400,
                reason=large_request_error.replace(" ", "_"),
                metadata=metadata,
            )
            return JSONResponse(
                status_code=400,
                content={"error": {"message": large_request_error}},
            )
        metadata.required_capabilities.update(extra_capabilities)

        try:
            worker = self._selector.select(
                self._workers,
                required_capabilities=metadata.required_capabilities,
                requested_model=metadata.model,
            )
        except NoEligibleWorkerError:
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=503,
                reason="no_eligible_upstream",
                metadata=metadata,
            )
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "no eligible upstream"}},
            )

        return await self._forward_relay(request, path, body, metadata, worker, release)

    async def _forward_relay(
        self,
        request: Request,
        path: str,
        body: bytes,
        metadata: RouteMetadata,
        worker: Worker,
        release: _ReleaseOnce,
    ) -> Response:
        # Note (Jiaxin Deng): stream through instead of buffering. A mid-stream
        # failure after a 2xx cannot become a 502; SSE (text/event-stream) bodies
        # get a terminal error event, other bodies (audio, JSON) truncate.
        stream = metadata.stream
        start_time = time.perf_counter()
        upstream_request = self._client.build_request(
            request.method,
            build_upstream_url(worker, path, request),
            content=body,
            headers=filter_request_headers(request),
        )
        worker.increment_active()
        try:
            upstream = await self._client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            worker.decrement_active()
            worker.record_routed_request()
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

        if upstream.status_code in WORKER_EVICTION_STATUS_CODES:
            record_worker_failure_once(
                status_code=upstream.status_code,
                error=f"status={upstream.status_code}",
            )

        is_event_stream = (upstream.headers.get("content-type") or "").startswith(
            "text/event-stream"
        )

        def record_completion(outcome: str) -> None:
            status_code = upstream.status_code if outcome == "completed" else None
            worker.record_routed_request(status_code=status_code)
            self._log_route_completion(
                worker=worker,
                path=path,
                metadata=metadata,
                status_code=upstream.status_code,
                outcome=outcome,
                start_time=start_time,
            )

        cleanup = _RelayCleanup(
            upstream=upstream,
            worker=worker,
            release=release,
            record_completion=record_completion,
        )

        async def iter_bytes():
            outcome = _response_outcome(upstream.status_code)
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            except asyncio.CancelledError:
                outcome = "stream_cancelled"
                raise
            except httpx.HTTPError as exc:
                outcome = "stream_error"
                record_worker_failure_once(error=type(exc).__name__)
                if is_event_stream:
                    yield _SSE_UPSTREAM_ERROR_EVENT
                    return
                raise
            finally:
                await cleanup(outcome)

        try:
            headers = filter_response_headers(upstream.headers, buffered=not stream)
            headers.update(self._diagnostic_headers(worker, metadata))
        except Exception:
            await upstream.aclose()
            worker.decrement_active()
            raise
        media_type = (
            upstream.headers.get("content-type", "text/event-stream")
            if stream
            else upstream.headers.get("content-type")
        )
        return _RelayResponse(
            iter_bytes(),
            status_code=upstream.status_code,
            headers=headers,
            media_type=media_type,
            cleanup=cleanup,
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

    def _log_route_rejection(
        self,
        *,
        request: Request,
        path: str,
        status_code: int,
        reason: str,
        metadata: RouteMetadata | None = None,
    ) -> None:
        request_id = (
            metadata.request_id if metadata else _request_id_from_headers(request)
        )
        model = metadata.model if metadata else None
        capabilities = metadata.required_capabilities if metadata else set()
        logger.warning(
            f"route_rejected request_id={request_id or '-'} path={path} "
            f"status_code={status_code} reason={reason} "
            f"model={model or '-'} capabilities={_format_capabilities(capabilities)}",
        )


def _exceeds_max_size(value: str, max_size: int) -> bool:
    try:
        return int(value) > max_size
    except ValueError:
        return True


async def _read_body_with_limit(request: Request, max_size: int) -> bytes:
    total_size = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total_size += len(chunk)
        if total_size > max_size:
            raise PayloadTooLargeError
        chunks.append(chunk)
    return b"".join(chunks)


def _response_outcome(status_code: int) -> str:
    if status_code in WORKER_EVICTION_STATUS_CODES:
        return "worker_failure_status"
    if status_code == HTTPStatus.INTERNAL_SERVER_ERROR.value:
        return "upstream_5xx"
    return "completed"


def _format_capabilities(capabilities: set[Capability]) -> str:
    if not capabilities:
        return "-"
    return ",".join(sorted(capabilities))


def _request_id_from_headers(request: Request) -> str | None:
    return (
        request.headers.get("x-sglang-omni-request-id")
        or request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
    )


def _large_request_extra_capabilities_or_error(
    workers: list[Worker],
    metadata: RouteMetadata,
) -> tuple[set[Capability], str | None]:
    if not metadata.body_exceeds_metadata_limit:
        return set(), None

    candidates = [
        worker
        for worker in workers
        if worker.is_routable
        and all(
            worker.supports(capability) for capability in metadata.required_capabilities
        )
    ]
    if metadata.model is not None and any(worker.model for worker in candidates):
        candidates = [worker for worker in candidates if worker.model == metadata.model]
    if not candidates:
        return set(), None

    models = {worker.model for worker in candidates}
    if metadata.model is None and len(models) > 1:
        return set(), (
            "large JSON requests across mixed-model workers require "
            "x-sglang-omni-route-model"
        )

    capability_sets = {frozenset(worker.capabilities) for worker in candidates}
    if metadata.route_capabilities_header_present or len(capability_sets) <= 1:
        return set(), None

    maximal_sets = [
        capability_set
        for capability_set in capability_sets
        if not any(capability_set < other for other in capability_sets)
    ]
    if len(maximal_sets) == 1:
        return set(maximal_sets[0]) - metadata.required_capabilities, None

    return set(), (
        "large JSON requests across mixed-capability workers require "
        "x-sglang-omni-route-capabilities"
    )
