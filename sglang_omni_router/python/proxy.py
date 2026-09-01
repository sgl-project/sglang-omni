# SPDX-License-Identifier: Apache-2.0
"""Proxy request forwarding and response relay."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from http import HTTPStatus
from urllib.parse import unquote

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni.serve.speech_errors import openai_error_payload
from sglang_omni.serve.speech_limits import MAX_VOICE_UPLOAD_BODY_BYTES
from sglang_omni_router.python.config import Capability, RouterConfig
from sglang_omni_router.python.route_metadata import (
    ROUTE_HEADER_NAMES,
    RouteKind,
    RouteMetadata,
    RouteMetadataError,
    classify_route,
    extract_route_metadata,
)
from sglang_omni_router.python.selector import (
    NoEligibleWorkerError,
    WorkerSelector,
    require_eligible_worker,
)
from sglang_omni_router.python.voice_routing import VoiceMutation, VoiceRoutingState
from sglang_omni_router.python.worker import Worker

logger = logging.getLogger(__name__)

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
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


@dataclass(frozen=True)
class _VoiceUploadRequest:
    pass


@dataclass(frozen=True)
class _VoiceDeleteRequest:
    mutation: VoiceMutation


_VoiceMutationRequest = _VoiceUploadRequest | _VoiceDeleteRequest


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
        self._is_released = False

    def __call__(self) -> None:
        if self._is_released:
            return
        self._is_released = True
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
        self._is_done = False

    async def __call__(self, outcome: str) -> None:
        if self._is_done:
            return
        self._is_done = True
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
        worker_provider: Callable[[], list[Worker]] | None = None,
        on_worker_failure: (
            Callable[[Worker, int | None, str | None], None] | None
        ) = None,
        admission: AdmissionController | None = None,
        voice_routing: VoiceRoutingState | None = None,
    ) -> None:
        self._config = config
        self._workers = workers
        self._selector = selector
        self._client = client
        # Note (Jiaxin Deng): admission accepts any object with the
        # AdmissionController surface, e.g. the shared-memory one.
        self._admission = (
            admission
            if admission is not None
            else AdmissionController(config.effective_max_inflight)
        )
        # Note (Jiaxin Deng): set only in data-plane mode; workers then come
        # from the snapshot view and failures are reported to the CP.
        self._worker_provider = worker_provider
        self._on_worker_failure = on_worker_failure
        self._voice_routing = voice_routing

    def _current_workers(self) -> list[Worker]:
        if self._worker_provider is not None:
            return self._worker_provider()
        return self._workers

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
        route_kind = classify_route(path)
        body_limit = self._config.max_payload_size
        is_voice_upload = path == "/v1/audio/voices" and request.method == "POST"
        if is_voice_upload:
            body_limit = min(body_limit, MAX_VOICE_UPLOAD_BODY_BYTES)
        content_length = request.headers.get("content-length")
        if content_length is not None and _exceeds_max_size(content_length, body_limit):
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=413,
                reason="payload_too_large",
            )
            return _payload_too_large_response(
                is_voice_upload=is_voice_upload,
                max_size=body_limit,
            )

        try:
            body = await _read_body_with_limit(request, body_limit)
        except PayloadTooLargeError:
            self._log_route_rejection(
                request=request,
                path=path,
                status_code=413,
                reason="payload_too_large",
            )
            return _payload_too_large_response(
                is_voice_upload=is_voice_upload,
                max_size=body_limit,
            )

        try:
            metadata = extract_route_metadata(request, route_kind, body)
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

        is_large_speech_request = (
            metadata.is_body_over_metadata_limit
            and metadata.route_kind in {RouteKind.SPEECH, RouteKind.SPEECH_BATCH}
        )
        if is_large_speech_request:
            metadata.required_capabilities.add("audio_input")
        large_request_candidates, large_request_error = (
            _large_request_candidates_and_model_error(self._current_workers(), metadata)
        )
        voice_owner_handles_large_body = (
            is_large_speech_request
            and self._voice_routing is not None
            and self._voice_routing.ensure_owner() is not None
        )
        if large_request_error is not None or voice_owner_handles_large_body:
            extra_capabilities = set()
        else:
            extra_capabilities, large_request_error = (
                _large_request_extra_capabilities_or_error(
                    large_request_candidates, metadata
                )
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

        voice_mutation = (
            self._voice_mutation_request(request, path)
            if self._voice_routing is not None
            else None
        )
        if voice_mutation is not None:
            assert self._voice_routing is not None
            self._voice_routing.begin_mutation()
        try:
            return await self._select_and_forward(
                request,
                path,
                body,
                metadata,
                release,
                voice_mutation=voice_mutation,
            )
        finally:
            if voice_mutation is not None:
                assert self._voice_routing is not None
                self._voice_routing.end_mutation()

    async def _select_and_forward(
        self,
        request: Request,
        path: str,
        body: bytes,
        metadata: RouteMetadata,
        release: _ReleaseOnce,
        *,
        voice_mutation: _VoiceMutationRequest | None = None,
    ) -> Response:
        try:
            worker = self._select_worker(metadata)
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

        return await self._forward_relay(
            request,
            path,
            body,
            metadata,
            worker,
            release,
            voice_mutation=voice_mutation,
        )

    def _select_worker(
        self,
        metadata: RouteMetadata,
    ) -> Worker:
        if self._voice_routing is not None:
            is_voice_control = metadata.route_kind is RouteKind.VOICE_CONTROL
            if is_voice_control:
                self._voice_routing.activate()
            requires_voice_owner = metadata.route_kind in {
                RouteKind.SPEECH,
                RouteKind.SPEECH_BATCH,
            } and self._voice_routing.requires_owner(
                metadata.voice_names_requiring_registry,
                is_body_over_metadata_limit=metadata.is_body_over_metadata_limit,
            )
            if requires_voice_owner:
                metadata.required_capabilities.add("audio_input")
            if is_voice_control or requires_voice_owner:
                return require_eligible_worker(
                    self._voice_routing.ensure_owner(),
                    required_capabilities=metadata.required_capabilities,
                    requested_model=metadata.model,
                )
        return self._selector.select(
            self._current_workers(),
            required_capabilities=metadata.required_capabilities,
            requested_model=metadata.model,
        )

    async def _forward_relay(
        self,
        request: Request,
        path: str,
        body: bytes,
        metadata: RouteMetadata,
        worker: Worker,
        release: _ReleaseOnce,
        voice_mutation: _VoiceMutationRequest | None = None,
    ) -> Response:
        start_time = time.perf_counter()
        upstream = await self._open_upstream(
            request=request,
            path=path,
            body=body,
            metadata=metadata,
            worker=worker,
            voice_mutation=voice_mutation,
            start_time=start_time,
        )
        if isinstance(upstream, JSONResponse):
            return upstream
        if voice_mutation is not None:
            assert self._voice_routing is not None
            self._voice_routing.mark_mutation_dispatched()
        return await self._build_relay_response(
            upstream=upstream,
            worker=worker,
            path=path,
            metadata=metadata,
            release=release,
            start_time=start_time,
            voice_mutation=voice_mutation,
        )

    async def _open_upstream(
        self,
        *,
        request: Request,
        path: str,
        body: bytes,
        metadata: RouteMetadata,
        worker: Worker,
        voice_mutation: _VoiceMutationRequest | None,
        start_time: float,
    ) -> httpx.Response | JSONResponse:
        upstream_request = self._client.build_request(
            request.method,
            build_upstream_url(worker, path, request),
            content=body,
            headers=filter_request_headers(request),
        )
        worker.increment_active()
        try:
            upstream = await self._client.send(upstream_request, stream=True)
        except httpx.PoolTimeout:
            # Note (Jiaxin Deng): router-local pool contention, not a worker
            # fault; feed neither the eviction signal nor failed_requests, and
            # the request never reached the worker, so it is not routed.
            worker.decrement_active()
            self._log_route_completion(
                worker=worker,
                path=path,
                metadata=metadata,
                status_code=503,
                outcome="router_pool_exhausted",
                start_time=start_time,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "router upstream pool exhausted",
                        "type": "overloaded_error",
                        "code": 503,
                    }
                },
                headers={"Retry-After": _OVERLOAD_RETRY_AFTER_SECS},
            )
        except httpx.HTTPError as exc:
            worker.decrement_active()
            if voice_mutation is not None and self._voice_routing is not None:
                self._voice_routing.mark_uncertain()
            worker.record_routed_request(service_class=metadata.service_class)
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
        except BaseException:
            # Note (Jiaxin Deng): a client disconnect cancels this await, and
            # neither httpx handler sees it. Without returning the gauge the
            # worker looks busy forever, and a retiring incarnation never
            # drains, so the data plane reports it to the CP indefinitely.
            worker.decrement_active()
            if voice_mutation is not None and self._voice_routing is not None:
                self._voice_routing.mark_uncertain()
            raise

        return upstream

    async def _build_relay_response(
        self,
        *,
        upstream: httpx.Response,
        worker: Worker,
        path: str,
        metadata: RouteMetadata,
        release: _ReleaseOnce,
        start_time: float,
        voice_mutation: _VoiceMutationRequest | None,
    ) -> Response:
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

        def record_completion(outcome: str) -> None:
            status_code = upstream.status_code if outcome == "completed" else None
            worker.record_routed_request(
                status_code=status_code,
                service_class=metadata.service_class,
            )
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

        if voice_mutation is not None:
            assert self._voice_routing is not None
            return await self._buffer_voice_mutation_response(
                upstream=upstream,
                worker=worker,
                metadata=metadata,
                mutation_request=voice_mutation,
                cleanup=cleanup,
                record_worker_failure=record_worker_failure_once,
            )

        return await self._streaming_relay_response(
            upstream=upstream,
            worker=worker,
            metadata=metadata,
            cleanup=cleanup,
            record_worker_failure=record_worker_failure_once,
        )

    async def _streaming_relay_response(
        self,
        *,
        upstream: httpx.Response,
        worker: Worker,
        metadata: RouteMetadata,
        cleanup: _RelayCleanup,
        record_worker_failure: Callable[..., None],
    ) -> Response:
        try:
            headers = filter_response_headers(
                upstream.headers,
                buffered=not metadata.stream,
            )
            headers.update(self._diagnostic_headers(worker, metadata))
        except Exception:
            await upstream.aclose()
            worker.decrement_active()
            raise
        media_type = (
            upstream.headers.get("content-type", "text/event-stream")
            if metadata.stream
            else upstream.headers.get("content-type")
        )
        return _RelayResponse(
            self._iter_upstream_bytes(
                upstream,
                cleanup=cleanup,
                record_worker_failure=record_worker_failure,
            ),
            status_code=upstream.status_code,
            headers=headers,
            media_type=media_type,
            cleanup=cleanup,
        )

    async def _iter_upstream_bytes(
        self,
        upstream: httpx.Response,
        *,
        cleanup: _RelayCleanup,
        record_worker_failure: Callable[..., None],
    ) -> AsyncIterator[bytes]:
        # A mid-stream failure after a 2xx cannot become a 502. SSE responses
        # receive a terminal error event; audio and JSON responses truncate.
        outcome = _response_outcome(upstream.status_code)
        is_event_stream = (upstream.headers.get("content-type") or "").startswith(
            "text/event-stream"
        )
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        except asyncio.CancelledError:
            outcome = "stream_cancelled"
            raise
        except httpx.HTTPError as exc:
            outcome = "stream_error"
            record_worker_failure(error=type(exc).__name__)
            if is_event_stream:
                yield _SSE_UPSTREAM_ERROR_EVENT
                return
            raise
        finally:
            await cleanup(outcome)

    async def _buffer_voice_mutation_response(
        self,
        *,
        upstream: httpx.Response,
        worker: Worker,
        metadata: RouteMetadata,
        mutation_request: _VoiceMutationRequest,
        cleanup: _RelayCleanup,
        record_worker_failure: Callable[..., None],
    ) -> Response:
        assert self._voice_routing is not None
        try:
            body = await upstream.aread()
        except asyncio.CancelledError:
            self._voice_routing.mark_uncertain()
            await cleanup("stream_cancelled")
            raise
        except httpx.HTTPError as exc:
            self._voice_routing.mark_uncertain()
            record_worker_failure(error=type(exc).__name__)
            await cleanup("stream_error")
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "upstream response failed"}},
                headers=self._diagnostic_headers(worker, metadata),
            )

        if 200 <= upstream.status_code < 300:
            try:
                mutation = _committed_voice_mutation(mutation_request, body)
            except ValueError as exc:
                logger.warning(
                    f"voice_registry_commit_uncertain worker={worker.display_id} "
                    f"error={exc}"
                )
                self._voice_routing.mark_uncertain()
            else:
                self._voice_routing.apply(mutation)

        headers = filter_response_headers(upstream.headers, buffered=True)
        headers.update(self._diagnostic_headers(worker, metadata))
        await cleanup(_response_outcome(upstream.status_code))
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers=headers,
            media_type=upstream.headers.get("content-type"),
        )

    def _voice_mutation_request(
        self,
        request: Request,
        path: str,
    ) -> _VoiceMutationRequest | None:
        if path == "/v1/audio/voices" and request.method == "POST":
            return _VoiceUploadRequest()

        prefix = "/v1/audio/voices/"
        if path.startswith(prefix) and request.method == "DELETE":
            name = unquote(path.removeprefix(prefix))
            mutation = VoiceMutation.create("delete", name)
            return None if mutation is None else _VoiceDeleteRequest(mutation)
        return None

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
        if self._on_worker_failure is not None:
            # Note (Jiaxin Deng): the CP owns the health verdict; its snapshot
            # never resets a DP-local consecutive_failures, so counting here
            # too would let a DP evict a worker the CP still calls healthy.
            logger.warning(
                f"worker={worker.display_id} worker_request_failure "
                f"status_code={status_code} error={error} reported_to=control_plane",
            )
            self._on_worker_failure(worker, status_code, error)
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


def _committed_voice_mutation(
    request: _VoiceMutationRequest,
    body: bytes,
) -> VoiceMutation:
    if isinstance(request, _VoiceDeleteRequest):
        return request.mutation

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("upload response is not valid JSON") from None
    if not isinstance(payload, dict):
        raise ValueError("upload response must be an object")
    mutation = VoiceMutation.create("upload", payload.get("name"))
    if mutation is None:
        raise ValueError("upload response must include the stored voice name")
    return mutation


def _payload_too_large_response(
    *,
    is_voice_upload: bool,
    max_size: int,
) -> JSONResponse:
    if is_voice_upload:
        return JSONResponse(
            status_code=413,
            content=openai_error_payload(
                f"request body must be at most {max_size} bytes",
                error_type="RequestTooLargeError",
                param="audio_sample",
                code=413,
            ),
        )
    return JSONResponse(
        status_code=413,
        content={"error": {"message": "payload too large"}},
    )


def _large_request_extra_capabilities_or_error(
    candidates: list[Worker],
    metadata: RouteMetadata,
) -> tuple[set[Capability], str | None]:
    if not metadata.is_body_over_metadata_limit:
        return set(), None
    if not candidates:
        return set(), None

    capability_sets = {frozenset(worker.capabilities) for worker in candidates}
    if metadata.has_route_capabilities_header or len(capability_sets) <= 1:
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


def _large_request_candidates_and_model_error(
    workers: list[Worker],
    metadata: RouteMetadata,
) -> tuple[list[Worker], str | None]:
    if not metadata.is_body_over_metadata_limit:
        return [], None
    candidates = [
        worker
        for worker in workers
        if worker.is_routable
        and all(
            worker.supports(capability) for capability in metadata.required_capabilities
        )
    ]
    candidate_models = {
        worker.model for worker in candidates if worker.model is not None
    }
    if (
        metadata.route_kind is RouteKind.SPEECH_BATCH
        and len(candidate_models) > 1
        and not metadata.has_route_model_header
    ):
        return (
            candidates,
            "large speech batch requests across mixed-model workers require "
            "x-sglang-omni-route-model",
        )
    if metadata.model is not None and any(worker.model for worker in candidates):
        candidates = [worker for worker in candidates if worker.model == metadata.model]
    models = {worker.model for worker in candidates}
    if metadata.model is None and len(models) > 1:
        return (
            candidates,
            "large JSON requests across mixed-model workers require "
            "x-sglang-omni-route-model",
        )
    return candidates, None
