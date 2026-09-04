# SPDX-License-Identifier: Apache-2.0
"""Pinned WebSocket relay for TTS streaming sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState
from websockets.exceptions import ConnectionClosed, WebSocketException

try:
    from websockets.asyncio.client import connect as websocket_connect

    _WEBSOCKET_HEADERS_ARGUMENT = "additional_headers"
except ImportError:  # websockets 12 compatibility
    from websockets.client import connect as websocket_connect

    _WEBSOCKET_HEADERS_ARGUMENT = "extra_headers"

from sglang_omni.serve.speech_errors import (
    SpeechAPIError,
    bad_request,
    speech_websocket_error_payload,
)
from sglang_omni.serve.speech_limits import (
    MAX_SPEECH_WS_CONFIG_MESSAGE_BYTES,
    MAX_SPEECH_WS_TEXT_MESSAGE_BYTES,
    SPEECH_WS_CONFIG_TIMEOUT_S,
)
from sglang_omni_router.python.config import Capability, RouterConfig
from sglang_omni_router.python.proxy import (
    WORKER_EVICTION_STATUS_CODES,
    AdmissionController,
)
from sglang_omni_router.python.route_metadata import (
    ROUTE_HEADER_NAMES,
    RouteKind,
    SpeechRouteFacts,
    extract_speech_route_facts,
)
from sglang_omni_router.python.selector import (
    NoEligibleWorkerError,
    WorkerSelector,
    require_eligible_worker,
)
from sglang_omni_router.python.voice_routing import VoiceRoutingState
from sglang_omni_router.python.worker import Worker

logger = logging.getLogger(__name__)

RelayOutcome = Literal[
    "application_failure",
    "client_disconnected",
    "client_message_too_large",
    "completed",
    "upstream_failure",
]
SessionOutcome = (
    RelayOutcome
    | Literal[
        "configuration_timeout",
        "router_overloaded",
        "session_cancelled",
        "upstream_unavailable",
    ]
)
ClientRelayOutcome = Literal[
    "client_disconnected",
    "message_too_large",
    "upstream_closed_while_sending",
]


@dataclass(frozen=True)
class _HandshakeFailure:
    error: SpeechAPIError
    close_code: int
    outcome: SessionOutcome


@dataclass(frozen=True)
class _SessionResult:
    outcome: SessionOutcome
    routed_status_code: int | None = None
    worker_failure_status_code: int | None = None
    worker_failure_error: str | None = None
    client_error: SpeechAPIError | None = None
    client_close_code: int | None = None


_APPLICATION_CLOSE_CODES = {1002, 1003, 1007, 1008, 1009, 1010, 1013}
_MAX_UPSTREAM_SERVER_MESSAGE_BYTES = 512 * 1024 * 1024
_REQUEST_HEADERS_TO_STRIP = {
    "connection",
    "host",
    "sec-websocket-accept",
    "sec-websocket-extensions",
    "sec-websocket-key",
    "sec-websocket-protocol",
    "sec-websocket-version",
    "upgrade",
} | ROUTE_HEADER_NAMES


class TTSWebSocketProxy:
    def __init__(
        self,
        *,
        config: RouterConfig,
        workers: list[Worker],
        selector: WorkerSelector,
        admission: AdmissionController,
        voice_routing: VoiceRoutingState,
    ) -> None:
        self._config = config
        self._workers = workers
        self._selector = selector
        self._admission = admission
        self._voice_routing = voice_routing

    async def forward(self, websocket: WebSocket) -> None:
        await websocket.accept()
        request_id = _request_id(websocket)
        start_time = time.perf_counter()
        worker: Worker | None = None
        result: _SessionResult | None = None
        is_admitted = self._admission.try_acquire()
        try:
            if not is_admitted:
                result = _SessionResult(
                    outcome="router_overloaded",
                    client_error=SpeechAPIError(
                        message="router overloaded: max in-flight requests reached",
                        status_code=503,
                        error_type="overloaded_error",
                        code=503,
                    ),
                    client_close_code=1013,
                )
                await self._complete_session(websocket, None, result=result)
                return

            first_message = await self._receive_initial_message(websocket)
            if isinstance(first_message, _SessionResult):
                result = first_message
                await self._complete_session(websocket, None, result=result)
                return

            facts = _session_route_facts(first_message)
            try:
                worker = self._select_worker(facts)
            except NoEligibleWorkerError:
                result = _SessionResult(
                    outcome="upstream_unavailable",
                    client_error=SpeechAPIError(
                        message="no eligible upstream",
                        status_code=503,
                        error_type="server_error",
                        code=503,
                    ),
                    client_close_code=1013,
                )
                await self._complete_session(websocket, None, result=result)
                return

            with worker.request_guard():
                result = await self._connect_and_relay(
                    websocket,
                    worker,
                    first_message,
                )
                await self._complete_session(
                    websocket,
                    worker,
                    result=result,
                )
        except asyncio.TimeoutError:
            result = _SessionResult(
                outcome="configuration_timeout",
                client_error=bad_request(
                    "session.config was not received before timeout"
                ),
                client_close_code=1008,
            )
            await self._complete_session(
                websocket,
                None,
                result=result,
            )
        except WebSocketDisconnect:
            result = _SessionResult(outcome="client_disconnected")
            await self._complete_session(
                websocket,
                worker,
                result=result,
            )
        except asyncio.CancelledError:
            if result is None:
                result = _SessionResult(outcome="session_cancelled")
            raise
        finally:
            if is_admitted:
                self._admission.release()
            if result is not None:
                self._log_session_completion(
                    worker,
                    request_id=request_id,
                    start_time=start_time,
                    result=result,
                )

    async def _receive_initial_message(
        self,
        websocket: WebSocket,
    ) -> dict[str, Any] | _SessionResult:
        message = await asyncio.wait_for(
            websocket.receive(),
            timeout=SPEECH_WS_CONFIG_TIMEOUT_S,
        )
        if message.get("type") == "websocket.disconnect":
            return _SessionResult(outcome="client_disconnected")
        message_limit = min(
            self._config.max_payload_size,
            MAX_SPEECH_WS_CONFIG_MESSAGE_BYTES,
        )
        if _message_size(message) <= message_limit:
            return message
        return _SessionResult(
            outcome="client_message_too_large",
            client_error=SpeechAPIError(
                message="session.config WebSocket message exceeds payload limit",
                status_code=413,
                error_type="BadRequestError",
                code=413,
            ),
            client_close_code=1009,
        )

    async def _connect_and_relay(
        self,
        websocket: WebSocket,
        worker: Worker,
        first_message: dict[str, Any],
    ) -> _SessionResult:
        connect_options = {
            _WEBSOCKET_HEADERS_ARGUMENT: _forward_headers(websocket),
            "open_timeout": self._config.health_check_timeout_secs,
            "close_timeout": self._config.health_check_timeout_secs,
            "compression": None,
            "max_queue": 1,
            "max_size": _MAX_UPSTREAM_SERVER_MESSAGE_BYTES,
        }
        try:
            async with websocket_connect(
                _upstream_url(worker, websocket),
                **connect_options,
            ) as upstream:
                await _send_upstream(upstream, first_message)
                outcome = await _relay(
                    websocket,
                    upstream,
                    max_client_message_bytes=min(
                        self._config.max_payload_size,
                        MAX_SPEECH_WS_TEXT_MESSAGE_BYTES,
                    ),
                )
        except (WebSocketException, OSError, asyncio.TimeoutError) as exc:
            return _connection_failure_result(exc)
        return _relay_result(outcome)

    async def _complete_session(
        self,
        websocket: WebSocket,
        worker: Worker | None,
        *,
        result: _SessionResult,
    ) -> None:
        if worker is not None:
            if (
                result.worker_failure_status_code is not None
                or result.worker_failure_error is not None
            ):
                worker.record_request_failure(
                    failure_threshold=self._config.health_failure_threshold,
                    status_code=result.worker_failure_status_code,
                    error=result.worker_failure_error,
                )
            worker.record_routed_request(
                status_code=result.routed_status_code,
                service_class="tts_websocket",
            )
        if result.client_error is not None:
            assert result.client_close_code is not None
            await _send_router_error(
                websocket,
                result.client_error,
                close_code=result.client_close_code,
            )

    def _log_session_completion(
        self,
        worker: Worker | None,
        *,
        request_id: str,
        start_time: float,
        result: _SessionResult,
    ) -> None:
        log = (
            logger.warning
            if result.outcome
            in {
                "configuration_timeout",
                "router_overloaded",
                "upstream_failure",
                "upstream_unavailable",
            }
            else logger.info
        )
        status_code = (
            str(result.client_error.status_code)
            if result.client_error is not None
            else (
                str(result.routed_status_code)
                if result.routed_status_code is not None
                else "-"
            )
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        log(
            f"tts_websocket_completed request_id={request_id} "
            f"worker={worker.display_id if worker is not None else '-'} "
            f"outcome={result.outcome} "
            f"status_code={status_code} duration_ms={duration_ms:.2f}"
        )

    def _select_worker(self, facts: SpeechRouteFacts) -> Worker:
        required_capabilities: set[Capability] = {"speech", "streaming"}
        if facts.has_reference_audio:
            required_capabilities.add("audio_input")
        requires_voice_owner = self._voice_routing.requires_owner(
            facts.voice_names_requiring_registry
        )
        if requires_voice_owner:
            required_capabilities.add("audio_input")
            return require_eligible_worker(
                self._voice_routing.ensure_owner(),
                required_capabilities=required_capabilities,
                requested_model=facts.model,
            )
        return self._selector.select(
            self._workers,
            required_capabilities=required_capabilities,
            requested_model=facts.model,
        )


def _session_route_facts(message: dict[str, Any]) -> SpeechRouteFacts:
    text = message.get("text")
    if not isinstance(text, str):
        return extract_speech_route_facts({}, RouteKind.SPEECH)
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return extract_speech_route_facts({}, RouteKind.SPEECH)
    if not isinstance(payload, dict) or payload.get("type") != "session.config":
        return extract_speech_route_facts({}, RouteKind.SPEECH)
    session = payload.get("session")
    if session is None:
        session = {key: value for key, value in payload.items() if key != "type"}
    if not isinstance(session, dict):
        return extract_speech_route_facts({}, RouteKind.SPEECH)
    return extract_speech_route_facts(session, RouteKind.SPEECH)


def _relay_result(outcome: RelayOutcome) -> _SessionResult:
    if outcome == "completed":
        return _SessionResult(outcome=outcome, routed_status_code=200)
    if outcome == "upstream_failure":
        return _SessionResult(
            outcome=outcome,
            worker_failure_error="WebSocketRelayError",
        )
    if outcome == "client_message_too_large":
        return _SessionResult(
            outcome=outcome,
            client_error=SpeechAPIError(
                message="WebSocket message exceeds payload limit",
                status_code=413,
                error_type="BadRequestError",
                code=413,
            ),
            client_close_code=1009,
        )
    return _SessionResult(outcome=outcome)


def _connection_failure_result(exc: Exception) -> _SessionResult:
    status_code = _handshake_status_code(exc)
    if status_code is not None:
        failure = _handshake_failure(status_code)
        worker_failure_status = (
            status_code if status_code in WORKER_EVICTION_STATUS_CODES else None
        )
        return _SessionResult(
            outcome=failure.outcome,
            worker_failure_status_code=worker_failure_status,
            worker_failure_error=(
                f"status={status_code}" if worker_failure_status is not None else None
            ),
            client_error=failure.error,
            client_close_code=failure.close_code,
        )
    if isinstance(exc, ConnectionClosed) and _is_application_close(exc):
        return _SessionResult(
            outcome="application_failure",
            client_error=SpeechAPIError(
                message="upstream WebSocket rejected the request",
                status_code=400,
                error_type="upstream_error",
                code=400,
            ),
            client_close_code=1008,
        )
    return _SessionResult(
        outcome="upstream_failure",
        worker_failure_error=type(exc).__name__,
        client_error=SpeechAPIError(
            message="upstream WebSocket failed",
            status_code=502,
            error_type="upstream_error",
            code=502,
        ),
        client_close_code=1011,
    )


async def _relay(
    websocket: WebSocket,
    upstream: Any,
    *,
    max_client_message_bytes: int,
) -> RelayOutcome:
    client_task = asyncio.create_task(
        _client_to_upstream(
            websocket,
            upstream,
            max_message_bytes=max_client_message_bytes,
        )
    )
    upstream_task = asyncio.create_task(_upstream_to_client(upstream, websocket))
    try:
        return await _coordinate_relay(
            websocket,
            upstream,
            client_task=client_task,
            upstream_task=upstream_task,
        )
    finally:
        await _cancel_relay_tasks(client_task, upstream_task)


async def _coordinate_relay(
    websocket: WebSocket,
    upstream: Any,
    *,
    client_task: asyncio.Task[ClientRelayOutcome],
    upstream_task: asyncio.Task[RelayOutcome],
) -> RelayOutcome:
    done, _ = await asyncio.wait(
        {client_task, upstream_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    _raise_completed_task_error(done)
    if client_task in done:
        client_outcome = client_task.result()
        if client_outcome == "client_disconnected":
            outcome: RelayOutcome = "client_disconnected"
        elif client_outcome == "message_too_large":
            upstream_task.cancel()
            await _close_upstream(upstream, code=1009)
            return "client_message_too_large"
        else:
            # The receive loop owns the upstream protocol outcome. A send may
            # observe closure before buffered terminal frames are read.
            outcome = await upstream_task
    else:
        outcome = upstream_task.result()

    if outcome == "client_disconnected":
        await _close_upstream(upstream, code=1000)
    else:
        client_task.cancel()
        if websocket.application_state != WebSocketState.CONNECTED:
            return outcome
        code = (
            1011
            if outcome == "upstream_failure"
            else getattr(upstream, "close_code", None) or 1000
        )
        reason = getattr(upstream, "close_reason", None) or ""
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=_safe_close_code(code), reason=reason)
    return outcome


def _raise_completed_task_error(tasks: set[asyncio.Task[Any]]) -> None:
    for task in tasks:
        if task.cancelled():
            continue
        error = task.exception()
        if error is not None:
            raise error


async def _close_upstream(upstream: Any, *, code: int) -> None:
    try:
        await upstream.close(code=code)
    except (WebSocketException, OSError, asyncio.TimeoutError):
        pass


async def _cancel_relay_tasks(*tasks: asyncio.Task[Any]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    first_error: Exception | None = None
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


async def _client_to_upstream(
    websocket: WebSocket,
    upstream: Any,
    *,
    max_message_bytes: int,
) -> ClientRelayOutcome:
    while True:
        try:
            message = await websocket.receive()
        except (OSError, RuntimeError, WebSocketDisconnect):
            return "client_disconnected"
        message_type = message.get("type")
        if message_type == "websocket.disconnect":
            return "client_disconnected"
        if _message_size(message) > max_message_bytes:
            return "message_too_large"
        try:
            await _send_upstream(upstream, message)
        except ConnectionClosed:
            return "upstream_closed_while_sending"


async def _upstream_to_client(
    upstream: Any,
    websocket: WebSocket,
) -> RelayOutcome:
    protocol = _TTSProtocolState()
    try:
        async for message in upstream:
            if isinstance(message, str):
                protocol.observe(message)
                if not await _send_downstream(websocket.send_text(message)):
                    return "client_disconnected"
            else:
                if not await _send_downstream(websocket.send_bytes(bytes(message))):
                    return "client_disconnected"
    except ConnectionClosed as exc:
        if protocol.has_application_failure or _is_application_close(exc):
            return "application_failure"
        return "upstream_failure"
    except (WebSocketException, OSError, asyncio.TimeoutError):
        return "upstream_failure"
    return protocol.clean_close_outcome()


class _TTSProtocolState:
    def __init__(self) -> None:
        self._has_received_error = False
        self._has_failed_audio = False
        self._is_session_done = False

    def observe(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        message_type = payload.get("type")
        if message_type == "error":
            self._has_received_error = True
        elif message_type == "audio.done" and payload.get("error") is True:
            self._has_failed_audio = True
        elif message_type == "session.done":
            self._is_session_done = True
            self._has_received_error = False

    @property
    def has_application_failure(self) -> bool:
        return self._has_received_error or self._has_failed_audio

    def clean_close_outcome(self) -> RelayOutcome:
        if self._has_failed_audio:
            return "application_failure"
        if self._is_session_done:
            return "completed"
        if self._has_received_error:
            return "application_failure"
        return "upstream_failure"


async def _send_downstream(send: Awaitable[None]) -> bool:
    try:
        await send
    except (OSError, RuntimeError, WebSocketDisconnect):
        return False
    return True


def _handshake_status_code(exc: Exception) -> int | None:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    legacy_status_code = getattr(exc, "status_code", None)
    return legacy_status_code if isinstance(legacy_status_code, int) else None


def _handshake_failure(status_code: int) -> _HandshakeFailure:
    if status_code in {429, 503}:
        if status_code == 429:
            message = "upstream WebSocket is overloaded"
            error_type = "overloaded_error"
        else:
            message = "upstream WebSocket is unavailable"
            error_type = "server_error"
        return _HandshakeFailure(
            error=SpeechAPIError(
                message=message,
                status_code=status_code,
                error_type=error_type,
                code=status_code,
            ),
            close_code=1013,
            outcome="upstream_unavailable",
        )
    if 400 <= status_code < 500:
        close_code = 1008
        message = "upstream WebSocket rejected the request"
        outcome = "application_failure"
    else:
        close_code = 1011
        message = "upstream WebSocket failed"
        outcome = "upstream_failure"
    return _HandshakeFailure(
        error=SpeechAPIError(
            message=message,
            status_code=status_code,
            error_type="upstream_error",
            code=status_code,
        ),
        close_code=close_code,
        outcome=outcome,
    )


def _is_application_close(exc: ConnectionClosed) -> bool:
    received = getattr(exc, "rcvd", None)
    close_code = getattr(received, "code", None)
    if not isinstance(close_code, int):
        legacy_close_code = getattr(exc, "code", None)
        close_code = legacy_close_code if isinstance(legacy_close_code, int) else None
    return close_code in _APPLICATION_CLOSE_CODES or (
        close_code is not None and 3000 <= close_code < 5000
    )


async def _send_upstream(upstream: Any, message: dict[str, Any]) -> None:
    if message.get("type") != "websocket.receive":
        return
    text = message.get("text")
    if text is not None:
        await upstream.send(text)
        return
    data = message.get("bytes")
    if data is not None:
        await upstream.send(data)


async def _send_router_error(
    websocket: WebSocket,
    error: SpeechAPIError,
    *,
    close_code: int,
) -> None:
    if websocket.application_state != WebSocketState.CONNECTED:
        return
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    with suppress(OSError, RuntimeError, WebSocketDisconnect):
        await websocket.send_json(speech_websocket_error_payload(error))
        await websocket.close(code=close_code)


def _upstream_url(worker: Worker, websocket: WebSocket) -> str:
    parsed = urlsplit(worker.url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    query = websocket.url.query
    return urlunsplit(
        (
            scheme,
            parsed.netloc,
            "/v1/audio/speech/stream",
            query,
            "",
        )
    )


def _forward_headers(websocket: WebSocket) -> dict[str, str]:
    return {
        key: value
        for key, value in websocket.headers.items()
        if key.lower() not in _REQUEST_HEADERS_TO_STRIP
    }


def _request_id(websocket: WebSocket) -> str:
    return (
        websocket.headers.get("x-sglang-omni-request-id")
        or websocket.headers.get("x-request-id")
        or websocket.headers.get("x-correlation-id")
        or str(uuid.uuid4())
    )


def _safe_close_code(code: int) -> int:
    return code if 1000 <= code < 5000 else 1011


def _message_size(message: dict[str, Any]) -> int:
    text = message.get("text")
    if isinstance(text, str):
        return len(text.encode("utf-8"))
    data = message.get("bytes")
    return len(data) if isinstance(data, bytes) else 0
