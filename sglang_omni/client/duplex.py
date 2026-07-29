# SPDX-License-Identifier: Apache-2.0
"""Session-native full-duplex client handle."""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from sglang_omni.client.types import DUPLEX_EVENT_TYPES, ClientError, DuplexEvent
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import (
    CompleteMessage,
    OmniRequest,
    SessionCommandMessage,
    StreamMessage,
)

_QUEUE_END = object()


class DuplexSessionError(ClientError):
    """Raised when a duplex session violates its lifecycle contract."""


class DuplexSession:
    """Long-lived bidirectional session backed by one pipeline request."""

    def __init__(
        self,
        coordinator: Coordinator,
        request: OmniRequest,
        *,
        session_id: str,
        output_queue_size: int = 64,
        handshake_timeout_s: float = 30.0,
        command_timeout_s: float = 30.0,
    ) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise TypeError("session_id must be a non-empty str")
        if type(output_queue_size) is not int or output_queue_size <= 0:
            raise ValueError("output_queue_size must be a positive int")
        if handshake_timeout_s <= 0 or command_timeout_s <= 0:
            raise ValueError("duplex timeouts must be positive")

        self._coordinator = coordinator
        self._request = request
        self.session_id = session_id
        self._handshake_timeout_s = float(handshake_timeout_s)
        self._command_timeout_s = float(command_timeout_s)
        self._output_queue_size = output_queue_size
        self._output_queue: asyncio.Queue[DuplexEvent | object] = asyncio.Queue(
            maxsize=output_queue_size + 2
        )
        self._regular_output_slots = asyncio.Semaphore(output_queue_size)
        self._generation: int | None = None
        self._input_seq = 0
        self._response_epoch = 0
        self._output_seq = 0
        self._last_processed_input_seq = 0
        self._pending_input_seq = 0
        self._pending_response_epoch: int | None = None
        self._stream: AsyncIterator[CompleteMessage | StreamMessage] | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._created_waiter: asyncio.Future[None] | None = None
        self._closed_waiter: asyncio.Future[None] | None = None
        self._ack_waiters: dict[int, asyncio.Future[DuplexEvent]] = {}
        self._command_lock = asyncio.Lock()
        self._terminal_published = False
        self._terminal_event: DuplexEvent | None = None
        self._started = False
        self._events_claimed = False
        self._closing = False
        self._close_command_task: (
            asyncio.Task[
                tuple[SessionCommandMessage, asyncio.Future[DuplexEvent] | None]
            ]
            | None
        ) = None

    @property
    def generation(self) -> int | None:
        return self._generation

    @property
    def input_seq(self) -> int:
        return self._input_seq

    @property
    def response_epoch(self) -> int:
        return self._response_epoch

    @property
    def is_closed(self) -> bool:
        return self._terminal_published

    async def start(self) -> "DuplexSession":
        """Open the pipeline stream and wait for ``session.created``."""
        if self._started:
            raise RuntimeError(f"Session {self.session_id} has already started")
        self._started = True
        loop = asyncio.get_running_loop()
        self._created_waiter = loop.create_future()
        self._closed_waiter = loop.create_future()
        try:
            generation, stream = await self._coordinator.open_session(
                self.session_id,
                self._request,
                output_queue_size=self._output_queue_size,
            )
            self._generation = generation
            self._stream = stream
            self._consumer_task = asyncio.create_task(
                self._consume_stream(),
                name=f"duplex-session-{self.session_id}",
            )
            await asyncio.wait_for(
                asyncio.shield(self._created_waiter),
                timeout=self._handshake_timeout_s,
            )
        except BaseException:
            await self._abort_and_stop()
            raise
        return self

    async def events(self) -> AsyncIterator[DuplexEvent]:
        """Yield validated session events until a terminal event is observed."""
        self._require_started()
        if self._events_claimed:
            raise DuplexSessionError(
                f"Session {self.session_id} event stream already has a consumer"
            )
        # The terminal event and the internal end marker share one queue.  Two
        # consumers could split those messages and leave one iterator blocked
        # forever, so ownership is deliberately one-shot for the session.
        self._events_claimed = True
        while True:
            event = await self._output_queue.get()
            if event is _QUEUE_END:
                return
            assert isinstance(event, dict)
            if event.get("type") not in {"session.closed", "session.error"}:
                self._regular_output_slots.release()
            yield event

    async def append(
        self,
        data: dict[str, Any],
        *,
        wait_processed: bool = True,
        timeout_s: float | None = None,
    ) -> int:
        """Append one input unit and optionally wait for its processed ack."""
        if not isinstance(data, dict):
            raise TypeError("duplex append data must be a dict")
        message, waiter = await self._send_command(
            "append",
            data,
            create_ack_waiter=wait_processed,
        )
        if waiter is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(waiter),
                    timeout=self._command_timeout_s if timeout_s is None else timeout_s,
                )
            finally:
                self._ack_waiters.pop(message.input_seq, None)
                if not waiter.done():
                    waiter.cancel()
        return message.input_seq

    async def interrupt(self) -> int:
        """Fence the current response while preserving the session context."""
        message, _ = await self._send_command("interrupt", {})
        return message.response_epoch

    async def playback_ack(self, audio_end_ms: float) -> int:
        """Report the latest audio playback commit cursor."""
        if isinstance(audio_end_ms, bool) or not isinstance(audio_end_ms, (int, float)):
            raise TypeError("audio_end_ms must be a non-negative number")
        if not math.isfinite(audio_end_ms) or audio_end_ms < 0:
            raise ValueError("audio_end_ms must be non-negative")
        message, _ = await self._send_command(
            "playback_ack",
            {"audio_end_ms": float(audio_end_ms)},
        )
        return message.input_seq

    async def close(
        self,
        *,
        reason: str = "client_close",
        timeout_s: float | None = None,
    ) -> None:
        """Request graceful close and wait for a terminal session event."""
        if not isinstance(reason, str) or not reason.strip():
            raise TypeError("close reason must be a non-empty str")
        if len(reason) > 128:
            raise ValueError("close reason must be at most 128 characters")
        if not self._started:
            return
        if self._terminal_published:
            self._raise_terminal_error()
            return
        assert self._closed_waiter is not None
        timeout = self._command_timeout_s if timeout_s is None else timeout_s
        try:
            if self._close_command_task is None:
                self._close_command_task = asyncio.create_task(
                    self._send_command("close", {"reason": reason.strip()}),
                    name=f"duplex-session-close-{self.session_id}",
                )
            await asyncio.wait_for(
                asyncio.shield(self._close_command_task), timeout=timeout
            )
            await asyncio.wait_for(
                asyncio.shield(self._closed_waiter),
                timeout=timeout,
            )
            self._raise_terminal_error()
        except BaseException:
            await self._abort_and_stop()
            raise

    async def _send_command(
        self,
        command: str,
        data: dict[str, Any],
        *,
        create_ack_waiter: bool = False,
    ) -> tuple[SessionCommandMessage, asyncio.Future[DuplexEvent] | None]:
        self._require_active()
        assert self._generation is not None
        waiter: asyncio.Future[DuplexEvent] | None = None
        async with self._command_lock:
            self._require_active()
            if command == "close":
                self._closing = True
            expected_seq = self._input_seq + 1
            expected_epoch = self._response_epoch
            if command in {"interrupt", "close"}:
                expected_epoch += 1
            self._pending_input_seq = expected_seq
            self._pending_response_epoch = expected_epoch
            if create_ack_waiter:
                waiter = asyncio.get_running_loop().create_future()
                self._ack_waiters[expected_seq] = waiter
            try:
                message = await self._coordinator.send_session_command(
                    self.session_id,
                    self._generation,
                    command,
                    data,
                )
            except BaseException:
                self._ack_waiters.pop(expected_seq, None)
                self._pending_input_seq = 0
                self._pending_response_epoch = None
                if command == "close" and not self._terminal_published:
                    self._closing = False
                raise
            if message.input_seq != expected_seq:
                self._ack_waiters.pop(expected_seq, None)
                self._pending_input_seq = 0
                self._pending_response_epoch = None
                raise DuplexSessionError(
                    f"Session command sequence jumped from {self._input_seq} "
                    f"to {message.input_seq}"
                )
            if message.response_epoch != expected_epoch:
                self._ack_waiters.pop(expected_seq, None)
                self._pending_input_seq = 0
                self._pending_response_epoch = None
                raise DuplexSessionError(
                    f"Session response epoch jumped from {self._response_epoch} "
                    f"to {message.response_epoch}"
                )
            self._input_seq = message.input_seq
            self._response_epoch = message.response_epoch
            self._pending_input_seq = 0
            self._pending_response_epoch = None
            if self._terminal_published:
                if waiter is not None and waiter.done() and not waiter.cancelled():
                    with suppress(Exception):
                        waiter.result()
                if (
                    command != "close"
                    or self._terminal_event is None
                    or self._terminal_event.get("type") != "session.closed"
                ):
                    raise self._terminal_command_error()
            return message, waiter

    async def _consume_stream(self) -> None:
        assert self._stream is not None
        try:
            async for message in self._stream:
                if isinstance(message, StreamMessage):
                    if not isinstance(message.chunk, dict):
                        raise DuplexSessionError("Duplex stream events must be dicts")
                    should_continue = await self._handle_event(dict(message.chunk))
                    if not should_continue:
                        return
                    continue

                if not message.success:
                    if isinstance(message.result, dict) and message.result.get(
                        "type"
                    ) in {"session.closed", "session.error"}:
                        event = dict(message.result)
                        # A failed completion must never be surfaced as a
                        # successful close, even if the backend attached a
                        # malformed terminal envelope.
                        event["type"] = "session.error"
                        if (
                            not isinstance(event.get("error"), str)
                            or not event["error"]
                        ):
                            event["error"] = message.error or "session failed"
                        await self._handle_event(event)
                        return
                    await self._publish_terminal(
                        self._error_event(message.error or "session failed")
                    )
                    return
                if isinstance(message.result, dict) and message.result.get("type") in {
                    "session.closed",
                    "session.error",
                }:
                    await self._handle_event(dict(message.result))
                    return
                await self._publish_terminal(self._closed_event())
                return
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._publish_terminal(
                self._error_event(str(exc) or type(exc).__name__)
            )
        finally:
            if not self._terminal_published:
                await self._publish_terminal(
                    self._error_event("session stream ended without a terminal event")
                )
            close_stream = getattr(self._stream, "aclose", None)
            if close_stream is not None:
                await close_stream()

    async def _handle_event(self, event: DuplexEvent) -> bool:
        event_type = event.get("type")
        if event_type not in DUPLEX_EVENT_TYPES:
            raise DuplexSessionError(f"Unknown duplex event type: {event_type!r}")
        if event.get("session_id") != self.session_id:
            raise DuplexSessionError("Duplex event session_id does not match")
        generation = event.get("generation")
        if type(generation) is not int:
            raise DuplexSessionError("Duplex event generation must be an int")
        if generation != self._generation:
            return True
        output_seq = event.get("output_seq")
        if type(output_seq) is not int or output_seq <= self._output_seq:
            raise DuplexSessionError(
                "Duplex event output_seq must be a positive, strictly increasing int"
            )
        self._output_seq = output_seq

        if event_type == "session.created":
            if self._created_waiter is None or self._created_waiter.done():
                raise DuplexSessionError("Duplicate session.created event")
            input_seq = event.get("input_seq")
            if type(input_seq) is not int or input_seq != 0:
                raise DuplexSessionError("session.created input_seq must be 0")
            epoch = event.get("response_epoch")
            if type(epoch) is not int or epoch != 0:
                raise DuplexSessionError("session.created response_epoch must be 0")
            self._created_waiter.set_result(None)
            await self._publish_regular_event(event)
            return True

        if self._created_waiter is None or not self._created_waiter.done():
            raise DuplexSessionError(f"Received {event_type} before session.created")

        if event_type == "session.input_processed":
            input_seq = event.get("input_seq")
            if type(input_seq) is not int or input_seq <= 0:
                raise DuplexSessionError("input_processed input_seq must be positive")
            if input_seq > max(self._input_seq, self._pending_input_seq):
                raise DuplexSessionError("input_processed acknowledges a future input")
            if input_seq <= self._last_processed_input_seq:
                raise DuplexSessionError(
                    "input_processed input_seq must be strictly increasing"
                )
            self._last_processed_input_seq = input_seq
            waiter = self._ack_waiters.get(input_seq)
            if waiter is not None and not waiter.done():
                waiter.set_result(event)
            await self._publish_regular_event(event)
            return True

        if event_type in {"response.output.delta", "response.output.done"}:
            epoch = event.get("response_epoch")
            if type(epoch) is not int or epoch < 0:
                raise DuplexSessionError(
                    f"{event_type} response_epoch must be non-negative"
                )
            active_epoch = (
                self._response_epoch
                if self._pending_response_epoch is None
                else self._pending_response_epoch
            )
            if epoch != active_epoch:
                return True
            if event_type == "response.output.done":
                reason = event.get("reason")
                if reason is not None and (
                    not isinstance(reason, str) or not reason.strip()
                ):
                    raise DuplexSessionError(
                        "response.output.done reason must be a non-empty str"
                    )
            await self._publish_regular_event(event)
            return True

        if event_type in {"session.closed", "session.error"}:
            input_seq = event.get("input_seq")
            max_known_input_seq = max(self._input_seq, self._pending_input_seq)
            if (
                type(input_seq) is not int
                or input_seq < 0
                or input_seq > max_known_input_seq
            ):
                raise DuplexSessionError(
                    "terminal input_seq must be a non-negative, committed "
                    "or pending input sequence"
                )
            await self._publish_terminal(event)
            return False

        return True

    async def _publish_regular_event(self, event: DuplexEvent) -> None:
        await self._regular_output_slots.acquire()
        try:
            self._output_queue.put_nowait(event)
        except BaseException:
            self._regular_output_slots.release()
            raise

    async def _publish_terminal(self, event: DuplexEvent) -> None:
        if self._terminal_published:
            return
        self._terminal_published = True
        self._terminal_event = event
        error = DuplexSessionError(str(event.get("error") or "session closed"))
        for waiter in self._ack_waiters.values():
            if not waiter.done():
                waiter.set_exception(error)
        self._ack_waiters.clear()
        if self._created_waiter is not None and not self._created_waiter.done():
            self._created_waiter.set_exception(error)
        if self._closed_waiter is not None and not self._closed_waiter.done():
            self._closed_waiter.set_result(None)
        # Two internal slots are reserved beyond the user-visible regular event
        # capacity, so terminal publication cannot block on a client that has
        # not drained its event iterator yet.
        self._output_queue.put_nowait(event)
        self._output_queue.put_nowait(_QUEUE_END)

    def _terminal_command_error(self) -> DuplexSessionError:
        event = self._terminal_event or {}
        return DuplexSessionError(str(event.get("error") or "session closed"))

    def _raise_terminal_error(self) -> None:
        if self._terminal_event is not None and self._terminal_event.get("type") == (
            "session.error"
        ):
            raise self._terminal_command_error()

    def _closed_event(self) -> DuplexEvent:
        return {
            "type": "session.closed",
            "session_id": self.session_id,
            "generation": self._generation,
            "input_seq": self._input_seq,
            "response_epoch": self._response_epoch,
            "output_seq": self._output_seq + 1,
        }

    def _error_event(self, error: str) -> DuplexEvent:
        return {
            "type": "session.error",
            "session_id": self.session_id,
            "generation": self._generation,
            "input_seq": self._input_seq,
            "response_epoch": self._response_epoch,
            "output_seq": self._output_seq + 1,
            "error": error,
        }

    async def _abort_and_stop(self) -> None:
        close_task = self._close_command_task
        if (
            close_task is not None
            and close_task is not asyncio.current_task()
            and not close_task.done()
        ):
            close_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await close_task
        if self._started:
            with suppress(Exception):
                await self._coordinator.abort(self.session_id)
        task = self._consumer_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError(f"Session {self.session_id} has not started")

    def _require_active(self) -> None:
        self._require_started()
        if self._terminal_published:
            raise RuntimeError(f"Session {self.session_id} is closed")
        if self._closing:
            raise RuntimeError(f"Session {self.session_id} is closing")

    async def __aenter__(self) -> "DuplexSession":
        return await self.start()

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            await self.close()
        else:
            await self._abort_and_stop()
