# SPDX-License-Identifier: Apache-2.0
"""Shared scheduler for simple stages that process streaming chunks.

This keeps SimpleScheduler's inbox/outbox contract, but adds the request
lifecycle needed by stream-processing stages such as vocoders:

1. new_request for both streaming setup and non-streaming compute
2. stream_chunk carrying a StreamItem
3. stream_done that may arrive before the terminal payload
4. non-streaming single and batched compute
"""

from __future__ import annotations

import asyncio
import collections
import logging
import queue as queue_mod
import threading
import time
from collections.abc import Coroutine, Sequence
from typing import Callable

from typing_extensions import Generic, TypeVar

from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage

logger = logging.getLogger(__name__)

ABORTED_REQUEST_ID_LIMIT = 10000
ABORTED_REQUEST_ID_RETAINED = 5000
COMPLETED_NON_STREAMING_REQUEST_ID_LIMIT = 10000
COMPLETED_NON_STREAMING_REQUEST_ID_RETAINED = 5000

StreamingInputT = TypeVar("StreamingInputT", default=StagePayload)


class StreamingSimpleScheduler(Generic[StreamingInputT]):
    """Scheduler base for simple stages with streaming input.

    Subclasses implement the streaming hooks. Non-streaming requests use
    compute_fn / batch_compute_fn exactly like SimpleScheduler, while
    streaming requests are kept out of the non-streaming batch path.
    """

    can_batch_stream_chunks: bool = False
    stream_chunk_batch_max: int | None = None
    stream_chunk_batch_distinct_requests: bool = False

    def __init__(
        self,
        compute_fn: Callable[[StreamingInputT], object] | None,
        *,
        batch_compute_fn: (
            Callable[
                [list[StreamingInputT]],
                Sequence[object] | Coroutine[object, None, Sequence[object]],
            ]
            | None
        ) = None,
        max_batch_size: int = 1,
        max_batch_wait_ms: int = 0,
        request_cost_fn: Callable[[StreamingInputT], int] | None = None,
        max_batch_cost: int | None = None,
        abort_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.inbox: queue_mod.Queue[IncomingMessage] = queue_mod.Queue()
        self.outbox: queue_mod.Queue[OutgoingMessage] = queue_mod.Queue()
        self.requires_tp_work_fanout: bool = True

        self.compute_fn = compute_fn
        self.batch_fn = batch_compute_fn
        self.max_batch_size = max(int(max_batch_size), 1)
        self.max_batch_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self.request_cost_fn = request_cost_fn
        self.max_batch_cost = (
            max(int(max_batch_cost), 0) if max_batch_cost is not None else None
        )
        self.abort_callback = abort_callback

        self.running = False
        self.pending_messages: collections.deque[IncomingMessage] = collections.deque()
        self.pending_done: set[str] = set()
        self.stream_payloads: dict[str, StreamingInputT] = {}
        self.aborted_request_ids: set[str] = set()
        self.completed_non_streaming_request_ids: set[str] = set()
        self.state_lock = threading.RLock()
        self.abort_lock = threading.Lock()

    def is_streaming_payload(
        self,
        payload: StreamingInputT,
    ) -> bool:
        return False

    def validate_non_streaming_payload(self, payload: StreamingInputT) -> None:
        del payload

    def on_streaming_new_request(
        self,
        request_id: str,
        payload: StreamingInputT,
    ) -> None:
        del request_id, payload

    def on_stream_chunk(
        self, request_id: str, item: StreamItem
    ) -> list[OutgoingMessage]:
        del request_id, item
        return []

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        """Caller holds no lock and ignores any return.

        Subclasses own their locking and emit via outbox internally.
        """
        for request_id, item in items:
            if self.is_aborted(request_id):
                continue
            try:
                self.handle_stream_chunk(request_id, item)
            except Exception as exc:
                self.emit_error(request_id, exc)
                self.abort(request_id)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        """Messages that complete the stream, or None to complete it later
        through complete_stream_request."""
        del request_id
        return []

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        """Output a stage owes the client the moment the producer signals EOS.

        stream_done routinely lands before the terminal payload, so the
        rest of on_stream_done is deferred until the payload latches the
        request. Anything already computed must not wait on that latch.
        """
        del request_id
        return []

    def clear_stream_state(self, request_id: str) -> None:
        del request_id

    def has_ready_work(self) -> bool:
        """True when a compute step can run on already-ingested state."""
        return False

    def run_ready_step(self) -> None:
        """One compute step on already-ingested state; runs off the inbox."""

    def start(self) -> None:
        self.running = True
        loop = asyncio.new_event_loop()
        try:
            while self.running:
                if self.has_ready_work():
                    # note (ratish): drain queued messages into state before a
                    # step, so ranking never sees a stale inbox
                    try:
                        msg = self.get_batch_message()
                    except queue_mod.Empty:
                        self.run_ready_step()
                        continue
                else:
                    msg = self.next_message()
                    if msg is None:
                        continue
                if self.is_aborted(msg.request_id):
                    continue
                try:
                    self.handle_message(msg, loop)
                except Exception as exc:
                    logger.exception(
                        "%s failed for %s",
                        self.__class__.__name__,
                        msg.request_id,
                    )
                    self.emit_error(msg.request_id, exc)
                    self.abort(msg.request_id)
        finally:
            loop.close()

    def stop(self) -> None:
        self.running = False

    def abort(self, request_id: str) -> None:
        self.abort_state(request_id)
        self.cleanup_aborted_request(request_id)

    def abort_state(self, request_id: str) -> None:
        self.record_aborted_request_id(request_id)
        self.clear_request_state(request_id, keep_aborted=True)

    def handle_message(
        self, msg: IncomingMessage, loop: asyncio.AbstractEventLoop
    ) -> None:
        if msg.type == "new_request":
            self.handle_new_request_batch(self.collect_new_request_batch(msg), loop)
            return
        if msg.type == "stream_chunk":
            if self.can_batch_stream_chunks:
                self.handle_stream_chunk_batch(self.collect_stream_chunk_batch(msg))
            else:
                self.handle_stream_chunk(msg.request_id, msg.data)
            return
        if msg.type == "stream_done":
            self.handle_stream_done(msg.request_id)
            return
        raise ValueError(f"Unsupported streaming scheduler message type: {msg.type}")

    def next_message(self) -> IncomingMessage | None:
        try:
            return self.get_batch_message(timeout=0.1)
        except queue_mod.Empty:
            return None

    def get_batch_message(self, *, timeout: float = 0.0) -> IncomingMessage:
        if self.pending_messages:
            return self.pending_messages.popleft()
        else:
            return self.inbox.get(timeout=timeout)

    def record_aborted_request_id(self, request_id: str) -> None:
        with self.abort_lock:
            self.aborted_request_ids.add(request_id)
            if len(self.aborted_request_ids) <= ABORTED_REQUEST_ID_LIMIT:
                return
            excess = len(self.aborted_request_ids) - ABORTED_REQUEST_ID_RETAINED
            for stale_request_id in list(self.aborted_request_ids)[:excess]:
                self.aborted_request_ids.discard(stale_request_id)

    def is_aborted(self, request_id: str) -> bool:
        with self.abort_lock:
            return request_id in self.aborted_request_ids

    def clear_request_state(
        self, request_id: str, *, keep_aborted: bool = False
    ) -> None:
        with self.state_lock:
            self.stream_payloads.pop(request_id, None)
            self.pending_done.discard(request_id)
            self.clear_stream_state(request_id)
            if not keep_aborted:
                with self.abort_lock:
                    self.aborted_request_ids.discard(request_id)

    def record_completed_non_streaming_request_id(self, request_id: str) -> None:
        with self.state_lock:
            self.completed_non_streaming_request_ids.add(request_id)
            if (
                len(self.completed_non_streaming_request_ids)
                <= COMPLETED_NON_STREAMING_REQUEST_ID_LIMIT
            ):
                return
            excess = (
                len(self.completed_non_streaming_request_ids)
                - COMPLETED_NON_STREAMING_REQUEST_ID_RETAINED
            )
            for stale_request_id in list(self.completed_non_streaming_request_ids)[
                :excess
            ]:
                self.completed_non_streaming_request_ids.discard(stale_request_id)

    def cleanup_aborted_request(self, request_id: str) -> None:
        if self.abort_callback is None:
            return
        try:
            self.abort_callback(request_id)
        except Exception:
            logger.exception(
                "%s: abort cleanup failed for %s",
                self.__class__.__name__,
                request_id,
            )

    def message_cost(self, msg: IncomingMessage) -> int:
        if self.request_cost_fn is None or msg.type != "new_request":
            return 0
        return max(int(self.request_cost_fn(msg.data)), 0)

    def collect_new_request_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        batch = [first_msg]
        if (
            self.batch_fn is None
            or self.max_batch_size <= 1
            or self.is_streaming_payload(first_msg.data)
        ):
            return batch

        deferred: list[IncomingMessage] = []
        batch_cost = self.message_cost(first_msg)
        deadline = time.monotonic() + self.max_batch_wait_s
        while len(batch) < self.max_batch_size:
            try:
                msg = self.get_batch_message()
            except queue_mod.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.get_batch_message(timeout=remaining)
                except queue_mod.Empty:
                    break

            if self.is_aborted(msg.request_id):
                continue
            if msg.type != "new_request":
                if (
                    msg.type == "stream_done"
                    and msg.request_id not in self.stream_payloads
                ):
                    # Note(Chenchen Hong): Done-before-payload only latches state,
                    # so defer it and keep looking for terminal payloads that can batch.
                    deferred.append(msg)
                    continue
                deferred.append(msg)
                break
            try:
                is_streaming = self.is_streaming_payload(msg.data)
            except Exception as exc:
                self.emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                continue
            if is_streaming:
                deferred.append(msg)
                break
            if self.max_batch_cost is not None:
                try:
                    msg_cost = self.message_cost(msg)
                except Exception as exc:
                    self.emit_error(msg.request_id, exc)
                    self.abort(msg.request_id)
                    continue
                if batch and batch_cost + msg_cost > self.max_batch_cost:
                    deferred.append(msg)
                    break
                batch_cost += msg_cost
            batch.append(msg)
        # note (ratish): restore deferred messages in arrival order ahead of
        # anything still sitting in pending or the inbox
        self.pending_messages.extendleft(reversed(deferred))
        return batch

    def collect_stream_chunk_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        """Front-pushback of the first non-chunk message preserves arrival order; no blocking
        wait, so only already-queued chunks coalesce."""
        batch = [first_msg]
        seen_request_ids = (
            {first_msg.request_id}
            if self.stream_chunk_batch_distinct_requests
            else None
        )
        cap = self.stream_chunk_batch_max or max(self.max_batch_size, 1)
        if cap <= 1:
            return batch
        while len(batch) < cap:
            try:
                msg = self.get_batch_message()
            except queue_mod.Empty:
                break
            if msg.type != "stream_chunk":
                self.pending_messages.appendleft(msg)
                break
            if self.is_aborted(msg.request_id):
                continue
            if seen_request_ids is not None and msg.request_id in seen_request_ids:
                self.pending_messages.appendleft(msg)
                break
            batch.append(msg)
            if seen_request_ids is not None:
                seen_request_ids.add(msg.request_id)
        return batch

    def handle_new_request_batch(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        owns_loop = loop is None
        if loop is None:
            loop = asyncio.new_event_loop()
        try:
            self.handle_new_request_batch_with_loop(batch, loop)
        finally:
            if owns_loop:
                loop.close()

    def handle_new_request_batch_with_loop(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        streaming: list[IncomingMessage] = []
        non_streaming: list[IncomingMessage] = []
        for msg in batch:
            try:
                is_streaming = self.is_streaming_payload(msg.data)
            except Exception as exc:
                self.emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                continue
            if is_streaming:
                streaming.append(msg)
            else:
                non_streaming.append(msg)

        for msg in streaming:
            if self.is_aborted(msg.request_id):
                continue
            self.handle_streaming_new_request(msg.request_id, msg.data)

        if non_streaming:
            self.run_non_streaming_batch(non_streaming, loop)

    def run_non_streaming_batch(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        active = [msg for msg in batch if not self.is_aborted(msg.request_id)]
        if not active:
            return
        with self.state_lock:
            for msg in active:
                self.pending_done.discard(msg.request_id)

        valid: list[IncomingMessage] = []
        for msg in active:
            try:
                self.validate_non_streaming_payload(msg.data)
            except Exception as exc:
                self.emit_error(msg.request_id, exc)
                self.record_completed_non_streaming_request_id(msg.request_id)
                continue
            valid.append(msg)
        if not valid:
            return

        if self.batch_fn is None or len(valid) <= 1:
            for msg in valid:
                if self.is_aborted(msg.request_id):
                    continue
                try:
                    result = self.run_compute(msg.data, loop)
                except Exception as exc:
                    if not self.is_aborted(msg.request_id):
                        self.emit_error(msg.request_id, exc)
                        self.record_completed_non_streaming_request_id(msg.request_id)
                    continue
                if not self.is_aborted(msg.request_id):
                    self.emit_result(msg.request_id, result)
                    self.record_completed_non_streaming_request_id(msg.request_id)
            return

        try:
            results = self.batch_fn([msg.data for msg in valid])
            if asyncio.iscoroutine(results):
                results = loop.run_until_complete(results)
        except Exception as exc:
            for msg in valid:
                if not self.is_aborted(msg.request_id):
                    self.emit_error(msg.request_id, exc)
                    self.record_completed_non_streaming_request_id(msg.request_id)
            return
        if len(results) != len(valid):
            exc = ValueError(
                f"batch_compute_fn returned {len(results)} results for "
                f"{len(valid)} requests"
            )
            for msg in valid:
                if not self.is_aborted(msg.request_id):
                    self.emit_error(msg.request_id, exc)
                    self.record_completed_non_streaming_request_id(msg.request_id)
            return
        for msg, result in zip(valid, results):
            if not self.is_aborted(msg.request_id):
                self.emit_result(msg.request_id, result)
                self.record_completed_non_streaming_request_id(msg.request_id)

    def run_compute(
        self,
        payload: StreamingInputT,
        loop: asyncio.AbstractEventLoop,
    ) -> object:
        if self.compute_fn is None:
            raise RuntimeError(
                f"{self.__class__.__name__} does not support non-streaming compute"
            )
        result = self.compute_fn(payload)
        if asyncio.iscoroutine(result):
            result = loop.run_until_complete(result)
        return result

    def validate_stream_chunk_item(self, request_id: str, item: object) -> StreamItem:
        if not isinstance(item, StreamItem):
            raise TypeError(
                f"{self.__class__.__name__} expected StreamItem for "
                f"{request_id!r}, got {type(item).__name__}"
            )
        return item

    def handle_streaming_new_request(
        self,
        request_id: str,
        payload: StreamingInputT,
    ) -> None:
        with self.abort_lock:
            self.aborted_request_ids.discard(request_id)
        with self.state_lock:
            self.completed_non_streaming_request_ids.discard(request_id)
            self.stream_payloads[request_id] = payload
            self.on_streaming_new_request(request_id, payload)
            if request_id in self.pending_done:
                self.pending_done.discard(request_id)
                self.handle_stream_done(request_id)

    def handle_stream_chunk(self, request_id: str, item: object) -> None:
        item = self.validate_stream_chunk_item(request_id, item)
        with self.state_lock:
            for out in self.on_stream_chunk(request_id, item):
                if not self.is_aborted(request_id):
                    self.outbox.put(out)

    def handle_stream_chunk_batch(self, batch: list[IncomingMessage]) -> None:
        items: list[tuple[str, StreamItem]] = []
        for msg in batch:
            if self.is_aborted(msg.request_id):
                continue
            try:
                item = self.validate_stream_chunk_item(msg.request_id, msg.data)
            except Exception as exc:
                self.emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                continue
            items.append((msg.request_id, item))
        items = [
            (request_id, item)
            for request_id, item in items
            if not self.is_aborted(request_id)
        ]
        if items:
            self.on_stream_chunk_batch(items)

    def handle_stream_done(self, request_id: str) -> None:
        with self.state_lock:
            if request_id not in self.stream_payloads:
                if request_id in self.completed_non_streaming_request_ids:
                    return
                self.pending_done.add(request_id)
                for out in self.on_stream_done_before_payload(request_id):
                    if not self.is_aborted(request_id):
                        self.outbox.put(out)
                return
            messages = self.on_stream_done(request_id)
            if messages is None:
                return
            self.complete_stream_request(request_id, messages)

    def complete_stream_request(
        self, request_id: str, messages: list[OutgoingMessage]
    ) -> None:
        with self.state_lock:
            for out in messages:
                if not self.is_aborted(request_id):
                    self.outbox.put(out)
            if not self.is_aborted(request_id):
                self.clear_request_state(request_id)

    def emit_result(self, request_id: str, result: object) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="result",
                data=result,
            )
        )

    def emit_error(self, request_id: str, error: BaseException) -> None:
        self.outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="error",
                data=error,
            )
        )
