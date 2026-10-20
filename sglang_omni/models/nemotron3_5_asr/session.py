# SPDX-License-Identifier: Apache-2.0
"""Session lifecycle and the private hook executor for Nemotron ASR."""

from __future__ import annotations

import asyncio
import queue
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

from sglang_omni.admission import QueueFullError
from sglang_omni.models.nemotron3_5_asr.batch_engine import NemotronBatchEngine
from sglang_omni.models.nemotron3_5_asr.streaming import AppendResult
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    ResourceUsage,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.session import (
    SessionContext,
    SessionHooks,
    SessionScheduler,
    StageSession,
)

INBOX_POLL_TIMEOUT_SECONDS: float = 0.05
NOTIFICATION_POLL_TIMEOUT_SECONDS: float = 0.1


class NemotronSessionHooks(SessionHooks):
    def __init__(self, engine: NemotronBatchEngine) -> None:
        self.engine = engine

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.engine.open(session_identity, request).result()

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        try:
            result = self.engine.append(
                context.session_identity, chunk, payload, context.cancelled
            ).result()
            assert isinstance(result, AppendResult)
            return self.finish_append(result, chunk, payload, context)
        except BaseException:
            self.engine.close(context.session_identity).result()
            raise

    def finish_append(
        self,
        result: AppendResult,
        chunk: TimedChunk,
        payload: StagePayload,
        context: SessionContext,
    ) -> StagePayload:
        if context.cancelled.is_set():
            raise RuntimeError("Nemotron append cancelled")
        elif result.text or result.is_final:
            context.emit(
                TimedChunk(
                    modality="text",
                    t_start_ms=chunk.t_start_ms,
                    duration_ms=chunk.duration_ms,
                    seq=chunk.seq,
                    format="text",
                    eos=result.is_final,
                    payload={
                        "text": result.text,
                        "full_text": result.full_text,
                        "is_first_output": result.is_first_output,
                    },
                )
            )
        else:
            pass
        if result.final_payload is not None:
            return result.final_payload
        else:
            payload.data = {"consumed_samples": len(chunk.payload) // 2}
            return payload

    def close(self, session_identity: SessionIdentity) -> None:
        self.engine.close(session_identity).result()

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return self.engine.usage(session_identity)


class NemotronSessionScheduler(SessionScheduler):
    def __init__(
        self,
        engine: NemotronBatchEngine,
        *,
        max_concurrency: int,
        max_open_sessions: int,
        max_state_bytes: int,
    ) -> None:
        self.engine = engine
        self.offline_cancel_events: dict[str, threading.Event] = {}
        self.offline_lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.has_started = False
        super().__init__(
            NemotronSessionHooks(engine),
            compute_fn=self.compute_offline,
            max_concurrency=max_concurrency,
            max_open_sessions=max_open_sessions,
            max_state_bytes_per_session=max_state_bytes,
        )

    def compute_offline(self, payload: StagePayload) -> StagePayload:
        cancelled = threading.Event()
        with self.offline_lock:
            self.offline_cancel_events[payload.request_id] = cancelled
            if self.is_aborted(payload.request_id):
                cancelled.set()
            else:
                pass
        try:
            result = self.engine.submit_offline(payload, cancelled).result()
            assert isinstance(result, StagePayload)
            return result
        finally:
            with self.offline_lock:
                self.offline_cancel_events.pop(payload.request_id, None)

    def cancel_operation(self, request_id: str) -> None:
        super().cancel_operation(request_id)
        with self.offline_lock:
            cancelled = self.offline_cancel_events.get(request_id)
            if cancelled is not None:
                cancelled.set()
            else:
                pass
        with self.engine.condition:
            self.engine.condition.notify_all()

    async def close_session_async(
        self, session_identity: SessionIdentity, session: StageSession
    ) -> None:
        await asyncio.wrap_future(self.engine.close(session_identity))
        session.is_open = False
        with self.session_table_lock:
            self.open_sessions.pop(session_identity, None)

    async def compute_session_async(
        self, payload: StagePayload, session_operation: SessionOperation
    ) -> StagePayload:
        session_identity = session_operation.session_identity
        operation = session_operation.operation
        with self.session_table_lock:
            if operation == "open":
                if self.is_shutting_down:
                    raise RuntimeError("session scheduler is stopping")
                elif session_identity in self.open_sessions:
                    raise ValueError("session already opened")
                elif len(self.open_sessions) >= self.max_open_sessions:
                    raise QueueFullError()
                else:
                    session = StageSession()
                    session.lock.acquire()
                    self.open_sessions[session_identity] = session
            else:
                session = self.open_sessions.get(session_identity)
                if session is None:
                    if operation == "close":
                        payload.data = {"closed": True}
                        return payload
                    else:
                        raise ValueError("unknown session open index")
                else:
                    acquired = session.lock.acquire(blocking=False)
                    assert acquired
        try:
            if operation == "open":
                await asyncio.wrap_future(
                    self.engine.open(session_identity, payload.request)
                )
                session.is_open = True
                self.check_usage(session_identity)
                if self.is_shutting_down:
                    raise RuntimeError("session scheduler is stopping")
                else:
                    payload.data = {"opened": True}
                    return payload
            elif operation == "close":
                await self.close_session_async(session_identity, session)
                payload.data = {"closed": True}
                return payload
            elif self.is_shutting_down:
                raise RuntimeError("session scheduler is stopping")
            else:
                append = self.start_append(payload, session_operation)
                result = await asyncio.wrap_future(
                    self.engine.append(
                        session_identity,
                        append.chunk,
                        payload,
                        append.context.cancelled,
                    )
                )
                assert isinstance(result, AppendResult)
                assert isinstance(self.session_hooks, NemotronSessionHooks)
                updated_payload = self.session_hooks.finish_append(
                    result, append.chunk, payload, append.context
                )
                self.check_usage(session_identity)
                return updated_payload
        except BaseException:
            await self.close_session_async(session_identity, session)
            raise
        finally:
            try:
                if self.is_shutting_down and session.is_open:
                    await self.close_session_async(session_identity, session)
                else:
                    pass
            finally:
                with self.session_table_lock:
                    self.append_cancel_events.pop(payload.request_id, None)
                session.lock.release()

    async def process_operation(self, message: IncomingMessage) -> None:
        if self.consume_if_aborted(message.request_id):
            return
        else:
            pass
        try:
            session_operation = find_session_operation(message.data.request.metadata)
            if session_operation is None:
                result = await asyncio.to_thread(self.compute_offline, message.data)
            else:
                result = await self.compute_session_async(
                    message.data, session_operation
                )
            if self.consume_if_aborted(message.request_id):
                return
            else:
                self.emit_result(message.request_id, result, self.outbox)
        except Exception as error:
            if self.consume_if_aborted(message.request_id):
                return
            else:
                self.emit_error(message.request_id, error, self.outbox)
        finally:
            self.finish_operation(message.request_id)

    def collect_inbox_messages(self) -> list[IncomingMessage]:
        messages = [self.inbox.get(timeout=INBOX_POLL_TIMEOUT_SECONDS)]
        while len(messages) < self.engine.max_pending_tasks:
            try:
                messages.append(self.inbox.get_nowait())
            except queue.Empty:
                break
        return messages

    async def run_workers(self, loop: asyncio.AbstractEventLoop) -> None:
        notifications: asyncio.Queue[IncomingMessage | asyncio.Task[None]] = (
            asyncio.Queue()
        )
        pending_by_session: OrderedDict[
            SessionIdentity | None, deque[IncomingMessage]
        ] = OrderedDict()
        active_tasks: dict[asyncio.Task[None], SessionIdentity | None] = {}
        active_sessions: set[SessionIdentity] = set()
        offline_count = 0

        async def bridge_inbox() -> None:
            while self.running:
                try:
                    messages = await loop.run_in_executor(
                        None, self.collect_inbox_messages
                    )
                except queue.Empty:
                    continue
                for message in messages:
                    notifications.put_nowait(message)

        def completed(task: asyncio.Task[None]) -> None:
            notifications.put_nowait(task)

        bridge_task = asyncio.create_task(bridge_inbox())
        try:
            while self.running or active_tasks:
                try:
                    notification = await asyncio.wait_for(
                        notifications.get(), timeout=NOTIFICATION_POLL_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError:
                    continue
                if isinstance(notification, IncomingMessage):
                    if notification.type != "new_request":
                        continue
                    else:
                        with self.session_table_lock:
                            arrival = self.arrivals_by_request_id.get(
                                notification.request_id
                            )
                        session_identity = (
                            arrival.session_identity if arrival is not None else None
                        )
                        pending = pending_by_session.setdefault(
                            session_identity, deque()
                        )
                        with self.session_table_lock:
                            if (
                                arrival is None
                                or not pending
                                or self.arrivals_by_request_id[
                                    pending[-1].request_id
                                ].sequence
                                < arrival.sequence
                            ):
                                pending.append(notification)
                            else:
                                for index, queued_message in enumerate(pending):
                                    queued_arrival = self.arrivals_by_request_id[
                                        queued_message.request_id
                                    ]
                                    if queued_arrival.sequence > arrival.sequence:
                                        pending.insert(index, notification)
                                        break
                                    else:
                                        pass
                                else:
                                    pending.append(notification)
                else:
                    session_identity = active_tasks.pop(notification)
                    notification.result()
                    if session_identity is None:
                        offline_count -= 1
                    else:
                        active_sessions.remove(session_identity)
                for session_identity in list(pending_by_session):
                    if not self.running:
                        break
                    elif len(active_tasks) >= self.engine.max_pending_tasks:
                        break
                    elif session_identity in active_sessions or (
                        session_identity is None
                        and offline_count >= self.max_concurrency
                    ):
                        continue
                    else:
                        pending = pending_by_session[session_identity]
                        with self.session_table_lock:
                            arrival = self.arrivals_by_request_id.get(
                                pending[0].request_id
                            )
                            cursor = self.cursors_by_session.get(session_identity)
                            if (
                                arrival is not None
                                and cursor is not None
                                and arrival.sequence != cursor.runnable_sequence
                            ):
                                continue
                            else:
                                pass
                        task = asyncio.create_task(
                            self.process_operation(pending.popleft())
                        )
                        active_tasks[task] = session_identity
                        task.add_done_callback(completed)
                        if session_identity is None:
                            offline_count += 1
                        else:
                            active_sessions.add(session_identity)
                        if not pending:
                            pending_by_session.pop(session_identity)
                        else:
                            pass
        finally:
            await bridge_task
            for pending in pending_by_session.values():
                for message in pending:
                    self.finish_operation(message.request_id)
            while not self.inbox.empty():
                self.finish_operation(self.inbox.get_nowait().request_id)
            while not notifications.empty():
                notification = notifications.get_nowait()
                if isinstance(notification, IncomingMessage):
                    self.finish_operation(notification.request_id)
                else:
                    pass

    def start(self) -> None:
        with self.lifecycle_lock:
            if self.is_shutting_down:
                return
            else:
                self.has_started = True
                self.running = True
        loop = asyncio.new_event_loop()
        loop.set_default_executor(
            ThreadPoolExecutor(
                max_workers=self.max_concurrency + 1, thread_name_prefix="nemotron-hook"
            )
        )
        try:
            loop.run_until_complete(self.run_workers(loop))
        finally:
            self.engine.begin_shutdown()
            super().stop()
            self.engine.shutdown()
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    def stop(self) -> None:
        self.engine.begin_shutdown()
        with self.lifecycle_lock:
            super().stop()
            if not self.has_started:
                self.engine.shutdown()
            else:
                pass
