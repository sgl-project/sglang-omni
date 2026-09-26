# SPDX-License-Identifier: Apache-2.0
"""Session lifecycle and the private hook executor for Nemotron ASR."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from sglang_omni.models.nemotron3_5_asr.batch_engine import NemotronBatchEngine
from sglang_omni.models.nemotron3_5_asr.streaming import AppendResult
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks, SessionScheduler


class NemotronSessionHooks(SessionHooks):
    def __init__(self, engine: NemotronBatchEngine) -> None:
        self.engine = engine

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.engine.open(session_identity, request).result()

    def append(self, chunk: TimedChunk, payload: StagePayload, context: SessionContext) -> StagePayload:
        try:
            result = self.engine.append(context.session_identity, chunk, payload, context.cancelled).result()
            assert isinstance(result, AppendResult)
            if context.cancelled.is_set():
                raise RuntimeError("Nemotron append cancelled")
            elif result.text or result.is_final:
                context.emit(TimedChunk(
                    modality="text", t_start_ms=chunk.t_start_ms,
                    duration_ms=chunk.duration_ms, seq=chunk.seq, format="text",
                    eos=result.is_final,
                    payload={"text": result.text, "full_text": result.full_text,
                             "is_first_output": result.is_first_output},
                ))
            else:
                pass
            if result.final_payload is not None:
                return result.final_payload
            else:
                payload.data = {"consumed_samples": len(chunk.payload) // 2}
                return payload
        except BaseException:
            self.engine.close(context.session_identity).result()
            raise

    def close(self, session_identity: SessionIdentity) -> None:
        self.engine.close(session_identity).result()

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        return self.engine.usage(session_identity)


class NemotronSessionScheduler(SessionScheduler):
    def __init__(self, engine: NemotronBatchEngine, *, max_concurrency: int,
                 max_open_sessions: int, max_state_bytes: int) -> None:
        self.engine = engine
        self.offline_cancel_events: dict[str, threading.Event] = {}
        self.offline_lock = threading.Lock()
        self.lifecycle_lock = threading.Lock()
        self.has_started = False
        super().__init__(
            NemotronSessionHooks(engine), compute_fn=self.compute_offline,
            max_concurrency=max_concurrency, max_open_sessions=max_open_sessions,
            max_state_bytes=max_state_bytes,
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

    def start(self) -> None:
        with self.lifecycle_lock:
            if self.is_shutting_down:
                return
            else:
                self.has_started = True
                self.running = True
        loop = asyncio.new_event_loop()
        loop.set_default_executor(ThreadPoolExecutor(
            max_workers=self.max_concurrency + 1, thread_name_prefix="nemotron-hook"))
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
