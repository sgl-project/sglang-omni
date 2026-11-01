# SPDX-License-Identifier: Apache-2.0
"""Request-level event recorder compatibility facade.

The producer-facing API stays here. Deferred-buffer ownership and all file I/O
live in the dedicated writer module so instrumentation call sites remain
lightweight.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sglang_omni.profiler.event_recorder_types import (
    DEFAULT_FINALIZATION_TIMEOUT_SECONDS,
    MAX_PENDING_BUFFER_STATES,
    PendingRequestEvent,
    RequestEvent,
    RequestEventFinalization,
    RequestEventSnapshot,
    _active_stage_cv,
    _thread_active_stage,
    get_active_stage,
    json_default,
    reset_active_stage,
    set_active_stage,
    snapshot_metadata,
    snapshot_metadata_value,
)
from sglang_omni.profiler.event_recorder_writer import RequestEventRecorder

thread_active_stage = _thread_active_stage
active_stage_context = _active_stage_cv

__all__ = [
    "DEFAULT_FINALIZATION_TIMEOUT_SECONDS",
    "PendingRequestEvent",
    "RequestEvent",
    "RequestEventBuffer",
    "RequestEventFinalization",
    "RequestEventRecorder",
    "RequestEventSnapshot",
    "active_stage_context",
    "emit",
    "emit_model_path",
    "emit_model_path_end",
    "emit_model_path_start",
    "get_active_stage",
    "get_recorder",
    "json_default",
    "read_host_boot_id",
    "reset_active_stage",
    "set_active_stage",
    "snapshot_metadata",
    "snapshot_metadata_value",
    "thread_active_stage",
]


@dataclass
class _DeferredBufferState:
    """Bounded deferred records belonging to one recorder run."""

    run_id: str
    records: list[PendingRequestEvent] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    in_flight: int = 0
    sealed: bool = False

    def seal(self) -> int:
        """Prevent future flushes and return records that were never flushed."""
        while True:
            with self.lock:
                if self.in_flight == 0:
                    pending_count = len(self.records)
                    self.records = []
                    self.sealed = True
                    return pending_count
                else:
                    pass
            time.sleep(0.001)


class RequestEventBuffer(threading.local):
    """Capture bounded scalar records on a thread, then flush them later."""

    def __init__(self) -> None:
        self._states: dict[str, _DeferredBufferState] = {}

    def _state_for_run(
        self,
        recorder: RequestEventRecorder,
        run_id: str,
    ) -> _DeferredBufferState | None:
        for stale_run_id in tuple(self._states):
            stale_state = self._states[stale_run_id]
            with stale_state.lock:
                stale = stale_state.sealed
            if stale:
                del self._states[stale_run_id]
            else:
                pass
        existing_state = self._states.get(run_id)
        if existing_state is not None:
            with existing_state.lock:
                if existing_state.sealed:
                    del self._states[run_id]
                else:
                    existing_state.in_flight += 1
                    return existing_state
        else:
            pass
        if len(self._states) >= MAX_PENDING_BUFFER_STATES:
            return None
        else:
            pass
        state = _DeferredBufferState(run_id=run_id)
        with state.lock:
            state.in_flight = 1
        if not recorder.register_deferred_buffer(run_id, state.seal):
            with state.lock:
                state.in_flight -= 1
            return None
        else:
            pass
        self._states[run_id] = state
        return state

    def capture(
        self,
        event_name: str,
        snapshots: Iterable[RequestEventSnapshot],
        metadata: dict[str, int | float | str],
    ) -> None:
        """Consume snapshots now; all requests in this cohort share capture clocks."""
        recorder = get_recorder()
        active_run = recorder.capture_active_run_timestamp_ns()
        if active_run is None:
            return
        else:
            pass
        run_id, timestamp_ns = active_run
        state = self._state_for_run(recorder, run_id)
        if state is None:
            lost_count = sum(1 for _ in snapshots)
            recorder.record_external_loss(
                run_id, lost_count, "deferred_buffer_registration_failure"
            )
            return
        else:
            pass
        monotonic_seconds = time.monotonic()
        worker_name = threading.current_thread().name
        overflow_count = 0
        snapshot_failure_count = 0
        try:
            for snapshot in snapshots:
                try:
                    captured_metadata = snapshot_metadata(
                        {
                            **metadata,
                            **snapshot.metadata,
                            "worker": worker_name,
                            "monotonic_s": monotonic_seconds,
                        }
                    )
                except Exception:
                    snapshot_failure_count += 1
                    continue
                else:
                    pass
                with state.lock:
                    if len(state.records) >= recorder.queue_capacity:
                        overflow_count += 1
                    else:
                        state.records.append(
                            PendingRequestEvent(
                                request_id=snapshot.request_id,
                                event_name=event_name,
                                metadata=captured_metadata,
                                timestamp_ns=timestamp_ns,
                                run_id=run_id,
                            )
                        )
            if overflow_count:
                recorder.record_external_loss(
                    run_id, overflow_count, "deferred_buffer_overflow"
                )
            else:
                pass
            if snapshot_failure_count:
                recorder.record_external_metadata_snapshot_failure(
                    run_id, snapshot_failure_count
                )
            else:
                pass
        except Exception:
            recorder.record_external_loss(run_id, 1, "deferred_buffer_capture_failure")
        finally:
            with state.lock:
                state.in_flight -= 1

    def flush(self, *, stage: str | None) -> None:
        recorder = get_recorder()
        for state in tuple(self._states.values()):
            with state.lock:
                if state.sealed:
                    continue
                else:
                    pass
                state.in_flight += 1
                records = state.records
                state.records = []
            try:
                for record in records:
                    retained = recorder.emit_record(
                        request_id=record.request_id,
                        stage=stage,
                        event_name=record.event_name,
                        metadata=record.metadata,
                        timestamp_ns=record.timestamp_ns,
                        expected_run_id=record.run_id,
                    )
                    if not retained and recorder.active_run_id() != record.run_id:
                        recorder.record_external_loss(
                            record.run_id, 1, "deferred_buffer_stale"
                        )
                    else:
                        pass
            finally:
                with state.lock:
                    state.in_flight -= 1


_RECORDER = RequestEventRecorder()


def get_recorder() -> RequestEventRecorder:
    """Return the process-local recorder singleton."""
    return _RECORDER


def emit(
    *,
    request_id: str,
    stage: str | None,
    event_name: str,
    metadata: Mapping[str, object] | None = None,
    timestamp_ns: int | None = None,
) -> None:
    """Module-level shortcut for ``get_recorder().emit(...)``."""
    _RECORDER.emit(
        request_id=request_id,
        stage=stage,
        event_name=event_name,
        metadata=metadata,
        timestamp_ns=timestamp_ns,
    )


@functools.cache
def read_host_boot_id() -> str | None:
    # Note: (Jiaxin Deng) constant for the process lifetime, and these events
    # are emitted from the scheduler loop while profiling is active, which is
    # exactly when extra syscalls would contaminate what is being measured.
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def emit_model_path(event_name: str, request_id: str, **extra: str) -> None:
    if not _RECORDER.is_active():
        return
    else:
        pass
    emit(
        request_id=request_id,
        stage=None,
        event_name=event_name,
        metadata={
            "clock": "CLOCK_MONOTONIC",
            "host_boot_id": read_host_boot_id(),
            "monotonic_ns": time.monotonic_ns(),
            **extra,
        },
    )


def emit_model_path_start(request_id: str) -> None:
    emit_model_path("model_path_start", request_id)


def emit_model_path_end(request_id: str, *, status: str) -> None:
    emit_model_path("model_path_end", request_id, status=status)
