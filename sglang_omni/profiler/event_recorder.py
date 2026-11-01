# SPDX-License-Identifier: Apache-2.0
"""Request-level event recorder API and deferred event buffer."""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from sglang_omni.profiler.event_recorder_types import (
    DEFAULT_FINALIZATION_TIMEOUT_SECONDS,
    MAX_EVENT_TEXT_LENGTH,
    PendingRequestEvent,
    ProducerBuffer,
    RequestEvent,
    RequestEventFinalization,
    RequestEventSnapshot,
    active_stage_context,
    get_active_stage,
    json_default,
    reset_active_stage,
    set_active_stage,
    snapshot_metadata,
    snapshot_metadata_value,
    thread_active_stage,
)
from sglang_omni.profiler.event_recorder_writer import RequestEventRecorder

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


class RequestEventBuffer(threading.local):
    """Defer bounded event records until serving locks are released."""

    def __init__(self) -> None:
        self.producer_buffer: ProducerBuffer | None = None

    def capture(
        self,
        event_name: str,
        snapshots: Iterable[RequestEventSnapshot],
        metadata: dict[str, object],
    ) -> None:
        """Snapshot one cohort once into the current producer's buffer."""
        recorder = get_recorder()
        try:
            producer_buffer = recorder.begin_buffer_operation()
        except Exception:
            return
        else:
            pass
        if producer_buffer is None:
            return
        else:
            pass
        self.producer_buffer = producer_buffer
        run_state = producer_buffer.run_state
        timestamp_ns = time.time_ns()
        monotonic_seconds = time.monotonic()
        worker_name = threading.current_thread().name
        try:
            for snapshot in snapshots:
                if (
                    len(producer_buffer.events) + len(producer_buffer.deferred_events)
                    >= producer_buffer.capacity
                ):
                    producer_buffer.dropped_events += 1
                    producer_buffer.incomplete_reasons.add("producer_buffer_overflow")
                    continue
                else:
                    pass
                if (
                    len(snapshot.request_id) > MAX_EVENT_TEXT_LENGTH
                    or len(event_name) > MAX_EVENT_TEXT_LENGTH
                ):
                    producer_buffer.dropped_events += 1
                    producer_buffer.incomplete_reasons.add("event_payload_too_large")
                    continue
                else:
                    pass
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
                    producer_buffer.metadata_snapshot_failures += 1
                    producer_buffer.dropped_events += 1
                    producer_buffer.incomplete_reasons.add("metadata_snapshot_failure")
                    continue
                else:
                    pass
                if "__metadata_truncated__" in captured_metadata:
                    producer_buffer.metadata_truncations += 1
                    producer_buffer.incomplete_reasons.add("metadata_truncated")
                else:
                    pass
                producer_buffer.deferred_events.append(
                    PendingRequestEvent(
                        request_id=snapshot.request_id,
                        event_name=event_name,
                        metadata=captured_metadata,
                        timestamp_ns=timestamp_ns,
                    )
                )
        except Exception:
            producer_buffer.dropped_events += 1
            producer_buffer.incomplete_reasons.add("deferred_buffer_capture_failure")
        finally:
            producer_buffer.in_flight -= 1

    def flush(self, *, stage: str | None) -> None:
        """Move captured events to the main producer buffer outside serving locks."""
        producer_buffer = self.producer_buffer
        if producer_buffer is None:
            return
        else:
            pass
        run_state = producer_buffer.run_state
        producer_buffer.in_flight += 1
        if get_recorder().active_state is not run_state:
            producer_buffer.in_flight -= 1
            return
        else:
            pass

        resolved_stage = stage or get_active_stage() or run_state.stage or "unknown"
        pending_events = producer_buffer.deferred_events
        producer_buffer.deferred_events = []
        try:
            for pending_event in pending_events:
                if not RequestEventRecorder.event_text_is_bounded(
                    pending_event.request_id,
                    resolved_stage,
                    pending_event.event_name,
                    run_state.run_id,
                ):
                    producer_buffer.dropped_events += 1
                    producer_buffer.incomplete_reasons.add("event_payload_too_large")
                    continue
                else:
                    pass
                try:
                    request_event = RequestEvent(
                        request_id=pending_event.request_id,
                        stage=resolved_stage,
                        event_name=pending_event.event_name,
                        timestamp_ns=pending_event.timestamp_ns,
                        run_id=run_state.run_id,
                        pid=get_recorder().pid,
                        metadata=pending_event.metadata,
                    )
                    producer_buffer.events.append(request_event)
                    producer_buffer.accepted_events += 1
                except Exception:
                    producer_buffer.dropped_events += 1
                    producer_buffer.incomplete_reasons.add(
                        "deferred_buffer_flush_failure"
                    )
        finally:
            producer_buffer.in_flight -= 1


RECORDER = RequestEventRecorder()


def get_recorder() -> RequestEventRecorder:
    """Return the process-local recorder singleton."""
    return RECORDER


def emit(
    *,
    request_id: str,
    stage: str | None,
    event_name: str,
    metadata: Mapping[str, object] | None = None,
    timestamp_ns: int | None = None,
) -> None:
    """Module-level shortcut for the process recorder."""
    RECORDER.emit(
        request_id=request_id,
        stage=stage,
        event_name=event_name,
        metadata=metadata,
        timestamp_ns=timestamp_ns,
    )


@functools.cache
def read_host_boot_id() -> str | None:
    """Read the process boot ID once for model-path events."""
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def emit_model_path(event_name: str, request_id: str, **extra: str) -> None:
    if not RECORDER.is_active():
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
