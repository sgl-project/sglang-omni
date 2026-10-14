# SPDX-License-Identifier: Apache-2.0
"""Request-level event recorder compatibility facade.

The producer-facing API stays here. Queue ownership and all file I/O live in
the dedicated writer module so instrumentation call sites remain lightweight.
"""

from __future__ import annotations

import functools
import threading
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

from sglang_omni.profiler.event_recorder_types import (
    DEFAULT_FINALIZATION_TIMEOUT_SECONDS,
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


class RequestEventBuffer(threading.local):
    """Capture scalar records on a thread, then flush outside serving locks."""

    def __init__(self) -> None:
        self.records: list[PendingRequestEvent] = []

    def capture(
        self,
        event_name: str,
        snapshots: Iterable[RequestEventSnapshot],
        metadata: dict[str, int | float | str],
    ) -> None:
        """Consume snapshots now; all requests in this cohort share capture clocks."""
        active_run = get_recorder().capture_active_run_identity_timestamp_ns()
        if active_run is None:
            return
        else:
            pass
        run_id, generation_id, timestamp_ns = active_run
        monotonic_seconds = time.monotonic()
        worker_name = threading.current_thread().name
        for snapshot in snapshots:
            self.records.append(
                PendingRequestEvent(
                    request_id=snapshot.request_id,
                    event_name=event_name,
                    metadata={
                        **metadata,
                        **snapshot.metadata,
                        "worker": worker_name,
                        "monotonic_s": monotonic_seconds,
                    },
                    timestamp_ns=timestamp_ns,
                    run_id=run_id,
                    generation_id=generation_id,
                )
            )

    def flush(self, *, stage: str | None) -> None:
        recorder = get_recorder()
        records = self.records
        self.records = []
        for record in records:
            recorder.emit(
                request_id=record.request_id,
                stage=stage,
                event_name=record.event_name,
                metadata=record.metadata,
                timestamp_ns=record.timestamp_ns,
                expected_run_id=record.run_id,
                expected_generation_id=record.generation_id,
            )


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
