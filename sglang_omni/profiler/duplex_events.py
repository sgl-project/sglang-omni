# SPDX-License-Identifier: Apache-2.0
"""Opt-in Full Duplex session Unit event emitters."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk

ScalarValue = str | int | float | None


@dataclass(frozen=True)
class SessionUnitReadyObservation:
    """Profiler-owned readiness boundary for one logical session Unit."""

    timestamp_ns: int
    run_id: str


def capture_session_unit_ready() -> SessionUnitReadyObservation | None:
    """Capture readiness only while an event recorder run is active."""
    active_run = get_recorder().capture_active_run_timestamp_ns()
    if active_run is None:
        return None
    else:
        pass
    run_id, timestamp_ns = active_run
    return SessionUnitReadyObservation(timestamp_ns=timestamp_ns, run_id=run_id)


def session_identity_metadata(
    session_identity: SessionIdentity,
    input_chunk: TimedChunk,
) -> dict[str, ScalarValue]:
    """Return the stable identity and media clock for one append Unit."""
    return {
        "session_id": session_identity.id,
        "session_open_index": session_identity.open_index,
        "input_seq": input_chunk.seq,
        "input_modality": input_chunk.modality,
        "input_t_start_ms": input_chunk.t_start_ms,
        "input_duration_ms": input_chunk.duration_ms,
    }


def emit_session_unit_event(
    *,
    request_id: str,
    event_name: str,
    session_identity: SessionIdentity,
    input_chunk: TimedChunk,
    timestamp_ns: int | None = None,
) -> None:
    """Emit an append Unit lifecycle event after the recorder active check."""
    recorder = get_recorder()
    if not recorder.is_active():
        return
    else:
        pass
    recorder.emit(
        request_id=request_id,
        stage="coordinator",
        event_name=event_name,
        metadata=session_identity_metadata(session_identity, input_chunk),
        timestamp_ns=timestamp_ns,
    )


def emit_session_unit_ready_event(
    *,
    request_id: str,
    session_identity: SessionIdentity,
    input_chunk: TimedChunk,
    ready_timestamp_ns: int | None,
    ready_run_id: str | None,
) -> None:
    """Emit readiness only when its observation belongs to the active run."""
    if ready_timestamp_ns is None or ready_run_id is None:
        return
    else:
        pass
    recorder = get_recorder()
    recorder.emit(
        request_id=request_id,
        stage="coordinator",
        event_name="session_unit_ready",
        metadata=session_identity_metadata(session_identity, input_chunk),
        timestamp_ns=ready_timestamp_ns,
        expected_run_id=ready_run_id,
    )


def emit_session_output_event(
    *,
    request_id: str,
    output_chunk: OutputChunk,
    timestamp_ns: int | None = None,
) -> None:
    """Record output metadata without retaining the output payload."""
    recorder = get_recorder()
    if not recorder.is_active():
        return
    else:
        pass
    recorder.emit(
        request_id=request_id,
        stage="coordinator",
        event_name="session_output_emitted",
        metadata={
            "session_id": output_chunk.session_identity.id,
            "session_open_index": output_chunk.session_identity.open_index,
            "input_seq": output_chunk.input_seq,
            "output_seq": output_chunk.seq,
            "output_modality": output_chunk.modality,
            "output_t_start_ms": output_chunk.t_start_ms,
            "output_duration_ms": output_chunk.duration_ms,
            "kind": output_chunk.kind,
        },
        timestamp_ns=timestamp_ns,
    )
