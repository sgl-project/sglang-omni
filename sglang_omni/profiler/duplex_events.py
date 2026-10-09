# SPDX-License-Identifier: Apache-2.0
"""Opt-in Full Duplex session Unit event emitters."""

from __future__ import annotations

from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto.session import (
    OutputChunk,
    SessionIdentity,
    TimedChunk,
)

ScalarValue = str | int | float | None


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
