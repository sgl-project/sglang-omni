# SPDX-License-Identifier: Apache-2.0
"""Low-overhead request events for the MOSS-TTS-Realtime critical path."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from sglang_omni.profiler.event_recorder import get_recorder

_EVENT_PREFIX = "moss_tts_realtime_"


def realtime_events_active() -> bool:
    """Return whether this process currently has an active request-event sink."""

    return get_recorder().is_active()


def realtime_identity_metadata(source: Any) -> dict[str, Any]:
    """Extract available session/turn identity from a mapping or state object."""

    if source is None:
        return {}
    values: dict[str, Any] = {}
    for name in ("session_id", "turn_id", "turn_index"):
        value = (
            source.get(name)
            if isinstance(source, Mapping)
            else getattr(source, name, None)
        )
        if value is not None:
            values[name] = value
    return values


def emit_realtime_event(
    *,
    request_id: str,
    stage: str | None,
    event_name: str,
    metadata: Mapping[str, Any] | None = None,
    monotonic_ns: int | None = None,
) -> bool:
    """Emit one model-local event, adding a same-host monotonic timestamp.

    The recorder keeps its top-level wall-clock timestamp for cross-process
    views. Critical-path interval analysis uses ``metadata.monotonic_ns``.
    """

    recorder = get_recorder()
    if not recorder.is_active():
        return False

    values = dict(metadata) if metadata else {}
    values["monotonic_ns"] = (
        time.perf_counter_ns() if monotonic_ns is None else int(monotonic_ns)
    )
    resolved_name = (
        event_name
        if event_name.startswith(_EVENT_PREFIX)
        else _EVENT_PREFIX + event_name
    )
    recorder.emit(
        request_id=str(request_id),
        stage=stage,
        event_name=resolved_name,
        metadata=values,
        timestamp_ns=time.time_ns(),
    )
    return True


__all__ = [
    "emit_realtime_event",
    "realtime_events_active",
    "realtime_identity_metadata",
]
