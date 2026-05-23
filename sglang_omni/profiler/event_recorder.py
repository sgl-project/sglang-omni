# SPDX-License-Identifier: Apache-2.0
"""Request-level event recorder.

Emits a stream of small JSONL events covering the per-request milestones laid
out in https://github.com/sgl-project/sglang-omni/issues/501. Events from every
process are written to ``<dir>/events_<stage>_<pid>.jsonl``; the views layer
merges them back into a single per-request timeline.

This module deliberately stays free of any sglang-omni dependency so it can be
imported safely from any process (coordinator, stage, scheduler, model runner)
without circular-import risk.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RequestEvent:
    """A single point-in-time profiling event for one request.

    The shape is intentionally narrow:
    ``request_id``/``stage``/``event_name``/``timestamp_ns`` are required, and
    free-form metadata lives in ``metadata`` so callers can attach token counts,
    chunk ids, audio duration, queue depth, error strings, etc. without
    forcing every event to grow new top-level fields.
    """

    request_id: str
    stage: str
    event_name: str
    timestamp_ns: int
    run_id: str | None = None
    pid: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RequestEventRecorder:
    """Process-local JSONL event sink.

    Thread-safe append: each call serializes one JSON object on its own line.
    Lifetimes are controlled by ``start(run_id, event_dir, stage)`` /
    ``stop()`` so the recorder can be toggled live via the existing
    profiler control plane (``ProfilerStartMessage`` / ``ProfilerStopMessage``).

    The recorder is exposed as a module-level singleton (`get_recorder`) so
    arbitrary callsites can emit without threading a handle through every
    constructor.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_id: str | None = None
        self._stage: str | None = None
        self._path: Path | None = None
        self._fp: Any = None
        self._pid: int = os.getpid()
        self._dropped: int = 0

    # ---- lifecycle -----------------------------------------------------

    def is_active(self) -> bool:
        return self._fp is not None

    def active_run_id(self) -> str | None:
        return self._run_id

    def active_path(self) -> str | None:
        return None if self._path is None else str(self._path)

    def start(self, run_id: str, event_dir: str, stage: str) -> str:
        """Open a fresh per-process JSONL file for this ``run_id``/``stage``.

        Returns the absolute path of the opened file.
        """
        with self._lock:
            if self._fp is not None:
                if self._run_id == run_id and self._stage == stage:
                    assert self._path is not None
                    return str(self._path)
                logger.warning(
                    "RequestEventRecorder already active (run_id=%s, stage=%s); "
                    "rotating to (run_id=%s, stage=%s)",
                    self._run_id,
                    self._stage,
                    run_id,
                    stage,
                )
                self._close_unlocked()

            directory = Path(event_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"events_{stage}_{self._pid}.jsonl"
            # Append mode keeps history if the same run_id is reused.
            self._fp = path.open("a", buffering=1, encoding="utf-8")
            self._run_id = run_id
            self._stage = stage
            self._path = path
            self._dropped = 0
            logger.info(
                "RequestEventRecorder started run_id=%s stage=%s path=%s",
                run_id,
                stage,
                path,
            )
            return str(path)

    def stop(self, *, run_id: str | None = None) -> str | None:
        """Close the active file. Returns the path that was written, if any.

        If ``run_id`` is provided, the call is ignored unless it matches the
        active run — mirrors :class:`TorchProfiler`.
        """
        with self._lock:
            if self._fp is None:
                return None
            if (
                run_id is not None
                and self._run_id is not None
                and run_id != self._run_id
            ):
                logger.warning(
                    "Ignoring RequestEventRecorder stop for run_id=%s; active run_id=%s",
                    run_id,
                    self._run_id,
                )
                return None
            path = str(self._path) if self._path is not None else None
            self._close_unlocked()
            return path

    def _close_unlocked(self) -> None:
        if self._fp is not None:
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                logger.warning(
                    "RequestEventRecorder failed to close cleanly", exc_info=True
                )
        self._fp = None
        self._run_id = None
        self._stage = None
        self._path = None

    # ---- emit ----------------------------------------------------------

    def emit(
        self,
        *,
        request_id: str,
        stage: str | None,
        event_name: str,
        metadata: Mapping[str, Any] | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        """Append a single event. Silent no-op when the recorder is inactive.

        The recorder is intentionally tolerant: any unexpected error during
        emission must NOT propagate to the caller — profiling must never
        break serving. Errors are logged once per occurrence and counted in
        ``self._dropped``.
        """
        if self._fp is None:
            return
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        # Capture file/run state under the lock so we can detect a concurrent
        # stop() that closed the file out from under us.
        with self._lock:
            fp = self._fp
            if fp is None:
                return
            event = RequestEvent(
                request_id=request_id,
                stage=stage if stage is not None else (self._stage or "unknown"),
                event_name=event_name,
                timestamp_ns=ts,
                run_id=self._run_id,
                pid=self._pid,
                metadata=dict(metadata) if metadata else {},
            )
            try:
                fp.write(json.dumps(event.to_dict(), default=_json_default))
                fp.write("\n")
            except Exception:
                self._dropped += 1
                # Log only the first failure per recorder lifetime to avoid
                # swamping the log file.
                if self._dropped == 1:
                    logger.warning(
                        "RequestEventRecorder failed to write event %s for %s",
                        event_name,
                        request_id,
                        exc_info=True,
                    )


def _json_default(obj: Any) -> Any:
    # Conservative fallback so unknown metadata types never break the recorder.
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover - last-resort path
            pass
    return repr(obj)


_RECORDER = RequestEventRecorder()


def get_recorder() -> RequestEventRecorder:
    """Return the process-local recorder singleton."""
    return _RECORDER


def emit(
    *,
    request_id: str,
    stage: str | None,
    event_name: str,
    metadata: Mapping[str, Any] | None = None,
    timestamp_ns: int | None = None,
) -> None:
    """Convenience wrapper: ``emit(request_id=..., event_name=...)``.

    Equivalent to ``get_recorder().emit(...)`` and is the form callers should
    use at instrumentation sites.
    """
    _RECORDER.emit(
        request_id=request_id,
        stage=stage,
        event_name=event_name,
        metadata=metadata,
        timestamp_ns=timestamp_ns,
    )
