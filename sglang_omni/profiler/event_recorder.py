# SPDX-License-Identifier: Apache-2.0
"""Request-level event recorder.

Each process appends events to ``<dir>/events_<stage>_<pid>.jsonl``; the
views layer merges files by ``request_id``. Kept free of sglang-omni
imports so it can be loaded from any process without circular risk.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


# Active-stage binding used when ``emit(stage=None)`` is called from code
# that can't plumb the stage name down (preprocessor, encoder callable,
# scheduler internals). Stage._run_scheduler binds the active stage on
# the scheduler thread; the contextvar propagates through
# ``asyncio.to_thread`` / ``loop.run_in_executor``, the thread-local
# covers plain ``threading.Thread`` workers.

_thread_active_stage = threading.local()
_active_stage_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sglang_omni_active_stage", default=None
)


def set_active_stage(stage: str | None) -> contextvars.Token:
    """Bind ``stage`` for this thread / task. Returns a Token for reset."""
    _thread_active_stage.stage = stage
    return _active_stage_cv.set(stage)


def reset_active_stage(token: contextvars.Token | None) -> None:
    """Undo :func:`set_active_stage`. ``token=None`` clears the binding."""
    if token is not None:
        _active_stage_cv.reset(token)
    else:
        _active_stage_cv.set(None)
    _thread_active_stage.stage = None


def get_active_stage() -> str | None:
    """Active stage for this thread / task, contextvar first."""
    stage = _active_stage_cv.get()
    if stage is not None:
        return stage
    return getattr(_thread_active_stage, "stage", None)


@dataclass(frozen=True)
class RequestEvent:
    """A single point-in-time profiling event for one request."""

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
    """Process-local JSONL event sink. Toggled via profiler control plane."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_id: str | None = None
        self._stage: str | None = None
        self._stages: set[str] = set()
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
        """Open (or join) the per-process JSONL file for ``run_id``.

        Co-located stages share one file per ``(run_id, pid)``; only a
        new ``run_id`` rotates. Returns the absolute path.
        """
        with self._lock:
            if self._fp is not None:
                if self._run_id == run_id:
                    if stage not in self._stages:
                        self._stages.add(stage)
                    assert self._path is not None
                    return str(self._path)
                logger.warning(
                    "RequestEventRecorder already active (run_id=%s); "
                    "rotating to run_id=%s",
                    self._run_id,
                    run_id,
                )
                self._close_unlocked()

            directory = Path(event_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            # Filename uses the first stage to join; per-event ``stage``
            # disambiguates owners once others join.
            path = directory / f"events_{stage}_{self._pid}.jsonl"
            self._fp = path.open("a", buffering=1, encoding="utf-8")
            self._run_id = run_id
            self._stage = stage
            self._stages = {stage}
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
        """Close the active file. ``run_id=None`` stops any active session."""
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
        self._stages = set()
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
        """Append one event. No-op when inactive; errors are swallowed."""
        if self._fp is None:
            return
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        with self._lock:
            fp = self._fp
            if fp is None:
                return
            if stage is None:
                # Prefer thread/task binding over the process-global
                # ``_stage``, which is wrong in shared-process topologies.
                stage = get_active_stage() or self._stage or "unknown"
            event = RequestEvent(
                request_id=request_id,
                stage=stage,
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
                if self._dropped == 1:
                    logger.warning(
                        "RequestEventRecorder failed to write event %s for %s",
                        event_name,
                        request_id,
                        exc_info=True,
                    )


def _json_default(obj: Any) -> Any:
    """Safe fallback for ``json.dumps``: summarise tensors, never materialise.

    Tensors / arrays return ``{__tensor_summary__, type, shape, dtype,
    device}``; 0-D variants serialise as plain scalars; everything else
    falls back to ``repr``.
    """
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            if len(shape) == 0 and hasattr(obj, "item"):
                return obj.item()
        except TypeError:
            # ``.shape`` without ``__len__`` — skip the 0-D fast path
            # and fall through to the summary serializer below.
            pass
        try:
            shape_list: Any = [int(d) for d in shape]
        except Exception:
            shape_list = repr(shape)
        device = getattr(obj, "device", None)
        return {
            "__tensor_summary__": True,
            "type": type(obj).__name__,
            "shape": shape_list,
            "dtype": str(dtype),
            "device": str(device) if device is not None else None,
        }
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
    """Module-level shortcut for ``get_recorder().emit(...)``."""
    _RECORDER.emit(
        request_id=request_id,
        stage=stage,
        event_name=event_name,
        metadata=metadata,
        timestamp_ns=timestamp_ns,
    )
