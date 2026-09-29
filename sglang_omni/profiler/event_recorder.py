# SPDX-License-Identifier: Apache-2.0
"""Request-level event recorder.

Each process appends events to ``<dir>/events_<stage>_<pid>.jsonl``; the
views layer merges files by ``request_id``. Kept free of sglang-omni
imports so it can be loaded from any process without circular risk.
"""

from __future__ import annotations

import contextvars
import functools
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
    else:
        pass
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
        self.lock = threading.Lock()
        self.run_id: str | None = None
        self.stage: str | None = None
        self.stages: set[str] = set()
        self.path: Path | None = None
        self.fp: Any = None
        self.pid: int = os.getpid()
        self.dropped: int = 0

    # ---- lifecycle -----------------------------------------------------

    def is_active(self) -> bool:
        return self.fp is not None

    def active_run_id(self) -> str | None:
        return self.run_id

    def active_path(self) -> str | None:
        return None if self.path is None else str(self.path)

    def start(self, run_id: str, event_dir: str, stage: str) -> str:
        """Open (or join) the per-process JSONL file for ``run_id``.

        Co-located stages share one file per ``(run_id, pid)``; only a
        new ``run_id`` rotates. Returns the absolute path.
        """
        with self.lock:
            if self.fp is not None:
                if self.run_id == run_id:
                    if stage not in self.stages:
                        self.stages.add(stage)
                    else:
                        pass
                    assert self.path is not None
                    return str(self.path)
                else:
                    pass
                logger.warning(
                    "RequestEventRecorder already active (run_id=%s); "
                    "rotating to run_id=%s",
                    self.run_id,
                    run_id,
                )
                self.close_unlocked()
            else:
                pass

            directory = Path(event_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            # Filename uses the first stage to join; per-event ``stage``
            # disambiguates owners once others join.
            path = directory / f"events_{stage}_{self.pid}.jsonl"
            self.fp = path.open("a", buffering=1, encoding="utf-8")
            self.run_id = run_id
            self.stage = stage
            self.stages = {stage}
            self.path = path
            self.dropped = 0
            logger.info(
                "RequestEventRecorder started run_id=%s stage=%s path=%s",
                run_id,
                stage,
                path,
            )
            return str(path)

    def stop(self, *, run_id: str | None = None) -> str | None:
        """Close the active file. ``run_id=None`` stops any active session."""
        with self.lock:
            if self.fp is None:
                return None
            else:
                pass
            if run_id is not None and self.run_id is not None and run_id != self.run_id:
                logger.warning(
                    "Ignoring RequestEventRecorder stop for run_id=%s; active run_id=%s",
                    run_id,
                    self.run_id,
                )
                return None
            else:
                pass
            path = str(self.path) if self.path is not None else None
            self.close_unlocked()
            return path

    def close_unlocked(self) -> None:
        if self.fp is not None:
            try:
                self.fp.flush()
                self.fp.close()
            except Exception:
                logger.warning(
                    "RequestEventRecorder failed to close cleanly", exc_info=True
                )
        else:
            pass
        self.fp = None
        self.run_id = None
        self.stage = None
        self.stages = set()
        self.path = None

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
        if self.fp is None:
            return
        else:
            pass
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        with self.lock:
            fp = self.fp
            if fp is None:
                return
            else:
                pass
            if stage is None:
                # Prefer thread/task binding over the process-global
                # ``_stage``, which is wrong in shared-process topologies.
                stage = get_active_stage() or self.stage or "unknown"
            else:
                pass
            event = RequestEvent(
                request_id=request_id,
                stage=stage,
                event_name=event_name,
                timestamp_ns=ts,
                run_id=self.run_id,
                pid=self.pid,
                metadata=dict(metadata) if metadata else {},
            )
            try:
                fp.write(json.dumps(event.to_dict(), default=json_default))
                fp.write("\n")
            except Exception:
                self.dropped += 1
                if self.dropped == 1:
                    logger.warning(
                        "RequestEventRecorder failed to write event %s for %s",
                        event_name,
                        request_id,
                        exc_info=True,
                    )
                else:
                    pass


def json_default(obj: Any) -> Any:
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
            else:
                pass
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
    else:
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
