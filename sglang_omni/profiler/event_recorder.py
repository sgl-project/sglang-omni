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


# ---------------------------------------------------------------------------
# Active-stage attribution
# ---------------------------------------------------------------------------
#
# When multiple stages live in the same OS process (the declarative topology
# does this for co-located ``role="single"`` stages), the recorder keeps ONE
# file per process and the per-event ``stage`` field is the source of truth.
# Most callsites pass ``stage=self.name`` explicitly, but library code that
# can't easily plumb the stage name down (preprocessors, encoder callables,
# OmniScheduler internals) calls ``emit(stage=None)`` and expects the
# recorder to fill it in.
#
# A process-global fallback (the recorder's ``_stage``) is wrong: in a shared
# process it's whichever stage called ``start()`` first, so every event from
# every co-located stage would get that one name.
#
# Instead we expose a per-thread / per-task active stage. Stage's scheduler
# thread sets it before invoking ``scheduler.start()``; emits in that thread
# (and in any ``asyncio.to_thread`` / ``loop.run_in_executor`` descendants
# that copy the context) see the correct stage name. Explicit ``stage=`` on
# emit always wins.

_thread_active_stage = threading.local()
_active_stage_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sglang_omni_active_stage", default=None
)


def set_active_stage(stage: str | None) -> contextvars.Token | None:
    """Bind ``stage`` as the active stage for the current thread / task.

    Returns a ``contextvars.Token`` that ``reset_active_stage`` can use to
    restore the previous value, or ``None`` if no contextvar binding was
    performed (e.g. when ``stage`` is ``None`` and clearing was requested).

    Sets BOTH a ``threading.local`` slot (for plain-thread callsites that
    don't propagate contextvars) AND the contextvar (for asyncio /
    ``run_in_executor`` callsites that copy context).
    """
    _thread_active_stage.stage = stage
    return _active_stage_cv.set(stage)


def reset_active_stage(token: contextvars.Token | None) -> None:
    """Reverse a prior :func:`set_active_stage` call.

    With a ``token``, restores the previous contextvar value (the standard
    ``ContextVar.reset`` contract). Without a token, clears the binding by
    setting the contextvar back to ``None`` — this is the form fixtures
    use to scrub leaked active-stage state between tests, so it must
    actually clear the contextvar and not just the ``threading.local``.
    """
    if token is not None:
        _active_stage_cv.reset(token)
    else:
        _active_stage_cv.set(None)
    _thread_active_stage.stage = None


def get_active_stage() -> str | None:
    """Return the active stage for this thread / task, or ``None``.

    The contextvar takes precedence so asyncio tasks see the binding even
    when running inside ``loop.run_in_executor`` (which copies context but
    not thread-local). Thread-local is the fallback for plain
    ``threading.Thread`` workers.
    """
    stage = _active_stage_cv.get()
    if stage is not None:
        return stage
    return getattr(_thread_active_stage, "stage", None)


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
        """Open (or join) the per-process JSONL file for this ``run_id``.

        Multiple stages can share one OS process (declarative topology,
        co-located non-AR stages). When that happens each Stage will call
        ``start()`` independently as it receives the profiler-start
        broadcast. We keep ONE file per ``(run_id, pid)`` and let every
        stage write into it — the per-event ``stage`` field makes the
        owner unambiguous, and the views layer groups by ``request_id``
        anyway. Only a different ``run_id`` (a brand-new profiling
        session) triggers a rotation.

        Returns the absolute path of the file the caller will write to.
        """
        with self._lock:
            if self._fp is not None:
                if self._run_id == run_id:
                    # Same session — just register the additional stage so
                    # the filename reflects the full set, and keep writing
                    # to the existing file.
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
            # Filename carries the FIRST stage to call start() in this
            # process. Other stages in the same process append to this
            # same file; their identity lives in each event's `stage`.
            path = directory / f"events_{stage}_{self._pid}.jsonl"
            # Append mode keeps history if the same run_id is reused.
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
            if stage is None:
                # Prefer the per-thread / per-task binding (Stage._run_scheduler
                # sets it before invoking the scheduler). The process-global
                # ``_stage`` is only used when nothing better is available,
                # and is wrong in shared-process topologies — see module
                # docstring.
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
    """Fallback serializer for objects ``json.dumps`` doesn't know.

    Profiler metadata must stay tiny — the profiler docs explicitly say
    "large blobs stay out of metadata". Earlier versions of this function
    called ``.tolist()`` on anything that had it, which materialised
    arbitrarily large numpy arrays AND synchronised + copied GPU tensors
    to CPU. Both can balloon a JSON line into megabytes and stall the hot
    path.

    Current behaviour:

    - 0-D tensors / arrays (``shape == ()``) are scalars; return ``item()``.
    - Higher-rank tensors / arrays return a SUMMARY dict
      (``type``, ``shape``, ``dtype``, ``device``) — never the data.
    - Everything else falls back to ``repr``.
    """
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None:
        # 0-D tensor / array → behaves like a scalar.
        # ``len(shape)`` works for ``torch.Size`` and ``numpy.ndarray.shape``
        # (tuples). Some duck-typed objects expose a ``.shape`` that isn't
        # sized — ``len()`` raises ``TypeError`` on those, and we fall
        # through to the summary path instead of crashing the recorder.
        try:
            if len(shape) == 0 and hasattr(obj, "item"):
                return obj.item()
        except TypeError:
            pass
        # Real tensor / array — refuse to materialise. Summary only.
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
