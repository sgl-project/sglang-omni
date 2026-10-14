# SPDX-License-Identifier: Apache-2.0
"""Types and producer-side helpers for request event recording."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_EVENT_QUEUE_CAPACITY = 8192
EVENT_FILE_BUFFER_BYTES = 64 * 1024
MAX_METADATA_SNAPSHOT_DEPTH = 8
MAX_METADATA_SNAPSHOT_ITEMS = 256
MAX_PENDING_FINALIZATIONS = 4
WRITER_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_FINALIZATION_TIMEOUT_SECONDS = 5.0


_thread_active_stage = threading.local()
_active_stage_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sglang_omni_active_stage", default=None
)


def set_active_stage(stage: str | None) -> contextvars.Token[str | None]:
    """Bind ``stage`` for this thread / task. Returns a Token for reset."""
    _thread_active_stage.stage = stage
    return _active_stage_cv.set(stage)


def reset_active_stage(token: contextvars.Token[str | None] | None) -> None:
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
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return the event's JSON-compatible field mapping."""
        return {
            "request_id": self.request_id,
            "stage": self.stage,
            "event_name": self.event_name,
            "timestamp_ns": self.timestamp_ns,
            "run_id": self.run_id,
            "pid": self.pid,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, kw_only=True)
class RequestEventSnapshot:
    request_id: str
    metadata: dict[str, int | float | str]


@dataclass(frozen=True, kw_only=True)
class PendingRequestEvent:
    request_id: str
    event_name: str
    metadata: dict[str, int | float | str]
    timestamp_ns: int
    run_id: str
    generation_id: int | None = None


@dataclass(kw_only=True)
class RequestEventFinalization:
    """Completion handle for one deactivated recorder run."""

    path: str
    status_path: str
    completion_event: threading.Event
    successful: bool = False

    def wait(self, timeout_seconds: float | None = None) -> bool:
        """Wait for the writer to close the event file and status sidecar."""
        if not self.completion_event.wait(timeout_seconds):
            return False
        else:
            pass
        return self.successful


@dataclass
class RecorderRunState:
    run_id: str
    stage: str
    stages: set[str]
    active_stages: set[str]
    path: Path
    status_path: Path
    generation_id: int
    finalization: RequestEventFinalization
    accepted_events: int = 0
    written_events: int = 0
    unwritten_events: int = 0
    dropped_events: int = 0
    metadata_snapshot_failures: int = 0
    metadata_truncations: int = 0
    serialization_failures: int = 0
    event_file_open_failures: int = 0
    write_failures: int = 0
    finalization_failures: int = 0
    incomplete_reasons: set[str] = field(default_factory=set)


@dataclass(frozen=True, kw_only=True)
class QueuedRequestEvent:
    run_state: RecorderRunState
    event: RequestEvent


def snapshot_metadata_value(
    metadata_value: object,
    *,
    depth: int,
    remaining_items: list[int],
    truncated: list[bool] | None = None,
) -> object:
    """Copy bounded JSON-like metadata without touching tensor contents."""
    if remaining_items[0] <= 0 or depth > MAX_METADATA_SNAPSHOT_DEPTH:
        if truncated is not None:
            truncated[0] = True
        else:
            pass
        return {
            "__metadata_truncated__": True,
            "reason": "snapshot_limit",
        }
    else:
        pass
    remaining_items[0] -= 1

    if metadata_value is None or isinstance(metadata_value, (bool, int, float, str)):
        return metadata_value
    else:
        pass

    shape = getattr(metadata_value, "shape", None)
    dtype = getattr(metadata_value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            shape_values = tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            shape_values = None
        else:
            pass
        if shape_values == ():
            device = getattr(metadata_value, "device", None)
            return {
                "__tensor_summary__": True,
                "type": type(metadata_value).__name__,
                "shape": [],
                "dtype": str(dtype),
                "device": str(device) if device is not None else None,
            }
        else:
            pass
        if shape_values is not None:
            device = getattr(metadata_value, "device", None)
            return {
                "__tensor_summary__": True,
                "type": type(metadata_value).__name__,
                "shape": list(shape_values),
                "dtype": str(dtype),
                "device": str(device) if device is not None else None,
            }
        else:
            pass
    else:
        pass

    if isinstance(metadata_value, Mapping):
        copied_mapping: dict[str, object] = {}
        for metadata_key, nested_value in metadata_value.items():
            if remaining_items[0] <= 0:
                if truncated is not None:
                    truncated[0] = True
                else:
                    pass
                break
            else:
                pass
            if not isinstance(metadata_key, str):
                if truncated is not None:
                    truncated[0] = True
                else:
                    pass
                continue
            else:
                pass
            copied_mapping[metadata_key] = snapshot_metadata_value(
                nested_value,
                depth=depth + 1,
                remaining_items=remaining_items,
                truncated=truncated,
            )
        return copied_mapping
    else:
        pass

    if isinstance(metadata_value, (list, tuple)):
        copied_values: list[object] = []
        for nested_value in metadata_value:
            if remaining_items[0] <= 0:
                if truncated is not None:
                    truncated[0] = True
                else:
                    pass
                break
            else:
                pass
            copied_values.append(
                snapshot_metadata_value(
                    nested_value,
                    depth=depth + 1,
                    remaining_items=remaining_items,
                    truncated=truncated,
                )
            )
        return copied_values
    else:
        pass
    try:
        return repr(metadata_value)
    except Exception as exc:
        raise TypeError(
            f"unsupported metadata value {type(metadata_value).__name__}"
        ) from exc


def snapshot_metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    """Capture bounded metadata ownership on the producer thread."""
    if metadata is None:
        return {}
    else:
        pass
    remaining_items = [MAX_METADATA_SNAPSHOT_ITEMS]
    truncated = [False]
    copied_metadata: dict[str, object] = {}
    for metadata_key, metadata_value in metadata.items():
        if remaining_items[0] <= 0:
            truncated[0] = True
            break
        else:
            pass
        if not isinstance(metadata_key, str):
            truncated[0] = True
            continue
        else:
            pass
        copied_metadata[metadata_key] = snapshot_metadata_value(
            metadata_value,
            depth=0,
            remaining_items=remaining_items,
            truncated=truncated,
        )
    if truncated[0]:
        copied_metadata["__metadata_truncated__"] = {
            "reason": "snapshot_limit",
        }
    else:
        pass
    return copied_metadata


def json_default(obj: object) -> object:
    """Safely summarize tensor-like values without materializing contents."""
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            if len(shape) == 0 and hasattr(obj, "item"):
                return obj.item()
            else:
                pass
        except TypeError:
            pass
        try:
            shape_list: list[int] | str = [int(dimension) for dimension in shape]
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
