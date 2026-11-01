# SPDX-License-Identifier: Apache-2.0
"""Types and producer-side helpers for request event recording."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from sglang_omni.utils.json import JsonValue

DEFAULT_EVENT_BUFFER_CAPACITY = 1024
MAX_PRODUCER_BUFFERS = 16
MAX_EVENT_TEXT_LENGTH = 1024
EVENT_FILE_BUFFER_BYTES = 64 * 1024
MAX_METADATA_SNAPSHOT_DEPTH = 6
MAX_METADATA_SNAPSHOT_ITEMS = 32
MAX_METADATA_SNAPSHOT_CHARACTERS = 512
MAX_METADATA_STRING_LENGTH = 256
WRITER_POLL_INTERVAL_SECONDS = 0.01
DEFAULT_FINALIZATION_TIMEOUT_SECONDS = 5.0


thread_active_stage = threading.local()
active_stage_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sglang_omni_active_stage", default=None
)


def set_active_stage(stage: str | None) -> contextvars.Token[str | None]:
    """Bind ``stage`` for this thread / task. Returns a Token for reset."""
    thread_active_stage.stage = stage
    return active_stage_context.set(stage)


def reset_active_stage(token: contextvars.Token[str | None] | None) -> None:
    """Undo :func:`set_active_stage`. ``token=None`` clears the binding."""
    if token is not None:
        active_stage_context.reset(token)
    else:
        active_stage_context.set(None)
    thread_active_stage.stage = None


def get_active_stage() -> str | None:
    """Active stage for this thread / task, contextvar first."""
    stage = active_stage_context.get()
    if stage is not None:
        return stage
    else:
        pass
    return getattr(thread_active_stage, "stage", None)


@dataclass(frozen=True)
class RequestEvent:
    """A single point-in-time profiling event for one request."""

    request_id: str
    stage: str
    event_name: str
    timestamp_ns: int
    run_id: str | None = None
    pid: int | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
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
    metadata: dict[str, JsonValue]


@dataclass(frozen=True, kw_only=True)
class PendingRequestEvent:
    request_id: str
    event_name: str
    metadata: dict[str, JsonValue]
    timestamp_ns: int


@dataclass(kw_only=True)
class RequestEventFinalization:
    """Report whether one deactivated writer published its status sidecar."""

    path: str
    status_path: str
    completion_event: threading.Event
    successful: bool = False

    def wait(self, timeout_seconds: float | None = None) -> bool:
        """Wait until the writer exits after attempting status publication."""
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
    finalization: RequestEventFinalization
    producer_buffers: list[ProducerBuffer] = field(default_factory=list)
    accepted_events: int = 0
    processed_events: int = 0
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
    writer_scheduling: str = "unknown"


@dataclass
class ProducerBuffer:
    """Bounded event storage owned by one producer thread for one run."""

    run_state: RecorderRunState
    capacity: int
    events: list[RequestEvent] = field(default_factory=list)
    deferred_events: list[PendingRequestEvent] = field(default_factory=list)
    in_flight: int = 0
    accepted_events: int = 0
    dropped_events: int = 0
    metadata_snapshot_failures: int = 0
    metadata_truncations: int = 0
    incomplete_reasons: set[str] = field(default_factory=set)


def snapshot_metadata_text(
    value: object,
    remaining_characters: list[int],
    truncated: list[bool],
) -> str:
    text = str(value)
    captured_text = text[: min(MAX_METADATA_STRING_LENGTH, remaining_characters[0])]
    remaining_characters[0] -= len(captured_text)
    if len(captured_text) != len(text):
        truncated[0] = True
    else:
        pass
    return captured_text


def snapshot_metadata_value(
    metadata_value: object,
    *,
    depth: int,
    remaining_items: list[int],
    remaining_characters: list[int],
    truncated: list[bool],
) -> JsonValue:
    """Copy bounded JSON-like metadata without touching tensor contents."""
    if (
        remaining_items[0] <= 0
        or remaining_characters[0] <= 0
        or depth > MAX_METADATA_SNAPSHOT_DEPTH
    ):
        truncated[0] = True
        return {
            "__metadata_truncated__": True,
            "reason": "snapshot_limit",
        }
    else:
        pass
    remaining_items[0] -= 1

    if metadata_value is None or isinstance(metadata_value, bool):
        return metadata_value
    elif isinstance(metadata_value, int):
        if metadata_value.bit_length() > MAX_METADATA_SNAPSHOT_CHARACTERS * 4:
            truncated[0] = True
            return {
                "__metadata_truncated__": True,
                "reason": "integer_limit",
            }
        else:
            numeric_text = str(metadata_value)
    elif isinstance(metadata_value, float):
        numeric_text = repr(metadata_value)
    elif isinstance(metadata_value, str):
        captured_text = metadata_value[
            : min(MAX_METADATA_STRING_LENGTH, remaining_characters[0])
        ]
        remaining_characters[0] -= len(captured_text)
        if len(captured_text) == len(metadata_value):
            return captured_text
        else:
            truncated[0] = True
            return {
                "__metadata_truncated__": True,
                "reason": "string_limit",
                "value": captured_text,
            }
    elif isinstance(metadata_value, (bytes, bytearray, memoryview)):
        raise TypeError(f"unsupported metadata value {type(metadata_value).__name__}")
    else:
        pass

    if isinstance(metadata_value, (int, float)):
        captured_text = numeric_text[: remaining_characters[0]]
        if len(captured_text) != len(numeric_text):
            truncated[0] = True
            return {
                "__metadata_truncated__": True,
                "reason": "number_limit",
                "value": captured_text,
            }
        else:
            remaining_characters[0] -= len(captured_text)
            return metadata_value
    else:
        pass

    shape = getattr(metadata_value, "shape", None)
    dtype = getattr(metadata_value, "dtype", None)
    if shape is not None and dtype is not None:
        shape_values: list[int] | None = []
        shape_was_truncated = False
        try:
            for dimension_index, dimension in enumerate(shape):
                if dimension_index >= min(
                    MAX_METADATA_SNAPSHOT_ITEMS, remaining_items[0]
                ):
                    shape_was_truncated = True
                    truncated[0] = True
                    break
                else:
                    pass
                shape_values.append(int(dimension))
        except (TypeError, ValueError):
            shape_values = None
        else:
            pass
        if shape_values == []:
            device = getattr(metadata_value, "device", None)
            device_text = (
                snapshot_metadata_text(device, remaining_characters, truncated)
                if device is not None
                else None
            )
            scalar_available = False
            scalar_value: object = None
            if hasattr(metadata_value, "item") and (
                device_text is None or device_text.startswith("cpu")
            ):
                try:
                    scalar_value = metadata_value.item()
                except Exception:
                    pass
                else:
                    scalar_available = True
            else:
                pass
            if scalar_available and (
                scalar_value is None
                or isinstance(scalar_value, (bool, int, float, str))
            ):
                return snapshot_metadata_value(
                    scalar_value,
                    depth=depth + 1,
                    remaining_items=remaining_items,
                    remaining_characters=remaining_characters,
                    truncated=truncated,
                )
            else:
                pass
            type_name = snapshot_metadata_text(
                type(metadata_value).__name__, remaining_characters, truncated
            )
            dtype_text = snapshot_metadata_text(dtype, remaining_characters, truncated)
            return {
                "__tensor_summary__": True,
                "type": type_name,
                "shape": [],
                "dtype": dtype_text,
                "device": device_text,
            }
        else:
            pass
        if shape_values is not None:
            device = getattr(metadata_value, "device", None)
            device_text = (
                snapshot_metadata_text(device, remaining_characters, truncated)
                if device is not None
                else None
            )
            type_name = snapshot_metadata_text(
                type(metadata_value).__name__, remaining_characters, truncated
            )
            dtype_text = snapshot_metadata_text(dtype, remaining_characters, truncated)
            captured_shape: list[int] = []
            for dimension in shape_values:
                dimension_characters = len(str(dimension))
                if dimension_characters > remaining_characters[0]:
                    truncated[0] = True
                    shape_was_truncated = True
                    break
                else:
                    pass
                remaining_characters[0] -= dimension_characters
                captured_shape.append(dimension)
            tensor_summary: dict[str, JsonValue] = {
                "__tensor_summary__": True,
                "type": type_name,
                "shape": captured_shape,
                "dtype": dtype_text,
                "device": device_text,
            }
            if shape_was_truncated:
                tensor_summary["shape_truncated"] = True
            else:
                pass
            return tensor_summary
        else:
            pass
    else:
        pass

    if isinstance(metadata_value, Mapping):
        copied_mapping: dict[str, JsonValue] = {}
        for metadata_key, nested_value in metadata_value.items():
            if remaining_items[0] <= 0 or remaining_characters[0] <= 0:
                truncated[0] = True
                break
            else:
                pass
            if not isinstance(metadata_key, str):
                truncated[0] = True
                continue
            else:
                pass
            if len(metadata_key) > min(
                MAX_METADATA_STRING_LENGTH, remaining_characters[0]
            ):
                truncated[0] = True
                break
            else:
                pass
            remaining_characters[0] -= len(metadata_key)
            copied_mapping[metadata_key] = snapshot_metadata_value(
                nested_value,
                depth=depth + 1,
                remaining_items=remaining_items,
                remaining_characters=remaining_characters,
                truncated=truncated,
            )
        return copied_mapping
    else:
        pass

    if isinstance(metadata_value, (list, tuple)):
        copied_values: list[JsonValue] = []
        for nested_value in metadata_value:
            if remaining_items[0] <= 0 or remaining_characters[0] <= 0:
                truncated[0] = True
                break
            else:
                pass
            copied_values.append(
                snapshot_metadata_value(
                    nested_value,
                    depth=depth + 1,
                    remaining_items=remaining_items,
                    remaining_characters=remaining_characters,
                    truncated=truncated,
                )
            )
        return copied_values
    else:
        pass
    raise TypeError(f"unsupported metadata value {type(metadata_value).__name__}")


def snapshot_metadata(metadata: Mapping[str, object] | None) -> dict[str, JsonValue]:
    """Capture bounded metadata ownership on the producer thread."""
    if metadata is None:
        return {}
    else:
        pass
    remaining_items = [MAX_METADATA_SNAPSHOT_ITEMS]
    remaining_characters = [MAX_METADATA_SNAPSHOT_CHARACTERS]
    truncated = [False]
    copied_metadata: dict[str, JsonValue] = {}
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
        if len(metadata_key) > min(MAX_METADATA_STRING_LENGTH, remaining_characters[0]):
            truncated[0] = True
            break
        else:
            pass
        remaining_characters[0] -= len(metadata_key)
        copied_metadata[metadata_key] = snapshot_metadata_value(
            metadata_value,
            depth=0,
            remaining_items=remaining_items,
            remaining_characters=remaining_characters,
            truncated=truncated,
        )
    if truncated[0]:
        copied_metadata["__metadata_truncated__"] = {
            "reason": "snapshot_limit",
        }
    else:
        pass
    return copied_metadata


def json_default(metadata_value: object) -> JsonValue:
    """Safely summarize tensor-like values without materializing contents."""
    return snapshot_metadata_value(
        metadata_value,
        depth=0,
        remaining_items=[MAX_METADATA_SNAPSHOT_ITEMS],
        remaining_characters=[MAX_METADATA_SNAPSHOT_CHARACTERS],
        truncated=[False],
    )
