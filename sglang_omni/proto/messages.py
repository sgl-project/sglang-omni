# SPDX-License-Identifier: Apache-2.0
"""Control plane messages."""

from dataclasses import dataclass
from typing import Any

import msgspec

from sglang_omni.proto.admin import AdminOperation, AdminResult
from sglang_omni.proto.request import StagePayload


@dataclass
class DataReadyMessage:
    """Notify next stage that a data-plane object is ready."""

    request_id: str
    from_stage: str
    to_stage: str
    data_ref: dict[str, Any] | None
    chunk_id: int | None = None
    is_done: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        _require_str(self.request_id, "request_id")
        _require_str(self.from_stage, "from_stage")
        _require_str(self.to_stage, "to_stage")
        _require_bool(self.is_done, "is_done")
        if self.is_done and self.error is not None:
            raise ValueError("stream signal cannot be both done and error")
        if self.is_done or self.error is not None:
            if self.data_ref is not None:
                raise ValueError("stream signal must not carry data_ref")
            if self.chunk_id is not None:
                raise ValueError("stream signal must not carry chunk_id")
        elif not isinstance(self.data_ref, dict):
            raise TypeError(
                "DataReadyMessage.data_ref must be dict for data messages, got "
                f"{type(self.data_ref).__name__}"
            )
        d = {
            "type": "data_ready",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
        }
        if self.data_ref is not None:
            d["data_ref"] = self.data_ref.copy()
        if self.chunk_id is not None:
            _require_non_negative_int(self.chunk_id, "chunk_id")
            d["chunk_id"] = self.chunk_id
        if self.is_done:
            d["is_done"] = True
        if self.error is not None:
            _require_str(self.error, "error")
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataReadyMessage":
        request_id = _require_str(d.get("request_id"), "request_id")
        from_stage = _require_str(d.get("from_stage"), "from_stage")
        to_stage = _require_str(d.get("to_stage"), "to_stage")
        data_ref = d.get("data_ref")
        raw_is_done = d.get("is_done", False)
        is_done = _require_bool(raw_is_done, "is_done")
        error = d.get("error")
        if error is not None:
            error = _require_str(error, "error")
        if is_done and error is not None:
            raise ValueError("stream signal cannot be both done and error")
        if is_done or error is not None:
            if data_ref is not None:
                raise ValueError("stream signal must not carry data_ref")
            if "chunk_id" in d:
                raise ValueError("stream signal must not carry chunk_id")
        elif not isinstance(data_ref, dict):
            raise TypeError(
                "data_ready data_ref must be dict for data messages, got "
                f"{type(data_ref).__name__}"
            )
        chunk_id = d.get("chunk_id")
        if chunk_id is not None:
            _require_non_negative_int(chunk_id, "chunk_id")

        return cls(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=to_stage,
            data_ref=data_ref,
            chunk_id=chunk_id,
            is_done=is_done,
            error=error,
        )


class DataAckMessage(msgspec.Struct):
    """Receiver completion for one data-plane object."""

    request_id: str
    from_stage: str
    to_stage: str
    object_id: str
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        _require_str(self.request_id, "request_id")
        _require_str(self.from_stage, "from_stage")
        _require_str(self.to_stage, "to_stage")
        _require_str(self.object_id, "object_id")
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        if self.success:
            if self.error is not None:
                raise ValueError("successful data ack must not carry error")
        else:
            _require_str(self.error, "error")
        d: dict[str, Any] = {
            "type": "data_ack",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "object_id": self.object_id,
            "success": self.success,
        }
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataAckMessage":
        success = d.get("success")
        if not isinstance(success, bool):
            raise TypeError("data_ack success must be bool")
        error = d.get("error")
        if success:
            if error is not None:
                raise ValueError("successful data_ack must not carry error")
        else:
            error = _require_str(error, "error")
        return cls(
            request_id=_require_str(d.get("request_id"), "request_id"),
            from_stage=_require_str(d.get("from_stage"), "from_stage"),
            to_stage=_require_str(d.get("to_stage"), "to_stage"),
            object_id=_require_str(d.get("object_id"), "object_id"),
            success=success,
            error=error,
        )


@dataclass
class AbortMessage:
    """Broadcast abort signal to all stages."""

    request_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "abort", "request_id": self.request_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AbortMessage":
        return cls(request_id=d["request_id"])


@dataclass
class CompleteMessage:
    """Notify coordinator that a request completed (or failed)."""

    request_id: str
    from_stage: str
    success: bool
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "complete",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompleteMessage":
        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            success=d["success"],
            result=d.get("result"),
            error=d.get("error"),
        )


@dataclass
class StreamMessage:
    """Send a partial output chunk to the coordinator."""

    request_id: str
    from_stage: str
    chunk: Any
    stage_id: int | None = None
    stage_name: str | None = None
    modality: str | None = None
    chunk_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "type": "stream",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "chunk": self.chunk,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "modality": self.modality,
        }
        if self.chunk_id is not None:
            d["chunk_id"] = self.chunk_id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamMessage":
        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            chunk=d.get("chunk"),
            stage_id=d.get("stage_id"),
            stage_name=d.get("stage_name"),
            modality=d.get("modality"),
            chunk_id=d.get("chunk_id"),
        )


@dataclass
class SubmitMessage:
    """Submit a new request to the entry stage."""

    request_id: str
    data: Any

    def to_dict(self) -> dict[str, Any]:
        data = self.data
        if isinstance(self.data, StagePayload):
            data = self.data.to_dict()
        return {"type": "submit", "request_id": self.request_id, "data": data}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubmitMessage":
        data = d["data"]
        if isinstance(data, dict) and data.get("_type") == "StagePayload":
            data = StagePayload.from_dict(data)
        return cls(request_id=d["request_id"], data=data)


@dataclass
class ShutdownMessage:
    """Signal graceful shutdown to a stage."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "shutdown"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShutdownMessage":
        return cls()


@dataclass
class ProfilerStartMessage:
    """Profiler start for a stage."""

    run_id: str
    trace_path_template: str  # e.g. "/tmp/profiles/{run_id}/{stage}/trace"
    event_dir: str | None = None  # Per-stage JSONL event sink dir for request profiling
    enable_torch: bool = True  # When False, only request-level events are captured

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "profiler_start",
            "run_id": self.run_id,
            "trace_path_template": self.trace_path_template,
            "event_dir": self.event_dir,
            "enable_torch": self.enable_torch,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProfilerStartMessage":
        return cls(
            run_id=d["run_id"],
            trace_path_template=d["trace_path_template"],
            event_dir=d.get("event_dir"),
            enable_torch=bool(d.get("enable_torch", True)),
        )


@dataclass
class ProfilerStopMessage:
    """Profiler stop. ``run_id=None`` is a wildcard (stop active session)."""

    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"type": "profiler_stop", "run_id": self.run_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProfilerStopMessage":
        return cls(run_id=d.get("run_id"))


@dataclass
class AdminMessage:
    """Send an administrative operation to a stage."""

    operation: AdminOperation

    def to_dict(self) -> dict[str, Any]:
        return {"type": "admin", "operation": self.operation.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AdminMessage":
        return cls(operation=AdminOperation.from_dict(d["operation"]))


@dataclass
class AdminResultMessage:
    """Return an administrative result to the coordinator."""

    result: AdminResult

    def to_dict(self) -> dict[str, Any]:
        return {"type": "admin_result", "result": self.result.to_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AdminResultMessage":
        return cls(result=AdminResult.from_dict(d["result"]))


def parse_message(
    d: dict[str, Any],
) -> (
    AdminMessage
    | AdminResultMessage
    | DataAckMessage
    | DataReadyMessage
    | AbortMessage
    | CompleteMessage
    | StreamMessage
    | SubmitMessage
    | ShutdownMessage
    | ProfilerStartMessage
    | ProfilerStopMessage
):
    """Parse a dict into the appropriate message type."""
    msg_type = d.get("type")
    if msg_type == "data_ready":
        return DataReadyMessage.from_dict(d)
    elif msg_type == "data_ack":
        return DataAckMessage.from_dict(d)
    elif msg_type == "abort":
        return AbortMessage.from_dict(d)
    elif msg_type == "complete":
        return CompleteMessage.from_dict(d)
    elif msg_type == "stream":
        return StreamMessage.from_dict(d)
    elif msg_type == "submit":
        return SubmitMessage.from_dict(d)
    elif msg_type == "shutdown":
        return ShutdownMessage.from_dict(d)
    elif msg_type == "profiler_start":
        return ProfilerStartMessage.from_dict(d)
    elif msg_type == "profiler_stop":
        return ProfilerStopMessage.from_dict(d)
    elif msg_type == "admin":
        return AdminMessage.from_dict(d)
    elif msg_type == "admin_result":
        return AdminResultMessage.from_dict(d)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{name} must be a non-empty str")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    return value


def _require_non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative int")
    return value
