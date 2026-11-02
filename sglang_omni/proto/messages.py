# SPDX-License-Identifier: Apache-2.0
"""Control plane messages."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

import msgspec

from sglang_omni.proto.admin import AdminOperation, AdminResult, SerializedAdminResult
from sglang_omni.proto.kv_transfer import (
    KVTransferPrepareMessage,
    KVTransferReadyMessage,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import SessionIdentityDict


class DirectCudaIpcTensor(TypedDict):
    path: str
    tensor_bytes: bytes


class DirectCudaIpcPayloadRef(TypedDict):
    _type: Literal["TorchCudaIpcPayload"]
    version: Literal[1]
    header: bytes
    tensors: list[DirectCudaIpcTensor]


class DirectCudaIpcStreamChunkOptionalFields(TypedDict, total=False):
    metadata: dict[str, object]


class DirectCudaIpcStreamChunkRef(DirectCudaIpcStreamChunkOptionalFields):
    _type: Literal["TorchCudaIpcStreamChunk"]
    version: Literal[1]
    tensor_bytes: bytes


StageDataRef: TypeAlias = DirectCudaIpcPayloadRef | DirectCudaIpcStreamChunkRef


@dataclass
class DataReadyMessage:
    """Notify next stage that a data-plane object is ready."""

    request_id: str
    from_stage: str
    to_stage: str
    data_ref: dict[str, object] | StageDataRef | None
    chunk_id: int | None = None
    is_done: bool = False
    error: str | None = None
    replica_bindings: dict[str, int] | None = None
    session_sampling_identity: SessionIdentityDict | None = None

    def to_dict(self) -> dict[str, object]:
        require_str(self.request_id, "request_id")
        require_str(self.from_stage, "from_stage")
        require_str(self.to_stage, "to_stage")
        require_bool(self.is_done, "is_done")
        if self.is_done and self.error is not None:
            raise ValueError("stream signal cannot be both done and error")
        else:
            pass
        if self.is_done or self.error is not None:
            if self.data_ref is not None:
                raise ValueError("stream signal must not carry data_ref")
            else:
                pass
            if self.chunk_id is not None:
                raise ValueError("stream signal must not carry chunk_id")
            else:
                pass
        elif not isinstance(self.data_ref, dict):
            raise TypeError(
                "DataReadyMessage.data_ref must be dict for data messages, got "
                f"{type(self.data_ref).__name__}"
            )
        else:
            pass
        d: dict[str, object] = {
            "type": "data_ready",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
        }
        if self.data_ref is not None:
            d["data_ref"] = self.data_ref.copy()
        else:
            pass
        if self.chunk_id is not None:
            require_non_negative_int(self.chunk_id, "chunk_id")
            d["chunk_id"] = self.chunk_id
        else:
            pass
        if self.is_done:
            d["is_done"] = True
        else:
            pass
        if self.error is not None:
            require_str(self.error, "error")
            d["error"] = self.error
        else:
            pass
        if self.replica_bindings:
            d["replica_bindings"] = dict(self.replica_bindings)
        else:
            pass
        if self.session_sampling_identity is not None:
            d["session_sampling_identity"] = require_session_sampling_identity(
                self.session_sampling_identity
            )
        else:
            pass
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "DataReadyMessage":
        request_id = require_str(d.get("request_id"), "request_id")
        from_stage = require_str(d.get("from_stage"), "from_stage")
        to_stage = require_str(d.get("to_stage"), "to_stage")
        data_ref = d.get("data_ref")
        raw_is_done = d.get("is_done", False)
        is_done = require_bool(raw_is_done, "is_done")
        error = d.get("error")
        if error is not None:
            error = require_str(error, "error")
        else:
            pass
        if is_done and error is not None:
            raise ValueError("stream signal cannot be both done and error")
        else:
            pass
        if is_done or error is not None:
            if data_ref is not None:
                raise ValueError("stream signal must not carry data_ref")
            else:
                pass
            if "chunk_id" in d:
                raise ValueError("stream signal must not carry chunk_id")
            else:
                pass
        elif not isinstance(data_ref, dict):
            raise TypeError(
                "data_ready data_ref must be dict for data messages, got "
                f"{type(data_ref).__name__}"
            )
        else:
            pass
        chunk_id = d.get("chunk_id")
        if chunk_id is not None:
            require_non_negative_int(chunk_id, "chunk_id")
        else:
            pass

        return cls(
            request_id=request_id,
            from_stage=from_stage,
            to_stage=to_stage,
            data_ref=data_ref,
            chunk_id=chunk_id,
            is_done=is_done,
            error=error,
            replica_bindings=d.get("replica_bindings"),
            session_sampling_identity=(
                None
                if d.get("session_sampling_identity") is None
                else require_session_sampling_identity(
                    d.get("session_sampling_identity")
                )
            ),
        )


@dataclass
class StreamChunkBatchMessage:
    """Small CPU stream chunks of many requests for one target stage.

    Index i of every list describes chunk i, and pickled_metadata is the pickled list of every chunk's metadata.
    """

    from_stage: str
    to_stage: str
    request_ids: list[str]
    chunk_ids: list[int]
    dtypes: list[str]
    shapes: list[list[int]]
    tensor_bytes: list[bytes]
    pickled_metadata: bytes
    replica_bindings: dict[str, dict[str, int]]
    session_sampling_identities: list[SessionIdentityDict | None] | None = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "type": "stream_chunk_batch",
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "request_ids": self.request_ids,
            "chunk_ids": self.chunk_ids,
            "dtypes": self.dtypes,
            "shapes": self.shapes,
            "tensor_bytes": self.tensor_bytes,
            "pickled_metadata": self.pickled_metadata,
            "replica_bindings": self.replica_bindings,
        }
        if self.session_sampling_identities is not None:
            if len(self.session_sampling_identities) != len(self.request_ids):
                raise ValueError(
                    "stream chunk batch session sampling identities must match chunks"
                )
            else:
                pass
            d["session_sampling_identities"] = [
                None
                if identity is None
                else require_session_sampling_identity(identity)
                for identity in self.session_sampling_identities
            ]
        else:
            pass
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "StreamChunkBatchMessage":
        columns = [
            d.get(name)
            for name in ("request_ids", "chunk_ids", "dtypes", "shapes", "tensor_bytes")
        ]
        if not all(type(column) is list for column in columns):
            raise TypeError("stream chunk batch columns must be lists")
        else:
            pass
        if len({len(column) for column in columns}) != 1:
            raise ValueError("stream chunk batch columns must have one length")
        else:
            pass
        request_ids, chunk_ids, dtypes, shapes, tensor_bytes = columns
        for request_id in request_ids:
            require_str(request_id, "request_id")
        for chunk_id in chunk_ids:
            require_non_negative_int(chunk_id, "chunk_id")
        session_sampling_identities = d.get("session_sampling_identities")
        if session_sampling_identities is not None:
            if type(session_sampling_identities) is not list:
                raise TypeError(
                    "stream chunk batch session sampling identities must be a list"
                )
            elif len(session_sampling_identities) != len(request_ids):
                raise ValueError(
                    "stream chunk batch session sampling identities must match chunks"
                )
            else:
                session_sampling_identities = [
                    None
                    if identity is None
                    else require_session_sampling_identity(identity)
                    for identity in session_sampling_identities
                ]
        else:
            pass
        pickled_metadata = d.get("pickled_metadata")
        if type(pickled_metadata) is not bytes:
            raise TypeError("stream chunk batch pickled_metadata must be bytes")
        else:
            pass
        replica_bindings = d.get("replica_bindings")
        if type(replica_bindings) is not dict:
            raise TypeError("stream chunk batch replica_bindings must be dict")
        else:
            pass
        return cls(
            from_stage=require_str(d.get("from_stage"), "from_stage"),
            to_stage=require_str(d.get("to_stage"), "to_stage"),
            request_ids=request_ids,
            chunk_ids=chunk_ids,
            dtypes=dtypes,
            shapes=shapes,
            tensor_bytes=tensor_bytes,
            pickled_metadata=pickled_metadata,
            replica_bindings=replica_bindings,
            session_sampling_identities=session_sampling_identities,
        )


class DataAckMessage(msgspec.Struct):
    """Receiver completion for one data-plane object."""

    request_id: str
    from_stage: str
    to_stage: str
    object_id: str
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, str | bool]:
        require_str(self.request_id, "request_id")
        require_str(self.from_stage, "from_stage")
        require_str(self.to_stage, "to_stage")
        require_str(self.object_id, "object_id")
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        else:
            pass
        if self.success:
            if self.error is not None:
                raise ValueError("successful data ack must not carry error")
            else:
                pass
        else:
            require_str(self.error, "error")
        d: dict[str, str | bool] = {
            "type": "data_ack",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "object_id": self.object_id,
            "success": self.success,
        }
        if self.error is not None:
            d["error"] = self.error
        else:
            pass
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "DataAckMessage":
        success = d.get("success")
        if not isinstance(success, bool):
            raise TypeError("data_ack success must be bool")
        else:
            pass
        error = d.get("error")
        if success:
            if error is not None:
                raise ValueError("successful data_ack must not carry error")
            else:
                pass
        else:
            error = require_str(error, "error")
        return cls(
            request_id=require_str(d.get("request_id"), "request_id"),
            from_stage=require_str(d.get("from_stage"), "from_stage"),
            to_stage=require_str(d.get("to_stage"), "to_stage"),
            object_id=require_str(d.get("object_id"), "object_id"),
            success=success,
            error=error,
        )


@dataclass
class AbortMessage:
    """Broadcast abort signal to all stages."""

    request_id: str

    def to_dict(self) -> dict[str, str]:
        return {"type": "abort", "request_id": self.request_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "AbortMessage":
        return cls(request_id=d["request_id"])


@dataclass
class CompleteMessage:
    """Notify coordinator that a request completed (or failed)."""

    request_id: str
    from_stage: str
    success: bool
    result: object = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "complete",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "CompleteMessage":
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
    chunk: object
    stage_id: int | None = None
    stage_name: str | None = None
    modality: str | None = None
    chunk_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
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
        else:
            pass
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "StreamMessage":
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
    data: object
    replica_bindings: dict[str, int] | None = None

    def to_dict(self) -> dict[str, object]:
        data = self.data
        if isinstance(self.data, StagePayload):
            data = self.data.to_dict()
        else:
            pass
        d = {"type": "submit", "request_id": self.request_id, "data": data}
        if self.replica_bindings:
            d["replica_bindings"] = dict(self.replica_bindings)
        else:
            pass
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SubmitMessage":
        data = d["data"]
        if isinstance(data, dict) and data.get("_type") == "StagePayload":
            data = StagePayload.from_dict(data)
        else:
            pass
        return cls(
            request_id=d["request_id"],
            data=data,
            replica_bindings=d.get("replica_bindings"),
        )


@dataclass
class ShutdownMessage:
    """Signal graceful shutdown to a stage."""

    def to_dict(self) -> dict[str, str]:
        return {"type": "shutdown"}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ShutdownMessage":
        return cls()


@dataclass
class FreezeGCMessage:
    """Freeze the stage process's heap once the server is warmed up."""

    def to_dict(self) -> dict[str, str]:
        return {"type": "freeze_gc"}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "FreezeGCMessage":
        return cls()


@dataclass
class ProfilerStartMessage:
    """Profiler start for a stage."""

    run_id: str
    trace_path_template: str  # e.g. "/tmp/profiles/{run_id}/{stage}/trace"
    event_dir: str | None = None  # Per-stage JSONL event sink dir for request profiling
    enable_torch: bool = True  # When False, only request-level events are captured
    event_buffer_capacity: int | None = None
    mode: Literal["performance", "runtime"] = "performance"
    sampling_unit: Literal["request", "session"] | None = None
    sample_rate: float | None = None
    max_event_file_bytes: int | None = None

    def to_dict(self) -> dict[str, str | bool | int | float | None]:
        message: dict[str, str | bool | int | float | None] = {
            "type": "profiler_start",
            "run_id": self.run_id,
            "trace_path_template": self.trace_path_template,
            "event_dir": self.event_dir,
            "enable_torch": self.enable_torch,
        }
        if self.event_buffer_capacity is not None:
            message["event_buffer_capacity"] = self.event_buffer_capacity
        else:
            pass
        if self.mode == "runtime":
            message["mode"] = self.mode
            message["sampling_unit"] = self.sampling_unit
            message["sample_rate"] = self.sample_rate
            message["max_event_file_bytes"] = self.max_event_file_bytes
        else:
            pass
        return message

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ProfilerStartMessage":
        event_buffer_capacity = d.get("event_buffer_capacity")
        if event_buffer_capacity is not None and (
            type(event_buffer_capacity) is not int or event_buffer_capacity <= 0
        ):
            raise TypeError("event_buffer_capacity must be a positive integer")
        else:
            pass
        mode = d.get("mode", "performance")
        if mode not in ("performance", "runtime"):
            raise TypeError("mode must be 'performance' or 'runtime'")
        else:
            pass
        sampling_unit = d.get("sampling_unit")
        if sampling_unit is not None and sampling_unit not in ("request", "session"):
            raise TypeError("sampling_unit must be 'request' or 'session'")
        else:
            pass
        raw_sample_rate = d.get("sample_rate")
        if raw_sample_rate is not None and (
            type(raw_sample_rate) not in (int, float)
            or not math.isfinite(raw_sample_rate)
            or not 0 < raw_sample_rate <= 1
        ):
            raise TypeError(
                "sample_rate must be finite and satisfy 0 < sample_rate <= 1"
            )
        else:
            pass
        max_event_file_bytes = d.get("max_event_file_bytes")
        if max_event_file_bytes is not None and (
            type(max_event_file_bytes) is not int or max_event_file_bytes <= 0
        ):
            raise TypeError("max_event_file_bytes must be a positive integer")
        else:
            pass
        if mode == "runtime":
            if sampling_unit is None or raw_sample_rate is None:
                raise TypeError("runtime mode requires sampling_unit and sample_rate")
            elif bool(d.get("enable_torch", True)):
                raise TypeError("runtime mode cannot enable torch profiling")
            else:
                pass
        elif (
            sampling_unit is not None
            or raw_sample_rate is not None
            or max_event_file_bytes is not None
        ):
            raise TypeError("runtime sampling and file limits require runtime mode")
        else:
            pass
        return cls(
            run_id=d["run_id"],
            trace_path_template=d["trace_path_template"],
            event_dir=d.get("event_dir"),
            enable_torch=bool(d.get("enable_torch", True)),
            event_buffer_capacity=event_buffer_capacity,
            mode=mode,
            sampling_unit=sampling_unit,
            sample_rate=(
                float(raw_sample_rate) if raw_sample_rate is not None else None
            ),
            max_event_file_bytes=max_event_file_bytes,
        )


@dataclass
class ProfilerStopMessage:
    """Profiler stop. ``run_id=None`` is a wildcard (stop active session)."""

    run_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"type": "profiler_stop", "run_id": self.run_id}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "ProfilerStopMessage":
        return cls(run_id=d.get("run_id"))


@dataclass
class AdminMessage:
    """Send an administrative operation to a stage."""

    operation: AdminOperation

    def to_dict(self) -> dict[str, str | dict[str, object]]:
        return {"type": "admin", "operation": self.operation.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "AdminMessage":
        return cls(operation=AdminOperation.from_dict(d["operation"]))


@dataclass
class AdminResultMessage:
    """Return an administrative result to the coordinator."""

    result: AdminResult

    def to_dict(self) -> dict[str, str | SerializedAdminResult]:
        return {"type": "admin_result", "result": self.result.to_dict()}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "AdminResultMessage":
        return cls(result=AdminResult.from_dict(d["result"]))


def parse_message(
    d: Mapping[str, object],
) -> (
    AdminMessage
    | AdminResultMessage
    | DataAckMessage
    | DataReadyMessage
    | StreamChunkBatchMessage
    | KVTransferPrepareMessage
    | KVTransferReadyMessage
    | AbortMessage
    | CompleteMessage
    | StreamMessage
    | SubmitMessage
    | ShutdownMessage
    | FreezeGCMessage
    | ProfilerStartMessage
    | ProfilerStopMessage
):
    """Parse a dict into the appropriate message type."""
    msg_type = d.get("type")
    if msg_type == "data_ready":
        return DataReadyMessage.from_dict(d)
    elif msg_type == "stream_chunk_batch":
        return StreamChunkBatchMessage.from_dict(d)
    elif msg_type == "data_ack":
        return DataAckMessage.from_dict(d)
    elif msg_type == "kv_transfer_prepare":
        return KVTransferPrepareMessage.from_dict(d)
    elif msg_type == "kv_transfer_ready":
        return KVTransferReadyMessage.from_dict(d)
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
    elif msg_type == "freeze_gc":
        return FreezeGCMessage.from_dict(d)
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


def require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise TypeError(f"{name} must be a non-empty str")
    else:
        pass
    return value


def require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool")
    else:
        pass
    return value


def require_non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{name} must be a non-negative int")
    else:
        pass
    return value


def require_session_sampling_identity(value: object) -> SessionIdentityDict:
    if not isinstance(value, Mapping):
        raise TypeError("session_sampling_identity must be a mapping")
    else:
        pass
    return {
        "id": require_str(value.get("id"), "session_sampling_identity.id"),
        "open_index": require_non_negative_int(
            value.get("open_index"), "session_sampling_identity.open_index"
        ),
    }
