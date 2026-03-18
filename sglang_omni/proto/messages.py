# SPDX-License-Identifier: Apache-2.0
"""Control plane messages."""

from dataclasses import dataclass
from typing import Any

from sglang_omni.proto.request import StagePayload


@dataclass
class DataReadyMessage:
    """Notify next stage that data is ready.

    Supports different metadata formats:
    - Simple dict (for current NixlRelay with transfer_info)
    - SHMMetadata (for backward compatibility)
    - RdmaMetadata (for other relay types)
    """

    request_id: str
    from_stage: str
    to_stage: str
    shm_metadata: Any  # Can be dict, SHMMetadata, or RdmaMetadata
    chunk_id: int | None = None
    is_done: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # Handle different metadata types
        if isinstance(self.shm_metadata, dict):
            # Simple dict (current NixlRelay format)
            metadata_dict = self.shm_metadata.copy()
            metadata_dict["_type"] = "dict"  # Mark as simple dict
        elif hasattr(self.shm_metadata, "to_dict"):
            # SHMMetadata
            metadata_dict = self.shm_metadata.to_dict()
        elif hasattr(self.shm_metadata, "model_dump"):
            # RdmaMetadata (Pydantic BaseModel)
            metadata_dict = self.shm_metadata.model_dump()
            metadata_dict["_type"] = "RdmaMetadata"  # Mark as RdmaMetadata
        else:
            # Fallback: try to convert to dict
            metadata_dict = (
                dict(self.shm_metadata)
                if hasattr(self.shm_metadata, "__dict__")
                else {}
            )

        d = {
            "type": "data_ready",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "shm_metadata": metadata_dict,
        }
        if self.chunk_id is not None:
            d["chunk_id"] = self.chunk_id
        if self.is_done:
            d["is_done"] = True
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataReadyMessage":
        metadata_dict = d["shm_metadata"]

        # Determine metadata type based on _type field first
        metadata_type = metadata_dict.get("_type", "")

        if metadata_type == "dict" or "transfer_info" in metadata_dict:
            # Simple dict format (current NixlRelay design)
            # Remove _type marker if present
            metadata = {k: v for k, v in metadata_dict.items() if k != "_type"}
        elif metadata_type == "RdmaMetadata":
            # Try to import RdmaMetadata if available
            try:
                from sglang_omni.relay.operations.nixl import RdmaMetadata

                clean_dict = {
                    k: v
                    for k, v in metadata_dict.items()
                    if k not in ["_type", "shm_segments"]
                }
                metadata = RdmaMetadata(**clean_dict)
            except (ImportError, Exception):
                # Fallback to dict if RdmaMetadata not available
                metadata = {k: v for k, v in metadata_dict.items() if k != "_type"}
        elif metadata_type == "SHMMetadata" or "shm_segments" in metadata_dict:
            # Try to import SHMMetadata if available
            try:
                from sglang_omni.relay.nixl import SHMMetadata

                metadata = SHMMetadata.from_dict(metadata_dict)
            except (ImportError, Exception):
                # Fallback to dict if SHMMetadata not available
                metadata = {k: v for k, v in metadata_dict.items() if k != "_type"}
        elif "descriptors" in metadata_dict:
            # Has descriptors but no _type - try RdmaMetadata first, fallback to dict
            try:
                from sglang_omni.relay.operations.nixl import RdmaMetadata

                clean_dict = {
                    k: v
                    for k, v in metadata_dict.items()
                    if k not in ["_type", "shm_segments"]
                }
                metadata = RdmaMetadata(**clean_dict)
            except (ImportError, Exception):
                # Fallback to dict
                metadata = {k: v for k, v in metadata_dict.items() if k != "_type"}
        else:
            # Default: use as dict (for current NixlRelay)
            metadata = {k: v for k, v in metadata_dict.items() if k != "_type"}

        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            to_stage=d["to_stage"],
            shm_metadata=metadata,
            chunk_id=d.get("chunk_id"),
            is_done=d.get("is_done", False),
            error=d.get("error"),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "stream",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "chunk": self.chunk,
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "modality": self.modality,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StreamMessage":
        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            chunk=d.get("chunk"),
            stage_id=d.get("stage_id"),
            stage_name=d.get("stage_name"),
            modality=d.get("modality"),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "profiler_start",
            "run_id": self.run_id,
            "trace_path_template": self.trace_path_template,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProfilerStartMessage":
        return cls(
            run_id=d["run_id"],
            trace_path_template=d["trace_path_template"],
        )


@dataclass
class ProfilerStopMessage:
    """Profiler stop for an entry."""

    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"type": "profiler_stop", "run_id": self.run_id}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProfilerStopMessage":
        return cls(run_id=d["run_id"])


def parse_message(
    d: dict[str, Any],
) -> (
    DataReadyMessage
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
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
