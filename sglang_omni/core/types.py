# SPDX-License-Identifier: Apache-2.0
"""Shared types for SGLang-Omni pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from sglang_omni.relay.descriptor import SerializedDescriptor
from sglang_omni.relay.descriptor import Descriptor
# === Enums ===


class RequestState(Enum):
    """State of a request in the pipeline."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# === Stage Info ===


@dataclass
class StageInfo:
    """Information about a registered stage."""

    name: str
    control_endpoint: str  # ZMQ endpoint for receiving control messages


# === SHM Metadata ===


@dataclass
class SHMMetadata:
    """Metadata for shared memory segment(s).
    
    Supports both single segment (legacy) and multiple descriptors (new format).
    The new format is compatible with RdmaMetadata structure.
    """

    name: str = ""  # SHM segment name (system-generated) - legacy single segment
    size: int = 0  # Size in bytes - legacy single segment
    descriptors: list[SerializedDescriptor] = field(default_factory=list)  # List of SerializedDescriptor - new format (compatible with RdmaMetadata)
    shm_segments: list[dict[str, Any]] = field(default_factory=list)  # List of {name, size} for each descriptor - new format

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        if self.descriptors and len(self.descriptors) > 0:
            # New format: multiple descriptors (compatible with RdmaMetadata)
            serialized_descriptors = []
            for desc in self.descriptors:
                serialized_descriptors.append(desc.model_dump())
            
            return {
                "descriptors": serialized_descriptors,
                "shm_segments": self.shm_segments,
                "_type": "SHMMetadata",
            }
        else:
            # Legacy format: single segment
            return {"name": self.name, "size": self.size, "_type": "SHMMetadata"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SHMMetadata":
        """Deserialize from dictionary."""
        if "descriptors" in d and d["descriptors"]:
            descriptors = []
            for desc_dict in d["descriptors"]:
                if isinstance(desc_dict, dict):
                    descriptors.append(SerializedDescriptor(**desc_dict))
                else:
                    descriptors.append(desc_dict)
            
            return cls(
                descriptors=descriptors,
                shm_segments=d.get("shm_segments", []),
            )
        else:
            # Legacy format: single segment
            return cls(name=d["name"], size=d["size"])

    def to_descriptors(self) -> Any:
        """Convert to Descriptor(s), compatible with RdmaMetadata interface.
        
        Returns:
            Descriptor or list[Descriptor]: Descriptor objects for receiving data.
            The size information comes from shm_segments (actual data size in SHM).
        """        
        desc_list = []
        for i, serialized_desc in enumerate(self.descriptors):
            # Use size from shm_segments if available (actual data size)
            if self.shm_segments and i < len(self.shm_segments):
                actual_size = self.shm_segments[i]["size"]
                # Create descriptor with actual size from SHM segment
                desc = Descriptor(data=(
                    serialized_desc.ptr if serialized_desc.ptr != 0 else 1,  # Placeholder if ptr is 0
                    actual_size,  # Use actual size from SHM segment
                    serialized_desc.device,
                    None
                ))
            else:
                # Fallback to serialized descriptor size
                desc = serialized_desc.to_descriptor()
            desc_list.append(desc)
        
        if len(desc_list) == 1:
            return desc_list[0]
        return desc_list


# === Control Plane Messages ===


@dataclass
class DataReadyMessage:
    """Notify next stage that data is ready.
    
    Supports both SHMMetadata (for SHMRelay) and RdmaMetadata (for NIXLRelay).
    """

    request_id: str
    from_stage: str
    to_stage: str
    metadata: Any  # Can be SHMMetadata or RdmaMetadata

    def to_dict(self) -> dict[str, Any]:
        # Handle different metadata types
        if hasattr(self.metadata, "to_dict"):
            # SHMMetadata
            metadata_dict = self.metadata.to_dict()
        elif hasattr(self.metadata, "model_dump"):
            # RdmaMetadata (Pydantic BaseModel)
            metadata_dict = self.metadata.model_dump()
            metadata_dict["_type"] = "RdmaMetadata"  # Mark as RdmaMetadata
        else:
            # Fallback: try to convert to dict
            metadata_dict = dict(self.metadata) if hasattr(self.metadata, "__dict__") else {}
        
        return {
            "type": "data_ready",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "metadata": metadata_dict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataReadyMessage":
        metadata_dict = d["metadata"]
        
        # Determine metadata type based on content
        if "_type" in metadata_dict and metadata_dict["_type"] == "RdmaMetadata":
            # RdmaMetadata
            from sglang_omni.relay.nixl import RdmaMetadata
            # Remove the type marker
            metadata_dict = {k: v for k, v in metadata_dict.items() if k != "_type"}
            metadata = RdmaMetadata(**metadata_dict)
        elif "descriptors" in metadata_dict or "nixl_metadata" in metadata_dict:
            # Looks like RdmaMetadata
            try:
                from sglang_omni.relay.nixl import RdmaMetadata
                metadata = RdmaMetadata(**metadata_dict)
            except Exception:
                # Fallback to SHMMetadata if RdmaMetadata import fails
                metadata = SHMMetadata.from_dict(metadata_dict)
        else:
            # SHMMetadata (has "name" and "size" fields)
            metadata = SHMMetadata.from_dict(metadata_dict)
        
        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            to_stage=d["to_stage"],
            metadata=metadata,
        )


@dataclass
class AbortMessage:
    """Broadcast abort signal to all stages."""

    request_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "abort",
            "request_id": self.request_id,
        }

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
class SubmitMessage:
    """Submit a new request to the entry stage."""

    request_id: str
    data: Any  # Initial input data

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "submit",
            "request_id": self.request_id,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubmitMessage":
        return cls(
            request_id=d["request_id"],
            data=d["data"],
        )


@dataclass
class ShutdownMessage:
    """Signal graceful shutdown to a stage."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "shutdown"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShutdownMessage":
        return cls()


# === Request Tracking ===


@dataclass
class RequestInfo:
    """Tracking info for a request in the coordinator."""

    request_id: str
    state: RequestState = RequestState.PENDING
    current_stage: str | None = None
    result: Any = None
    error: str | None = None


# === Message Parsing Helper ===


def parse_message(
    d: dict[str, Any]
) -> (
    DataReadyMessage | AbortMessage | CompleteMessage | SubmitMessage | ShutdownMessage
):
    """Parse a dict into the appropriate message type."""
    msg_type = d.get("type")
    if msg_type == "data_ready":
        return DataReadyMessage.from_dict(d)
    elif msg_type == "abort":
        return AbortMessage.from_dict(d)
    elif msg_type == "complete":
        return CompleteMessage.from_dict(d)
    elif msg_type == "submit":
        return SubmitMessage.from_dict(d)
    elif msg_type == "shutdown":
        return ShutdownMessage.from_dict(d)
    else:
        raise ValueError(f"Unknown message type: {msg_type}")
