# SPDX-License-Identifier: Apache-2.0
"""Control plane messages."""

from dataclasses import dataclass
from typing import Any

# Import metadata types from relay.nixl
from sglang_omni.relay.nixl import RdmaMetadata, SHMMetadata


@dataclass
class DataReadyMessage:
    """Notify next stage that data is ready.

    Supports both SHMMetadata (for SHMRelay) and RdmaMetadata (for NIXLRelay).
    """

    request_id: str
    from_stage: str
    to_stage: str
    shm_metadata: Any  # Can be SHMMetadata or RdmaMetadata

    def to_dict(self) -> dict[str, Any]:
        # Handle different metadata types
        if hasattr(self.shm_metadata, "to_dict"):
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

        return {
            "type": "data_ready",
            "request_id": self.request_id,
            "from_stage": self.from_stage,
            "to_stage": self.to_stage,
            "shm_metadata": metadata_dict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataReadyMessage":
        metadata_dict = d["shm_metadata"]

        # Determine metadata type based on _type field first
        metadata_type = metadata_dict.get("_type", "")

        if metadata_type == "RdmaMetadata":
            # RdmaMetadata: remove type marker and fields not in RdmaMetadata
            clean_dict = {
                k: v
                for k, v in metadata_dict.items()
                if k not in ["_type", "shm_segments"]
            }
            metadata = RdmaMetadata(**clean_dict)
        elif metadata_type == "SHMMetadata" or "shm_segments" in metadata_dict:
            # SHMMetadata (new format with descriptors and shm_segments, or legacy format)
            metadata = SHMMetadata.from_dict(metadata_dict)
        elif "descriptors" in metadata_dict:
            # Has descriptors but no _type - try RdmaMetadata first, fallback to SHMMetadata
            try:
                # Remove fields not in RdmaMetadata
                clean_dict = {
                    k: v
                    for k, v in metadata_dict.items()
                    if k not in ["_type", "shm_segments"]
                }
                metadata = RdmaMetadata(**clean_dict)
            except Exception:
                # Fallback to SHMMetadata if RdmaMetadata fails
                metadata = SHMMetadata.from_dict(metadata_dict)
        else:
            # Legacy SHMMetadata format (has "name" and "size" fields)
            metadata = SHMMetadata.from_dict(metadata_dict)

        return cls(
            request_id=d["request_id"],
            from_stage=d["from_stage"],
            to_stage=d["to_stage"],
            shm_metadata=metadata,
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
class SubmitMessage:
    """Submit a new request to the entry stage."""

    request_id: str
    data: Any

    def to_dict(self) -> dict[str, Any]:
        return {"type": "submit", "request_id": self.request_id, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubmitMessage":
        return cls(request_id=d["request_id"], data=d["data"])


@dataclass
class ShutdownMessage:
    """Signal graceful shutdown to a stage."""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "shutdown"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShutdownMessage":
        return cls()


def parse_message(
    d: dict[str, Any],
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
