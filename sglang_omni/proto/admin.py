# SPDX-License-Identifier: Apache-2.0
"""Administrative control-plane payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ADMIN_MODEL_INFO = "model_info"
ADMIN_PAUSE_GENERATION = "pause_generation"
ADMIN_CONTINUE_GENERATION = "continue_generation"
ADMIN_UPDATE_WEIGHTS_FROM_DISK = "update_weights_from_disk"
ADMIN_UPDATE_WEIGHTS_FROM_TENSOR = "update_weights_from_tensor"
ADMIN_UPDATE_WEIGHTS_FROM_DISTRIBUTED = "update_weights_from_distributed"
ADMIN_WEIGHTS_CHECKER = "weights_checker"


@dataclass
class AdminOperation:
    """One coordinator-to-stage administrative operation."""

    op_id: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    target_stages: list[str] | None = None
    timeout_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "action": self.action,
            "payload": self.payload,
            "target_stages": self.target_stages,
            "timeout_s": self.timeout_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdminOperation":
        target_stages = data.get("target_stages")
        return cls(
            op_id=str(data["op_id"]),
            action=str(data["action"]),
            payload=dict(data.get("payload") or {}),
            target_stages=list(target_stages) if target_stages is not None else None,
            timeout_s=data.get("timeout_s"),
        )


@dataclass
class AdminResult:
    """One stage-to-coordinator administrative result."""

    op_id: str
    stage: str
    action: str
    success: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    rank: int | None = None
    role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "stage": self.stage,
            "action": self.action,
            "success": self.success,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "rank": self.rank,
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdminResult":
        return cls(
            op_id=str(data["op_id"]),
            stage=str(data["stage"]),
            action=str(data["action"]),
            success=bool(data["success"]),
            message=str(data.get("message") or ""),
            data=dict(data.get("data") or {}),
            error=data.get("error"),
            rank=data.get("rank"),
            role=data.get("role"),
        )


def is_update_action(action: str) -> bool:
    return action in {
        ADMIN_UPDATE_WEIGHTS_FROM_DISK,
        ADMIN_UPDATE_WEIGHTS_FROM_TENSOR,
        ADMIN_UPDATE_WEIGHTS_FROM_DISTRIBUTED,
    }
