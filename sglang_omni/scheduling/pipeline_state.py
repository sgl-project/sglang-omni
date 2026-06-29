# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for per-request state carried between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.typed_tensor import decode_typed_tensor, encode_typed_tensor

StateT = TypeVar("StateT", bound="PipelineStateBase")

# Re-exported so subclasses can reach the exact-round-trip escape hatch from the
# same module as the base they extend. See ``scheduling/typed_tensor.py``.
__all__ = [
    "PipelineStateBase",
    "build_usage",
    "decode_typed_tensor",
    "encode_typed_tensor",
    "load_state",
    "store_state",
]


@dataclass
class PipelineStateBase:
    """Common mechanics for model-specific pipeline states.

    Tensor handling is intentionally *not* a base policy — a subclass keeps its
    own strategy (``.tolist()`` round-trip, keep a CPU tensor via
    :meth:`serialize_value`, or the exact ``encode_typed_tensor`` /
    ``decode_typed_tensor`` bytes wrapper). The base only owns the structural
    dedup: usage fields, the opt-in ``schema_version`` guard, and the
    ``load_state`` / ``store_state`` / ``build_usage`` helpers.
    """

    sample_rate: int = 24000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0
    # Opt-in fail-fast guard. ``None`` = no guard, which preserves today's
    # behavior; a subclass sets a class-level value only when it wants the
    # check, and adding it is a behavior change, not free.
    schema_version: int | None = None

    @staticmethod
    def serialize_value(value: Any) -> Any:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu()
        return value

    def append_usage_fields(self, data: dict[str, Any]) -> None:
        if self.prompt_tokens:
            data["prompt_tokens"] = int(self.prompt_tokens)
        if self.completion_tokens:
            data["completion_tokens"] = int(self.completion_tokens)
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)

    def append_schema_version(self, data: dict[str, Any]) -> None:
        """Write the schema tag only when a subclass opts in (non-None)."""
        if self.schema_version is not None:
            data["schema_version"] = int(self.schema_version)

    @staticmethod
    def check_schema_version(data: dict[str, Any], expected: int) -> None:
        """Fail-fast guard for subclasses that opt into versioning.

        Raises ``ValueError`` if the payload carries a different
        ``schema_version``. A payload with no tag is accepted (forward path
        from before the guard existed).
        """
        found = data.get("schema_version")
        if found is not None and int(found) != int(expected):
            raise ValueError(
                "pipeline-state schema_version mismatch: "
                f"payload={found}, expected={expected}"
            )

    def usage_dict(self) -> dict[str, Any] | None:
        return build_usage(self)


def load_state(payload: StagePayload, state_cls: type[StateT]) -> StateT:
    return state_cls.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineStateBase) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def build_usage(state: PipelineStateBase) -> dict[str, Any] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    return usage
