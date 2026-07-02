# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for per-request state carried between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from sglang_omni.proto import StagePayload

StateT = TypeVar("StateT", bound="PipelineStateBase")

__all__ = [
    "PipelineStateBase",
    "build_usage",
    "load_state",
    "store_state",
]


@dataclass
class PipelineStateBase:
    """Shared usage/serialization mechanics; tensor strategy stays subclass-owned."""

    sample_rate: int = 24000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # Note(Chenchen Hong): subclasses must override; the stub turns a forgotten
    # override into a clear contract error rather than an AttributeError in store_state.
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} must implement to_dict()")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineStateBase":
        raise NotImplementedError(f"{cls.__name__} must implement from_dict()")

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
