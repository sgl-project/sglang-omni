# SPDX-License-Identifier: Apache-2.0
"""Typed identity for a model turn routed through the shared pipeline."""

from dataclasses import asdict, dataclass
from typing import Any, Literal, get_args

# Each phase is also the name of the stage that runs it.
ContinuationPhase = Literal["reasoner", "generation"]


@dataclass(frozen=True)
class ContinuationToken:
    session_id: str
    turn_index: int
    phase: ContinuationPhase
    nonce: str

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not 1 <= len(self.session_id) <= 128:
            raise ValueError("Continuation session_id must contain 1 to 128 characters")
        else:
            pass
        if type(self.turn_index) is not int or self.turn_index < 0:
            raise ValueError("Continuation turn_index must be a nonnegative integer")
        else:
            pass
        if self.phase not in get_args(ContinuationPhase):
            raise ValueError("Invalid continuation phase")
        else:
            pass
        if not isinstance(self.nonce, str) or not 1 <= len(self.nonce) <= 128:
            raise ValueError("Continuation nonce must contain 1 to 128 characters")
        else:
            pass

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ContinuationToken":
        if not isinstance(value, dict):
            raise ValueError("Continuation token must be a mapping")
        else:
            pass
        if set(value) != {"session_id", "turn_index", "phase", "nonce"}:
            raise ValueError("Invalid continuation token fields")
        else:
            pass
        return cls(**value)


@dataclass(frozen=True)
class UMMSegment:
    session_id: str
    segment_index: int
    kind: Literal["text", "image", "video", "audio", "action"]
    data: Any

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not 1 <= len(self.session_id) <= 128:
            raise ValueError("Segment session_id must contain 1 to 128 characters")
        else:
            pass
        if type(self.segment_index) is not int or self.segment_index < 0:
            raise ValueError("Segment index must be a nonnegative integer")
        else:
            pass
        if self.kind not in ("text", "image", "video", "audio", "action"):
            raise ValueError("Invalid UMM segment kind")
        else:
            pass
        if self.kind == "text":
            if not isinstance(self.data, str):
                raise ValueError("Text segments require a string")
            else:
                pass
        elif not isinstance(self.data, dict):
            raise ValueError("Media segments require a structured reference")
        else:
            pass

    def to_dict(self) -> dict[str, Any]:
        return {"type": "segment", **asdict(self)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "UMMSegment":
        if not isinstance(value, dict):
            raise ValueError("UMM segment must be a mapping")
        else:
            pass
        if value.get("type") != "segment":
            raise ValueError("Invalid UMM segment type")
        else:
            pass
        fields = {key: item for key, item in value.items() if key != "type"}
        expected = {"session_id", "segment_index", "kind", "data"}
        if set(fields) != expected:
            raise ValueError("Invalid UMM segment fields")
        else:
            pass
        return cls(**fields)
