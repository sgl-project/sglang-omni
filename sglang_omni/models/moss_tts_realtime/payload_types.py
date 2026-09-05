# SPDX-License-Identifier: Apache-2.0
"""Declarative pipeline payload for MOSS-TTS-Realtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


def _validate_non_negative_token_ids(token_ids: Sequence[Any], name: str) -> None:
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError(f"{name} entries must be integers")
        if token_id < 0:
            raise ValueError(f"{name} entries must be non-negative")


def _validate_matrix_width(value: Any, *, width: int | None, name: str) -> None:
    if value is None:
        return
    ndim = getattr(value, "ndim", None)
    shape = getattr(value, "shape", None)
    if ndim is not None and shape is not None:
        if int(ndim) != 2:
            raise ValueError(f"{name} must be rank 2")
        if width is not None and int(shape[1]) != width:
            raise ValueError(f"{name} must have shape [T, {width}]")
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a rank-2 tensor or sequence")
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise TypeError(f"{name} rows must be sequences")
        if width is not None and len(row) != width:
            raise ValueError(f"{name} must have shape [T, {width}]")


@dataclass
class MossTTSRealtimeState(DeclarativeStateBase):
    """Msgpack-safe state passed through preprocessing, AR, and vocoder stages."""

    sample_rate: int = wire(24000, codec="int_or")
    session_id: str = wire("", codec="str")
    turn_id: str = wire("", codec="str")
    voice: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None
    language: str | None = None
    instructions: str | None = None
    turn_index: int = wire(0, codec="int")
    user_text: str | None = None
    user_audio: Any | None = None
    initial_text: str | None = None
    initial_token_ids: list[int] = wire(default_factory=list, codec="list")
    input_done: bool = wire(False, codec="bool")
    keep_session: bool = wire(True, codec="bool")
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    prompt_rows: Any | None = wire(None, codec="tensor_cpu")
    audio_codes: Any | None = wire(None, codec="tensor_cpu")
    stream_metadata: dict[str, Any] = wire(default_factory=dict, codec="dict")

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer")
        if self.sample_rate < 1:
            raise ValueError("sample_rate must be positive")
        _validate_non_negative_token_ids(self.initial_token_ids, "initial_token_ids")
        if isinstance(self.turn_index, bool) or not isinstance(self.turn_index, int):
            raise TypeError("turn_index must be an integer")
        if self.turn_index < 0:
            raise ValueError("turn_index must be non-negative")
        if self.initial_text is not None and not isinstance(self.initial_text, str):
            raise TypeError("initial_text must be a string")
        if self.initial_text is not None and self.initial_token_ids:
            raise ValueError(
                "initial_text and initial_token_ids are mutually exclusive"
            )
        if not isinstance(self.input_done, bool):
            raise TypeError("input_done must be a boolean")
        if not isinstance(self.keep_session, bool):
            raise TypeError("keep_session must be a boolean")
        if not isinstance(self.generation_kwargs, dict):
            raise TypeError("generation_kwargs must be a dictionary")
        if not isinstance(self.stream_metadata, dict):
            raise TypeError("stream_metadata must be a dictionary")
        _validate_matrix_width(
            self.prompt_rows,
            width=None,
            name="prompt_rows",
        )
        _validate_matrix_width(
            self.audio_codes,
            width=None,
            name="audio_codes",
        )
