# SPDX-License-Identifier: Apache-2.0
"""Cross-stage state for MiniMax Music 3."""

from __future__ import annotations

from dataclasses import dataclass

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire

from .constants import DEFAULT_MAX_AUDIO_FRAMES


@dataclass
class MiniMaxMusic3State(DeclarativeStateBase):
    """Small serializable request state.

    Hidden tensors are emitted through the internal stream and are deliberately
    not retained in the final terminal payload.
    """

    internal_chunk_stream: bool = wire(True, codec="bool")
    caption: str = wire("", codec="str")
    lyrics: str = wire("", codec="str")
    prompt: str | None = wire(None, codec="raw")
    seed: int = wire(0, codec="int")
    max_audio_frames: int = wire(DEFAULT_MAX_AUDIO_FRAMES, codec="int")
    generated_frames: int = wire(0, emit="truthy", codec="int")
    finish_reason: str | None = wire(None, codec="str")


__all__ = ["MiniMaxMusic3State"]
