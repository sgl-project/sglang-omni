# SPDX-License-Identifier: Apache-2.0
"""Session-owned state for MiniCPM-o 4.5 native duplex inference.

The main language-model KV cache is intentionally absent from these data
classes.  It is owned by SGLang's ``StreamingSession`` cache wrapper.  This
module only stores state that has no representation in a decoder-only paged KV
pool: streaming perception caches, embedding overrides, duplex control state,
and the TTS/token2wav side state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(frozen=True)
class EmbeddingSpan:
    """An absolute session-token interval replaced by projected embeddings."""

    start: int
    end: int
    embedding: torch.Tensor
    modality: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid embedding span [{self.start}, {self.end})")
        if self.embedding.ndim != 2:
            raise ValueError("embedding span tensor must have shape [tokens, hidden]")
        if self.end - self.start != int(self.embedding.shape[0]):
            raise ValueError(
                "embedding span length does not match its tensor: "
                f"{self.end - self.start} != {int(self.embedding.shape[0])}"
            )


@dataclass(frozen=True)
class MiniCPMOSpecialTokens:
    unit_start: int
    unit_end: int
    image_start: int
    image_end: int
    slice_start: int
    slice_end: int
    listen: int
    speak: int
    tts_bos: int
    tts_eos: int
    tts_pad: int
    chunk_eos: int
    chunk_tts_eos: int
    turn_eos: int
    media_placeholder: int
    bad_token_ids: tuple[int, ...] = ()

    @property
    def chunk_terminators(self) -> frozenset[int]:
        return frozenset((self.listen, self.chunk_eos, self.chunk_tts_eos))

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "MiniCPMOSpecialTokens":
        def required(token: str) -> int:
            value = tokenizer.convert_tokens_to_ids(token)
            if value is None or int(value) < 0 or value == tokenizer.unk_token_id:
                raise RuntimeError(
                    f"MiniCPM-o tokenizer is missing required token {token!r}"
                )
            return int(value)

        placeholder = getattr(tokenizer, "unk_token_id", None)
        if placeholder is None or int(placeholder) < 0:
            placeholder = getattr(tokenizer, "pad_token_id", None)
        if placeholder is None or int(placeholder) < 0:
            raise RuntimeError(
                "MiniCPM-o tokenizer needs unk_token_id or pad_token_id for "
                "embedding-only media positions"
            )
        raw_bad = getattr(tokenizer, "bad_token_ids", ()) or ()
        return cls(
            unit_start=required("<unit>"),
            unit_end=required("</unit>"),
            image_start=required("<image>"),
            image_end=required("</image>"),
            slice_start=required("<slice>"),
            slice_end=required("</slice>"),
            listen=required("<|listen|>"),
            speak=required("<|speak|>"),
            tts_bos=required("<|tts_bos|>"),
            tts_eos=required("<|tts_eos|>"),
            tts_pad=required("<|tts_pad|>"),
            chunk_eos=required("<|chunk_eos|>"),
            chunk_tts_eos=required("<|chunk_tts_eos|>"),
            turn_eos=required("<|turn_eos|>"),
            media_placeholder=int(placeholder),
            bad_token_ids=tuple(int(token_id) for token_id in raw_bad),
        )


@dataclass
class MiniCPMOSessionState:
    request_id: str
    session_id: str
    generation: int
    response_epoch: int
    next_input_seq: int
    system_prompt: str
    last_activity: float = field(default_factory=time.monotonic)
    next_output_seq: int = 1
    playback_audio_end_ms: float = 0.0
    emitted_audio_end_ms: float = 0.0
    force_listen_next: bool = False
    closing: bool = False
    close_reason: str | None = None
    aborted: bool = False
    inflight_rid: str | None = None
    inflight_input_seq: int | None = None
    inflight_response_epoch: int | None = None
    pending_failure: BaseException | None = None
    failure_terminal_emitted: bool = False

    # Model semantic state.  The LLM KV itself belongs to StreamingSession.
    current_turn_ended: bool = True
    generated_unit_count: int = 0
    generated_text_ids: list[int] = field(default_factory=list)
    prefix_input_ids: list[int] = field(default_factory=list)
    prefix_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    prefix_pending: bool = True
    embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    unit_journal: list[dict[str, Any]] = field(default_factory=list)
    temp_paths: list[str] = field(default_factory=list)

    # Perception and synthesis objects are deliberately opaque here so the
    # state contract does not depend on optional model packages at import time.
    perception: Any = None
    tts: Any = None


@dataclass
class MiniCPMOUnitRequestData(SGLangARRequestData):
    """One finite SGLang request representing a duplex unit."""

    session_state: MiniCPMOSessionState | None = None
    outer_request_id: str = ""
    input_seq: int = 0
    response_epoch: int = 0
    local_input_ids: list[int] = field(default_factory=list)
    local_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    absolute_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    generated_unit_ids: list[int] = field(default_factory=list)
    tts_pairs: list[tuple[int, torch.Tensor, bool]] = field(default_factory=list)
    pending_tts_token_id: int | None = None
    forced_listen: bool = False
    input_mode: str = "audio"
    enforce_request_limits: bool = True


__all__ = [
    "EmbeddingSpan",
    "MiniCPMOSessionState",
    "MiniCPMOSpecialTokens",
    "MiniCPMOUnitRequestData",
]
