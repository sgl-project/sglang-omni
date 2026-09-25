# SPDX-License-Identifier: Apache-2.0
"""SGLang per-request data — bridges StagePayload and SGLang Req."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from sglang_omni.scheduling.types import ARRequestData


@dataclass(frozen=True, kw_only=True)
class EmbeddingSpan:
    """Model-space rows that replace the token embeddings on [start, end)."""

    start: int
    end: int
    input_embeds: torch.Tensor

    def __post_init__(self) -> None:
        # A unit-relative start may be negative: it then covers retained output
        # tokens that the native session re-feeds ahead of this unit's ids.
        if self.end <= self.start:
            raise ValueError(f"invalid embedding span [{self.start}, {self.end})")
        else:
            pass
        if (
            self.input_embeds.ndim != 2
            or self.input_embeds.shape[0] != self.end - self.start
        ):
            raise ValueError(
                f"embedding span [{self.start}, {self.end}) needs "
                f"{self.end - self.start} rows, got {tuple(self.input_embeds.shape)}"
            )
        else:
            pass


def splice_embedding_spans(
    embeddings: torch.Tensor, start: int, spans: list[EmbeddingSpan]
) -> torch.Tensor:
    """Overwrite the rows of the extend window starting at start that spans cover."""
    end = start + embeddings.shape[0]
    for span in spans:
        left, right = max(start, span.start), min(end, span.end)
        if left < right:
            embeddings[left - start : right - start] = span.input_embeds[
                left - span.start : right - span.start
            ].to(embeddings)
        else:
            continue
    return embeddings


class TokenEmbedding(Protocol):
    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        pass


@dataclass
class SGLangARRequestData(ARRequestData):
    """Per-request state for SGLang-backed AR stages."""

    req: Any = None
    # The adapter sets unit spans relative to the unit's own token ids; the AR
    # session bridge binds them to the native sequence and exposes the retained
    # session history plus this unit as session_embedding_spans.
    unit_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    session_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    input_embeds_are_projected: bool = False
    stage_payload: Any = None
    talker_model_inputs: dict[str, Any] = field(default_factory=dict)
    pending_feedback_queue: Any = field(default_factory=collections.deque)
    pending_text_queue: Any = field(default_factory=collections.deque)
    pending_codec_rows: list["torch.Tensor"] = field(default_factory=list)
    codec_first_flush_done: bool = False
    codec_frames_seen: int = 0
    tts_pad_embed: Any = None
    tts_eos_embed: Any = None
    thinker_chunks_done: bool = True


@dataclass
class SGLangDLLMRequestData:
    """Per-request state for SGLang-backed dLLM stages."""

    output_ids: list[int] = field(default_factory=list)
    req: Any = None
    stage_payload: Any = None
    finish_reason: str | None = None


def session_prefill_rows(
    request_data: SGLangARRequestData,
    embed_tokens: TokenEmbedding,
    device: torch.device,
) -> torch.Tensor:
    """Embed the uncached window of a native session request with its spans spliced in."""
    session_request = request_data.req
    start = session_request.extend_range.start
    token_ids = torch.tensor(
        session_request.get_fill_ids()[start : session_request.extend_range.end],
        dtype=torch.long,
        device=device,
    )
    return splice_embedding_spans(
        embed_tokens(token_ids), start, request_data.session_embedding_spans
    )
