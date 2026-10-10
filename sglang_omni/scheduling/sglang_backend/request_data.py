# SPDX-License-Identifier: Apache-2.0
"""SGLang per-request data — bridges StagePayload and SGLang Req."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Protocol

import torch
from sglang.srt.managers.schedule_batch import Req

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue
from sglang_omni.scheduling.types import ARRequestData


@dataclass(frozen=True, kw_only=True)
class EmbeddingSpan:
    """Model-space rows that replace the token embeddings on [start, end)."""

    start: int
    end: int
    input_embeds: torch.Tensor

    def __post_init__(self) -> None:
        # note (Junnan Li): Negative starts cover retained tokens re-fed before this unit.
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


def validate_prompt_token_ids(input_ids: torch.Tensor, vocab_size: int) -> None:
    """Reject prompt token ids outside [0, vocab_size) before the embedding lookup.

    Call it before multimodal pad remapping, which writes ids at or above vocab_size.
    """
    flat_input_ids = input_ids.reshape(-1)
    out_of_vocabulary = (flat_input_ids < 0) | (flat_input_ids >= vocab_size)
    if bool(out_of_vocabulary.any()):
        position = int(out_of_vocabulary.nonzero()[0])
        raise ValueError(
            "prompt contains out-of-vocabulary token id "
            f"{int(flat_input_ids[position])} at position {position}. "
            f"Valid token ids are in [0, {vocab_size})."
        )
    else:
        pass


@dataclass
class SGLangARRequestData(ARRequestData):
    """Per-request state for SGLang-backed AR stages."""

    req: Req | None = None
    # note (Junnan Li): The bridge binds unit-relative spans to retained session history.
    unit_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    session_embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    input_embeds_are_projected: bool = False
    stage_payload: StagePayload | None = None
    talker_model_inputs: dict[str, object] = field(default_factory=dict)
    pending_feedback_queue: collections.deque[torch.Tensor] = field(
        default_factory=collections.deque
    )
    pending_text_queue: (
        collections.deque[int]
        | collections.deque[torch.Tensor]
        | list[torch.Tensor]
        | PendingTextTensorQueue
        | None
    ) = field(default_factory=collections.deque)
    pending_codec_rows: list["torch.Tensor"] = field(default_factory=list)
    codec_first_flush_done: bool = False
    codec_frames_seen: int = 0
    tts_pad_embed: torch.Tensor | None = None
    tts_eos_embed: torch.Tensor | None = None
    thinker_chunks_done: bool = True


@dataclass
class SGLangDLLMRequestData:
    """Per-request state for SGLang-backed dLLM stages."""

    output_ids: list[int] = field(default_factory=list)
    req: Req | None = None
    stage_payload: StagePayload | None = None
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
