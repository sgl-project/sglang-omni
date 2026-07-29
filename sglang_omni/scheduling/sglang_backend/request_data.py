# SPDX-License-Identifier: Apache-2.0
"""SGLang per-request data — bridges StagePayload and SGLang Req."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sglang_omni.scheduling.types import ARRequestData

if TYPE_CHECKING:
    import torch


@dataclass
class SGLangARRequestData(ARRequestData):
    """Per-request state for SGLang-backed AR stages."""

    req: Any = None
    # Optional upstream TokenizedGenerateReqInput for an append-only SGLang
    # streaming-session turn. OmniScheduler materializes the final Req on its
    # scheduler thread so Session/StreamingSession state is never mutated by a
    # parallel request-builder worker.
    tokenized_session_req: Any = None
    session_tokenizer: Any = None
    session_req_setup: Any = None
    # Admission is transactional: Session.create_req marks a streaming
    # session inflight before request-limit/KV checks run. These fields let the
    # scheduler roll that mark and model-specific setup back on rejection.
    session_req_rollback: Any = None
    session_req_session: Any = None
    session_req_owns_inflight: bool = False
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    input_embeds_are_projected: bool = False
    prefill_input_embeds: "torch.Tensor | None" = None
    decode_input_embeds: list["torch.Tensor"] = field(default_factory=list)
    stage_payload: Any = None
    talker_model_inputs: dict[str, Any] = field(default_factory=dict)
    pending_feedback_queue: Any = field(default_factory=collections.deque)
    pending_text_queue: Any = field(default_factory=collections.deque)
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
