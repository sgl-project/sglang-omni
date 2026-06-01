# SPDX-License-Identifier: Apache-2.0
"""SGLang AR per-request data — bridges StagePayload and SGLang Req."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SGLangARRequestData:
    """Per-request state for SGLang-backed AR stages."""

    input_ids: Any = None
    attention_mask: Any = None
    model_inputs: dict[str, Any] = field(default_factory=dict)
    output_ids: list[int] = field(default_factory=list)
    extra_model_outputs: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    req: Any = None
    synced: bool = False
    generation_steps: int = 0
    suppress_tokens: list[int] | None = None
    capture_model_output_keys: tuple = ()
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    input_embeds_are_projected: bool = False
    stage_payload: Any = None
    talker_model_inputs: dict[str, Any] = field(default_factory=dict)
    pending_feedback_queue: Any = field(default_factory=collections.deque)
    pending_text_queue: Any = field(default_factory=collections.deque)
    tts_pad_embed: Any = None
    tts_eos_embed: Any = None
    thinker_chunks_done: bool = True
