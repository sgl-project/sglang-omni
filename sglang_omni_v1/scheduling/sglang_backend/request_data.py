# SPDX-License-Identifier: Apache-2.0
"""SGLang AR per-request data — bridges StagePayload and SGLang Req."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from sglang_omni_v1.scheduling.types import ARRequestData


@dataclass
class SGLangARRequestData(ARRequestData):
    """Per-request state for SGLang-backed AR stages."""

    req: Any = None
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
    tts_pad_embed: Any = None
    tts_eos_embed: Any = None
    thinker_chunks_done: bool = True
