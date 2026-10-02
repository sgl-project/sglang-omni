# SPDX-License-Identifier: Apache-2.0
"""SGLang per-request data — bridges StagePayload and SGLang Req."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sglang.srt.managers.schedule_batch import Req

from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue
from sglang_omni.scheduling.types import ARRequestData

if TYPE_CHECKING:
    import torch

else:
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
