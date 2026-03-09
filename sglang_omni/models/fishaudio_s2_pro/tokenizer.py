# SPDX-License-Identifier: Apache-2.0
"""S2-Pro tokenizer adapter wrapping HuggingFace PreTrainedTokenizerFast.

S2-Pro uses Qwen3 chat-format prompts built via the ``Conversation`` class:
- System message: reference text + VQ codes (voice cloning)
- User message: target text to synthesize
- Assistant message: ``<|voice|>`` modality marker (generation starts here)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.fishaudio_s2_pro.fish_speech.tokenizer import (
    IM_END_TOKEN,
    SEMANTIC_TOKEN_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """A voice-cloning reference for S2-Pro TTS."""

    audio_bytes: bytes
    text: str
    vq_codes: torch.Tensor | None = None


class S2ProTokenizerAdapter:
    """
    Builds Qwen3 chat-format prompts using the ``Conversation`` class
    """

    def __init__(self, hf_tokenizer: PreTrainedTokenizerFast) -> None:
        self._tok = hf_tokenizer

    @property
    def eos_token_ids(self) -> list[int]:
        return [self._tok.convert_tokens_to_ids(IM_END_TOKEN)]

    @property
    def semantic_begin_id(self) -> int:
        return self._tok.convert_tokens_to_ids(SEMANTIC_TOKEN_TEMPLATE.format(i=0))

    @property
    def semantic_end_id(self) -> int:
        return self._tok.convert_tokens_to_ids(SEMANTIC_TOKEN_TEMPLATE.format(i=4095))

    def build_prompt(
        self,
        text: str,
        references: list[Reference] | None = None,
        *,
        num_codebooks: int = 10,
        speaker: int | str = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an S2-Pro prompt using Qwen3 chat format."""
        from sglang_omni.models.fishaudio_s2_pro.fish_speech.content_sequence import (
            TextPart,
            VQPart,
        )
        from sglang_omni.models.fishaudio_s2_pro.fish_speech.conversation import (
            Conversation,
            Message,
        )

        conversation = Conversation()

        # System message: reference audio for voice cloning
        if references:
            system_parts: list = []
            all_codes = []

            system_parts.append(
                TextPart(
                    text="convert the provided text to speech reference to the following:\n\nText:\n",
                    cal_loss=False,
                )
            )

            for ref in references:
                ref_text = f"<|speaker:{speaker}|>{ref.text}" if ref.text else ""
                if ref_text:
                    system_parts.append(TextPart(text=ref_text, cal_loss=False))
                if ref.vq_codes is not None:
                    all_codes.append(ref.vq_codes)

            system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))

            if all_codes:
                combined = torch.cat(all_codes, dim=1)
                system_parts.append(VQPart(codes=combined, cal_loss=False))

            conversation.append(
                Message(
                    role="system",
                    parts=system_parts,
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

        # User message: text to synthesize
        text_with_tag = f"<|speaker:{speaker}|>{text}"
        conversation.append(
            Message(
                role="user",
                parts=[TextPart(text=text_with_tag, cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )

        # Assistant message: voice modality marker (generation starts after this)
        conversation.append(
            Message(
                role="assistant",
                parts=[],
                cal_loss=False,
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )

        encoded = conversation.encode(self._tok, add_shift=False)
        vq_parts_list = encoded.vq_parts  # list of [num_codebooks, T_i]

        return {
            "input_ids": encoded.tokens,
            "vq_mask_tokens": encoded.vq_mask_tokens,
            "vq_parts": vq_parts_list,
        }
