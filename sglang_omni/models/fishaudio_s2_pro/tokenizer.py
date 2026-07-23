# SPDX-License-Identifier: Apache-2.0
"""Inference-only S2-Pro prompt encoding.

S2-Pro uses Qwen3 chat-format prompts:
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
    IM_START_TOKEN,
    MODALITY_VOICE_TOKEN,
    SEMANTIC_TOKEN_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """A voice-cloning reference for S2-Pro TTS."""

    audio_bytes: bytes
    text: str
    vq_codes: torch.Tensor | None = None


class _InferencePromptEncoder:
    """Accumulate the tensor fields consumed by the Fish serving path."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self._tokenizer = tokenizer
        self._token_segments: list[torch.Tensor] = []
        self._vq_mask_segments: list[torch.Tensor] = []
        self._vq_parts: list[torch.Tensor] = []

    def append_text(self, text: str) -> None:
        tokens = torch.tensor(self._tokenizer.encode(text), dtype=torch.int)
        self._token_segments.append(tokens)
        self._vq_mask_segments.append(torch.zeros_like(tokens, dtype=torch.bool))

    def append_vq(self, codes: torch.Tensor) -> None:
        codes = codes.clone().to(torch.int)
        tokens = torch.tensor(
            self._tokenizer.convert_tokens_to_ids(
                [SEMANTIC_TOKEN_TEMPLATE.format(i=code) for code in codes[0].int()]
            ),
            dtype=torch.int,
        )
        self._token_segments.append(tokens)
        self._vq_mask_segments.append(torch.ones_like(tokens, dtype=torch.bool))
        self._vq_parts.append(codes)

    def finish(self) -> dict[str, Any]:
        return {
            "input_ids": torch.cat(self._token_segments, dim=0),
            "vq_mask_tokens": torch.cat(self._vq_mask_segments, dim=0),
            "vq_parts": self._vq_parts,
        }


class S2ProTokenizerAdapter:
    """Build the inference prompt fields consumed by S2-Pro serving."""

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
        """Build an S2-Pro inference prompt using Qwen3 chat format."""
        if references:
            for index, ref in enumerate(references):
                codes = ref.vq_codes
                if codes is None:
                    continue
                shape = tuple(codes.shape)
                if codes.ndim != 2 or shape[0] != num_codebooks:
                    raise ValueError(
                        f"Reference {index} VQ codes must have shape "
                        f"({num_codebooks}, T); got {shape}"
                    )

        encoder = _InferencePromptEncoder(self._tok)

        # System message: reference audio for voice cloning
        if references:
            encoder.append_text(f"{IM_START_TOKEN}system\n")
            encoder.append_text(
                "convert the provided text to speech reference to the following:\n\nText:\n"
            )
            all_codes: list[torch.Tensor] = []

            for ref in references:
                ref_text = f"<|speaker:{speaker}|>{ref.text}" if ref.text else ""
                if ref_text:
                    encoder.append_text(ref_text)
                if ref.vq_codes is not None:
                    all_codes.append(ref.vq_codes)

            encoder.append_text("\n\nSpeech:\n")

            if all_codes:
                encoder.append_vq(torch.cat(all_codes, dim=1))

            encoder.append_text(f"{IM_END_TOKEN}\n")

        # User message: text to synthesize
        text_with_tag = f"<|speaker:{speaker}|>{text}"
        encoder.append_text(f"{IM_START_TOKEN}user\n")
        encoder.append_text(text_with_tag)
        encoder.append_text(f"{IM_END_TOKEN}\n")

        # Assistant message: voice modality marker (generation starts after this)
        encoder.append_text(f"{IM_START_TOKEN}assistant\n{MODALITY_VOICE_TOKEN}")
        return encoder.finish()
