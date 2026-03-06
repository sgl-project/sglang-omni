# SPDX-License-Identifier: Apache-2.0
"""FishAudio tokenizer adapter for DualAR models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """A voice-cloning reference for FishAudio TTS."""

    audio_bytes: bytes
    text: str
    vq_codes: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# FishAudio / tiktoken adapter
# ---------------------------------------------------------------------------


class FishTokenizerAdapter:
    """Wraps ``fish_speech.tokenizer.FishTokenizer``.

    Also implements ``PromptBuilder`` for the interleaved
    ``ContentSequence`` format used by DualARTransformer.
    """

    def __init__(self, fish_tokenizer: Any) -> None:
        self._tok = fish_tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size + self._tok.num_special_tokens

    @property
    def eos_token_ids(self) -> list[int]:
        return [self._tok.get_token_id("<|im_end|>")]

    @property
    def semantic_begin_id(self) -> int:
        return self._tok.semantic_begin_id

    @property
    def semantic_end_id(self) -> int:
        return self._tok.semantic_end_id

    @property
    def semantic_id_to_token_id(self) -> dict[int, int]:
        return self._tok.semantic_id_to_token_id

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)

    # -- PromptBuilder -------------------------------------------------------

    def build_prompt(
        self,
        text: str,
        references: list[Reference] | None = None,
        *,
        num_codebooks: int = 4,
        speaker: int | str = 0,
        modality: str = "interleave",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Build a DualAR prompt from text and optional voice references.

        Returns:
            ``(values, audio_masks, audio_parts)`` where
            ``values`` has shape ``[num_codebooks+1, seq_len]``.
        """
        from fish_speech.content_sequence import ContentSequence, TextPart, VQPart

        seq = ContentSequence(modality=modality)

        if references:
            for ref in references:
                parts = [TextPart(text=ref.text)]
                if ref.vq_codes is not None:
                    parts.append(VQPart(codes=ref.vq_codes))
                seq.append(parts, add_end=True, speaker=speaker)

        seq.append([TextPart(text=text)], add_end=False, speaker=speaker)

        return seq.encode_for_inference(self._tok, num_codebooks=num_codebooks)
