# SPDX-License-Identifier: Apache-2.0
"""Extensible tokenizer adapter for sglang-omni engines.

Provides a uniform interface over HuggingFace tokenizers, FishAudio's
tiktoken-based FishTokenizer, and any future tokenizer backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TokenizerAdapter(Protocol):
    """Minimal contract between the engine and any tokenizer."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_ids(self) -> list[int]:
        """Token IDs that signal generation should stop."""
        ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@runtime_checkable
class PromptBuilder(Protocol):
    """Optional extension for non-standard prompt formats.

    Models with interleaved multimodal prompts (e.g., DualAR TTS with
    reference audio VQ codes) implement this instead of relying on
    ``apply_chat_template``.
    """

    def build_prompt(
        self,
        text: str,
        references: list[Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Build model-ready input tensor(s) from user request.

        Returns shape varies by model:
        - Standard LLM:  ``[seq_len]``
        - DualAR:        ``[num_codebooks+1, seq_len]``
        """
        ...


# ---------------------------------------------------------------------------
# HuggingFace adapter
# ---------------------------------------------------------------------------


class HFTokenizerAdapter:
    """Wraps any HuggingFace ``PreTrainedTokenizer``."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    @property
    def eos_token_ids(self) -> list[int]:
        eos = getattr(self._tok, "eos_token_id", None)
        if eos is None:
            return [2]
        return [eos] if isinstance(eos, int) else list(eos)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids)


# ---------------------------------------------------------------------------
# FishAudio / tiktoken adapter
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """A voice-cloning reference for FishAudio TTS."""

    audio_bytes: bytes
    text: str
    vq_codes: torch.Tensor | None = None


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


# ---------------------------------------------------------------------------
# Auto-detection helper
# ---------------------------------------------------------------------------


def wrap_tokenizer(tokenizer: Any) -> TokenizerAdapter:
    """Auto-wrap a tokenizer into the appropriate adapter.

    Accepts:
    - ``None`` → returns a no-op stub
    - Already a ``TokenizerAdapter`` → passthrough
    - Has ``.tkt_model`` attr (FishTokenizer) → ``FishTokenizerAdapter``
    - Otherwise → ``HFTokenizerAdapter``
    """
    if tokenizer is None:
        return _StubTokenizer()
    if isinstance(tokenizer, TokenizerAdapter):
        return tokenizer
    if hasattr(tokenizer, "tkt_model"):
        return FishTokenizerAdapter(tokenizer)
    return HFTokenizerAdapter(tokenizer)


class _StubTokenizer:
    """Fallback when no tokenizer is provided."""

    @property
    def vocab_size(self) -> int:
        return 0

    @property
    def eos_token_ids(self) -> list[int]:
        return [2]

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError("No tokenizer configured")

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError("No tokenizer configured")
