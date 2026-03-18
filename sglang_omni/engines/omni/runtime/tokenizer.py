# SPDX-License-Identifier: Apache-2.0
"""Extensible tokenizer adapter for sglang-omni engines.

Provides a uniform interface over HuggingFace tokenizers and any future
tokenizer backend.  Model-specific adapters (e.g., FishTokenizerAdapter)
live under their respective ``models/`` package.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import torch

logger = logging.getLogger(__name__)


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


def wrap_tokenizer(tokenizer: Any) -> TokenizerAdapter:
    """Auto-wrap a tokenizer into the appropriate adapter.

    Accepts:
    - ``None`` → returns a no-op stub
    - Already a ``TokenizerAdapter`` → passthrough
    - Otherwise → ``HFTokenizerAdapter``
    """
    if tokenizer is None:
        return _StubTokenizer()
    if isinstance(tokenizer, TokenizerAdapter):
        return tokenizer
    # For model-specific tokenizers, use the corresponding model's adapter
    # directly (e.g., S2ProTokenizerAdapter from fishaudio_s2_pro.tokenizer).
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
