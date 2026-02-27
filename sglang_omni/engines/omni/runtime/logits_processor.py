# SPDX-License-Identifier: Apache-2.0
"""Composable logits processors for sglang-omni engines.

Follows the same pattern as HuggingFace ``LogitsProcessorList`` but is
decoupled from HF and works with any model type (LLM, DualAR TTS, DiT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Sampling context
# ---------------------------------------------------------------------------


@dataclass
class SamplingContext:
    """Per-request state available to logits processors and samplers."""

    request_id: str
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    previous_tokens: torch.Tensor | None = None
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LogitsProcessor(Protocol):
    """Single logits transformation step."""

    def __call__(
        self,
        logits: torch.Tensor,
        context: SamplingContext,
    ) -> torch.Tensor:
        """Transform logits. May modify in-place or return a new tensor.

        Args:
            logits: ``[batch, vocab_size]`` raw or partially processed logits.
            context: per-request sampling state.
        """
        ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class LogitsProcessorPipeline:
    """Ordered chain of ``LogitsProcessor`` instances.

    When the chain is empty, ``__call__`` is a no-op (returns logits unchanged).
    """

    def __init__(self, processors: list[LogitsProcessor] | None = None) -> None:
        self._processors: list[LogitsProcessor] = list(processors or [])

    def add(self, processor: LogitsProcessor) -> LogitsProcessorPipeline:
        self._processors.append(processor)
        return self

    def __len__(self) -> int:
        return len(self._processors)

    def __call__(
        self, logits: torch.Tensor, context: SamplingContext
    ) -> torch.Tensor:
        for proc in self._processors:
            logits = proc(logits, context)
        return logits


# ---------------------------------------------------------------------------
# Built-in processors
# ---------------------------------------------------------------------------


class TemperatureProcessor:
    """Scale logits by ``1 / temperature``."""

    def __call__(
        self, logits: torch.Tensor, context: SamplingContext
    ) -> torch.Tensor:
        t = context.temperature
        if t <= 0.0:
            return logits
        return logits / max(t, 1e-5)


class TopPProcessor:
    """Nucleus (top-p) sampling: zero out tokens outside the smallest set
    whose cumulative probability exceeds ``top_p``."""

    def __call__(
        self, logits: torch.Tensor, context: SamplingContext
    ) -> torch.Tensor:
        top_p = context.top_p
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1
        )
        mask = cumulative_probs > top_p
        mask[..., 0] = False  # always keep at least one token
        indices_to_remove = mask.scatter(
            dim=-1, index=sorted_indices, src=mask
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits


class TopKProcessor:
    """Top-k truncation: keep only the ``top_k`` highest-scoring tokens."""

    def __call__(
        self, logits: torch.Tensor, context: SamplingContext
    ) -> torch.Tensor:
        top_k = context.top_k
        if top_k <= 0 or top_k >= logits.shape[-1]:
            return logits

        values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        return logits


class RepetitionPenaltyProcessor:
    """Penalize tokens that already appeared in ``previous_tokens``.

    Follows the HuggingFace convention: scores > 0 are divided by the penalty,
    scores < 0 are multiplied.
    """

    def __call__(
        self, logits: torch.Tensor, context: SamplingContext
    ) -> torch.Tensor:
        penalty = context.repetition_penalty
        if penalty == 1.0 or context.previous_tokens is None:
            return logits

        prev = context.previous_tokens.long()
        if prev.dim() == 1:
            prev = prev.unsqueeze(0).expand(logits.shape[0], -1)

        logits = logits.clone()
        scores = torch.gather(logits, dim=-1, index=prev)
        scores = torch.where(
            scores < 0, scores * penalty, scores / penalty
        )
        logits.scatter_(dim=-1, index=prev, src=scores)
        return logits


class FrequencyPenaltyProcessor:
    """Subtract ``frequency_penalty * count(token)`` from logits."""

    def __call__(
        self, logits: torch.Tensor, context: SamplingContext
    ) -> torch.Tensor:
        penalty = context.frequency_penalty
        if penalty == 0.0 or context.previous_tokens is None:
            return logits

        prev = context.previous_tokens.long()
        counts = torch.zeros_like(logits)
        if prev.dim() == 1:
            prev = prev.unsqueeze(0).expand(logits.shape[0], -1)
        counts.scatter_add_(dim=-1, index=prev, src=torch.ones_like(prev, dtype=logits.dtype))
        logits = logits - penalty * counts
        return logits


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def default_logits_pipeline() -> LogitsProcessorPipeline:
    """Standard pipeline: repetition penalty → temperature → top-p."""
    return LogitsProcessorPipeline([
        RepetitionPenaltyProcessor(),
        TemperatureProcessor(),
        TopPProcessor(),
    ])
