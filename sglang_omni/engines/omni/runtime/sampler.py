# SPDX-License-Identifier: Apache-2.0
"""Sampler abstractions for sglang-omni engines.

Converts processed logits into sampled token IDs. The ``Sampler`` protocol
is intentionally minimal so model-specific samplers (e.g., DualAR multi-codebook)
can be added without touching core engine code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from .logits_processor import SamplingContext

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class SamplerOutput:
    """Output from a single sampling step."""

    token_ids: torch.Tensor  # [batch] or [batch, num_codebooks+1]
    logprobs: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Sampler(Protocol):
    """Convert processed logits to token IDs."""

    def sample(
        self,
        logits: torch.Tensor,
        context: SamplingContext,
    ) -> SamplerOutput: ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class ArgmaxSampler:
    """Greedy decoding: pick the token with the highest score."""

    def sample(self, logits: torch.Tensor, context: SamplingContext) -> SamplerOutput:
        token_ids = logits.argmax(dim=-1)
        return SamplerOutput(token_ids=token_ids)


class MultinomialSampler:
    """Stochastic sampling from the (already processed) logits distribution."""

    def sample(self, logits: torch.Tensor, context: SamplingContext) -> SamplerOutput:
        probs = torch.softmax(logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return SamplerOutput(token_ids=token_ids)


class MultinomialNoSyncSampler:
    """Gumbel-max trick sampling without CUDA synchronisation.

    Used by FishAudio's inference code for lower latency on GPU.
    """

    def sample(self, logits: torch.Tensor, context: SamplingContext) -> SamplerOutput:
        probs = torch.softmax(logits, dim=-1)
        q = torch.empty_like(probs).exponential_(1)
        token_ids = torch.argmax(probs / q, dim=-1)
        return SamplerOutput(token_ids=token_ids)
