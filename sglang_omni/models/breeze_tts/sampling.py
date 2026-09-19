# SPDX-License-Identifier: Apache-2.0
"""Breeze sampling with per-request seeds and guidance at every codebook.

Rows are sampled through SGLang's ``multinomial_with_seed``, so a token depends
only on its own seed and position and never on its batch neighbours. That keeps
seeded output reproducible at any batch size and leaves no host RNG state in the
decode loop, which is what lets the depth loop run under a CUDA graph.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.sampling.seed import derive_sampling_seed

# One position block per codec frame, so a frame's backbone token and its depth
# codebooks never collide with another frame's.
POSITIONS_PER_FRAME = 64

_MASK32 = 0xFFFFFFFF


def _rotl32(value: torch.Tensor, bits: int) -> torch.Tensor:
    return ((value << bits) | (value >> (32 - bits))) & _MASK32


def _murmur3_mix(state: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    key = _rotl32((key * 0xCC9E2D51) & _MASK32, 15)
    state = _rotl32(state ^ ((key * 0x1B873593) & _MASK32), 13)
    return (state * 5 + 0xE6546B64) & _MASK32


def _sample_seeded_eager(
    logprobs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """CPU-capable twin of SGLang's ``multinomial_with_seed``.

    SGLang hashes through a Triton kernel, so its sampler only runs on CUDA.
    This mirrors the same MurmurHash3 mixing and Gumbel-max draw in integer
    torch ops; masking to 32 bits keeps the low bits identical to the kernel's
    unsigned arithmetic, so both paths pick the same token.
    """
    columns = torch.arange(logprobs.shape[1], device=logprobs.device, dtype=torch.int64)
    seeds = seeds.to(torch.int64)
    state = torch.zeros_like(seeds).unsqueeze(1)
    state = _murmur3_mix(state, (seeds & _MASK32).unsqueeze(1))
    state = _murmur3_mix(state, ((seeds >> 32) & _MASK32).unsqueeze(1))
    state = _murmur3_mix(state, (positions.to(torch.int64) & _MASK32).unsqueeze(1))
    # Broadcasting the [rows, 1] state against the [1, columns] key expands the
    # hash to one value per candidate token.
    state = _murmur3_mix(state, columns.unsqueeze(0))
    state = state ^ 16
    state = state ^ (state >> 16)
    state = (state * 0x85EBCA6B) & _MASK32
    state = state ^ (state >> 13)
    state = (state * 0xC2B2AE35) & _MASK32
    state = state ^ (state >> 16)

    noise = state.to(torch.float64) / float(_MASK32)
    noise.log_().clamp_(min=torch.finfo(torch.float64).min, max=-(2.0**-32)).neg_()
    noise.log_().neg_()
    return (noise + logprobs.to(torch.float64)).argmax(dim=1)


def sample_seeded(
    logprobs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """One seeded draw per row, dependent only on that row's seed and position."""
    if logprobs.is_cuda:
        return multinomial_with_seed(logprobs, seeds, positions).view(-1)
    return _sample_seeded_eager(logprobs, seeds, positions)


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    max_new_tokens: int = 750
    cfg_scale: float = 1.0
    seed: int = 0

    @property
    def row_seed(self) -> int:
        """A positive int32 seed derived from the public 64-bit request seed."""
        return derive_sampling_seed("breeze-tts-2", self.seed)


class BatchedSampling:
    """Per-request sampling parameters as device tensors.

    Built once per decode step and reused across a frame's codebooks so the
    fifteen depth steps do not re-upload the same parameters.
    """

    def __init__(self, params: Sequence[SamplingConfig], device: torch.device):
        if not params:
            raise ValueError("Breeze batched sampling requires at least one request")
        self.size = len(params)
        self.device = device
        self.cfg_scale = self._column(entry.cfg_scale for entry in params)
        self.temperature = self._column(entry.temperature for entry in params)
        self.top_p = self._column(entry.top_p for entry in params)
        self.repetition_penalty = self._column(
            entry.repetition_penalty for entry in params
        )
        self.top_k = torch.tensor(
            [int(entry.top_k) for entry in params], device=device, dtype=torch.long
        )
        self.seeds = torch.tensor(
            [entry.row_seed for entry in params], device=device, dtype=torch.long
        )

    def _column(self, values) -> torch.Tensor:
        return torch.tensor(
            [float(value) for value in values], device=self.device, dtype=torch.float32
        )

    def positions(self, frames: torch.Tensor, codebook: int) -> torch.Tensor:
        """Sampling positions for one codebook of each request's current frame."""
        return frames * POSITIONS_PER_FRAME + codebook


def apply_cfg(logits: torch.Tensor, scale: float) -> torch.Tensor:
    """Rows are [conditional, unconditional]; return a single guided row."""
    if logits.shape[0] != 2:
        raise ValueError("Breeze CFG requires exactly two branch rows")
    if scale == 1.0:
        return logits[:1].float()
    if scale == 0.0:
        return logits[1:].float()
    cond, uncond = logits.float().split(1)
    return uncond + scale * (cond - uncond)


def apply_cfg_batched(logits: torch.Tensor, sampling: BatchedSampling) -> torch.Tensor:
    """Guide interleaved [cond, uncond] rows into one row per request.

    Every scale goes through the same expression, including the 1.0 default, so
    one captured graph stays valid for any mix of per-request guidance.
    """
    if logits.shape[0] != 2 * sampling.size:
        raise ValueError("Breeze CFG requires exactly two branch rows per request")
    cond = logits[0::2].float()
    uncond = logits[1::2].float()
    return uncond + sampling.cfg_scale.unsqueeze(1) * (cond - uncond)


def sample_scores_batched(
    scores: torch.Tensor,
    sampling: BatchedSampling,
    positions: torch.Tensor,
    *,
    codebook_size: int,
    eos_token_id: int | None = None,
    penalized: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one token per row from already-guided scores."""
    if penalized is not None:
        penalty = sampling.repetition_penalty.unsqueeze(1)
        scores = torch.where(
            penalized,
            torch.where(scores < 0, scores * penalty, scores / penalty),
            scores,
        )
    else:
        scores = scores.clone()
    # Reserved audio IDs are not codec codes. Only the backbone's extra EOS
    # class is allowed; depth heads must always sample an actual codec code.
    eos = None if eos_token_id is None else scores[:, eos_token_id].clone()
    scores[:, codebook_size:] = -torch.inf
    if eos is not None:
        scores[:, eos_token_id] = eos

    greedy = scores.argmax(dim=-1)
    temperature = sampling.temperature
    sampled = temperature > 0
    scores = scores / torch.where(sampled, temperature, 1.0).unsqueeze(1)

    width = scores.shape[-1]
    # top-k first, then top-p over what top-k left, matching the reference order.
    ordered, _ = scores.sort(dim=-1, descending=True)
    rank = sampling.top_k.clamp(min=1, max=width) - 1
    threshold = ordered.gather(1, rank.unsqueeze(1))
    active_k = ((sampling.top_k > 0) & (sampling.top_k < width)).unsqueeze(1)
    scores = scores.masked_fill(active_k & (scores < threshold), -torch.inf)

    active_p = (sampling.top_p < 1.0).unsqueeze(1)
    # Applied unconditionally: rows with p == 1 are masked out by active_p, so a
    # captured graph does not depend on which rows requested nucleus sampling.
    ordered, indices = scores.sort(dim=-1, descending=True)
    tail = ordered.softmax(-1).cumsum(-1) > sampling.top_p.unsqueeze(1)
    tail = torch.cat((torch.zeros_like(tail[:, :1]), tail[:, :-1]), dim=1)
    scores = scores.masked_fill(
        active_p & torch.zeros_like(tail).scatter(1, indices, tail), -torch.inf
    )

    drawn = sample_seeded(scores.log_softmax(dim=-1), sampling.seeds, positions)
    return torch.where(sampled, drawn, greedy)


def sample_logits_batched(
    logits: torch.Tensor,
    sampling: BatchedSampling,
    positions: torch.Tensor,
    *,
    codebook_size: int,
    eos_token_id: int | None = None,
    penalized: torch.Tensor | None = None,
) -> torch.Tensor:
    """Guide interleaved CFG rows, then sample one token per request."""
    return sample_scores_batched(
        apply_cfg_batched(logits, sampling),
        sampling,
        positions,
        codebook_size=codebook_size,
        eos_token_id=eos_token_id,
        penalized=penalized,
    )


def sample_logits(
    logits: torch.Tensor,
    params: SamplingConfig,
    position: int = 0,
    *,
    history: Sequence[int] = (),
    codebook_size: int = 2048,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Single-request sampling over one already-guided row."""
    scores = logits.float()
    if scores.ndim != 2 or scores.shape[0] != 1:
        raise ValueError("Breeze sampling expects one guided row")
    sampling = BatchedSampling([params], scores.device)
    penalized = None
    if history and params.repetition_penalty != 1.0:
        penalized = torch.zeros_like(scores, dtype=torch.bool)
        ids = torch.tensor(list(history), device=scores.device, dtype=torch.long)
        penalized[0, ids] = True
    return sample_scores_batched(
        scores,
        sampling,
        torch.tensor([position], device=scores.device, dtype=torch.long),
        codebook_size=codebook_size,
        eos_token_id=eos_token_id,
        penalized=penalized,
    )
