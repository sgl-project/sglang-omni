# SPDX-License-Identifier: Apache-2.0
"""How LFM3 draws the 12 codes of a speech frame from the depth decoder's logits.

The exact sampler makes the checkpoint's random draws in the checkpoint's order (codebook permutation, then one draw per revealed codebook, row by row) on a per-request generator, so a seeded request reproduces the checkpoint's own generate. The seeded sampler hashes its draws from the request seed and frame position, so one CUDA graph can hold the whole frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import torch
from transformers.generation.logits_process import (
    LogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
)

from sglang_omni.sampling.fused_seeded_sample import sample_seeded_fused

# The seeded reveal order's keys: (seed, position, column) mixed by spatial-hash primes, two rounds of an integer finalizer, then cut to 24 bits.
SEED_HASH_MULTIPLIER = 19349663
POSITION_HASH_MULTIPLIER = 73856093
COLUMN_HASH_MULTIPLIER = 83492791
FINALIZER_MULTIPLIER = 0x45D9F3B
UINT32_MASK = 0xFFFFFFFF
KEY_MASK = 0xFFFFFF


class SpeechSampler(str, Enum):
    """How speech frames are drawn.

    exact: each request's generator draws in the checkpoint's order, so a seed reproduces the checkpoint's own generate; only the depth passes are graphed.
    seeded: hashed draws from the request seed and frame position, so one graph holds the whole frame and decode needs no host sync; the draws do not depend on the batch, and differ from the checkpoint's.
    """

    EXACT = "exact"
    SEEDED = "seeded"


@dataclass(kw_only=True)
class FrameSampling:
    """A request's speech sampling settings, generator and repetition history."""

    temperature: float
    top_k: int | None
    repetition_penalty: float
    repetition_window: int
    generator: torch.Generator
    # Row seed of the seeded sampler; the exact sampler draws from the generator instead.
    seed: int = 0
    # Frames sampled so far: the seeded sampler's draw positions.
    frames_sampled: int = 0
    # The last repetition_window frames, (frames, num_codebooks).
    history: torch.Tensor | None = None

    def warpers(self) -> list[LogitsProcessor]:
        """The logits warpers of these settings; none for greedy decoding."""
        if self.temperature == 0:
            return []
        else:
            pass
        warpers: list[LogitsProcessor] = [TemperatureLogitsWarper(self.temperature)]
        if self.top_k is not None and self.top_k > 0:
            warpers.append(TopKLogitsWarper(self.top_k))
        else:
            pass
        return warpers

    def append(self, frame: torch.Tensor) -> None:
        if self.history is None:
            self.history = frame[None]
        else:
            self.history = torch.cat([self.history, frame[None]])[
                -self.repetition_window :
            ]


class DepthLogits(Protocol):
    def __call__(
        self, hidden_states: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor: ...


@torch.no_grad()
def sample_frames(
    depth_logits: DepthLogits,
    hidden_states: torch.Tensor,
    samplings: Sequence[FrameSampling],
    schedule: Sequence[int],
) -> torch.Tensor:
    """Sample one frame per row of hidden_states (N, H) -> codes (N, 12), drawing exactly as the checkpoint's generate.

    Each depth pass runs batched. A row's penalty and warpers run once over the codebooks it reveals in the pass, and its draws go codebook by codebook in the checkpoint's order on its own generator, so a seeded request reproduces the checkpoint's own generate.
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    num_codebooks = sum(schedule)
    codes = torch.zeros((batch_size, num_codebooks), dtype=torch.long, device=device)
    revealed = torch.zeros_like(codes, dtype=torch.bool)
    reveal_order = torch.stack(
        [
            torch.randperm(num_codebooks, device=device, generator=sampling.generator)
            for sampling in samplings
        ]
    )
    warpers = [sampling.warpers() for sampling in samplings]
    for revealed_now in reveal_order.split(list(schedule), dim=1):
        logits = depth_logits(hidden_states, codes, ~revealed)
        for row, sampling in enumerate(samplings):
            codebooks = revealed_now[row]
            scores = logits[row].index_select(0, codebooks).float()
            history = sampling.history
            if sampling.repetition_penalty != 1.0 and history is not None:
                scores = RepetitionPenaltyLogitsProcessor(sampling.repetition_penalty)(
                    history.index_select(1, codebooks).T, scores
                )
            else:
                pass
            if not warpers[row]:
                codes[row, codebooks] = scores.argmax(-1)
                continue
            else:
                pass
            for warper in warpers[row]:
                scores = warper(None, scores)
            probabilities = scores.softmax(-1)
            codes[row, codebooks] = torch.cat(
                [
                    torch.multinomial(
                        probabilities[index : index + 1],
                        1,
                        generator=sampling.generator,
                    )[0]
                    for index in range(probabilities.shape[0])
                ]
            )
        revealed.scatter_(1, revealed_now, True)
    return codes


def seeded_uniform_keys(
    seeds: torch.Tensor, positions: torch.Tensor, width: int
) -> torch.Tensor:
    """Per-row hashed integers in [0, 2**24) of (seed, position, column): (N,) x (N,) -> (N, width)."""
    columns = torch.arange(width, device=seeds.device, dtype=torch.int64)
    mixed = (
        (seeds[:, None] * SEED_HASH_MULTIPLIER)
        ^ (positions[:, None] * POSITION_HASH_MULTIPLIER)
        ^ (columns * COLUMN_HASH_MULTIPLIER)
    ) & UINT32_MASK
    for _ in range(2):
        mixed = ((mixed ^ (mixed >> 16)) * FINALIZER_MULTIPLIER) & UINT32_MASK
    return (mixed ^ (mixed >> 16)) & KEY_MASK


@torch.no_grad()
def seeded_frame_codes(
    depth_logits: DepthLogits,
    hidden_states: torch.Tensor,
    seeds: torch.Tensor,
    frame_positions: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    repetition_penalty: torch.Tensor,
    history: torch.Tensor,
    schedule: Sequence[int],
) -> torch.Tensor:
    """One frame per row with hashed draws and no host sync, so one CUDA graph can hold the whole frame.

    A row's draws depend only on its seed and frame position, not on the rows beside it. The draws differ from the checkpoint's generator stream.
    hidden_states (N, H); seeds, frame_positions, top_k (N,) int64; temperature, repetition_penalty (N,) float; history (N, window, 12) int64; a row without history passes penalty 1, which leaves its scores unchanged.
    """
    batch_size = hidden_states.shape[0]
    num_codebooks = sum(schedule)
    draws_per_frame = num_codebooks + 1
    position_base = frame_positions * draws_per_frame
    reveal_order = seeded_uniform_keys(
        seeds, position_base + num_codebooks, num_codebooks
    ).argsort(dim=1, stable=True)
    codes = torch.zeros(
        (batch_size, num_codebooks), dtype=torch.long, device=hidden_states.device
    )
    revealed = torch.zeros_like(codes, dtype=torch.bool)
    draw_positions = (
        position_base[:, None]
        + torch.arange(num_codebooks, device=hidden_states.device)[None, :]
    ).reshape(-1)
    row_seeds = seeds.repeat_interleave(num_codebooks)
    row_temperature = temperature.repeat_interleave(num_codebooks)
    row_top_k = top_k.repeat_interleave(num_codebooks)
    row_top_p = torch.ones_like(row_temperature)
    penalty = repetition_penalty[:, None, None]
    pass_start = 0
    for codebooks_this_pass in schedule:
        revealed_now = reveal_order[:, pass_start : pass_start + codebooks_this_pass]
        pass_start += codebooks_this_pass
        logits = depth_logits(hidden_states, codes, ~revealed).float()
        codebook_size = logits.shape[-1]
        repeated = torch.zeros(
            (batch_size, num_codebooks, codebook_size),
            dtype=torch.bool,
            device=logits.device,
        ).scatter_(2, history.transpose(1, 2), True)
        penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
        scores = torch.where(repeated, penalized, logits)
        drawn = sample_seeded_fused(
            scores.reshape(-1, codebook_size),
            temperature=row_temperature,
            top_p=row_top_p,
            top_k=row_top_k,
            seeds=row_seeds,
            positions=draw_positions,
        ).view(batch_size, num_codebooks)
        reveal_mask = torch.zeros_like(revealed).scatter_(1, revealed_now, True)
        codes = torch.where(reveal_mask, drawn, codes)
        revealed = revealed | reveal_mask
    return codes
