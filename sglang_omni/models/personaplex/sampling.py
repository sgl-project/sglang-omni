# SPDX-License-Identifier: Apache-2.0
"""Top-k sampling for the depformer, matching the reference draw."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from typing import Protocol

import torch

NOISE_TILE_SIZE = 256

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None
    libdevice = None
    HAS_TRITON = False


if HAS_TRITON:
    UNIFORM_MANTISSA_BITS = tl.constexpr(23)

    @triton.jit(do_not_specialize=["codebook_start"])
    def batched_seeded_exponential_noise_kernel(
        noise: tl.tensor,
        seeds: tl.tensor,
        frame_positions: tl.tensor,
        codebook_start: tl.tensor,
        candidate_count: tl.constexpr,
        noise_codebook_stride: tl.constexpr,
        noise_row_stride: tl.constexpr,
        tile_size: tl.constexpr,
    ) -> None:
        row = tl.program_id(0)
        codebook_offset = tl.program_id(1)
        candidate_positions = tl.program_id(2) * tile_size + tl.arange(0, tile_size)
        seed = tl.load(seeds + row)
        frame_position = tl.load(frame_positions + row)
        codebook_position = codebook_start + codebook_offset
        random_bits, _, _, _ = tl.philox(
            seed.to(tl.uint64),
            candidate_positions.to(tl.uint32),
            frame_position.to(tl.uint32),
            codebook_position.to(tl.uint32),
            0,
        )
        uniform = (
            (random_bits >> (32 - UNIFORM_MANTISSA_BITS)).to(tl.float32) + 0.5
        ) * (1.0 / (1 << UNIFORM_MANTISSA_BITS))
        exponential_noise = -libdevice.log(uniform)
        tl.store(
            noise
            + codebook_offset * noise_codebook_stride
            + row * noise_row_stride
            + candidate_positions,
            exponential_noise,
            mask=(seed >= 0) & (candidate_positions < candidate_count),
        )

else:
    pass


@dataclass(frozen=True)
class AudioSampling:
    temperature: float
    top_k: int

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


class AudioTokenSampler(Protocol):
    def __call__(self, logits: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, kw_only=True)
class SamplingRun:
    """Adjacent batch rows that share one sampling call."""

    row_slice: slice
    sampling: AudioSampling
    generators: list[torch.Generator | None]
    counter_seeds: torch.Tensor | None = None
    counter_frame_positions: torch.Tensor | None = None


def fill_sampling_noise(
    noise: torch.Tensor, generators: Sequence[torch.Generator | None]
) -> None:
    start_row = 0
    for _, run in groupby(generators, key=id):
        run_generators = list(run)
        end_row = start_row + len(run_generators)
        noise[start_row:end_row].exponential_(1.0, generator=run_generators[0])
        start_row = end_row
    assert start_row == noise.shape[0], (start_row, noise.shape)


def sample_runs(logits: torch.Tensor, runs: Sequence[SamplingRun]) -> torch.Tensor:
    picks = [
        sample_token(logits[run.row_slice], run.sampling, run.generators)
        for run in runs
    ]
    return picks[0] if len(picks) == 1 else torch.cat(picks)


def sample_token(
    logits: torch.Tensor,
    sampling: AudioSampling,
    generators: Sequence[torch.Generator | None] = (),
    *,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """[B, card] float logits → [B] token ids.

    generators[i] draws row i's noise, or the default generator when None.
    noise supplies counter-based draws for CUDA seeded rows.
    The reference draws from the top-k via the exponential race (argmax of
    p / Exp(1)), which avoids a host sync; kept so seeded runs line up.
    """
    if sampling.greedy:
        return logits.argmax(dim=-1)
    else:
        pass
    probs = torch.softmax(logits / sampling.temperature, dim=-1)
    if sampling.top_k > 0:
        probs, indices = torch.topk(probs, min(sampling.top_k, probs.shape[-1]), dim=-1)
    else:
        indices = None
    if noise is None:
        noise = torch.empty_like(probs)
        fill_sampling_noise(noise, generators)
    else:
        assert noise.shape[0] == probs.shape[0]
        assert noise.shape[1] >= probs.shape[1]
        noise = noise[:, : probs.shape[1]]
    choice = (probs / noise).argmax(dim=-1, keepdim=True)
    if indices is not None:
        choice = indices.gather(-1, choice)
    else:
        pass
    return choice[:, 0]


def batched_seeded_exponential_noise(
    logits: torch.Tensor,
    seeds: torch.Tensor,
    frame_positions: torch.Tensor,
    codebook_position: int,
) -> torch.Tensor:
    """Draw all seeded CUDA rows from request-local counters in one launch."""
    return batched_seeded_exponential_noises(
        logits,
        seeds,
        frame_positions,
        codebook_count=1,
        codebook_start=codebook_position,
    )[0]


def fill_batched_seeded_exponential_noises(
    noise: torch.Tensor,
    seeds: torch.Tensor,
    frame_positions: torch.Tensor,
    codebook_start: int = 0,
) -> None:
    """Fill reusable codebook noise storage for the supplied request rows."""
    if not HAS_TRITON:
        raise RuntimeError("PersonaPlex seeded CUDA sampling requires Triton")
    else:
        pass
    assert noise.ndim == 3
    assert noise.device.type == "cuda"
    assert seeds.shape == frame_positions.shape
    assert seeds.ndim == 1 and seeds.shape[0] <= noise.shape[1]
    assert seeds.device == frame_positions.device == noise.device
    candidate_count = noise.shape[2]
    candidate_tile_count = triton.cdiv(candidate_count, NOISE_TILE_SIZE)
    batched_seeded_exponential_noise_kernel[
        (seeds.shape[0], noise.shape[0], candidate_tile_count)
    ](
        noise,
        seeds,
        frame_positions,
        codebook_start,
        candidate_count=candidate_count,
        noise_codebook_stride=noise.stride(0),
        noise_row_stride=noise.stride(1),
        tile_size=NOISE_TILE_SIZE,
        num_warps=4,
    )


def batched_seeded_exponential_noises(
    logits: torch.Tensor,
    seeds: torch.Tensor,
    frame_positions: torch.Tensor,
    codebook_count: int,
    codebook_start: int = 0,
) -> torch.Tensor:
    """Draw several codebooks of seeded CUDA noise in one launch."""
    assert logits.ndim == 2
    assert logits.device.type == "cuda"
    assert seeds.shape == frame_positions.shape == (logits.shape[0],)
    assert seeds.device == frame_positions.device == logits.device
    assert codebook_count > 0
    noise = torch.empty(
        (codebook_count, *logits.shape), dtype=logits.dtype, device=logits.device
    )
    fill_batched_seeded_exponential_noises(
        noise, seeds, frame_positions, codebook_start
    )
    return noise


def create_audio_token_sampler(
    runs: list[SamplingRun], codebook_count: int
) -> AudioTokenSampler:
    """Build one frame sampler, fusing counter noise across codebooks."""
    counter_run = next((run for run in runs if run.counter_seeds is not None), None)
    codebook_position = 0
    counter_noises: torch.Tensor | None = None

    def sample(logits: torch.Tensor) -> torch.Tensor:
        nonlocal codebook_position, counter_noises
        if counter_run is not None and counter_noises is None:
            assert counter_run.counter_seeds is not None
            assert counter_run.counter_frame_positions is not None
            counter_noises = batched_seeded_exponential_noises(
                logits,
                counter_run.counter_seeds,
                counter_run.counter_frame_positions,
                codebook_count,
            )
        else:
            pass
        counter_noise = (
            None if counter_noises is None else counter_noises[codebook_position]
        )
        picks = [
            sample_token(
                logits[run.row_slice],
                run.sampling,
                run.generators,
                noise=(
                    None
                    if counter_noise is None or run.counter_seeds is None
                    else counter_noise[run.row_slice]
                ),
            )
            for run in runs
        ]
        codebook_position += 1
        return picks[0] if len(picks) == 1 else torch.cat(picks)

    return sample


__all__ = [
    "AudioSampling",
    "batched_seeded_exponential_noise",
    "batched_seeded_exponential_noises",
    "create_audio_token_sampler",
    "fill_batched_seeded_exponential_noises",
    "sample_token",
]
