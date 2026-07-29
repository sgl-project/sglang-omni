# SPDX-License-Identifier: Apache-2.0
"""Benchmark the fused MOSS-TTS two-candidate sampler against eager PyTorch."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from sglang_omni.models.moss_tts.model_runner import (
    MossTTSModelRunner,
    _multinomial_with_seed_and_token_ids,
)
from sglang_omni.models.moss_tts.sampling_kernels import sample_two_candidates

TensorTuple = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def eager_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """The operator chain used before the fused fast path."""

    do_sample = temperatures > 0
    safe_temperature = torch.where(
        do_sample, temperatures, torch.ones_like(temperatures)
    )
    scores = logits / safe_temperature.unsqueeze(1)
    scores = MossTTSModelRunner._apply_two_token_top_k(scores, top_ks)
    scores = MossTTSModelRunner._apply_top_p(
        scores,
        top_ps,
        skip_inactive_check=True,
    )
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    fallback = (~do_sample) | (probs.sum(dim=-1) <= 0)
    greedy = torch.argmax(logits, dim=-1)
    sampled = _multinomial_with_seed_and_token_ids(
        scores,
        seeds,
        positions,
        token_ids,
    )
    return token_ids[torch.where(fallback, greedy, sampled)]


def fused_sample(*args: torch.Tensor) -> torch.Tensor:
    result = sample_two_candidates(*args)
    if result is None:
        raise RuntimeError("The fused MOSS-TTS sampling kernel was not selected")
    return result


def make_inputs(batch_size: int, seed: int) -> TensorTuple:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        batch_size,
        2,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    row = torch.arange(batch_size, device=device)
    temperatures = torch.where(
        row.remainder(11) == 0,
        torch.zeros(batch_size, device=device),
        torch.full((batch_size,), 0.8, device=device),
    )
    top_ps = torch.where(
        row.remainder(3) == 0,
        torch.full((batch_size,), 0.85, device=device),
        torch.ones(batch_size, device=device),
    )
    top_ks = torch.where(
        row.remainder(5) == 0,
        torch.ones(batch_size, dtype=torch.long, device=device),
        torch.full((batch_size,), 2, dtype=torch.long, device=device),
    )
    seeds = row.to(torch.long) + seed
    positions = row.to(torch.long) + 1000
    token_ids = torch.tensor([151643, 151645], dtype=torch.long, device=device)
    return logits, temperatures, top_ps, top_ks, seeds, positions, token_ids


def time_cuda(
    function: Callable[..., torch.Tensor],
    args: TensorTuple,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        function(*args)
    torch.cuda.synchronize()

    timings_us = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function(*args)
        end.record()
        end.synchronize()
        timings_us.append(start.elapsed_time(end) * 1000.0 / iterations)
    return timings_us


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def benchmark_batch(
    batch_size: int,
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    args = make_inputs(batch_size, seed=1234)

    expected = eager_sample(*args)
    actual = fused_sample(*args)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    eager_us = time_cuda(
        eager_sample,
        args,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    fused_us = time_cuda(
        fused_sample,
        args,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    eager_median = percentile(eager_us, 0.5)
    fused_median = percentile(fused_us, 0.5)
    speedup = eager_median / fused_median

    print(
        f"{batch_size:>5}  "
        f"{eager_median:>12.3f}  "
        f"{fused_median:>12.3f}  "
        f"{speedup:>8.2f}x  "
        f"{percentile(eager_us, 0.95):>12.3f}  "
        f"{percentile(fused_us, 0.95):>12.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(
        "batch      eager_us      fused_us   speedup"
        "  eager_p95_us  fused_p95_us"
    )
    for batch_size in args.batch_sizes:
        benchmark_batch(
            batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
