# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Local fused sampling kernel tests."""

from __future__ import annotations

import pytest
import torch


def test_fused_sampler_falls_back_on_cpu() -> None:
    from sglang_omni.models.moss_tts_local.sampling_kernels import (
        sample_seeded_full_vocab,
    )

    rows = 2
    result = sample_seeded_full_vocab(
        torch.randn(rows, 16),
        torch.ones(rows),
        torch.ones(rows),
        torch.full((rows,), 8, dtype=torch.long),
        torch.arange(rows, dtype=torch.long),
        torch.arange(rows, dtype=torch.long),
    )
    assert result is None


@pytest.mark.gpu
def test_fused_sampler_randomized_parity_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from sglang_omni.models.moss_tts_local import local_transformer
    from sglang_omni.models.moss_tts_local.sampling_kernels import (
        sample_seeded_full_vocab,
    )

    device = torch.device("cuda")
    rows = 512
    vocab = 1024
    generator = torch.Generator(device=device).manual_seed(20260729)
    logits = torch.randn(
        rows,
        vocab,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    row_ids = torch.arange(rows, device=device)
    temperature_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.7],
        device=device,
        dtype=torch.float32,
    )
    top_p_values = torch.tensor(
        [0.2, 0.8, 0.95, 1.0],
        device=device,
        dtype=torch.float32,
    )
    top_k_values = torch.tensor(
        [-1, 1, 25, 50, 1024],
        device=device,
        dtype=torch.long,
    )
    temperatures = temperature_values[row_ids.remainder(temperature_values.numel())]
    top_ps = top_p_values[
        (row_ids // temperature_values.numel()).remainder(top_p_values.numel())
    ]
    top_ks = top_k_values[
        (row_ids // (temperature_values.numel() * top_p_values.numel())).remainder(
            top_k_values.numel()
        )
    ]
    seeds = torch.randint(
        0,
        1 << 31,
        (rows,),
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    positions = torch.randint(
        0,
        1 << 30,
        (rows,),
        generator=generator,
        device=device,
        dtype=torch.long,
    )

    actual = sample_seeded_full_vocab(
        logits,
        temperatures,
        top_ps,
        top_ks,
        seeds,
        positions,
    )
    if actual is None:
        pytest.skip("Triton fused sampling kernel is unavailable")

    expected = local_transformer._sample_seeded_branchless_eager(
        logits,
        temperature=temperatures,
        top_p=top_ps,
        top_k=top_ks,
        seeds=seeds,
        positions=positions,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.gpu
def test_fused_sampler_preserves_top_k_boundary_ties_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from sglang_omni.models.moss_tts_local import local_transformer
    from sglang_omni.models.moss_tts_local.sampling_kernels import (
        sample_seeded_full_vocab,
    )

    device = torch.device("cuda")
    rows = 128
    logits = (
        torch.tensor(
            [[5.0, 4.0, 4.0, 1.0]],
            device=device,
        )
        .expand(rows, -1)
        .contiguous()
    )
    temperatures = torch.ones(rows, device=device)
    top_ps = torch.ones(rows, device=device)
    top_ks = torch.full((rows,), 2, dtype=torch.long, device=device)
    seeds = torch.arange(rows, dtype=torch.long, device=device)
    positions = torch.arange(rows, dtype=torch.long, device=device)

    actual = sample_seeded_full_vocab(
        logits,
        temperatures,
        top_ps,
        top_ks,
        seeds,
        positions,
    )
    if actual is None:
        pytest.skip("Triton fused sampling kernel is unavailable")

    expected = local_transformer._sample_seeded_branchless_eager(
        logits,
        temperature=temperatures,
        top_p=top_ps,
        top_k=top_ks,
        seeds=seeds,
        positions=positions,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
