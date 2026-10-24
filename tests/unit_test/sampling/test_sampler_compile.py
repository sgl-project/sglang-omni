# SPDX-License-Identifier: Apache-2.0
"""Tests for compiling the sampler functions while a model loads."""

from __future__ import annotations

import pytest
import torch
from flashinfer.sampling import get_sampling_module
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils import DynamicGradMode
from torch._dynamo.utils import counters

from sglang_omni.sampling.sampler_compile import (
    SamplerCompileForms,
    compile_sampler_functions,
)

VOCAB_SIZE = 3072


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_forms_serve_views_and_standalone_tensors_at_every_batch_size() -> (
    None
):
    torch.compiler.reset()
    device = torch.device("cuda", torch.cuda.current_device())
    compile_sampler_functions(
        SamplerCompileForms(
            device=device,
            vocab_size=VOCAB_SIZE,
            seeded_log_probability_dtype=torch.float64,
            penalized_logits_dtype=torch.bfloat16,
        )
    )
    compiled_frames = counters["frames"]["ok"]

    seed_buffer = torch.arange(64, device=device)
    logits_buffer = torch.randn(64, VOCAB_SIZE, dtype=torch.bfloat16, device=device)
    with DynamicGradMode():
        for rows in (1, 2, 5, 13):
            log_probabilities = torch.randn(
                rows, VOCAB_SIZE, dtype=torch.float64, device=device
            )
            positions = torch.arange(rows, device=device)
            multinomial_with_seed(log_probabilities, seed_buffer[:rows], positions)
            multinomial_with_seed(
                log_probabilities, seed_buffer[:rows].clone(), positions
            )
            penalties = torch.ones(rows, VOCAB_SIZE, device=device)
            apply_scaling_penalties(logits_buffer[:rows], penalties)
            apply_scaling_penalties(logits_buffer[:rows].clone(), penalties)
        assert counters["frames"]["ok"] == compiled_frames

        apply_scaling_penalties(
            torch.randn(2, VOCAB_SIZE, dtype=torch.float16, device=device),
            torch.ones(2, VOCAB_SIZE, device=device),
        )
    assert counters["frames"]["ok"] == compiled_frames + 1


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("should_load_flashinfer_sampling", [False, True])
def test_flashinfer_form_loads_the_sampling_module(
    should_load_flashinfer_sampling: bool,
) -> None:
    get_sampling_module.cache_clear()

    compile_sampler_functions(
        SamplerCompileForms(
            device=torch.device("cuda", torch.cuda.current_device()),
            vocab_size=VOCAB_SIZE,
            should_load_flashinfer_sampling=should_load_flashinfer_sampling,
        )
    )

    assert get_sampling_module.cache_info().currsize == int(
        should_load_flashinfer_sampling
    )
