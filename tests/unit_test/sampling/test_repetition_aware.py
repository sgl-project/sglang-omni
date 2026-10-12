# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.model_runner import repetition_aware_redraw

ROWS = 4096


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_seeded_redraw_of_a_row_subset_matches_the_full_draw() -> None:
    probs = torch.zeros((ROWS, 8), device="cuda")
    probs[:, 3], probs[:, 5], probs[:, 6] = 0.9, 0.06, 0.04
    sampled_ids = torch.full((ROWS,), 3, dtype=torch.int32, device="cuda")
    seeds = torch.arange(1, ROWS + 1, dtype=torch.int64, device="cuda")
    positions = torch.arange(ROWS, device="cuda")
    subset = torch.arange(1, ROWS, 3, device="cuda")

    full = repetition_aware_redraw(
        probs, sampled_ids, torch.arange(ROWS, device="cuda"), seeds, positions
    )
    partial = repetition_aware_redraw(probs, sampled_ids, subset, seeds, positions)

    assert not bool((full == 3).any())
    assert torch.equal(partial[subset], full[subset])
