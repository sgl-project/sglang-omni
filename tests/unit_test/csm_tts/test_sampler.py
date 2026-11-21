# SPDX-License-Identifier: Apache-2.0
"""Sampler / EOS-machine regression."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.csm_tts.sampler import (
    CsmBatchedSamplerState,
    sample_codebook0_batch,
    sample_codebook0_row,
)
from sglang_omni.models.csm_tts.utils import NUM_CODEBOOKS, STOP_CODE

DEVICE = "cpu"


def eos_frame(*, cb31: int = 7) -> torch.Tensor:
    codes = torch.zeros(NUM_CODEBOOKS, dtype=torch.long)
    codes[-1] = cb31  # cb31 is excluded from the EOS test
    return codes


def test_reset_sampler_slot_clears_selected_slot_only() -> None:
    pool = CsmBatchedSamplerState(4, device=DEVICE)
    pool.last_codes[1] = torch.arange(NUM_CODEBOOKS)
    pool.generation_done[1] = True
    pool.frames_emitted[1] = 9
    pool.last_codes[2] = torch.full((NUM_CODEBOOKS,), 3)
    pool.generation_done[2] = True

    pool.reset_sampler_slot(1)

    assert int(pool.last_codes[1].sum()) == 0
    assert bool(pool.generation_done[1]) is False
    assert int(pool.frames_emitted[1]) == 0
    assert bool(pool.generation_done[2]) is True
    assert int(pool.last_codes[2][0]) == 3


@pytest.mark.parametrize(
    "phase",
    ["running", "eos_now", "already_done", "mixed"],
)
def test_sample_codebook0_batch_matches_per_row_reference(phase: str) -> None:
    running = torch.arange(NUM_CODEBOOKS, dtype=torch.long) + 1
    eos = eos_frame()
    if phase == "running":
        codes = torch.stack([running, running + 10])
        done = torch.tensor([False, False])
    elif phase == "eos_now":
        codes = torch.stack([eos, running])
        done = torch.tensor([False, False])
    elif phase == "already_done":
        codes = torch.stack([running, running])
        done = torch.tensor([True, True])
    else:
        codes = torch.stack([eos, running, running])
        done = torch.tensor([False, True, False])

    out, new_done, was_done = sample_codebook0_batch(codes, done)
    for b in range(codes.shape[0]):
        ref_out, ref_new, ref_was = sample_codebook0_row(codes[b], bool(done[b]))
        assert torch.equal(out[b], ref_out)
        assert bool(new_done[b]) == ref_new
        assert bool(was_done[b]) == ref_was


def test_eos_excludes_codebook_31() -> None:
    codes = eos_frame(cb31=2047).unsqueeze(0)
    done = torch.tensor([False])
    out, new_done, was_done = sample_codebook0_batch(codes, done)
    assert bool(was_done[0]) is False
    assert bool(new_done[0]) is True
    assert torch.equal(out[0], codes[0])
    assert int(out[0, -1]) == 2047


def test_already_done_emits_stop_code() -> None:
    codes = (torch.arange(NUM_CODEBOOKS) + 4).unsqueeze(0)
    done = torch.tensor([True])
    out, new_done, was_done = sample_codebook0_batch(codes, done)
    assert bool(was_done[0]) is True
    assert bool(new_done[0]) is True
    assert torch.equal(out[0], torch.full((NUM_CODEBOOKS,), STOP_CODE))
