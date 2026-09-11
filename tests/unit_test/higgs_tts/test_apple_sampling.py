# SPDX-License-Identifier: Apache-2.0
"""Portable sampler qualification; requires only torch and pytest, no checkpoint.

Set HIGGS_REQUIRE_MPS=1 on a Mac to fail rather than skip absent Metal hardware.
Load the leaf module directly so missing serving dependencies cannot mask the
numeric tests. Full-package collection is a separate integration check.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[3]
    / "sglang_omni"
    / "models"
    / "higgs_tts"
    / "apple_sampling.py"
).resolve()
_SPEC = importlib.util.spec_from_file_location("higgs_apple_sampling_test", _PATH)
sampling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sampling)


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        if os.environ.get("HIGGS_REQUIRE_MPS") == "1":
            pytest.fail("HIGGS_REQUIRE_MPS=1 but Metal is unavailable")
        pytest.skip("requires Apple Metal")
    return torch.device(request.param)


def test_hash_matches_independent_murmur3_vectors():
    # Generated with mmh3.hash(struct.pack('<QII', seed, pos, col), signed=False).
    # Covers high seed bits, uint32 position wrap, and overflow in every round.
    actual = sampling._murmur_hash_cpu(
        torch.tensor([0, 42, 2147483647, 4294967297]),
        torch.tensor([0, 7, 4294967295, 11]),
        4,
    )
    assert actual.tolist() == [
        [2167721464, 10027521, 2423355346, 1368026766],
        [1713287821, 2249025004, 521978470, 966665676],
        [1269191520, 1557978957, 4157290558, 2199673162],
        [3924483750, 1646346432, 3006310221, 3814039520],
    ]


def test_repeatability_and_batch_reordering(device):
    logits = torch.randn(8, 1026, generator=torch.Generator().manual_seed(9)).to(device)
    seeds = torch.arange(8, device=device) + 31
    positions = torch.arange(8, device=device) + 72
    first = sampling.multinomial_with_seed_cpu(logits, seeds, positions)
    torch.rand(100)  # Global RNG advancement cannot affect seeded draws.
    assert torch.equal(
        first, sampling.multinomial_with_seed_cpu(logits, seeds, positions)
    )
    order = torch.tensor([7, 2, 0], device=device)
    subset = sampling.multinomial_with_seed_cpu(
        logits[order], seeds[order], positions[order]
    )
    assert torch.equal(first[order], subset)
    assert first.dtype == torch.int64 and first.shape == (8, 1)
    assert first.device.type == device.type


def test_seed_and_position_change_draws(device):
    logits = torch.zeros(64, 64, device=device)
    seeds = torch.arange(64, device=device)
    positions = torch.zeros(64, dtype=torch.long, device=device)
    first = sampling.multinomial_with_seed_cpu(logits, seeds, positions)
    assert not torch.equal(
        first, sampling.multinomial_with_seed_cpu(logits, seeds + 1, positions)
    )
    assert not torch.equal(
        first, sampling.multinomial_with_seed_cpu(logits, seeds, positions + 1)
    )


def test_distribution_and_masked_tokens(device):
    probs = torch.tensor([0.9, 0.1, 0.0], device=device)
    logits = probs.log().expand(10000, -1)
    seeds = torch.arange(10000, device=device)
    positions = torch.zeros_like(seeds)
    result = sampling.multinomial_with_seed_cpu(logits, seeds, positions).cpu()
    assert not (result == 2).any()
    assert 0.88 < (result == 0).float().mean().item() < 0.92


def test_hash_endpoints_do_not_select_masked_tokens(monkeypatch, device):
    monkeypatch.setattr(
        sampling,
        "_murmur_hash_cpu",
        lambda *args: torch.tensor([[0xFFFFFFFF, 0, 10], [0, 0xFFFFFFFF, 10]]),
    )
    logits = torch.tensor(
        [[float("-inf"), 0, float("-inf")], [0, float("-inf"), float("-inf")]],
        device=device,
    )
    result = sampling.multinomial_with_seed_cpu(
        logits,
        torch.zeros(2, dtype=torch.long, device=device),
        torch.zeros(2, dtype=torch.long, device=device),
    )
    assert result.cpu().tolist() == [[1], [0]]


def test_mps_matches_cpu_for_identical_logits(device):
    logits = torch.randn(16, 1026, generator=torch.Generator().manual_seed(7))
    seeds = torch.arange(16)
    positions = torch.arange(16) * 8
    expected = sampling.multinomial_with_seed_cpu(logits, seeds, positions)
    actual = sampling.multinomial_with_seed_cpu(
        logits.to(device), seeds.to(device), positions.to(device)
    )
    assert torch.equal(expected, actual.cpu())


@pytest.mark.parametrize(
    "logits,seeds,positions",
    [
        (torch.zeros(3), torch.zeros(3), torch.zeros(3)),
        (torch.zeros(2, 0), torch.zeros(2), torch.zeros(2)),
        (torch.zeros(2, 3), torch.zeros(1), torch.zeros(2)),
        (torch.zeros(2, 3), torch.zeros(2), torch.zeros(1)),
    ],
)
def test_invalid_shapes(logits, seeds, positions):
    with pytest.raises(ValueError):
        sampling.multinomial_with_seed_cpu(logits, seeds, positions)
