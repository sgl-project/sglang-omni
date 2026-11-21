# SPDX-License-Identifier: Apache-2.0
"""Request-seeded proposal and rejection sampling for the MiniCPM-o thinker."""

from __future__ import annotations

import math

import pytest
import torch

from sglang_omni.models.minicpm_o.speculative_sampling import (
    prepare_seeded_batch,
    sample_draft_block,
    verify_draft_block,
)
from sglang_omni.sampling.speculative_noise import (
    ACCEPTANCE_NOISE_SEED_SALT,
    DRAFT_NOISE_SEED_SALT,
    LARGEST_FLOAT32_BELOW_ONE,
    RESIDUAL_NOISE_SEED_SALT,
    seeded_exponential_noise,
    seeded_uniform_noise,
)

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)
DEVICES = ["cpu", pytest.param("cuda", marks=[pytest.mark.accelerator, CUDA_REQUIRED])]


def draft_noise(
    logits: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    _, draft_count, vocab_size = logits.shape
    return seeded_exponential_noise(
        seeds,
        positions,
        salt=DRAFT_NOISE_SEED_SALT,
        column_count=draft_count * vocab_size,
    ).view_as(logits)


def verification_noise(
    target: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        seeded_uniform_noise(
            seeds,
            positions,
            salt=ACCEPTANCE_NOISE_SEED_SALT,
            column_count=target.shape[1] - 1,
        )
        .float()
        .clamp_max(LARGEST_FLOAT32_BELOW_ONE),
        seeded_exponential_noise(
            seeds,
            positions,
            salt=RESIDUAL_NOISE_SEED_SALT,
            column_count=target.shape[2],
        ),
    )


def test_mixed_seed_state_follows_requests_and_preserves_zero_seed() -> None:
    positions = torch.tensor([10, 20, 30])
    zero_only = prepare_seeded_batch(["zero"], [0], positions[:1])
    assert zero_only[0].item() == 0
    original = prepare_seeded_batch(
        ["zero", "random", "fixed"], [0, None, 42], positions
    )
    reordered = prepare_seeded_batch(
        ["fixed", "zero", "random"], [42, 0, None], positions[[2, 0, 1]]
    )
    seeds, saved_positions = original
    assert seeds[0].item() == 0
    assert seeds[2].item() == 42
    assert 0 <= seeds[1].item() <= 2**31 - 1
    torch.testing.assert_close(seeds, reordered[0][[1, 2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(saved_positions, reordered[1][[1, 2, 0]], rtol=0, atol=0)


def test_each_seeded_round_keeps_its_resolved_sequence_positions() -> None:
    positions = torch.tensor([10, 20], dtype=torch.int32)
    first = prepare_seeded_batch(["first", "second"], [42, 99], positions)
    positions.add_(4)
    second = prepare_seeded_batch(["first", "second"], [42, 99], positions)
    assert first[1].dtype == second[1].dtype == torch.int64
    assert first[1].tolist() == [10, 20]
    assert second[1].tolist() == [14, 24]
    same_dtype_positions = torch.tensor([30, 40])
    third = prepare_seeded_batch(["first", "second"], [42, 99], same_dtype_positions)
    same_dtype_positions.add_(5)
    assert third[1].tolist() == [30, 40]


def test_batch_seeds_use_the_existing_positive_int32_convention() -> None:
    state = prepare_seeded_batch(
        ["negative", "large"], [-1, 2**31 + 42], torch.tensor([1, 1])
    )
    assert state[0].tolist() == [2**31 - 1, 42]


@pytest.mark.accelerator
@CUDA_REQUIRED
def test_noise_tracks_seed_position_and_domain_without_using_global_rng() -> None:
    seeds = torch.tensor([0, 42, 2**31 - 1], device="cuda")
    positions = torch.tensor([1, 10, 100], device="cuda")
    noise = seeded_uniform_noise(
        seeds, positions, salt=DRAFT_NOISE_SEED_SALT, column_count=128
    )
    with torch.random.fork_rng(devices=[seeds.device]):
        torch.rand(1000, device=seeds.device)
        repeated = seeded_uniform_noise(
            seeds, positions, salt=DRAFT_NOISE_SEED_SALT, column_count=128
        )
    torch.testing.assert_close(noise, repeated, rtol=0, atol=0)
    assert torch.all((noise > 0) & (noise < 1))
    for other_seeds, other_positions, salt in (
        (seeds, positions, ACCEPTANCE_NOISE_SEED_SALT),
        (seeds, positions, RESIDUAL_NOISE_SEED_SALT),
        (seeds + 1, positions, DRAFT_NOISE_SEED_SALT),
        (seeds, positions + 1, DRAFT_NOISE_SEED_SALT),
    ):
        changed = seeded_uniform_noise(
            other_seeds, other_positions, salt=salt, column_count=128
        )
        assert not torch.equal(noise, changed)


@pytest.mark.accelerator
@CUDA_REQUIRED
def test_a_complete_step_follows_requests_through_reordering_and_slot_reuse() -> None:
    device = "cuda"
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(3, 3, 17, generator=generator).to(device)
    embedding = torch.randn(17, 4, generator=generator).to(device)
    projection = torch.randn(17, 4, generator=generator).to(device)
    previous = torch.tensor([2, 4, 6], device=device)
    request_ids = ["first", "seeded", "unseeded"]
    request_seeds: list[int | None] = [11, 42, None]
    positions = torch.tensor([8, 12, 20], device=device)
    temperatures = torch.tensor([0.7, 0.8, 1.0], device=device)
    target = torch.softmax(torch.randn(3, 4, 17, generator=generator).to(device), -1)

    def step(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = rows.tolist()
        seeds, round_positions = prepare_seeded_batch(
            [request_ids[index] for index in indices],
            [request_seeds[index] for index in indices],
            positions[rows],
        )
        drafts, corrected = sample_draft_block(
            logits[rows],
            previous[rows],
            embedding,
            projection,
            temperatures[rows],
            draft_noise(logits[rows], seeds, round_positions),
        )
        accepted, bonus = verify_draft_block(
            target[rows],
            corrected,
            drafts,
            temperatures[rows],
            *verification_noise(target[rows], seeds, round_positions),
        )
        return drafts, accepted, bonus

    original = step(torch.tensor([0, 1, 2], device=device))
    reordered = step(torch.tensor([2, 0, 1], device=device))
    alone = step(torch.tensor([1], device=device))
    for first, permuted, single in zip(original, reordered, alone, strict=True):
        torch.testing.assert_close(first, permuted[[1, 2, 0]], rtol=0, atol=0)
        torch.testing.assert_close(first[1:2], single, rtol=0, atol=0)
    request_ids[0] = "replacement"
    request_seeds[0] = 12345
    positions[0] = 1
    replaced = step(torch.tensor([0, 1, 2], device=device))
    for first, current in zip(original, replaced, strict=True):
        torch.testing.assert_close(first[1:], current[1:], rtol=0, atol=0)


@pytest.mark.accelerator
@CUDA_REQUIRED
@pytest.mark.parametrize("changed_input", ["seed", "position"])
def test_round_inputs_reach_proposal_acceptance_and_bonus(changed_input: str) -> None:
    seeds = torch.arange(128, device="cuda")
    positions = torch.full_like(seeds, 8)
    proposal = torch.tensor([0.5, 0.1, 0.1, 0.2, 0.1], device="cuda")
    logits = proposal.log().expand(128, 2, 5)
    embedding = torch.zeros(5, 1, device="cuda")
    previous = torch.zeros(128, dtype=torch.long, device="cuda")
    temperatures = torch.ones(128, device="cuda")
    target = torch.tensor([0.1, 0.4, 0.3, 0.2, 0.0], device="cuda").expand(128, 3, 5)
    drafts, corrected = sample_draft_block(
        logits,
        previous,
        embedding,
        embedding,
        temperatures,
        draft_noise(logits, seeds, positions),
    )
    accepted, bonus = verify_draft_block(
        target,
        corrected,
        drafts,
        temperatures,
        *verification_noise(target, seeds, positions),
    )
    if changed_input == "seed":
        seeds = seeds + 1
    else:
        positions = positions + 1
    changed_drafts, _ = sample_draft_block(
        logits,
        previous,
        embedding,
        embedding,
        temperatures,
        draft_noise(logits, seeds, positions),
    )
    changed_acceptance, changed_bonus = verify_draft_block(
        target,
        corrected,
        drafts,
        temperatures,
        *verification_noise(target, seeds, positions),
    )
    assert not torch.equal(drafts, changed_drafts)
    assert not torch.equal(accepted, changed_acceptance)
    assert not torch.equal(bonus, changed_bonus)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_proposals_sample_the_returned_markov_distribution(
    device: str, dtype: torch.dtype
) -> None:
    samples = 40000
    generator = torch.Generator(device=device).manual_seed(42)
    base = (
        torch.tensor([0.2, 0.3, 0.5], device=device)
        .log()
        .to(dtype)
        .expand(samples, 2, 3)
    )
    embedding = torch.eye(3, dtype=dtype, device=device)
    transition = torch.tensor(
        [[0.0, 1.0, -1.0], [1.0, 0.0, 0.5], [-0.5, -1.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    drafts, corrected = sample_draft_block(
        base,
        torch.zeros(samples, dtype=torch.long, device=device),
        embedding,
        transition,
        torch.ones(samples, device=device),
        torch.empty(base.shape, device=device).exponential_(generator=generator),
    )
    for anchor in range(3):
        selected = drafts[:, 0] == anchor
        expected_logits = base[selected, 1] + transition[:, anchor]
        torch.testing.assert_close(corrected[selected, 1], expected_logits)
        count = int(selected.sum())
        observed = torch.bincount(drafts[selected, 1], minlength=3).float() / count
        expected = torch.softmax(expected_logits[0].float(), -1)
        torch.testing.assert_close(
            observed, expected, rtol=0, atol=6 * math.sqrt(0.25 / count)
        )


@pytest.mark.parametrize("device", DEVICES)
def test_rejection_preserves_target_distribution_and_acceptance_mass(
    device: str,
) -> None:
    samples = 40000
    generator = torch.Generator(device=device).manual_seed(42)
    target_row = torch.tensor([0.1, 0.4, 0.3, 0.2, 0.0], device=device)
    proposal_row = torch.tensor([0.5, 0.1, 0.1, 0.2, 0.1], device=device)
    logits = proposal_row.log().expand(samples, 2, 5)
    temperatures = torch.ones(samples, device=device)
    target = target_row.expand(samples, 3, 5)
    if device == "cuda":
        seeds = torch.arange(samples, device=device)
        positions = seeds.remainder(101) + 1
        proposals = draft_noise(logits, seeds, positions)
        uniforms, residuals = verification_noise(target, seeds, positions)
    else:
        proposals = torch.empty(logits.shape).exponential_(generator=generator)
        uniforms = torch.rand((samples, 2), generator=generator)
        residuals = torch.empty((samples, 5)).exponential_(generator=generator)
    drafts, corrected = sample_draft_block(
        logits,
        torch.zeros(samples, dtype=torch.long, device=device),
        torch.zeros(5, 1, device=device),
        torch.zeros(5, 1, device=device),
        temperatures,
        proposals,
    )
    accepted, bonus = verify_draft_block(
        target,
        corrected,
        drafts,
        temperatures,
        uniforms,
        residuals,
    )
    first = torch.where(accepted > 0, drafts[:, 0], bonus)
    frequencies = torch.bincount(first, minlength=5).float() / samples
    tolerance = 6 * math.sqrt(0.25 / samples)
    torch.testing.assert_close(frequencies, target_row, rtol=0, atol=tolerance)
    expected_acceptance = torch.minimum(target_row, proposal_row).sum()
    assert abs(float((accepted > 0).float().mean() - expected_acceptance)) < tolerance


@pytest.mark.parametrize("device", DEVICES)
def test_equal_distributions_accept_every_draft_and_sample_the_bonus(
    device: str,
) -> None:
    samples = 20000
    generator = torch.Generator(device=device).manual_seed(42)
    probabilities = torch.tensor([0.25, 0.75], device=device)
    logits = probabilities.log().expand(samples, 2, 2)
    temperatures = torch.ones(samples, device=device)
    drafts, corrected = sample_draft_block(
        logits,
        torch.zeros(samples, dtype=torch.long, device=device),
        torch.zeros(2, 1, device=device),
        torch.zeros(2, 1, device=device),
        temperatures,
        torch.empty(logits.shape, device=device).exponential_(generator=generator),
    )
    target = torch.cat(
        [torch.softmax(corrected, -1), probabilities.expand(samples, 1, 2)], 1
    )
    accepted, bonus = verify_draft_block(
        target,
        corrected,
        drafts,
        temperatures,
        torch.rand(drafts.shape, device=device, generator=generator),
        torch.empty((samples, logits.shape[-1]), device=device).exponential_(
            generator=generator
        ),
    )
    assert torch.all(accepted == 2)
    observed = torch.bincount(bonus, minlength=2).float() / samples
    torch.testing.assert_close(
        observed, probabilities, rtol=0, atol=6 * math.sqrt(0.25 / samples)
    )


def test_disjoint_distributions_reject_at_first_token_and_keep_greedy_rows() -> None:
    target = torch.tensor([[[0.0, 1.0]] * 3] * 2)
    logits = torch.tensor([[[0.0, -torch.inf]] * 2] * 2)
    drafts = torch.zeros(2, 2, dtype=torch.long)
    accepted, bonus = verify_draft_block(
        target,
        logits,
        drafts,
        torch.tensor([1.0, 0.0]),
        torch.full((2, 2), 0.5),
        torch.ones(2, 2),
    )
    assert accepted.tolist() == [0, 0]
    assert bonus.tolist() == [1, 1]


def test_partial_acceptance_samples_the_first_rejected_target_row() -> None:
    target = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    logits = torch.tensor(
        [[[0.0, -torch.inf, -torch.inf], [-torch.inf, 0.0, -torch.inf]]]
    )
    accepted, bonus = verify_draft_block(
        target,
        logits,
        torch.tensor([[0, 1]]),
        torch.ones(1),
        torch.full((1, 2), 0.5),
        torch.ones(1, 3),
    )
    assert accepted.tolist() == [1]
    assert bonus.tolist() == [2]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_reads_updated_request_seeds_and_positions() -> None:
    seeds = torch.tensor([42, 99], device="cuda")
    positions = torch.tensor([10, 20], device="cuda")
    logits = torch.zeros(2, 3, 32, device="cuda")
    embedding = torch.zeros(32, 2, device="cuda")
    projection = torch.zeros_like(embedding)
    previous = torch.zeros(2, dtype=torch.long, device="cuda")
    temperatures = torch.ones(2, device="cuda")
    probabilities = torch.arange(1, 33, dtype=torch.float32, device="cuda")
    target = (probabilities / probabilities.sum()).expand(2, 4, 32)

    def step() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        drafts, corrected = sample_draft_block(
            logits,
            previous,
            embedding,
            projection,
            temperatures,
            draft_noise(logits, seeds, positions),
        )
        accepted, bonus = verify_draft_block(
            target,
            corrected,
            drafts,
            temperatures,
            *verification_noise(target, seeds, positions),
        )
        return drafts, accepted, bonus

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    accepted_counts: list[list[int]] = []
    for new_position in (20, 30):
        positions.fill_(new_position)
        seeds.add_(1)
        eager = step()
        graph.replay()
        for expected, actual in zip(eager, captured, strict=True):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        accepted_counts.append(captured[1].tolist())
    assert accepted_counts[0] != accepted_counts[1]
    assert any(count < 3 for counts in accepted_counts for count in counts)
