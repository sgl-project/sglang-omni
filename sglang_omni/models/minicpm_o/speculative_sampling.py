# SPDX-License-Identifier: Apache-2.0
"""Request-seeded sampling for the thinker's single-device VanillaMarkov draft."""

from __future__ import annotations

import torch
from torch.nn import functional

from sglang_omni.sampling.seed import derive_sampling_seed, resolve_row_seed


def prepare_seeded_batch(
    request_ids: list[str],
    request_seeds: list[int | None],
    sequence_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage seeds and resolved positions for a batch selected for seeded decoding."""
    assert sequence_positions.shape == (len(request_seeds),)
    row_seeds = [
        (
            resolve_row_seed(seed)
            if seed is not None
            else derive_sampling_seed("sglang-omni-unseeded-row", request_id)
        )
        for request_id, seed in zip(request_ids, request_seeds, strict=True)
    ]
    seeds = torch.tensor(row_seeds, dtype=torch.int64, device=sequence_positions.device)
    # note (IcyDesert): overlap may update the source lengths before this round's verifier consumes them.
    positions = sequence_positions.to(dtype=torch.int64, copy=True)
    return seeds, positions


def sample_draft_block(
    base_logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    markov_embedding: torch.Tensor,
    markov_projection: torch.Tensor,
    temperatures: torch.Tensor,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sampled candidates and the rounded logits they were drawn from."""
    batch_size, draft_count, vocab_size = base_logits.shape

    assert temperatures.shape == (batch_size,)
    assert exponential_noise.shape == base_logits.shape
    assert previous_tokens.shape == (batch_size,)
    assert markov_projection.shape[0] == vocab_size

    greedy = temperatures == 0
    safe_temperatures = torch.where(greedy, 1.0, temperatures).view(-1, 1)

    step_dtype = torch.promote_types(base_logits.dtype, markov_embedding.dtype)

    drafted_tokens: list[torch.Tensor] = []
    corrected_logits: list[torch.Tensor] = []
    for step in range(draft_count):
        bias = functional.linear(markov_embedding[previous_tokens], markov_projection)
        logits = (base_logits[:, step].float() + bias.float()).to(step_dtype)

        sampled = (
            logits.float() / safe_temperatures - exponential_noise[:, step].log()
        ).argmax(-1)
        previous_tokens = torch.where(greedy, logits.argmax(-1), sampled)
        drafted_tokens.append(previous_tokens)
        corrected_logits.append(logits)
    return torch.stack(drafted_tokens, 1), torch.stack(corrected_logits, 1)


def verify_draft_block(
    target_probabilities: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,
    uniform_noise: torch.Tensor,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept a candidate prefix and draw the residual or full-acceptance bonus."""
    batch_size, draft_count = draft_tokens.shape
    vocab_size = target_probabilities.shape[-1]
    assert target_probabilities.shape == (batch_size, draft_count + 1, vocab_size)
    assert draft_logits.shape == (batch_size, draft_count, vocab_size)
    assert temperatures.shape == (batch_size,)
    assert uniform_noise.shape == draft_tokens.shape
    assert exponential_noise.shape == (batch_size, vocab_size)
    greedy = temperatures == 0
    safe_temperatures = torch.where(greedy, 1.0, temperatures).view(-1, 1, 1)
    proposal = torch.softmax(draft_logits.float() / safe_temperatures, -1)
    selected = draft_tokens.unsqueeze(-1)
    accepts = uniform_noise * proposal.gather(2, selected).squeeze(
        -1
    ) < target_probabilities[:, :draft_count].gather(2, selected).squeeze(-1)
    accepts = torch.where(
        greedy.view(-1, 1),
        draft_tokens == target_probabilities[:, :draft_count].argmax(-1),
        accepts,
    )
    accepted_count = accepts.long().cumprod(1).sum(1)
    row_indices = accepted_count.view(-1, 1, 1).expand(-1, 1, vocab_size)
    target_row = target_probabilities.gather(1, row_indices).squeeze(1)
    # note (IcyDesert): appending a zero row for full acceptance copies the entire proposal tensor each round, which can be optimized.
    padded_proposal = torch.cat([proposal, torch.zeros_like(proposal[:, :1])], 1)
    residual = (
        target_row - padded_proposal.gather(1, row_indices).squeeze(1)
    ).clamp_min(0)
    residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, target_row)
    bonus = (residual / exponential_noise).argmax(-1)
    return accepted_count, torch.where(greedy, target_row.argmax(-1), bonus)
