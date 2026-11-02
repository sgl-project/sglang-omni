# SPDX-License-Identifier: Apache-2.0
"""Sampling math that keeps Fun-CosyVoice3 speculative decoding exact.

The draft draws each proposal from logits shaped as the stage shapes them, and the
verify rebuilds the stage's per-position distribution, including the repetition-aware
redraw, before rejection sampling against the draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.models.fun_cosyvoice3.model_runner import COSYVOICE3_RAS_WINDOW_SIZE
from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE, VOCAB_SIZE

# Pads a request's repetition window before it has that many output tokens.
EMPTY_HISTORY_TOKEN = -1


@dataclass(kw_only=True)
class DraftSamplingState:
    """Per-request sampling inputs one speculative step mirrors on the draft.

    Rows follow the running batch. presence_mask marks every committed output token
    (the anchor included); output_lengths counts them. Greedy rows take the argmax.
    """

    presence_mask: torch.Tensor
    repetition_penalties: torch.Tensor
    output_lengths: torch.Tensor
    min_new_tokens: torch.Tensor
    temperatures: torch.Tensor
    greedy: torch.Tensor


class DraftBlockSampler(Protocol):
    def __call__(
        self,
        base_logits: torch.Tensor,
        previous_tokens: torch.Tensor,
        exponential_noise: torch.Tensor,
        markov_embedding: torch.Tensor,
        markov_projection: torch.Tensor,
        presence_mask: torch.Tensor,
        repetition_penalties: torch.Tensor,
        output_lengths: torch.Tensor,
        min_new_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        greedy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def shape_logits(
    logits: torch.Tensor,
    *,
    presence_mask: torch.Tensor,
    stop_suppressed: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """Mask stop ids on suppressed rows and penalize tokens already in the history."""
    logits = torch.cat(
        [
            logits[:, :VOCAB_SIZE],
            torch.where(
                stop_suppressed.unsqueeze(1), float("-inf"), logits[:, VOCAB_SIZE:]
            ),
        ],
        dim=1,
    )
    penalties = repetition_penalties.unsqueeze(1)
    return torch.where(
        presence_mask,
        torch.where(logits < 0, logits * penalties, logits / penalties),
        logits,
    )


def sample_draft_block(
    base_logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    exponential_noise: torch.Tensor,
    markov_embedding: torch.Tensor,
    markov_projection: torch.Tensor,
    presence_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_lengths: torch.Tensor,
    min_new_tokens: torch.Tensor,
    temperatures: torch.Tensor,
    greedy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a draft block left to right, shaping each step as the stage would.

    Each step's history includes the block's earlier drafts. Returns the tokens and the
    shaped logits they were drawn from, which rejection sampling divides by.
    """
    draft_tokens: list[torch.Tensor] = []
    shaped_logits: list[torch.Tensor] = []
    for step_index in range(base_logits.shape[1]):
        step_logits = shape_logits(
            base_logits[:, step_index]
            + (markov_embedding[previous_tokens] @ markov_projection.T).float(),
            presence_mask=presence_mask,
            stop_suppressed=(output_lengths + step_index) < min_new_tokens,
            repetition_penalties=repetition_penalties,
        )
        sampled = torch.argmax(
            step_logits / temperatures.unsqueeze(1)
            - torch.log(exponential_noise[:, step_index]),
            dim=1,
        )
        next_tokens = torch.where(greedy, torch.argmax(step_logits, dim=1), sampled)
        draft_tokens.append(next_tokens)
        shaped_logits.append(step_logits)
        presence_mask = presence_mask.scatter(1, next_tokens.unsqueeze(1), True)
        previous_tokens = next_tokens
    return torch.stack(draft_tokens, dim=1), torch.stack(shaped_logits, dim=1)


def target_probabilities(
    logits: torch.Tensor,
    *,
    presence_mask: torch.Tensor,
    stop_suppressed: torch.Tensor,
    repetition_penalties: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    recent_tokens: torch.Tensor,
) -> torch.Tensor:
    """The distribution the stage samples each row's next token from.

    A row with top_k 1 is greedy; every other row redraws a sampled speech token found
    in recent_tokens from the tempered distribution without it.
    """
    logits = shape_logits(
        logits.float(),
        presence_mask=presence_mask,
        stop_suppressed=stop_suppressed,
        repetition_penalties=repetition_penalties,
    )
    greedy = top_ks == 1
    greedy_probabilities = torch.zeros_like(logits).scatter(
        1, logits.argmax(dim=-1, keepdim=True), 1.0
    )
    tempered = torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)
    sorted_probabilities, sorted_ids = tempered.sort(dim=-1, descending=True)
    cumulative = sorted_probabilities.cumsum(dim=-1)
    ranks = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    kept = (ranks < top_ks.unsqueeze(1)) & (
        (cumulative - sorted_probabilities) <= top_ps.unsqueeze(1)
    )
    nucleus_sorted = torch.where(kept, sorted_probabilities, 0.0)
    nucleus = torch.zeros_like(tempered).scatter(
        1, sorted_ids, nucleus_sorted / nucleus_sorted.sum(dim=-1, keepdim=True)
    )

    # Distinct speech ids of the window; repeats and padding contribute nothing.
    sorted_recent, _ = recent_tokens.sort(dim=-1)
    is_repeat = torch.cat(
        [
            torch.zeros_like(sorted_recent[:, :1], dtype=torch.bool),
            sorted_recent[:, 1:] == sorted_recent[:, :-1],
        ],
        dim=1,
    )
    valid = (
        (sorted_recent != EMPTY_HISTORY_TOKEN)
        & (sorted_recent < VOCAB_SIZE)
        & ~is_repeat
    )
    window_ids = sorted_recent.clamp_min(0)
    tempered_double = tempered.double()
    nucleus_double = nucleus.double()
    window_tempered = torch.where(valid, tempered_double.gather(1, window_ids), 0.0)
    window_nucleus = torch.where(valid, nucleus_double.gather(1, window_ids), 0.0)
    in_window = (
        torch.zeros_like(tempered_double).scatter_add(1, window_ids, valid.double()) > 0
    )
    # note (wilsonzheng0327): the redraw normalizes by the mass left after zeroing the
    # candidate, and keeps the candidate only when nothing is left. Summing the mass
    # outside the window directly avoids cancelling it against a near-1 candidate.
    outside_window = torch.where(in_window, 0.0, tempered_double).sum(
        dim=-1, keepdim=True
    )
    other_window_slots = ~torch.eye(
        window_ids.shape[1], dtype=torch.bool, device=window_ids.device
    )
    remaining = outside_window + (
        window_tempered.unsqueeze(1) * other_window_slots
    ).sum(dim=-1)
    redraw_weight = torch.where(remaining > 0, window_nucleus / remaining, 0.0)
    kept_weight = torch.where(remaining > 0, 0.0, window_nucleus)
    redrawn = torch.where(in_window, 0.0, nucleus_double) + tempered_double * (
        redraw_weight.sum(dim=-1, keepdim=True)
    )
    redrawn = redrawn.scatter_add(
        1, window_ids, kept_weight - redraw_weight * window_tempered
    )
    return torch.where(greedy.unsqueeze(1), greedy_probabilities, redrawn.float())


# note (wilsonzheng0327): inductor (torch 2.13) emits an invalid Triton kernel when it
# fuses the residual draw with the gathers, so this part stays eager.
@torch.compiler.disable
def rejection_sample(
    *,
    target: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    temperatures: torch.Tensor,
    greedy: torch.Tensor,
    uniform_noise: torch.Tensor,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept a prefix of the drafts and sample the token that follows it.

    target is [batch, drafts + 1, vocab]; the drafts were drawn from draft_logits at the
    same temperatures. uniform_noise and exponential_noise are the Uniform and Exp(1) draws.
    """
    draft_count = draft_tokens.shape[1]
    draft_probabilities = torch.softmax(
        draft_logits.float() / temperatures.view(-1, 1, 1), dim=-1
    )
    draft_probabilities = torch.where(
        greedy.view(-1, 1, 1),
        torch.zeros_like(draft_probabilities).scatter(
            2, draft_tokens.unsqueeze(2), 1.0
        ),
        draft_probabilities,
    )
    token_index = draft_tokens.long().unsqueeze(2)
    target_at_draft = target[:, :draft_count].gather(2, token_index).squeeze(2)
    draft_at_draft = draft_probabilities.gather(2, token_index).squeeze(2)
    accepted = uniform_noise * draft_at_draft < target_at_draft
    accepted_count = accepted.long().cumprod(dim=1).sum(dim=1)
    row_index = accepted_count.view(-1, 1, 1).expand(-1, 1, target.shape[2])
    target_row = target.gather(1, row_index).squeeze(1)
    # The row after the last draft has no proposal, so its residual is the target.
    padded_draft = torch.cat(
        [draft_probabilities, torch.zeros_like(draft_probabilities[:, :1])], dim=1
    )
    residual = (target_row - padded_draft.gather(1, row_index).squeeze(1)).clamp_min(0)
    residual = torch.where(residual.sum(dim=-1, keepdim=True) > 0, residual, target_row)
    # An exponential race picks a token with probability proportional to its weight.
    next_tokens = torch.argmax(residual / exponential_noise, dim=1)
    return accepted_count, next_tokens


def exact_accept(
    target_logits: torch.Tensor,
    shaped_draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    presence_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
    output_lengths: torch.Tensor,
    min_new_tokens: torch.Tensor,
    temperatures: torch.Tensor,
    greedy: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    recent_tokens: torch.Tensor,
    uniform_noise: torch.Tensor,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify one block per request against the stage's exact sampling.

    target_logits is [batch * (drafts + 1), vocab]; a request's row j follows j drafts,
    so its history and minimum-length state include them.
    """
    batch_size, draft_count = draft_tokens.shape
    rows_per_request = draft_count + 1
    row_offsets = torch.arange(rows_per_request, device=draft_tokens.device)
    follows_draft = row_offsets.view(-1, 1) > torch.arange(
        draft_count, device=draft_tokens.device
    ).view(1, -1)
    # Drafts a row does not follow scatter into one extra column, dropped below.
    drafted = torch.zeros(
        (batch_size, rows_per_request, TOTAL_VOCAB_SIZE + 1),
        dtype=torch.bool,
        device=draft_tokens.device,
    ).scatter(
        2,
        torch.where(
            follows_draft.unsqueeze(0), draft_tokens.unsqueeze(1), TOTAL_VOCAB_SIZE
        ),
        True,
    )
    row_presence = presence_mask.unsqueeze(1) | drafted[:, :, :TOTAL_VOCAB_SIZE]
    history = torch.cat([recent_tokens, draft_tokens], dim=1)
    row_recent = history.unfold(1, COSYVOICE3_RAS_WINDOW_SIZE, 1)
    stop_suppressed = (
        output_lengths.unsqueeze(1) + row_offsets.unsqueeze(0)
    ) < min_new_tokens.unsqueeze(1)
    target = target_probabilities(
        target_logits,
        presence_mask=row_presence.reshape(-1, TOTAL_VOCAB_SIZE),
        stop_suppressed=stop_suppressed.reshape(-1),
        repetition_penalties=repetition_penalties.repeat_interleave(rows_per_request),
        temperatures=temperatures.repeat_interleave(rows_per_request),
        top_ks=top_ks.repeat_interleave(rows_per_request),
        top_ps=top_ps.repeat_interleave(rows_per_request),
        recent_tokens=row_recent.reshape(-1, COSYVOICE3_RAS_WINDOW_SIZE),
    ).view(batch_size, rows_per_request, TOTAL_VOCAB_SIZE)
    return rejection_sample(
        target=target,
        draft_logits=shaped_draft_logits,
        draft_tokens=draft_tokens,
        temperatures=temperatures,
        greedy=greedy,
        uniform_noise=uniform_noise,
        exponential_noise=exponential_noise,
    )


class ExactAccept(Protocol):
    def __call__(
        self,
        target_logits: torch.Tensor,
        shaped_draft_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        presence_mask: torch.Tensor,
        repetition_penalties: torch.Tensor,
        output_lengths: torch.Tensor,
        min_new_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        greedy: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        recent_tokens: torch.Tensor,
        uniform_noise: torch.Tensor,
        exponential_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
