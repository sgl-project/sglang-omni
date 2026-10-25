# SPDX-License-Identifier: Apache-2.0
"""Sampling math that keeps MiniCPM-o talker speculative decoding exact.

The talker shapes each step's logits before sampling (TalkerSlotState.apply): codec
suppress mask, a repetition penalty over the last 16 output tokens applied once per
occurrence, and the stop id masked until the minimum length; then temperature, top-k
and top-p. The draft draws each proposal from its own logits shaped the same way, and
the verify rebuilds the talker's distribution for every row of the block, whose
history includes the drafts before it, before rejection sampling against the draft.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.minicpm_o.talker_model_runner import REP_PENALTY_WINDOW


@dataclass(kw_only=True)
class TalkerSamplingState:
    """Per-request sampling inputs of one speculative step, rows in batch order.

    recent_tokens holds the last REP_PENALTY_WINDOW committed outputs in order, padded
    on the left with the vocabulary size (a bin no token falls in); generated counts
    every committed output. Greedy rows take the argmax of the shaped logits.
    """

    recent_tokens: torch.Tensor
    generated: torch.Tensor
    min_new_tokens: torch.Tensor
    penalties: torch.Tensor
    suppress: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    top_ps: torch.Tensor
    greedy: torch.Tensor
    eos_id: int


def window_counts(windows: torch.Tensor, vocab: int) -> torch.Tensor:
    """Occurrences of each token in each row's window; padding lands in a dropped bin."""
    counts = torch.zeros(
        windows.shape[0], vocab + 1, dtype=torch.float32, device=windows.device
    )
    counts.scatter_add_(1, windows, torch.ones_like(windows, dtype=torch.float32))
    return counts[:, :vocab]


def shape_logits(
    logits: torch.Tensor,
    *,
    suppress: torch.Tensor,
    counts: torch.Tensor,
    penalties: torch.Tensor,
    eos_masked: torch.Tensor,
    eos_id: int,
) -> torch.Tensor:
    """TalkerSlotState.apply as a pure function of [rows, vocab] logits."""
    logits = logits.float().masked_fill(suppress, float("-inf"))
    factors = penalties.unsqueeze(1) ** counts
    penalized = torch.where(logits < 0, logits * factors, logits / factors)
    logits = torch.where(counts > 0, penalized, logits)
    eos = torch.zeros_like(suppress)
    eos[:, eos_id] = eos_masked
    return logits.masked_fill(eos, float("-inf"))


def nucleus(
    logits: torch.Tensor,
    *,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    greedy: torch.Tensor,
) -> torch.Tensor:
    """The distribution SGLang's sampler draws from: tempered softmax, top-k, top-p."""
    tempered = torch.softmax(logits / temperatures.unsqueeze(1), dim=-1)
    sorted_probabilities, sorted_ids = tempered.sort(dim=-1, descending=True)
    ranks = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    kept = (ranks < top_ks.unsqueeze(1)) & (
        (sorted_probabilities.cumsum(dim=-1) - sorted_probabilities)
        <= top_ps.unsqueeze(1)
    )
    kept_probabilities = torch.where(kept, sorted_probabilities, 0.0)
    sampled = torch.zeros_like(tempered).scatter(
        1, sorted_ids, kept_probabilities / kept_probabilities.sum(-1, keepdim=True)
    )
    argmax = torch.zeros_like(tempered).scatter(
        1, logits.argmax(dim=-1, keepdim=True), 1.0
    )
    return torch.where(greedy.unsqueeze(1), argmax, sampled)


def sample_draft_block(
    base_logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    markov_embedding: torch.Tensor,
    markov_projection: torch.Tensor,
    state: TalkerSamplingState,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a draft block left to right, shaping each step as the talker would.

    base_logits is [batch, drafts, vocab]. Each step's window and length include the
    block's earlier drafts. Returns the tokens and the shaped logits they were drawn
    from at the request's temperature, which rejection sampling divides by.
    """
    vocab = base_logits.shape[2]
    history = state.recent_tokens
    draft_tokens: list[torch.Tensor] = []
    shaped_logits: list[torch.Tensor] = []
    for step in range(base_logits.shape[1]):
        step_logits = shape_logits(
            base_logits[:, step].float()
            + (markov_embedding[previous_tokens] @ markov_projection.T).float(),
            suppress=state.suppress,
            counts=window_counts(history[:, -REP_PENALTY_WINDOW:], vocab),
            penalties=state.penalties,
            eos_masked=(state.generated + step) < state.min_new_tokens,
            eos_id=state.eos_id,
        )
        sampled = torch.argmax(
            step_logits / state.temperatures.unsqueeze(1)
            - torch.log(exponential_noise[:, step]),
            dim=1,
        )
        next_tokens = torch.where(state.greedy, step_logits.argmax(dim=1), sampled)
        draft_tokens.append(next_tokens)
        shaped_logits.append(step_logits)
        history = torch.cat([history, next_tokens.unsqueeze(1)], dim=1)
        previous_tokens = next_tokens
    return torch.stack(draft_tokens, dim=1), torch.stack(shaped_logits, dim=1)


def target_probabilities(
    target_logits: torch.Tensor, draft_tokens: torch.Tensor, state: TalkerSamplingState
) -> torch.Tensor:
    """The talker's distribution for every verify row, [batch, drafts + 1, vocab].

    target_logits is [batch * (drafts + 1), vocab]; row j of a request follows its
    first j drafts, so its window and length include them.
    """
    batch, drafts = draft_tokens.shape
    rows = drafts + 1
    vocab = target_logits.shape[1]
    history = torch.cat([state.recent_tokens, draft_tokens], dim=1)
    windows = history.unfold(1, REP_PENALTY_WINDOW, 1).reshape(-1, REP_PENALTY_WINDOW)
    offsets = torch.arange(rows, device=draft_tokens.device).unsqueeze(0)
    eos_masked = (
        state.generated.unsqueeze(1) + offsets
    ) < state.min_new_tokens.unsqueeze(1)
    shaped = shape_logits(
        target_logits,
        suppress=state.suppress.repeat_interleave(rows, dim=0),
        counts=window_counts(windows, vocab),
        penalties=state.penalties.repeat_interleave(rows),
        eos_masked=eos_masked.reshape(-1),
        eos_id=state.eos_id,
    )
    # note (0xtoward): the plain talker rounds penalized logits back to the head's dtype.
    shaped = shaped.to(target_logits.dtype).float()
    return nucleus(
        shaped,
        temperatures=state.temperatures.repeat_interleave(rows),
        top_ks=state.top_ks.repeat_interleave(rows),
        top_ps=state.top_ps.repeat_interleave(rows),
        greedy=state.greedy.repeat_interleave(rows),
    ).view(batch, rows, vocab)


# note (0xtoward): kept eager like the Fun-CosyVoice3 verify; inductor can miscompile the
# residual draw fused with the gathers.
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

    The drafts were drawn from softmax(draft_logits / temperature). A draft is kept with
    probability min(1, p / q); the first rejection draws from max(0, p - q), and a fully
    accepted block draws its bonus token from the next row's p.
    """
    drafts = draft_tokens.shape[1]
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
    index = draft_tokens.long().unsqueeze(2)
    target_at_draft = target[:, :drafts].gather(2, index).squeeze(2)
    draft_at_draft = draft_probabilities.gather(2, index).squeeze(2)
    accepted = uniform_noise * draft_at_draft < target_at_draft
    accepted_count = accepted.long().cumprod(dim=1).sum(dim=1)
    row = accepted_count.view(-1, 1, 1).expand(-1, 1, target.shape[2])
    target_row = target.gather(1, row).squeeze(1)
    # The row after the last draft has no proposal, so its residual is the target.
    padded = torch.cat(
        [draft_probabilities, torch.zeros_like(draft_probabilities[:, :1])], dim=1
    )
    residual = (target_row - padded.gather(1, row).squeeze(1)).clamp_min(0)
    residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, target_row)
    # An exponential race picks a token with probability proportional to its weight.
    next_tokens = torch.argmax(residual / exponential_noise, dim=1)
    return accepted_count, next_tokens


def exact_accept(
    target_logits: torch.Tensor,
    shaped_draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    state: TalkerSamplingState,
    uniform_noise: torch.Tensor,
    exponential_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify one block per request against the talker's exact sampling."""
    draft_tokens = draft_tokens.long()
    return rejection_sample(
        target=target_probabilities(target_logits, draft_tokens, state),
        draft_logits=shaped_draft_logits,
        draft_tokens=draft_tokens,
        temperatures=state.temperatures,
        greedy=state.greedy,
        uniform_noise=uniform_noise,
        exponential_noise=exponential_noise,
    )
