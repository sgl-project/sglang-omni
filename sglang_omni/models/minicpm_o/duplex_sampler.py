# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling over the thinker states of one batch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState

# note (0xtoward): candidates past the largest top-k keep the tokens tied with a row's k-th logit; a longer tie run draws from the whole vocabulary.
CANDIDATES_PAST_TOP_K = 16


@dataclass(kw_only=True, frozen=True)
class DuplexSampleRow:
    """One batch row: its session's thinker state and where its unit stands."""

    thinker_state: MiniCPMOThinkerSessionState
    generation_step: int
    is_listen_forced: bool


def build_forbidden_token_index(
    special_tokens: MiniCPMOSpecialTokenIds, vocab_size: int, device: torch.device
) -> torch.Tensor:
    """Rows the second-stage sample never picks, resolved once instead of per step."""
    forbidden_token_ids = sorted(
        token_id
        for token_id in {special_tokens.chunk_eos, *special_tokens.forbidden}
        if token_id < vocab_size
    )
    return torch.tensor(forbidden_token_ids, dtype=torch.long, device=device)


def to_device(
    values: list[int] | list[float], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Copy host values to the device without waiting for the work already queued there."""
    host_values = torch.tensor(values, dtype=dtype)
    if device.type == "cuda":
        return host_values.pin_memory().to(device, non_blocking=True)
    else:
        return host_values.to(device)


def mask_beyond_top_p(sorted_logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Drop each descending row's tail past its top-p mass; top_p outside (0, 1) keeps the row."""
    cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    top_p_limits = torch.where((top_p > 0.0) & (top_p < 1.0), top_p, torch.inf)
    should_remove = cumulative_probabilities > top_p_limits.unsqueeze(1)
    should_remove[:, 1:] = should_remove[:, :-1].clone()
    should_remove[:, 0] = False
    return sorted_logits.masked_fill(should_remove, -torch.inf)


def filter_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    max_top_k: int,
    has_top_p: bool,
) -> torch.Tensor:
    """Apply each row's top-k and top-p; values outside (0, vocab) or (0, 1) leave the row alone.

    max_top_k and has_top_p are host-side summaries of top_k and top_p, so disabled filters launch nothing.
    """
    vocab_size = logits.shape[-1]
    filtered_logits = logits.clone()
    if max_top_k > 0:
        has_top_k = (top_k > 0) & (top_k < vocab_size)
        kept_counts = torch.where(has_top_k, top_k, 1)
        thresholds = torch.topk(filtered_logits, max_top_k, dim=-1).values.gather(
            1, (kept_counts - 1).unsqueeze(1)
        )
        filtered_logits.masked_fill_(
            has_top_k.unsqueeze(1) & (filtered_logits < thresholds), -torch.inf
        )
    else:
        pass
    if has_top_p:
        sorted_logits, sorted_token_ids = torch.sort(
            filtered_logits, descending=True, dim=-1
        )
        filtered_logits.scatter_(
            1, sorted_token_ids, mask_beyond_top_p(sorted_logits, top_p)
        )
    else:
        pass
    return filtered_logits


def sample_filtered_vocabulary(
    scaled_logits: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    max_top_k: int,
    has_top_p: bool,
) -> torch.Tensor:
    """Draw one token per row from its whole vocabulary after its top-k and top-p."""
    filtered_logits = filter_top_k_top_p(
        scaled_logits,
        top_k=top_k,
        top_p=top_p,
        max_top_k=max_top_k,
        has_top_p=has_top_p,
    )
    return torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)[:, 0]


def duplex_sample(
    logits: torch.Tensor,
    rows: Sequence[DuplexSampleRow],
    *,
    special_tokens: MiniCPMOSpecialTokenIds,
    forbidden_token_index: torch.Tensor,
) -> list[int]:
    """Apply the two-stage unit sampler to every row of a batch with one device read."""
    token_ids = [-1] * len(rows)
    sampled_row_indices: list[int] = []
    for index, row in enumerate(rows):
        sampling = row.thinker_state.sampling
        if row.generation_step >= sampling.max_new_tokens_per_unit - 1:
            token_ids[index] = special_tokens.chunk_eos
        elif row.generation_step == 0 and row.is_listen_forced:
            token_ids[index] = special_tokens.listen
        else:
            sampled_row_indices.append(index)
    if not sampled_row_indices:
        return token_ids
    else:
        pass
    sampled_states = [rows[index].thinker_state for index in sampled_row_indices]
    sampling_settings = [state.sampling for state in sampled_states]
    sampled_count = len(sampled_row_indices)
    vocab_size = logits.shape[-1]
    penalty_entries = [
        (position, token_id, state.sampling.repetition_penalty)
        for position, state in enumerate(sampled_states)
        if state.sampling.repetition_penalty != 1.0
        for token_id in set(
            state.generated_history[-state.sampling.repetition_window_size :]
        )
    ]
    penalty_count = len(penalty_entries)
    top_k_settings = [sampling.top_k for sampling in sampling_settings]
    top_p_settings = [sampling.top_p for sampling in sampling_settings]
    # note (0xtoward): one copy per dtype carries every row's settings, so the device never waits on a host read.
    integer_settings = to_device(
        sampled_row_indices
        + top_k_settings
        + [position for position, _, _ in penalty_entries]
        + [token_id for _, token_id, _ in penalty_entries],
        torch.long,
        logits.device,
    )
    float_settings = to_device(
        [float(sampling.greedy) for sampling in sampling_settings]
        + [
            float(sampling.greedy or sampling.temperature <= 0)
            for sampling in sampling_settings
        ]
        + [
            sampling.temperature if sampling.temperature > 0 else 1.0
            for sampling in sampling_settings
        ]
        + [sampling.listen_prob_scale for sampling in sampling_settings]
        + top_p_settings
        + [penalty for _, _, penalty in penalty_entries],
        torch.float32,
        logits.device,
    )
    row_indices, top_k_values, penalty_positions, penalty_token_ids = (
        integer_settings.split(
            [sampled_count, sampled_count, penalty_count, penalty_count]
        )
    )
    (
        first_stage_greedy,
        second_stage_greedy,
        temperatures,
        listen_scales,
        top_p_values,
        penalties,
    ) = float_settings.split([sampled_count] * 5 + [penalty_count])
    row_logits = logits.index_select(0, row_indices).float()
    # note (Junnan Li): The first sample must use the unscaled model distribution.
    first_token_ids = torch.where(
        first_stage_greedy > 0,
        row_logits.argmax(dim=-1),
        torch.multinomial(F.softmax(row_logits, dim=-1), 1)[:, 0],
    )
    row_logits[:, forbidden_token_index] = -torch.inf
    if penalty_entries:
        # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
        row_logits[penalty_positions, penalty_token_ids] /= penalties
    else:
        pass
    row_logits[:, special_tokens.listen] *= listen_scales
    scaled_logits = row_logits / temperatures.unsqueeze(1)
    max_top_k = max(
        (top_k for top_k in top_k_settings if 0 < top_k < vocab_size), default=0
    )
    has_top_p = any(0.0 < top_p < 1.0 for top_p in top_p_settings)
    if max_top_k > 0 and all(0 < top_k < vocab_size for top_k in top_k_settings):
        # note (0xtoward): a row keeps every token at or above its k-th logit, so top-p and the draw need only the leading candidates.
        candidate_count = min(max_top_k + CANDIDATES_PAST_TOP_K, vocab_size)
        candidate_logits, candidate_token_ids = torch.topk(
            scaled_logits, candidate_count, dim=-1
        )
        thresholds = candidate_logits.gather(1, (top_k_values - 1).unsqueeze(1))
        # note (0xtoward): a tie run that reaches the last candidate may continue past it.
        is_tie_cut = (candidate_logits[:, -1] >= thresholds[:, 0]) & (
            candidate_count < vocab_size
        )
        candidate_logits = candidate_logits.masked_fill(
            candidate_logits < thresholds, -torch.inf
        )
        if has_top_p:
            candidate_logits = mask_beyond_top_p(candidate_logits, top_p_values)
        else:
            pass
        sampled_token_ids = candidate_token_ids.gather(
            1, torch.multinomial(F.softmax(candidate_logits, dim=-1), 1)
        )[:, 0]
    else:
        is_tie_cut = torch.zeros_like(first_token_ids, dtype=torch.bool)
        sampled_token_ids = sample_filtered_vocabulary(
            scaled_logits, top_k_values, top_p_values, max_top_k, has_top_p
        )
    second_token_ids = torch.where(
        second_stage_greedy > 0, row_logits.argmax(dim=-1), sampled_token_ids
    )
    first_picks, second_picks, tie_cut_flags = torch.stack(
        (first_token_ids, second_token_ids, is_tie_cut.long())
    ).tolist()
    if any(tie_cut_flags):
        second_picks = torch.where(
            second_stage_greedy > 0,
            row_logits.argmax(dim=-1),
            sample_filtered_vocabulary(
                scaled_logits, top_k_values, top_p_values, max_top_k, has_top_p
            ),
        ).tolist()
    else:
        pass
    for index, state, first_pick, second_pick in zip(
        sampled_row_indices, sampled_states, first_picks, second_picks, strict=True
    ):
        if first_pick == special_tokens.chunk_eos:
            token_ids[index] = special_tokens.chunk_eos
        else:
            # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
            history = state.generated_history
            history.append(second_pick)
            del history[: -state.sampling.repetition_window_size]
            if second_pick == special_tokens.listen and not state.is_turn_ended:
                token_id = special_tokens.tts_bos
            else:
                token_id = second_pick
            if token_id == special_tokens.turn_eos:
                state.is_turn_ended = True
            elif token_id not in special_tokens.chunk_terminators:
                state.is_turn_ended = False
            else:
                pass
            token_ids[index] = token_id
    return token_ids
