# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling over a batch of thinker sessions."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState


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


def filter_top_k_top_p(
    logits: torch.Tensor, *, top_k: int, top_p: float
) -> torch.Tensor:
    """Mask every row of a (rows, vocab) batch outside its top-k and its nucleus."""
    filtered_logits = logits.clone()
    if 0 < top_k < filtered_logits.shape[-1]:
        thresholds = torch.topk(filtered_logits, top_k, dim=-1).values[:, -1:]
        filtered_logits.masked_fill_(filtered_logits < thresholds, -torch.inf)
    else:
        pass
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits, descending=True, dim=-1
        )
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        should_remove = cumulative_probabilities > top_p
        should_remove[:, 1:] = should_remove[:, :-1].clone()
        should_remove[:, 0] = False
        filtered_logits.scatter_(
            -1, sorted_indices, sorted_logits.masked_fill(should_remove, -torch.inf)
        )
    else:
        pass
    return filtered_logits


def draw_token_ids(logits: torch.Tensor) -> torch.Tensor:
    """One categorical draw per row from softmax(logits), with no device sync."""
    probabilities = F.softmax(logits, dim=-1)
    # note (ruoyu): The argmax of probability over an exponential draw is a categorical sample; torch.multinomial syncs twice per call.
    return torch.argmax(
        probabilities / torch.empty_like(probabilities).exponential_(), dim=-1
    )


def sample_rows(logits: torch.Tensor, greedy_flags: list[bool]) -> torch.Tensor:
    """One token id per row: the argmax for greedy rows, a softmax draw for the rest."""
    if all(greedy_flags):
        return torch.argmax(logits, dim=-1)
    elif not any(greedy_flags):
        return draw_token_ids(logits)
    else:
        token_ids = torch.argmax(logits, dim=-1)
        sampled_rows = torch.tensor(
            [row for row, is_greedy in enumerate(greedy_flags) if not is_greedy],
            dtype=torch.long,
            device=logits.device,
        )
        token_ids[sampled_rows] = draw_token_ids(logits[sampled_rows])
        return token_ids


def duplex_sample_batch(
    logits: torch.Tensor,
    thinker_states: list[MiniCPMOThinkerSessionState],
    *,
    special_tokens: MiniCPMOSpecialTokenIds,
    forbidden_token_index: torch.Tensor,
    generation_steps: list[int],
    listen_forced_flags: list[bool],
) -> torch.Tensor:
    """Apply the two-stage unit sampler with one final token-id transfer."""
    device = logits.device
    decided_token_ids: dict[int, int] = {}
    undecided_rows: list[int] = []
    for row, thinker_state in enumerate(thinker_states):
        sampling = thinker_state.sampling
        if generation_steps[row] >= sampling.max_new_tokens_per_unit - 1:
            decided_token_ids[row] = special_tokens.chunk_eos
        elif generation_steps[row] == 0 and listen_forced_flags[row]:
            decided_token_ids[row] = special_tokens.listen
        else:
            undecided_rows.append(row)
    if not undecided_rows:
        return torch.tensor(
            [decided_token_ids[row] for row in range(len(thinker_states))],
            dtype=torch.long,
            device=device,
        )
    else:
        pass

    undecided_logits = logits[
        torch.tensor(undecided_rows, dtype=torch.long, device=device)
    ].float()
    # note (Junnan Li): The first sample must use the unscaled model distribution.
    # Keep the first-stage decision on the device.  The previous implementation
    # called ``.tolist()`` here to find the rows that were not chunk EOS, then
    # called it a second time for the final sample.  Both calls synchronize the
    # scheduler thread with the GPU.  A device mask lets the second-stage
    # sampler skip EOS rows without transferring the intermediate token ids.
    first_token_ids = sample_rows(
        undecided_logits,
        [thinker_states[row].sampling.greedy for row in undecided_rows],
    )
    first_is_chunk_eos = first_token_ids == special_tokens.chunk_eos

    # Apply the second-stage transforms to the complete device batch.  Rows
    # which selected chunk EOS are masked out by ``sample_candidates`` before
    # any random draw, so their presence does not consume RNG state or change
    # the distribution of the remaining rows.
    second_stage_states = [thinker_states[row] for row in undecided_rows]
    second_stage_logits = undecided_logits
    second_stage_logits[:, forbidden_token_index] = -torch.inf
    apply_repetition_penalty(second_stage_logits, second_stage_states)

    listen_scales = [state.sampling.listen_prob_scale for state in second_stage_states]
    if any(scale != 1.0 for scale in listen_scales):
        second_stage_logits[:, special_tokens.listen] *= torch.tensor(
            listen_scales, dtype=torch.float32, device=device
        )
    else:
        pass

    candidate_token_ids = sample_candidates(
        second_stage_logits, second_stage_states, active_mask=~first_is_chunk_eos
    )
    final_token_ids = torch.where(
        first_is_chunk_eos,
        torch.full_like(candidate_token_ids, special_tokens.chunk_eos),
        candidate_token_ids,
    )
    # Keep the model-facing result on the device.  The CPU copy is used only
    # for the Python session bookkeeping below; the runner returns the device
    # tensor directly instead of constructing a second GPU tensor from a list.
    final_token_ids_cpu = final_token_ids.cpu().tolist()
    output_token_ids = torch.empty(len(thinker_states), dtype=torch.long, device=device)
    if decided_token_ids:
        decided_rows = torch.tensor(
            list(decided_token_ids), dtype=torch.long, device=device
        )
        decided_values = torch.tensor(
            list(decided_token_ids.values()), dtype=torch.long, device=device
        )
        output_token_ids.index_copy_(0, decided_rows, decided_values)
    else:
        pass
    undecided_index = torch.tensor(undecided_rows, dtype=torch.long, device=device)
    output_token_ids.index_copy_(0, undecided_index, final_token_ids)
    rewrite_positions: list[int] = []
    for candidate_position, candidate_token_id in enumerate(final_token_ids_cpu):
        row = undecided_rows[candidate_position]
        if candidate_token_id == special_tokens.chunk_eos:
            continue
        else:
            pass
        thinker_state = thinker_states[row]
        sampling = thinker_state.sampling
        # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
        history = thinker_state.generated_history
        history.append(candidate_token_id)
        del history[: -sampling.repetition_window_size]
        if (
            candidate_token_id == special_tokens.listen
            and not thinker_state.is_turn_ended
        ):
            candidate_token_id = special_tokens.tts_bos
            rewrite_positions.append(candidate_position)
        else:
            pass
        if candidate_token_id == special_tokens.turn_eos:
            thinker_state.is_turn_ended = True
        elif candidate_token_id not in special_tokens.chunk_terminators:
            thinker_state.is_turn_ended = False
        else:
            pass
    if rewrite_positions:
        rewrite_index = torch.tensor(rewrite_positions, dtype=torch.long, device=device)
        output_token_ids[undecided_index[rewrite_index]] = special_tokens.tts_bos
    else:
        pass
    return output_token_ids


def apply_repetition_penalty(
    logits: torch.Tensor, thinker_states: list[MiniCPMOThinkerSessionState]
) -> None:
    """Divide each row's recent token logits by its session penalty, in place."""
    penalty_windows: list[list[int]] = []
    for thinker_state in thinker_states:
        sampling = thinker_state.sampling
        history = thinker_state.generated_history
        if sampling.repetition_penalty != 1.0 and history:
            penalty_windows.append(
                sorted(set(history[-sampling.repetition_window_size :]))
            )
        else:
            penalty_windows.append([])
    window_width = max(len(window) for window in penalty_windows)
    if window_width == 0:
        return
    else:
        pass
    vocab_size = logits.shape[-1]
    penalties = torch.tensor(
        [[state.sampling.repetition_penalty] for state in thinker_states],
        dtype=torch.float32,
        device=logits.device,
    )
    # note (ruoyu): A spare vocabulary column absorbs the padding of shorter windows.
    padded_index = torch.tensor(
        [
            window + [vocab_size] * (window_width - len(window))
            for window in penalty_windows
        ],
        dtype=torch.long,
        device=logits.device,
    )
    widened_logits = torch.cat([logits, logits.new_zeros(logits.shape[0], 1)], dim=-1)
    # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
    widened_logits.scatter_(
        -1, padded_index, widened_logits.gather(-1, padded_index) / penalties
    )
    logits.copy_(widened_logits[:, :vocab_size])


def sample_candidates(
    logits: torch.Tensor,
    thinker_states: list[MiniCPMOThinkerSessionState],
    *,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample the second-stage token, grouping filtered draws by settings.

    ``active_mask`` keeps rows whose first-stage sample did not terminate on
    device.  Inactive rows still receive an argmax placeholder, but are never
    passed to a stochastic draw.  This is used by the duplex sampler to avoid
    a host round-trip between its two sampling stages.
    """
    if active_mask is not None:
        if active_mask.ndim != 1 or active_mask.shape[0] != logits.shape[0]:
            raise ValueError(
                "active_mask must be a one-dimensional mask aligned with logits"
            )
        else:
            pass
        active_mask = active_mask.to(device=logits.device, dtype=torch.bool)
    else:
        pass
    token_ids = torch.argmax(logits, dim=-1)
    sampled_groups: dict[tuple[float, int, float], list[int]] = {}
    for row, thinker_state in enumerate(thinker_states):
        sampling = thinker_state.sampling
        if sampling.greedy or sampling.temperature <= 0:
            pass
        else:
            sampled_groups.setdefault(
                (sampling.temperature, sampling.top_k, sampling.top_p), []
            ).append(row)
    for (temperature, top_k, top_p), group_rows in sampled_groups.items():
        row_index = torch.tensor(group_rows, dtype=torch.long, device=logits.device)
        if active_mask is not None:
            row_index = row_index[active_mask[row_index]]
        else:
            pass
        if row_index.numel() == 0:
            continue
        else:
            pass
        filtered_logits = filter_top_k_top_p(
            logits[row_index] / temperature, top_k=top_k, top_p=top_p
        )
        token_ids[row_index] = draw_token_ids(filtered_logits)
    return token_ids
