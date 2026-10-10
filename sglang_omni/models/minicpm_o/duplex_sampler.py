# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling over one session's thinker state."""

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
    filtered_logits = logits.clone()
    if 0 < top_k < filtered_logits.numel():
        threshold = torch.topk(filtered_logits, top_k).values[-1]
        filtered_logits[filtered_logits < threshold] = -torch.inf
    else:
        pass
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        should_remove = cumulative_probabilities > top_p
        should_remove[1:] = should_remove[:-1].clone()
        should_remove[0] = False
        filtered_logits[sorted_indices[should_remove]] = -torch.inf
    else:
        pass
    return filtered_logits


def sample_token_id(logits: torch.Tensor, *, greedy: bool) -> int:
    if greedy:
        return int(torch.argmax(logits).item())
    else:
        return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


def duplex_sample(
    logits: torch.Tensor,
    thinker_state: MiniCPMOThinkerSessionState,
    *,
    special_tokens: MiniCPMOSpecialTokenIds,
    forbidden_token_index: torch.Tensor,
    generation_step: int,
    is_listen_forced: bool,
) -> int:
    """Apply the two-stage unit sampler to one vocabulary row."""
    sampling = thinker_state.sampling
    if generation_step >= sampling.max_new_tokens_per_unit - 1:
        return special_tokens.chunk_eos
    elif generation_step == 0 and is_listen_forced:
        return special_tokens.listen
    else:
        logits_row = logits.float().clone()
        # note (Junnan Li): The first sample must use the unscaled model distribution.
        if sample_token_id(logits_row, greedy=sampling.greedy) == (
            special_tokens.chunk_eos
        ):
            return special_tokens.chunk_eos
        else:
            logits_row[forbidden_token_index] = -torch.inf

            penalty = sampling.repetition_penalty
            history = thinker_state.generated_history
            if penalty != 1.0 and history:
                repeated_token_index = torch.tensor(
                    sorted(set(history[-sampling.repetition_window_size :])),
                    dtype=torch.long,
                    device=logits_row.device,
                )
                # note (Junnan Li): Matches the checkpoint sampler, which ignores the logit sign.
                logits_row[repeated_token_index] /= penalty
            else:
                pass

            if sampling.listen_prob_scale != 1.0:
                logits_row[special_tokens.listen] *= sampling.listen_prob_scale
            else:
                pass

            if sampling.greedy or sampling.temperature <= 0:
                candidate_token_id = int(torch.argmax(logits_row).item())
            else:
                filtered_logits = filter_top_k_top_p(
                    logits_row / sampling.temperature,
                    top_k=sampling.top_k,
                    top_p=sampling.top_p,
                )
                candidate_token_id = sample_token_id(filtered_logits, greedy=False)

            # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
            history.append(candidate_token_id)
            del history[: -sampling.repetition_window_size]
            if (
                candidate_token_id == special_tokens.listen
                and not thinker_state.is_turn_ended
            ):
                candidate_token_id = special_tokens.tts_bos
            else:
                pass
            if candidate_token_id == special_tokens.turn_eos:
                thinker_state.is_turn_ended = True
            elif candidate_token_id not in special_tokens.chunk_terminators:
                thinker_state.is_turn_ended = False
            else:
                pass
            return candidate_token_id
