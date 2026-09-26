# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o duplex token sampling and its mutable unit state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds


@dataclass
class DuplexSamplerState:
    """Mutable sampling history and policy for one generated unit."""

    special_tokens: MiniCPMOSpecialTokenIds
    generation_step: int = 0
    force_listen_count: int = 0
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    current_turn_ended: bool = True
    forbidden_token_ids: set[int] = field(default_factory=set)
    temperature: float = 0.7
    top_k: int = 100
    top_p: float = 0.8
    repetition_penalty: float = 1.05
    listen_prob_scale: float = 1.0
    greedy: bool = False


def top_k_top_p(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits.clone()
    if 0 < top_k < filtered.numel():
        threshold = torch.topk(filtered, top_k).values[-1]
        filtered[filtered < threshold] = -torch.inf
    else:
        pass
    if 0.0 < top_p < 1.0:
        values, indices = torch.sort(filtered, descending=True)
        cumulative = torch.cumsum(F.softmax(values, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        filtered[indices[remove]] = -torch.inf
    else:
        pass
    return filtered


def draw(logits: torch.Tensor, *, greedy: bool) -> int:
    if greedy:
        return int(torch.argmax(logits).item())
    else:
        probabilities = F.softmax(logits, dim=-1)
        if not torch.isfinite(probabilities).all() or float(probabilities.sum()) <= 0:
            raise RuntimeError(
                "MiniCPM-o duplex sampler produced invalid probabilities"
            )
        else:
            pass
        return int(torch.multinomial(probabilities, 1).item())


def duplex_sample(logits: torch.Tensor, state: DuplexSamplerState) -> int:
    """Apply the two-stage unit sampler to one vocabulary row."""

    if logits.ndim == 2:
        if logits.shape[0] != 1:
            raise ValueError("duplex_sample expects one logits row")
        else:
            pass
        logits = logits[0]
    else:
        pass
    if logits.ndim != 1:
        raise ValueError("duplex_sample expects logits shaped [V] or [1, V]")
    else:
        pass
    special = state.special_tokens

    if state.generation_step >= 19:
        return special.chunk_eos
    else:
        if (
            state.generation_step == 0
            and state.force_listen_counter < state.force_listen_count
        ):
            state.force_listen_counter += 1
            return special.listen
        else:
            row = logits.float().clone()
            # note (Junnan Li): The first draw must use the unscaled model distribution.
            if draw(row, greedy=state.greedy) == special.chunk_eos:
                return special.chunk_eos
            else:
                forbidden = {
                    special.chunk_eos,
                    *special.forbidden,
                    *state.forbidden_token_ids,
                }
                valid_forbidden = [
                    token for token in forbidden if 0 <= token < row.numel()
                ]
                if valid_forbidden:
                    row[valid_forbidden] = -torch.inf
                else:
                    pass

                penalty = float(state.repetition_penalty)
                if penalty <= 0:
                    raise ValueError("repetition_penalty must be positive")
                else:
                    pass
                if penalty != 1.0:
                    for token_id in set(state.generated_history[-512:]):
                        if 0 <= int(token_id) < row.numel():
                            # note (Junnan Li): Repetition scaling is deliberately sign-insensitive.
                            if penalty > 1.0:
                                row[int(token_id)] /= penalty
                            else:
                                row[int(token_id)] *= 1.0 / penalty
                        else:
                            pass
                else:
                    pass

                if state.listen_prob_scale != 1.0 and 0 <= special.listen < row.numel():
                    row[special.listen] *= float(state.listen_prob_scale)
                else:
                    pass

                if state.greedy or state.temperature <= 0:
                    candidate = int(torch.argmax(row).item())
                else:
                    filtered = top_k_top_p(
                        row / float(state.temperature),
                        top_k=int(state.top_k),
                        top_p=float(state.top_p),
                    )
                    candidate = draw(filtered, greedy=False)

                # note (Junnan Li): History retains controls before the mid-turn listen rewrite.
                state.generated_history.append(candidate)
                del state.generated_history[:-512]
                if candidate == special.listen and not state.current_turn_ended:
                    candidate = special.tts_bos
                else:
                    pass
                if candidate == special.turn_eos:
                    state.current_turn_ended = True
                elif candidate not in special.chunk_terminators:
                    state.current_turn_ended = False
                else:
                    pass
                return candidate
