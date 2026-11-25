# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni D2 Thinker: duplex tokens over a fixed cache and greedy text under the reply rules."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_omni_d2.components.decoder_graph import (
    DecoderGraph,
    causal_cache_mask,
    reset_cache,
    static_cache,
)
from sglang_omni.models.qwen3_omni_d2.components.mandarin import MandarinOpening
from sglang_omni.models.qwen3_omni_d2.components.model import Qwen3OmniD2Model

# Environment audio, delayed self audio and text each take one Thinker token per frame.
THINKER_TOKENS_PER_FRAME = 3


@dataclass(frozen=True, kw_only=True)
class TurnState:
    """Where the Thinker and Talker are in a reply after one text token."""

    is_responding: bool
    is_text_done: bool
    is_talker_responding: bool


@dataclass(frozen=True, kw_only=True)
class TurnTransition:
    """One text token's effect on the reply and on what the Talker does with it."""

    before: TurnState
    after: TurnState
    reads_text: bool
    emits_audio: bool


def advance_turn(
    state: TurnState, token_id: int, model: Qwen3OmniD2Model
) -> TurnTransition:
    is_response = token_id == model.response_token_id
    is_interrupt = token_id == model.interrupt_token_id
    assert not (is_response and state.is_responding)
    is_responding = (state.is_responding or is_response) and not is_interrupt
    if is_response:
        is_text_done = False
    elif state.is_responding and token_id == model.pad_token_id:
        is_text_done = True
    else:
        is_text_done = state.is_text_done
    return TurnTransition(
        before=state,
        after=TurnState(
            is_responding=is_responding,
            is_text_done=is_text_done or not is_responding,
            is_talker_responding=(state.is_talker_responding or is_response)
            and not is_interrupt,
        ),
        reads_text=state.is_talker_responding or is_response,
        emits_audio=state.is_talker_responding and not is_interrupt,
    )


def allowed_tokens(
    state: TurnState, previous_token_id: int, model: Qwen3OmniD2Model
) -> torch.Tensor:
    """Words only inside an unfinished reply; RESPONSE opens, INTERRUPT always available."""
    allowed = torch.full(
        (model.lm_head.out_features,),
        state.is_responding and not state.is_text_done,
        dtype=torch.bool,
        device=model.device,
    )
    allowed[model.pad_token_id] = not (
        state.is_responding and previous_token_id == model.response_token_id
    )
    allowed[model.response_token_id] = not state.is_responding
    allowed[model.interrupt_token_id] = True
    return allowed


class Qwen3OmniD2Thinker:
    """Append each interval's environment and self tokens, then one text token per frame."""

    def __init__(
        self,
        model: Qwen3OmniD2Model,
        capacity_frames: int,
        prompt_token_ids: list[int],
        opening: MandarinOpening | None,
    ) -> None:
        self.model = model
        self.interval_frames = model.interval_frames
        self.opening = opening
        self.prompt_token_ids = torch.tensor([prompt_token_ids], device=model.device)
        self.cache = static_cache(
            model.thinker, THINKER_TOKENS_PER_FRAME * capacity_frames
        )
        accepted_layer = model.thinker.layers[model.accept_hidden_layer - 1]
        audio_tokens = 2 * self.interval_frames
        if self.interval_frames == 1:
            # note (cuzimi): At 80 ms the previous token is known, so one causal forward covers env, self and text.
            self.fused_graph = DecoderGraph(
                model.thinker, self.cache, (4, 1, 3), 0, accepted_layer
            )
        else:
            self.audio_graph = DecoderGraph(
                model.thinker, self.cache, (4, 1, audio_tokens), 0
            )
            self.text_graph = DecoderGraph(
                model.thinker, self.cache, (4, 1, 1), audio_tokens, accepted_layer
            )

    def reset(self) -> None:
        """Empty the cache, prefill the system prompt and restart the text history."""
        model = self.model
        reset_cache(self.cache)
        prompt_tokens = self.prompt_token_ids.shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.thinker(
                inputs_embeds=model.thinker.embed_tokens(self.prompt_token_ids),
                attention_mask=causal_cache_mask(
                    0, prompt_tokens, self.cache.layers[0].max_cache_len, model.device
                ),
                position_ids=torch.arange(prompt_tokens, device=model.device).expand(
                    4, 1, -1
                ),
                past_key_values=self.cache,
                use_cache=True,
            )
        self.previous_token_id = model.pad_token_id
        self.generated_token_ids: list[int] = []

    def append_audio(self, audio_inputs: torch.Tensor, frames: int) -> None:
        """Add one interval of [1, 2 * interval_frames, hidden] environment then self tokens.

        At 80 ms the fused graph runs them together with the first text token.
        """
        prompt_tokens = self.prompt_token_ids.shape[1]
        self.cursor = prompt_tokens + THINKER_TOKENS_PER_FRAME * frames
        slots = torch.arange(2 * self.interval_frames, device=self.model.device)
        frame_positions = prompt_tokens + frames + slots % self.interval_frames
        self.audio_inputs = audio_inputs
        self.audio_positions = torch.stack(
            (self.cursor + slots, frame_positions, frame_positions, frame_positions)
        )[:, None]
        if self.interval_frames == 1:
            pass
        else:
            self.audio_graph.replay(audio_inputs, self.audio_positions, self.cursor)

    def decode(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Feed the previous token at text slot index; return final and accepted-layer states."""
        model = self.model
        text_cursor = self.cursor + 2 * self.interval_frames + index
        text_embedding = model.thinker.embed_tokens(
            torch.tensor([[self.previous_token_id]], device=model.device)
        )
        text_positions = torch.full(
            (4, 1, 1), text_cursor, dtype=torch.long, device=model.device
        )
        if self.interval_frames == 1:
            last_hidden = self.fused_graph.replay(
                torch.cat((self.audio_inputs, text_embedding), dim=1),
                torch.cat((self.audio_positions, text_positions), dim=2),
                self.cursor,
            )
            accepted_hidden = self.fused_graph.hooked_states
        else:
            last_hidden = self.text_graph.replay(
                text_embedding, text_positions, text_cursor
            )
            accepted_hidden = self.text_graph.hooked_states
        return last_hidden[:, -1], accepted_hidden[:, -1:]

    def text_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Vocabulary logits for [1, hidden] states with the trained PAD and control rows."""
        model = self.model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.lm_head(hidden_states)
            pad_logits = F.linear(hidden_states, model.pad_output_row)
            control_logits = F.linear(hidden_states, model.control_output_delta)
            logits = logits.scatter(-1, model.pad_token_index, pad_logits)
            logits = logits.scatter_add(-1, model.control_token_index, control_logits)
        return logits.masked_fill(model.invalid_vocabulary_mask, float("-inf"))

    def next_token(self, hidden_states: torch.Tensor, turn: TurnState) -> int:
        """Greedy text token under the reply rules, after the optional opening-word policy."""
        logits = self.text_logits(hidden_states).masked_fill(
            ~allowed_tokens(turn, self.previous_token_id, self.model), float("-inf")
        )
        if self.opening is not None:
            logits = self.opening.mask(logits, self.generated_token_ids)
        else:
            pass
        token_id = int(logits.argmax(dim=-1).item())
        self.previous_token_id = token_id
        self.generated_token_ids.append(token_id)
        return token_id
