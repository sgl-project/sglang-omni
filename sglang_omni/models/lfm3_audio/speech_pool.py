# SPDX-License-Identifier: Apache-2.0
"""Device copy of each request's speech state, so a seeded decode step needs no host sync.

The pool is indexed by req_pool_indices; row 0 is never a request's. Each step gathers the batch's rows, advances them as SpeechState does (generation.py), and stores them back with the next decode input embedding. The host SpeechState replays the same step one resolve later and checks that it agrees.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch

from sglang_omni.models.lfm3_audio.generation import GenerationMode, SpeechState
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens

# Columns of a packed step (pack_step): the emitted id, the sampled id, whether the row took a frame, the fed codes, then the advanced state (state_columns).
EMITTED_ID_COLUMN = 0
SAMPLED_ID_COLUMN = 1
TOOK_FRAME_COLUMN = 2
CODES_COLUMN = 3


@dataclass(kw_only=True)
class SpeechRows:
    """Speech state of a set of requests, one row each."""

    mode: torch.Tensor
    speaking: torch.Tensor
    text_done: torch.Tensor
    finished: torch.Tensor
    slots_left: torch.Tensor
    frames_sampled: torch.Tensor
    seed: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    repetition_penalty: torch.Tensor
    # The last repetition_window frames, the oldest repeated to fill the window; history_len counts the real ones.
    history: torch.Tensor
    history_len: torch.Tensor

    @classmethod
    def from_states(
        cls, states: list[SpeechState], window: int, num_codebooks: int
    ) -> SpeechRows:
        """Host rows of the given states."""
        histories = []
        for state in states:
            history = state.sampling.history
            if history is None:
                histories.append(torch.zeros(window, num_codebooks, dtype=torch.long))
            else:
                history = history.cpu()
                padding = history[:1].expand(window - history.shape[0], -1)
                histories.append(torch.cat([padding, history]))
        return cls(
            mode=torch.tensor([int(state.mode) for state in states]),
            speaking=torch.tensor([state.speaking for state in states]),
            text_done=torch.tensor([state.text_done for state in states]),
            finished=torch.tensor([state.finished for state in states]),
            slots_left=torch.tensor([state.slots_left for state in states]),
            frames_sampled=torch.tensor(
                [state.sampling.frames_sampled for state in states]
            ),
            seed=torch.tensor([state.sampling.seed for state in states]),
            temperature=torch.tensor(
                [float(state.sampling.temperature) for state in states]
            ),
            top_k=torch.tensor([state.sampling.top_k or 0 for state in states]),
            repetition_penalty=torch.tensor(
                [float(state.sampling.repetition_penalty) for state in states]
            ),
            history=torch.stack(histories),
            history_len=torch.tensor(
                [
                    (
                        0
                        if state.sampling.history is None
                        else state.sampling.history.shape[0]
                    )
                    for state in states
                ]
            ),
        )


class SpeechStatePool:
    """Speech state and next decode input of every request slot."""

    def __init__(
        self,
        *,
        slot_count: int,
        hidden_size: int,
        window: int,
        num_codebooks: int,
        embed_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        rows = slot_count + 1
        self.window = window
        self.num_codebooks = num_codebooks
        self.next_input_embeds = torch.zeros(
            rows, hidden_size, dtype=embed_dtype, device=device
        )
        self.state = SpeechRows(
            mode=torch.zeros(rows, dtype=torch.long, device=device),
            speaking=torch.zeros(rows, dtype=torch.bool, device=device),
            text_done=torch.zeros(rows, dtype=torch.bool, device=device),
            finished=torch.zeros(rows, dtype=torch.bool, device=device),
            slots_left=torch.zeros(rows, dtype=torch.long, device=device),
            frames_sampled=torch.zeros(rows, dtype=torch.long, device=device),
            seed=torch.zeros(rows, dtype=torch.long, device=device),
            temperature=torch.ones(rows, dtype=torch.float32, device=device),
            top_k=torch.zeros(rows, dtype=torch.long, device=device),
            repetition_penalty=torch.ones(rows, dtype=torch.float32, device=device),
            history=torch.zeros(
                rows, window, num_codebooks, dtype=torch.long, device=device
            ),
            history_len=torch.zeros(rows, dtype=torch.long, device=device),
        )

    def load(self, slots: torch.Tensor, states: list[SpeechState]) -> None:
        """Write the host states into their slots, as a prefill starts or replays a request."""
        self.store(
            slots,
            SpeechRows.from_states(states, self.window, self.num_codebooks),
        )

    def gather(self, slots: torch.Tensor) -> SpeechRows:
        return SpeechRows(
            **{
                field.name: getattr(self.state, field.name).index_select(0, slots)
                for field in fields(SpeechRows)
            }
        )

    def store(self, slots: torch.Tensor, rows: SpeechRows) -> None:
        for field in fields(SpeechRows):
            pooled = getattr(self.state, field.name)
            pooled.index_copy_(
                0, slots, getattr(rows, field.name).to(pooled.device, pooled.dtype)
            )


@dataclass(kw_only=True)
class SpeechStep:
    """One step of a batch: its new state, the ids it emits and the codes it feeds back."""

    rows: SpeechRows
    # A frame row emits audio_out; a text row its sampled token, or the stop id once the turn ends.
    token_ids: torch.Tensor
    needs_frame: torch.Tensor
    # The sampled frame, all end-of-speech codes for an end-of-speech frame.
    fed_codes: torch.Tensor


def needs_frame_mask(rows: SpeechRows) -> torch.Tensor:
    """Rows that take a frame this step (SpeechState.needs_frame)."""
    audio = rows.mode == GenerationMode.AUDIO
    interleaving = rows.mode == GenerationMode.INTERLEAVED
    return ~rows.finished & (audio | (interleaving & rows.speaking))


def advance_speech(
    rows: SpeechRows,
    sampled_ids: torch.Tensor,
    codes: torch.Tensor,
    tokens: Lfm3AudioTokens,
    stop_token_id: int,
) -> SpeechStep:
    """Advance every row one step, as SpeechState.apply_frame / apply_text / advance_slot do, without a host sync.

    codes holds a frame for every row; only the speaking rows use theirs.
    """
    text_mode = GenerationMode.TEXT
    interleaved = GenerationMode.INTERLEAVED
    needs_frame = needs_frame_mask(rows)
    was_interleaving = rows.mode == interleaved

    # note (ruilin): a frame row: an end-of-speech frame returns to text, any other joins the repetition history.
    eos_frame = needs_frame & (codes == tokens.audio_eos_code).any(-1)
    fed_codes = torch.where(
        eos_frame[:, None], torch.full_like(codes, tokens.audio_eos_code), codes
    )
    appended = needs_frame & ~eos_frame
    shifted = torch.cat([rows.history[:, 1:], codes[:, None]], dim=1)
    filled = codes[:, None].expand_as(rows.history)
    history = torch.where(
        appended[:, None, None],
        torch.where((rows.history_len == 0)[:, None, None], filled, shifted),
        rows.history,
    )
    history_len = torch.where(
        appended,
        (rows.history_len + 1).clamp(max=rows.history.shape[1]),
        rows.history_len,
    )
    mode = torch.where(eos_frame, text_mode, rows.mode)

    # note (ruilin): a text row: text_end ends the text of an interleaved block and the turn otherwise; an end id ends the turn; outside a block, mode markers open one.
    text = ~needs_frame
    is_text_end = sampled_ids == tokens.text_end_token_id
    # note (ruilin): compared one id at a time, since a device tensor built from the host ids would block on the stream.
    is_end_id = torch.zeros_like(is_text_end)
    for end_id in tokens.eos_token_ids:
        is_end_id |= sampled_ids == end_id
    ends_block_text = text & is_text_end & was_interleaving
    ends_turn = text & ((is_text_end & ~was_interleaving) | (~is_text_end & is_end_id))
    opens = text & ~is_text_end & ~is_end_id & ~was_interleaving
    opens_audio = opens & (sampled_ids == tokens.audio_start_token_id)
    opens_block = opens & (sampled_ids == tokens.mixed_start_token_id)
    mode = torch.where(
        opens_audio,
        GenerationMode.AUDIO,
        torch.where(opens_block, interleaved, mode),
    )
    speaking = torch.where(
        ends_block_text, True, torch.where(opens_block, False, rows.speaking)
    )
    text_done = torch.where(
        ends_block_text, True, torch.where(opens_block, False, rows.text_done)
    )
    slots_left = torch.where(opens_block, tokens.interleaved_n_text, rows.slots_left)
    finished = rows.finished | ends_turn

    counted = was_interleaving & ~text_done
    slots_left = slots_left - counted.long()
    flips = counted & (slots_left == 0)
    speaking = speaking ^ flips
    slots_left = torch.where(
        flips,
        torch.where(speaking, tokens.interleaved_n_audio, tokens.interleaved_n_text),
        slots_left,
    )

    token_ids = torch.where(
        needs_frame,
        tokens.audio_out_token_id,
        torch.where(text & finished, stop_token_id, sampled_ids),
    ).to(sampled_ids.dtype)
    return SpeechStep(
        rows=replace(
            rows,
            mode=mode,
            speaking=speaking,
            text_done=text_done,
            finished=finished,
            slots_left=slots_left,
            frames_sampled=rows.frames_sampled + needs_frame.long(),
            history=history,
            history_len=history_len,
        ),
        token_ids=token_ids,
        needs_frame=needs_frame,
        fed_codes=fed_codes,
    )


def pack_step(step: SpeechStep, sampled_ids: torch.Tensor) -> torch.Tensor:
    """One int64 row per request for the host: the step's ids and frame, then its advanced state (state_columns)."""
    rows = step.rows
    return torch.cat(
        [
            step.token_ids[:, None].long(),
            sampled_ids[:, None].long(),
            step.needs_frame[:, None].long(),
            step.fed_codes,
            torch.stack(
                [
                    rows.mode,
                    rows.speaking.long(),
                    rows.text_done.long(),
                    rows.finished.long(),
                    rows.slots_left,
                    rows.frames_sampled,
                    rows.history_len,
                ],
                dim=1,
            ),
        ],
        dim=1,
    )


def state_columns(num_codebooks: int) -> slice:
    """The packed step's advanced-state columns, in host_state order."""
    return slice(CODES_COLUMN + num_codebooks, None)


def host_state(state: SpeechState) -> list[int]:
    """A host SpeechState as the advanced-state columns of a packed step."""
    history = state.sampling.history
    return [
        int(state.mode),
        int(state.speaking),
        int(state.text_done),
        int(state.finished),
        state.slots_left,
        state.sampling.frames_sampled,
        0 if history is None else history.shape[0],
    ]
