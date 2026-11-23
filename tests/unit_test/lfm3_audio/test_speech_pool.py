# SPDX-License-Identifier: Apache-2.0
"""The device speech step against the host state machine it mirrors."""

from __future__ import annotations

from dataclasses import fields

import torch

from sglang_omni.models.lfm3_audio.depth import FrameSampling
from sglang_omni.models.lfm3_audio.generation import SpeechState
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens
from sglang_omni.models.lfm3_audio.speech_pool import (
    SpeechRows,
    SpeechStatePool,
    advance_speech,
)

TOKENS = Lfm3AudioTokens()
STOP_TOKEN_ID = TOKENS.eos_token_ids[0]
WINDOW = 3


def new_state(last_prompt_token: int, seed: int) -> SpeechState:
    return SpeechState.from_prompt(
        TOKENS,
        last_prompt_token,
        text_only=False,
        sampling=FrameSampling(
            temperature=0.8,
            top_k=5,
            repetition_penalty=1.2,
            repetition_window=WINDOW,
            generator=torch.Generator(),
            seed=seed,
        ),
    )


def host_step(state: SpeechState, sampled_id: int, codes: torch.Tensor) -> int:
    """One step of the host state machine, as the exact-sampler runner takes it; returns the id it emits."""
    was_interleaving = state.interleaving
    if state.needs_frame():
        state.apply_frame(codes)
        state.sampling.frames_sampled += 1
        token_id = TOKENS.audio_out_token_id
    else:
        state.apply_text(sampled_id, TOKENS.eos_token_ids)
        token_id = STOP_TOKEN_ID if state.finished else sampled_id
    state.advance_slot(was_interleaving)
    return token_id


def test_device_step_follows_the_host_state_machine() -> None:
    """Random text ids and frames through every mode, end-of-speech frames and finished turns; a finished slot takes a new request."""
    generator = torch.Generator().manual_seed(0)
    prompts = (
        TOKENS.audio_start_token_id,
        TOKENS.mixed_start_token_id,
        1000,
        TOKENS.audio_out_token_id,
    )
    slots = torch.tensor([2, 5, 1, 4])
    states = [new_state(token, seed) for seed, token in enumerate(prompts)]
    pool = SpeechStatePool(
        slot_count=5,
        hidden_size=2,
        window=WINDOW,
        num_codebooks=TOKENS.num_codebooks,
        embed_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    pool.load(slots, states)
    choices = torch.tensor(
        [
            1000,
            1001,
            1002,
            TOKENS.audio_start_token_id,
            TOKENS.mixed_start_token_id,
            TOKENS.text_end_token_id,
            TOKENS.eos_token_ids[0],
        ]
    )
    finished_turns = 0
    for step in range(400):
        sampled = choices[
            torch.randint(0, len(choices), (len(states),), generator=generator)
        ]
        codes = torch.randint(
            0, 8, (len(states), TOKENS.num_codebooks), generator=generator
        )
        eos_rows = torch.rand(len(states), generator=generator) < 0.05
        codes[eos_rows, 0] = TOKENS.audio_eos_code

        expected_ids = [
            host_step(state, int(token_id), frame)
            for state, token_id, frame in zip(states, sampled, codes, strict=True)
        ]
        speech_step = advance_speech(
            pool.gather(slots), sampled, codes, TOKENS, STOP_TOKEN_ID
        )
        pool.store(slots, speech_step.rows)

        assert speech_step.token_ids.tolist() == expected_ids, step
        mirrored = SpeechRows.from_states(states, WINDOW, TOKENS.num_codebooks)
        pooled = pool.gather(slots)
        for field in fields(SpeechRows):
            assert torch.equal(
                getattr(pooled, field.name), getattr(mirrored, field.name)
            ), (step, field.name)

        for row, state in enumerate(states):
            if state.finished:
                finished_turns += 1
                states[row] = new_state(prompts[row % len(prompts)], 100 + step)
                pool.load(slots[row : row + 1], [states[row]])
            else:
                pass
    assert finished_turns > 20
