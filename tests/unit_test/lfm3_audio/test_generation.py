# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the LFM3-Audio speech state machine, frame sampling and prompt helpers."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.lfm3_audio.depth import FrameSampling, sample_code
from sglang_omni.models.lfm3_audio.generation import GenerationMode, SpeechState
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens

TOKENS = Lfm3AudioTokens()
TEXT_ID = 1000


def frame(code: int = 5) -> torch.Tensor:
    return torch.full((TOKENS.num_codebooks,), code, dtype=torch.long)


def state_for(last_prompt_token: int, *, text_only: bool = False) -> SpeechState:
    return SpeechState.from_prompt(
        TOKENS, last_prompt_token, text_only=text_only, sampling=FrameSampling()
    )


def step(state: SpeechState, text_id: int = TEXT_ID) -> str:
    """One generation step as the model runner takes it; returns what the step produced."""
    was_interleaving = state.interleaving
    if state.needs_frame():
        state.apply_frame(frame())
        kind = "frame"
    else:
        state.apply_text(text_id, TOKENS.eos_token_ids)
        kind = "text"
    state.advance_slot(was_interleaving)
    return kind


@pytest.mark.parametrize(
    ("last_token", "text_only", "mode"),
    [
        (TOKENS.audio_start_token_id, False, GenerationMode.AUDIO),
        (TOKENS.mixed_start_token_id, False, GenerationMode.INTERLEAVED),
        (TEXT_ID, False, GenerationMode.TEXT),
        (TOKENS.mixed_start_token_id, True, GenerationMode.TEXT),
    ],
)
def test_prompt_marker_sets_the_mode(last_token, text_only, mode) -> None:
    assert state_for(last_token, text_only=text_only).mode is mode


def test_interleaved_alternates_6_text_and_12_frames() -> None:
    state = state_for(TOKENS.mixed_start_token_id)
    kinds = [step(state) for _ in range(36)]
    assert kinds == (["text"] * 6 + ["frame"] * 12) * 2
    assert state.text_out.count(TOKENS.audio_out_token_id) == 24
    assert len(state.frames) == 24


def test_text_end_in_interleaved_speaks_until_the_eos_frame() -> None:
    state = state_for(TOKENS.mixed_start_token_id)
    step(state)
    step(state, TOKENS.text_end_token_id)
    assert state.text_done and state.needs_frame()
    for _ in range(30):
        assert step(state) == "frame"
    feed, emitted = state.apply_frame(
        frame(TOKENS.audio_eos_code - 1).index_fill(
            0, torch.tensor([3]), TOKENS.audio_eos_code
        )
    )
    assert not emitted
    assert bool((feed == TOKENS.audio_eos_code).all())
    assert state.mode is GenerationMode.TEXT and not state.needs_frame()
    assert len(state.frames) == 30


def test_text_end_outside_interleaved_ends_the_turn() -> None:
    state = state_for(TEXT_ID)
    step(state, TOKENS.text_end_token_id)
    assert state.finished and state.ended_on_text_end


def test_markers_inside_an_interleaved_block_stay_text() -> None:
    state = state_for(TOKENS.mixed_start_token_id)
    step(state, TOKENS.audio_start_token_id)
    assert state.mode is GenerationMode.INTERLEAVED


def test_text_only_rows_never_sample_frames() -> None:
    state = state_for(TOKENS.audio_start_token_id, text_only=True)
    assert [step(state) for _ in range(4)] == ["text"] * 4


def test_maskgit_schedule_covers_every_codebook() -> None:
    assert TOKENS.maskgit_schedule == [4, 4, 4]


def test_greedy_and_non_positive_top_k_sampling() -> None:
    logits = torch.randn(TOKENS.codebook_size)
    greedy = FrameSampling(temperature=0.0, top_k=128)
    assert int(sample_code(logits, greedy, greedy.warpers(), None)) == int(
        logits.argmax()
    )
    unbounded = FrameSampling(
        temperature=1.0, top_k=-1, generator=torch.Generator().manual_seed(0)
    )
    code = sample_code(logits, unbounded, unbounded.warpers(), None)
    assert 0 <= int(code) < TOKENS.codebook_size


def test_repetition_penalty_lowers_recent_codes() -> None:
    logits = torch.zeros(TOKENS.codebook_size)
    logits[7] = 1.0
    logits[8] = 0.9
    sampling = FrameSampling(temperature=0.0, repetition_penalty=2.0)
    history = torch.tensor([7])
    assert int(sample_code(logits, sampling, sampling.warpers(), history)) == 8
