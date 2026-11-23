# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the LFM3 speech state machine and frame sampling."""

from __future__ import annotations

import pytest
import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from sglang_omni.models.lfm3_audio.depth import (
    FrameSampling,
    sample_frames,
    seeded_frame_codes,
)
from sglang_omni.models.lfm3_audio.generation import GenerationMode, SpeechState
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens

TOKENS = Lfm3AudioTokens()
TEXT_ID = 1000


def sampling(
    temperature: float = 0.0,
    top_k: int | None = None,
    repetition_penalty: float = 1.0,
    seed: int = 0,
) -> FrameSampling:
    return FrameSampling(
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=4,
        generator=torch.Generator().manual_seed(seed),
        seed=seed,
    )


def frame(code: int = 5) -> torch.Tensor:
    return torch.full((TOKENS.num_codebooks,), code, dtype=torch.long)


def state_for(last_prompt_token: int, *, text_only: bool = False) -> SpeechState:
    return SpeechState.from_prompt(
        TOKENS, last_prompt_token, text_only=text_only, sampling=sampling()
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
def test_prompt_marker_sets_the_mode(
    last_token: int, text_only: bool, mode: GenerationMode
) -> None:
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
    eos_frame = frame().index_fill(0, torch.tensor([3]), TOKENS.audio_eos_code)
    fed_codes = state.apply_frame(eos_frame)
    assert bool((fed_codes == TOKENS.audio_eos_code).all())
    assert state.mode is GenerationMode.TEXT and not state.needs_frame()
    assert len(state.frames) == 30


def test_text_end_outside_interleaved_ends_the_turn() -> None:
    state = state_for(TEXT_ID)
    step(state, TOKENS.text_end_token_id)
    assert state.finished


def test_markers_inside_an_interleaved_block_stay_text() -> None:
    state = state_for(TOKENS.mixed_start_token_id)
    step(state, TOKENS.audio_start_token_id)
    assert state.mode is GenerationMode.INTERLEAVED


def test_text_only_rows_never_sample_frames() -> None:
    state = state_for(TOKENS.audio_start_token_id, text_only=True)
    assert [step(state) for _ in range(4)] == ["text"] * 4


def test_maskgit_schedule_covers_every_codebook() -> None:
    assert TOKENS.maskgit_schedule == [4, 4, 4]


class PassLogits:
    """Depth logits that depend on the pass inputs, so each pass scores differently."""

    def __init__(self, rows: int) -> None:
        generator = torch.Generator().manual_seed(1)
        self.base = torch.randn(
            rows, TOKENS.num_codebooks, TOKENS.codebook_size, generator=generator
        )

    def __call__(
        self, hidden_states: torch.Tensor, codes: torch.Tensor, masked: torch.Tensor
    ) -> torch.Tensor:
        del hidden_states
        return self.base + codes[..., None] * 1e-3 + masked[..., None] * 0.5


def reference_frames(
    depth_logits: PassLogits, samplings: list[FrameSampling]
) -> torch.Tensor:
    """The checkpoint's generate: one draw per codebook, row by row, in reveal order."""
    rows = len(samplings)
    codes = torch.zeros((rows, TOKENS.num_codebooks), dtype=torch.long)
    revealed = torch.zeros_like(codes, dtype=torch.bool)
    reveal_order = torch.stack(
        [
            torch.randperm(TOKENS.num_codebooks, generator=frame_sampling.generator)
            for frame_sampling in samplings
        ]
    )
    for revealed_now in reveal_order.split(TOKENS.maskgit_schedule, dim=1):
        logits = depth_logits(None, codes, ~revealed)
        for row, frame_sampling in enumerate(samplings):
            warpers = frame_sampling.warpers()
            for codebook in revealed_now[row].tolist():
                scores = logits[row, codebook].float()[None].clone()
                history = frame_sampling.history
                if frame_sampling.repetition_penalty != 1.0 and history is not None:
                    scores = RepetitionPenaltyLogitsProcessor(
                        frame_sampling.repetition_penalty
                    )(history[:, codebook].reshape(1, -1), scores)
                else:
                    pass
                if not warpers:
                    codes[row, codebook] = scores.argmax(-1)[0]
                    continue
                else:
                    pass
                for warper in warpers:
                    scores = warper(None, scores)
                codes[row, codebook] = torch.multinomial(
                    scores.softmax(-1), 1, generator=frame_sampling.generator
                )[0, 0]
        revealed.scatter_(1, revealed_now, True)
    return codes


def test_exact_sampler_draws_as_the_checkpoint() -> None:
    """Per-row batched warping keeps every generator draw of the checkpoint's per-codebook loop."""
    depth_logits = PassLogits(3)
    settings = [(0.0, None, 1.0), (0.8, 64, 1.2), (1.0, None, 1.1)]
    ours = [sampling(*setting, seed=row) for row, setting in enumerate(settings)]
    theirs = [sampling(*setting, seed=row) for row, setting in enumerate(settings)]
    for _ in range(6):
        expected = reference_frames(depth_logits, theirs)
        frames = sample_frames(
            depth_logits, torch.zeros(3, 1), ours, TOKENS.maskgit_schedule
        )
        assert torch.equal(frames, expected)
        for row in range(3):
            ours[row].append(frames[row])
            theirs[row].append(expected[row])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the seeded sampler is a Triton kernel"
)
def test_seeded_sampler_rows_depend_only_on_their_seed() -> None:
    """A row's frame is the same alone or in a batch, the same on a rerun, and changes with its seed."""
    device = torch.device("cuda")
    depth_logits = PassLogits(3)
    base = depth_logits.base.to(device)
    depth_logits.base = base
    window = 4
    history = torch.randint(
        0, TOKENS.codebook_size, (3, window, TOKENS.num_codebooks), device=device
    )
    inputs = dict(
        seeds=torch.tensor([11, 12, 13], device=device),
        frame_positions=torch.tensor([0, 5, 9], device=device),
        temperature=torch.tensor([0.8, 0.0, 1.0], device=device),
        top_k=torch.tensor([64, 0, 0], device=device),
        repetition_penalty=torch.tensor([1.2, 1.3, 1.0], device=device),
        history=history,
    )
    batch = seeded_frame_codes(
        depth_logits,
        torch.zeros(3, 1, device=device),
        schedule=TOKENS.maskgit_schedule,
        **inputs,
    )
    for row in range(3):
        depth_logits.base = base[row : row + 1]
        alone = seeded_frame_codes(
            depth_logits,
            torch.zeros(1, 1, device=device),
            schedule=TOKENS.maskgit_schedule,
            **{name: value[row : row + 1] for name, value in inputs.items()},
        )
        assert torch.equal(alone[0], batch[row])
    depth_logits.base = base
    again = seeded_frame_codes(
        depth_logits,
        torch.zeros(3, 1, device=device),
        schedule=TOKENS.maskgit_schedule,
        **inputs,
    )
    assert torch.equal(again, batch)
    inputs["seeds"] = inputs["seeds"] + 100
    reseeded = seeded_frame_codes(
        depth_logits,
        torch.zeros(3, 1, device=device),
        schedule=TOKENS.maskgit_schedule,
        **inputs,
    )
    assert not torch.equal(reseeded[0], batch[0])


def test_repetition_history_keeps_the_last_window_of_frames() -> None:
    frame_sampling = sampling()
    for code in range(6):
        frame_sampling.append(frame(code))
    assert frame_sampling.history[:, 0].tolist() == [2, 3, 4, 5]
