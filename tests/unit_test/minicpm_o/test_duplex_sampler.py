# SPDX-License-Identifier: Apache-2.0
"""Tests for device-side two-stage MiniCPM-o sampling helpers."""

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.minicpm_o import duplex_sampler
from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds


def sampling_state(*, greedy: bool = False) -> SimpleNamespace:
    return SimpleNamespace(greedy=greedy, temperature=1.0, top_k=-1, top_p=1.0)


def thinker_state(*, greedy: bool = False) -> SimpleNamespace:
    return SimpleNamespace(sampling=sampling_state(greedy=greedy))


def test_sample_candidates_active_mask_does_not_draw_inactive_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0], [5.0, 0.0, 1.0]])
    states = [thinker_state(), thinker_state()]
    sampled_rows: list[int] = []

    def draw(rows: torch.Tensor) -> torch.Tensor:
        sampled_rows.append(rows.shape[0])
        return torch.zeros(rows.shape[0], dtype=torch.long)

    monkeypatch.setattr(duplex_sampler, "draw_token_ids", draw)
    output = duplex_sampler.sample_candidates(
        logits,
        states,
        active_mask=torch.tensor([True, False]),
    )

    assert sampled_rows == [1]
    assert output.tolist() == [0, 0]


def test_sample_candidates_rejects_misaligned_active_mask() -> None:
    with pytest.raises(ValueError, match="aligned with logits"):
        duplex_sampler.sample_candidates(
            torch.zeros(2, 4),
            [thinker_state(), thinker_state()],
            active_mask=torch.ones(3, dtype=torch.bool),
        )


def test_duplex_sampler_keeps_first_stage_eos_on_device() -> None:
    special = MiniCPMOSpecialTokenIds(
        unit=10,
        unit_end=11,
        image_start=12,
        image_end=13,
        slice_start=14,
        slice_end=15,
        listen=4,
        speak=5,
        tts_bos=6,
        tts_eos=7,
        chunk_eos=8,
        chunk_tts_eos=9,
        turn_eos=3,
        tts_pad=16,
        audio_start=17,
        audio_end=18,
        chunk_terminators=frozenset({4, 8, 9}),
        turn_terminators=frozenset({3}),
        forbidden=frozenset({16}),
    )
    sampling = SimpleNamespace(
        greedy=True,
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        repetition_penalty=1.0,
        listen_prob_scale=1.0,
        max_new_tokens_per_unit=20,
        repetition_window_size=16,
    )
    states = [
        SimpleNamespace(
            sampling=sampling,
            generated_history=[],
            is_turn_ended=True,
        ),
        SimpleNamespace(
            sampling=sampling,
            generated_history=[],
            is_turn_ended=True,
        ),
    ]
    logits = torch.zeros(2, 20)
    logits[0, special.chunk_eos] = 10
    logits[1, 2] = 10
    output = duplex_sampler.duplex_sample_batch(
        logits,
        states,
        special_tokens=special,
        forbidden_token_index=torch.tensor([special.chunk_eos, special.tts_pad]),
        generation_steps=[0, 0],
        listen_forced_flags=[False, False],
    )

    assert output.tolist() == [special.chunk_eos, 2]
    assert states[0].generated_history == []
    assert states[1].generated_history == [2]
