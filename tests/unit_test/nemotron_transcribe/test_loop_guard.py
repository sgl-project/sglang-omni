# SPDX-License-Identifier: Apache-2.0
"""The loop guard ends repeating decodes and leaves ordinary text alone."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.nemotron_transcribe.loop_guard import (
    NemotronTranscribeLoopGuard,
    loop_period,
)

EOS = 11


def test_a_short_sentence_repeated_three_times_is_a_loop() -> None:
    sentence = [101, 102, 103, 104, 105]
    assert loop_period([7, 8] + sentence * 2) == 0
    assert loop_period([7, 8] + sentence * 3) == 5


def test_a_long_pattern_needs_four_repeats() -> None:
    pattern = list(range(200, 220))
    assert loop_period(pattern * 3) == 0
    assert loop_period(pattern * 4) == 20


def test_a_single_token_run_needs_the_full_span() -> None:
    assert loop_period([9] * 31) == 0
    assert loop_period([9] * 32) == 1


def test_a_three_token_phrase_read_five_times_is_speech() -> None:
    # "percent sign," five times: periods under four tokens need the 32-token span.
    assert loop_period([7] + [301, 302, 303] * 5) == 0


def test_ordinary_text_is_not_a_loop() -> None:
    assert loop_period(list(range(1000, 1100))) == 0


def test_trailing_placeholders_are_ignored() -> None:
    sentence = [101, 102, 103, 104, 105]
    assert loop_period(sentence * 3 + [-1]) == 5


def test_only_looping_rows_are_forced_to_end() -> None:
    sentence = [101, 102, 103, 104, 105]
    looping = SimpleNamespace(output_ids=sentence * 3)
    speaking = SimpleNamespace(output_ids=list(range(1000, 1040)))
    logits = torch.zeros(3, 20)
    params = [
        {"eos_token_id": EOS, "__req__": looping},
        {"eos_token_id": EOS, "__req__": speaking},
        None,
    ]

    guarded = NemotronTranscribeLoopGuard()(logits, params)

    assert int(guarded[0].argmax()) == EOS
    assert torch.isinf(guarded[0]).sum() == 19
    assert torch.equal(guarded[1], torch.zeros(20))
    assert torch.equal(guarded[2], torch.zeros(20))
