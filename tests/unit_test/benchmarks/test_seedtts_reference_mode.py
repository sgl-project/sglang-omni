# SPDX-License-Identifier: Apache-2.0
"""Tests reference-mode selection for the SeedTTS TTS benchmark.

The warm/cold pair is the control for the codec lock contention study, so the
mode has to actually change cache pressure: "shared" must collapse every request
onto one reference, "per-sample" must leave SeedTTS's distinct prompts alone, and
neither may corrupt the memoized sample list.
"""

from __future__ import annotations

import pytest

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval.benchmark_tts_seedtts import REFERENCE_MODES, _apply_reference_mode


def _samples(count: int = 3) -> list[SampleInput]:
    return [
        SampleInput(
            sample_id=f"s{index}",
            ref_text=f"ref text {index}",
            ref_audio=f"/data/prompt_{index}.wav",
            target_text=f"target {index}",
        )
        for index in range(count)
    ]


def test_per_sample_mode_keeps_distinct_references() -> None:
    """Distinct reference content is what forces cache misses."""
    samples = _samples()

    result = _apply_reference_mode(samples, "per-sample")

    assert [s.ref_audio for s in result] == [
        "/data/prompt_0.wav",
        "/data/prompt_1.wav",
        "/data/prompt_2.wav",
    ]


def test_shared_mode_pins_one_reference_for_every_request() -> None:
    samples = _samples()

    result = _apply_reference_mode(samples, "shared")

    assert {s.ref_audio for s in result} == {"/data/prompt_0.wav"}
    assert {s.ref_text for s in result} == {"ref text 0"}


def test_shared_mode_keeps_the_target_text_distinct() -> None:
    """Only the reference is pinned; pinning the text would change the workload."""
    result = _apply_reference_mode(_samples(), "shared")

    assert [s.target_text for s in result] == ["target 0", "target 1", "target 2"]


def test_shared_mode_does_not_mutate_the_memoized_samples() -> None:
    """``load_seedtts_samples`` memoizes, so in-place edits would leak modes.

    A warm run followed by a cold run in the same process would otherwise
    silently measure warm twice.
    """
    samples = _samples()

    _apply_reference_mode(samples, "shared")

    assert [s.ref_audio for s in samples] == [
        "/data/prompt_0.wav",
        "/data/prompt_1.wav",
        "/data/prompt_2.wav",
    ]


def test_empty_sample_list_is_returned_unchanged() -> None:
    assert _apply_reference_mode([], "shared") == []


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="reference_mode"):
        _apply_reference_mode(_samples(), "warm")


def test_declared_modes_are_the_supported_ones() -> None:
    assert REFERENCE_MODES == ("per-sample", "shared")
