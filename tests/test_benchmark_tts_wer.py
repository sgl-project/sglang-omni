from __future__ import annotations

from benchmarks.accuracy.tts.benchmark_tts_wer import (
    compute_word_error_rate,
    normalize_wer_text,
)


def test_normalize_wer_text_expands_digits_to_words() -> None:
    assert (
        normalize_wer_text("Ninety five lines and no more, that's it.")
        == normalize_wer_text("95 lines and no more, that's it.")
    )


def test_normalize_wer_text_collapses_internal_apostrophes() -> None:
    assert (
        normalize_wer_text(
            "Replace the Ts in Tim Tebow's name with any other consonant."
        )
        == normalize_wer_text(
            "Replace the T's in Tim Tebow's name with any other consonant."
        )
    )


def test_compute_word_error_rate_matches_expected_fraction() -> None:
    wer = compute_word_error_rate(
        ["the primary coil has fifty turns"],
        ["the primary coil has turns"],
    )
    assert wer == 1 / 6
