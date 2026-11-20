# SPDX-License-Identifier: Apache-2.0
"""CTC forced alignment without a checkpoint."""

import torch

from sglang_omni.models.nemotron_transcribe.ctc_head import build_ctc_head
from sglang_omni.models.nemotron_transcribe.word_timestamps import (
    align_token_pieces,
    align_transcript,
    english_timestamp_language,
    forced_align,
)


def test_omitted_language_is_english_and_other_languages_are_refused() -> None:
    assert english_timestamp_language(None) == "en"
    assert english_timestamp_language("  ") == "en"
    assert english_timestamp_language("en-US") == "en"
    assert english_timestamp_language("eng") == "en"
    assert english_timestamp_language("fr") is None


def test_forced_align_recovers_two_token_spans() -> None:
    log_probs = torch.full((4, 3), -8.0)
    log_probs[0:2, 1] = 0.0
    log_probs[2:4, 2] = 0.0

    spans = forced_align(log_probs, [1, 2], blank_id=0)

    assert spans == [(0, 1), (2, 3)]


def test_forced_align_refuses_a_target_longer_than_the_frames() -> None:
    log_probs = torch.zeros((1, 3))
    assert forced_align(log_probs, [1, 2], blank_id=0) is None


def test_alignment_does_not_invent_times_for_an_empty_transcript() -> None:
    result = align_token_pieces(
        torch.zeros((2, 2)),
        [],
        [],
        blank_id=0,
        frame_seconds=0.08,
    )
    assert result["words"] == []
    assert result["word_timestamp_alignment"] == {
        "status": "unavailable",
        "failure_reason": "empty_transcript",
    }


def test_missing_evidence_is_explicit() -> None:
    result = align_transcript("hello", None)
    assert result["word_timestamp_alignment"]["failure_reason"] == (
        "ctc_evidence_unavailable"
    )
    assert result["words"] == []


def test_conv_head_emits_one_log_prob_row_per_frame() -> None:
    proto = torch.tensor(list(b"proto"), dtype=torch.uint8)
    head = build_ctc_head(
        {
            "feat_in": 4,
            "num_classes": 2,
            "vocabulary": ["a", "b"],
            "use_transformer": False,
        },
        {
            "decoder_layers.0.weight": torch.zeros(3, 4, 1),
            "decoder_layers.0.bias": torch.zeros(3),
        },
        proto,
    )
    missing, unexpected = head.load_state_dict(
        {
            "decoder_layers.0.weight": torch.zeros(3, 4, 1),
            "decoder_layers.0.bias": torch.zeros(3),
        },
        strict=False,
    )
    head.usable = not missing and not unexpected
    log_probs = head(torch.randn(1, 4, 5))
    assert log_probs.shape == (1, 5, 3)
    assert head.blank_id == 2
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(1, 5), atol=1e-5)


def test_token_spans_become_word_times() -> None:
    log_probs = torch.full((4, 3), -8.0)
    log_probs[0:2, 1] = 0.0
    log_probs[2:4, 2] = 0.0
    result = align_token_pieces(
        log_probs,
        ["hello", "there"],
        [[1], [2]],
        blank_id=0,
        frame_seconds=0.08,
    )
    assert result["word_timestamp_alignment"] == {"status": "aligned"}
    assert result["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.16},
        {"word": "there", "start": 0.16, "end": 0.32},
    ]
