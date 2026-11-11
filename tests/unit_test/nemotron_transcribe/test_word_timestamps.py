# SPDX-License-Identifier: Apache-2.0
"""CTC forced alignment without a checkpoint."""

import math

import pytest
import torch

import sglang_omni.models.nemotron_transcribe.word_timestamps as word_timestamps
from sglang_omni.models.nemotron_transcribe.ctc_head import (
    build_ctc_head,
    windowed_ctc_scores,
)
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


def per_state_viterbi(
    log_probs: torch.Tensor, targets: list[int], blank_id: int
) -> list[tuple[int, int]] | None:
    """The scalar CTC Viterbi the vectorized aligner replaced."""
    expanded = [blank_id]
    for token in targets:
        expanded += [token, blank_id]
    steps, states = log_probs.shape[0], len(expanded)
    if steps < len(targets):
        return None
    dp = [[-math.inf] * states for _ in range(steps)]
    back = [[0] * states for _ in range(steps)]
    dp[0][0] = float(log_probs[0, blank_id])
    dp[0][1] = float(log_probs[0, expanded[1]])
    for t in range(1, steps):
        for s in range(states):
            best, origin = dp[t - 1][s], s
            if s > 0 and dp[t - 1][s - 1] > best:
                best, origin = dp[t - 1][s - 1], s - 1
            label = expanded[s]
            if (
                s > 1
                and label != blank_id
                and label != expanded[s - 2]
                and dp[t - 1][s - 2] > best
            ):
                best, origin = dp[t - 1][s - 2], s - 2
            dp[t][s] = best + float(log_probs[t, label])
            back[t][s] = origin
    end = states - 1 if dp[-1][-1] >= dp[-1][-2] else states - 2
    if dp[-1][end] == -math.inf:
        return None
    path = [end]
    for t in range(steps - 1, 0, -1):
        path.append(back[t][path[-1]])
    path.reverse()
    spans, open_frame, open_state = [], None, None
    for t, s in enumerate(path):
        if s % 2 == 1 and s == open_state:
            continue
        if open_frame is not None:
            spans.append((open_frame, t - 1))
            open_frame = open_state = None
        if s % 2 == 1:
            open_frame, open_state = t, s
    if open_frame is not None:
        spans.append((open_frame, steps - 1))
    return spans if len(spans) == len(targets) else None


def test_vectorized_alignment_matches_the_per_state_viterbi() -> None:
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        classes = int(torch.randint(3, 7, (1,), generator=generator))
        steps = int(torch.randint(1, 25, (1,), generator=generator))
        length = int(torch.randint(1, 8, (1,), generator=generator))
        targets = torch.randint(1, classes, (length,), generator=generator).tolist()
        log_probs = torch.log_softmax(
            torch.randn(steps, classes, generator=generator), dim=-1
        )
        assert forced_align(log_probs, targets, blank_id=0) == per_state_viterbi(
            log_probs, targets, blank_id=0
        )


def test_alignment_scores_only_the_transcript_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        word_timestamps,
        "tokenize_words",
        lambda proto, words: [[3000], [17]],
    )
    requested: list[list[int]] = []

    def score_columns(columns: list[int]) -> torch.Tensor:
        requested.append(columns)
        # Columns are [blank, 17, 3000]: "there" (17) follows "hello" (3000).
        scores = torch.full((4, 3), -8.0)
        scores[0:2, 2] = 0.0
        scores[2:4, 1] = 0.0
        return scores

    evidence = {
        "tokenizer_proto": b"proto",
        "blank_id": 0,
        "class_count": 23864,
        "frame_seconds": 0.08,
    }
    result = align_transcript("hello there", evidence, score_columns)

    assert requested == [[0, 17, 3000]]
    assert result["words"] == [
        {"word": "hello", "start": 0.0, "end": 0.16},
        {"word": "there", "start": 0.16, "end": 0.32},
    ]


def test_ctc_head_runs_one_window_at_a_time_with_context() -> None:
    windows: list[tuple[int, int]] = []

    def head(states: torch.Tensor) -> torch.Tensor:
        frames = states[0, 0]
        windows.append((int(frames[0]), int(frames[-1])))
        # One class per frame that echoes the frame's own index.
        return frames.view(1, -1, 1)

    states = torch.arange(12, dtype=torch.float32).view(1, 1, 12)
    scores = windowed_ctc_scores(
        head,
        states,
        torch.tensor([0]),
        core_frames=5,
        left_frames=2,
        right_frames=1,
    )

    assert windows == [(0, 5), (3, 10), (8, 11)]
    assert scores[:, 0].tolist() == list(range(12))


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
