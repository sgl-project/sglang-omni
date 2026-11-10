# SPDX-License-Identifier: Apache-2.0
"""Forced alignment of a transcript onto CTC frame scores.

The scores come from the bundled ``ctc_timestamp`` head. A missing head, an
empty transcript, or a failed alignment is reported. This module never
invents a span by stretching the text across the file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

ScoreColumns = Callable[[list[int]], torch.Tensor]

_ENGLISH = {"en", "eng", "english"}


def english_timestamp_language(language: str | None) -> str | None:
    """Return ``en`` for an English hint, or None when timestamps must be refused."""
    if language is None or not str(language).strip():
        return "en"
    else:
        pass
    tag = str(language).strip().lower().replace("_", "-")
    if tag in _ENGLISH or tag.startswith("en-"):
        return "en"
    else:
        return None


def split_transcript_words(text: str) -> list[str]:
    return [part for part in text.split() if part]


def forced_align(
    log_probs: torch.Tensor,
    targets: list[int],
    blank_id: int,
) -> list[tuple[int, int]] | None:
    """Return an inclusive frame span for each target token.

    ``log_probs`` is ``(time, classes)``. ``None`` means the transcript does
    not fit the frames. Each frame updates every CTC state at once; only the
    one-byte choice per frame and state is kept for the backtrace.
    """
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape (time, classes)")
    else:
        pass
    if not targets:
        return []
    else:
        pass
    time_steps, class_count = log_probs.shape
    if blank_id < 0 or blank_id >= class_count:
        return None
    else:
        pass
    if any(token < 0 or token >= class_count or token == blank_id for token in targets):
        return None
    else:
        pass
    if time_steps < len(targets):
        return None
    else:
        pass
    labels = np.full(2 * len(targets) + 1, blank_id, dtype=np.int64)
    labels[1::2] = targets
    states = labels.size
    # A label may skip the blank before it unless it repeats the label before that.
    can_skip = np.zeros(states, dtype=bool)
    can_skip[2:] = (labels[2:] != blank_id) & (labels[2:] != labels[:-2])
    scores = log_probs.detach().to(dtype=torch.float32, device="cpu").numpy()
    every_state = np.arange(states)
    previous = np.full(states, -np.inf, dtype=np.float32)
    previous[:2] = scores[0, labels[:2]]
    # 0 stays in the state, 1 steps from the state before, 2 skips a blank.
    choices = np.zeros((time_steps, states), dtype=np.uint8)
    candidates = np.full((3, states), -np.inf, dtype=np.float32)
    for time in range(1, time_steps):
        candidates[0] = previous
        candidates[1, 1:] = previous[:-1]
        candidates[2, 2:] = np.where(can_skip[2:], previous[:-2], -np.inf)
        choice = candidates.argmax(axis=0)
        choices[time] = choice
        previous = candidates[choice, every_state] + scores[time, labels]
    end_state = states - 1 if previous[-1] >= previous[-2] else states - 2
    if not np.isfinite(previous[end_state]):
        return None
    else:
        pass
    path = [0] * time_steps
    state = end_state
    for time in range(time_steps - 1, 0, -1):
        path[time] = state
        state -= int(choices[time, state])
    path[0] = state

    spans: list[tuple[int, int]] = []
    open_frame: int | None = None
    open_state: int | None = None
    for time, state in enumerate(path):
        emitting = state % 2 == 1
        if emitting and state == open_state:
            continue
        else:
            pass
        if open_frame is not None:
            spans.append((open_frame, time - 1))
            open_frame = None
            open_state = None
        else:
            pass
        if emitting:
            open_frame = time
            open_state = state
        else:
            pass
    if open_frame is not None:
        spans.append((open_frame, time_steps - 1))
    else:
        pass
    if len(spans) != len(targets):
        return None
    else:
        return spans


def words_from_token_spans(
    words: list[str],
    pieces: list[list[int]],
    spans: list[tuple[int, int]],
    frame_seconds: float,
) -> list[dict[str, Any]]:
    """Collapse token spans into one ``{word, start, end}`` entry per word."""
    cursor = 0
    aligned: list[dict[str, Any]] = []
    for word, piece in zip(words, pieces, strict=True):
        word_spans = spans[cursor : cursor + len(piece)]
        cursor += len(piece)
        start = word_spans[0][0] * frame_seconds
        end = (word_spans[-1][1] + 1) * frame_seconds
        aligned.append(
            {
                "word": word,
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )
    return aligned


def alignment_result(
    *,
    status: str,
    failure_reason: str | None = None,
    words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "words": words or [],
        "word_timestamp_alignment": {"status": status},
    }
    if failure_reason is not None:
        body["word_timestamp_alignment"]["failure_reason"] = failure_reason
    else:
        pass
    return body


def unavailable(reason: str) -> dict[str, Any]:
    return alignment_result(status="unavailable", failure_reason=reason)


def align_token_pieces(
    log_probs: torch.Tensor,
    words: list[str],
    pieces: list[list[int]],
    *,
    blank_id: int,
    frame_seconds: float,
) -> dict[str, Any]:
    if not words:
        return unavailable("empty_transcript")
    else:
        pass
    if len(words) != len(pieces) or any(not piece for piece in pieces):
        return unavailable("ctc_alignment_failed")
    else:
        pass
    if frame_seconds <= 0:
        return unavailable("ctc_evidence_unavailable")
    else:
        pass
    targets = [token for piece in pieces for token in piece]
    spans = forced_align(log_probs, targets, blank_id)
    if spans is None:
        return unavailable("ctc_alignment_failed")
    else:
        pass
    return alignment_result(
        status="aligned",
        words=words_from_token_spans(words, pieces, spans, frame_seconds),
    )


def tokenize_words(proto: bytes, words: list[str]) -> list[list[int]] | None:
    try:
        import sentencepiece as spm
    except ImportError:
        logger.warning(
            "sentencepiece is not installed; word timestamps are unavailable"
        )
        return None
    else:
        pass
    processor = spm.SentencePieceProcessor()
    processor.LoadFromSerializedProto(proto)
    pieces: list[list[int]] = []
    for index, word in enumerate(words):
        text = word if index == 0 else f" {word}"
        ids = list(processor.EncodeAsIds(text))
        if not ids:
            return None
        else:
            pieces.append(ids)
    return pieces


def align_transcript(
    text: str,
    evidence: dict[str, Any] | None,
    score_columns: ScoreColumns | None = None,
) -> dict[str, Any]:
    """Align ``text`` to the encoder evidence stashed during perception.

    ``score_columns`` returns ``(time, len(columns))`` CTC log-probs for the
    requested classes, so only the transcript's own classes are scored.
    """
    if not evidence:
        return unavailable("ctc_evidence_unavailable")
    else:
        pass
    if evidence.get("failure_reason"):
        return unavailable(str(evidence["failure_reason"]))
    else:
        pass
    words = split_transcript_words(text)
    if not words:
        return unavailable("empty_transcript")
    else:
        pass
    proto = evidence.get("tokenizer_proto")
    blank_id = evidence.get("blank_id")
    class_count = evidence.get("class_count")
    frame_seconds = evidence.get("frame_seconds")
    if (
        score_columns is None
        or not isinstance(proto, (bytes, bytearray))
        or not isinstance(blank_id, int)
        or not isinstance(class_count, int)
        or not isinstance(frame_seconds, (int, float))
    ):
        return unavailable("ctc_evidence_unavailable")
    else:
        pass
    pieces = tokenize_words(bytes(proto), words)
    if pieces is None:
        return unavailable("ctc_alignment_failed")
    else:
        pass
    targets = {token for piece in pieces for token in piece}
    if any(token < 0 or token >= class_count or token == blank_id for token in targets):
        return unavailable("ctc_alignment_failed")
    else:
        pass
    columns = sorted({*targets, blank_id})
    compact = {token: index for index, token in enumerate(columns)}
    try:
        log_probs = score_columns(columns)
    except RuntimeError:
        logger.exception("CTC scoring failed; word timestamps are unavailable")
        return unavailable("ctc_evidence_unavailable")
    else:
        pass
    return align_token_pieces(
        log_probs,
        words,
        [[compact[token] for token in piece] for piece in pieces],
        blank_id=compact[blank_id],
        frame_seconds=float(frame_seconds),
    )
