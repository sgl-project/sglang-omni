# SPDX-License-Identifier: Apache-2.0
"""Forced alignment of a transcript onto CTC frame scores.

The scores come from the bundled ``ctc_timestamp`` head. A missing head, an
empty transcript, or a failed alignment is reported. This module never
invents a span by stretching the text across the file.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_ENGLISH = {"en", "eng", "english"}
_NEG = -1.0e30


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
    not fit the frames.
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
    expanded = [blank_id]
    for token in targets:
        expanded.append(token)
        expanded.append(blank_id)
    states = len(expanded)
    if time_steps < len(targets):
        return None
    else:
        pass

    scores = log_probs.detach().to(dtype=torch.float32, device="cpu")
    dp = torch.full((time_steps, states), _NEG)
    backpointer = torch.full((time_steps, states), -1, dtype=torch.int16)
    dp[0, 0] = scores[0, blank_id]
    if states > 1:
        dp[0, 1] = scores[0, expanded[1]]
    else:
        pass
    for time in range(1, time_steps):
        for state in range(states):
            best = dp[time - 1, state]
            origin = state
            if state > 0 and dp[time - 1, state - 1] > best:
                best = dp[time - 1, state - 1]
                origin = state - 1
            else:
                pass
            label = expanded[state]
            if (
                state > 1
                and label != blank_id
                and label != expanded[state - 2]
                and dp[time - 1, state - 2] > best
            ):
                best = dp[time - 1, state - 2]
                origin = state - 2
            else:
                pass
            if best <= _NEG / 2:
                continue
            else:
                pass
            dp[time, state] = best + scores[time, label]
            backpointer[time, state] = origin

    if states == 1:
        end_state = 0
    elif dp[-1, -1] >= dp[-1, -2]:
        end_state = states - 1
    else:
        end_state = states - 2
    if dp[-1, end_state] <= _NEG / 2:
        return None
    else:
        pass

    path: list[int] = []
    state = end_state
    for time in range(time_steps - 1, -1, -1):
        path.append(state)
        if time == 0:
            break
        else:
            pass
        state = int(backpointer[time, state])
        if state < 0:
            return None
        else:
            pass
    path.reverse()

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
) -> dict[str, Any]:
    """Align ``text`` to encoder evidence stashed during perception."""
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
    log_probs = evidence.get("log_probs")
    proto = evidence.get("tokenizer_proto")
    blank_id = evidence.get("blank_id")
    frame_seconds = evidence.get("frame_seconds")
    if (
        not isinstance(log_probs, torch.Tensor)
        or not isinstance(proto, (bytes, bytearray))
        or not isinstance(blank_id, int)
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
    return align_token_pieces(
        log_probs,
        words,
        pieces,
        blank_id=blank_id,
        frame_seconds=float(frame_seconds),
    )
