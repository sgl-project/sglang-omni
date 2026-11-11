# SPDX-License-Identifier: Apache-2.0
"""Subtitle cues from forced-aligned words.

The break rules match the Speech stack: at most 7 seconds, two lines, 42
graphemes per line, and a sentence boundary once a cue is already 0.833
seconds long. One word that cannot fit is still its own cue.
"""

from __future__ import annotations

import unicodedata
from typing import Any

MAX_CUE_SECONDS = 7.0
MAX_LINE_GRAPHEMES = 42
MIN_SENTENCE_SECONDS = 0.833
_SENTENCE_END = frozenset(".!?…")


def grapheme_count(text: str) -> int:
    count = 0
    for char in text:
        if unicodedata.combining(char):
            continue
        else:
            count += 1
    return count


def ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in _SENTENCE_END


def build_subtitle_cues(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return ``{start, end, text}`` cues. Raises ``ValueError`` on bad times."""
    validate_cue_words(words)
    cues: list[dict[str, Any]] = []
    first = 0
    while first < len(words):
        last_valid: int | None = None
        last_sentence: int | None = None
        candidate = first
        while candidate < len(words):
            duration = float(words[candidate]["end"]) - float(words[first]["start"])
            if duration > MAX_CUE_SECONDS:
                break
            else:
                pass
            layout = cue_layout(words[first : candidate + 1])
            if layout is None:
                break
            else:
                pass
            last_valid = candidate
            if duration >= MIN_SENTENCE_SECONDS and ends_sentence(
                str(words[candidate]["word"])
            ):
                last_sentence = candidate
            else:
                pass
            candidate += 1
        chosen = last_sentence if last_sentence is not None else last_valid
        if chosen is None:
            text = str(words[first]["word"])
            chosen = first
        else:
            text = cue_layout(words[first : chosen + 1])
            if text is None:
                raise ValueError("subtitle cue layout was lost")
            else:
                pass
        cues.append(
            {
                "start": float(words[first]["start"]),
                "end": float(words[chosen]["end"]),
                "text": text,
            }
        )
        first = chosen + 1
    return cues


def segments_from_word_timestamps(
    transcription: dict[str, Any],
) -> list[Any]:
    """Build subtitle segments, or refuse when alignment did not succeed."""
    alignment = transcription.get("word_timestamp_alignment") or {}
    if not isinstance(alignment, dict) or alignment.get("status") != "aligned":
        reason = "ctc_alignment_failed"
        if isinstance(alignment, dict) and alignment.get("failure_reason"):
            reason = str(alignment["failure_reason"])
        else:
            pass
        raise ValueError(f"subtitles require aligned word timestamps ({reason})")
    else:
        pass
    words = transcription.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError("subtitles require aligned word timestamps (empty_transcript)")
    else:
        pass
    from sglang_omni.serve.protocol import TranscriptionSegment

    return [
        TranscriptionSegment(
            id=index,
            start=float(cue["start"]),
            end=float(cue["end"]),
            text=str(cue["text"]),
        )
        for index, cue in enumerate(build_subtitle_cues(words))
    ]


def cue_layout(words: list[dict[str, Any]]) -> str | None:
    display = [str(word["word"]) for word in words]
    single = " ".join(display)
    if grapheme_count(single) <= MAX_LINE_GRAPHEMES:
        return single
    else:
        pass
    best: tuple[int, int, str, str] | None = None
    for split in range(1, len(display)):
        first_line = " ".join(display[:split])
        second_line = " ".join(display[split:])
        first_length = grapheme_count(first_line)
        second_length = grapheme_count(second_line)
        if first_length > MAX_LINE_GRAPHEMES or second_length > MAX_LINE_GRAPHEMES:
            continue
        else:
            pass
        candidate = (
            int(ends_sentence(display[split - 1])),
            MAX_LINE_GRAPHEMES - abs(first_length - second_length),
            first_line,
            second_line,
        )
        if best is None or candidate > best:
            best = candidate
        else:
            pass
    if best is None:
        return None
    else:
        return f"{best[2]}\n{best[3]}"


def validate_cue_words(words: list[dict[str, Any]]) -> None:
    previous_end = 0.0
    for index, word in enumerate(words):
        text = str(word.get("word", "")).strip()
        start = float(word.get("start", -1))
        end = float(word.get("end", -1))
        if not text:
            raise ValueError(f"word {index} has empty display text")
        else:
            pass
        if start < 0 or end <= start:
            raise ValueError(f"word {index} has invalid timestamps")
        else:
            pass
        if index > 0 and start < previous_end:
            raise ValueError(f"word {index} overlaps the preceding word")
        else:
            pass
        previous_end = end
