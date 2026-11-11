# SPDX-License-Identifier: Apache-2.0
"""Clean decoded Nemotron Transcribe text before it is served or aligned.

Speaker, event, and style tags stay. Reasoning markers, runs of ``<unk>``, and
an assistant refusal appended after the speech are not part of the transcript.
"""

from __future__ import annotations

import re

THINK_MARKERS = re.compile(r"</?think>", flags=re.IGNORECASE)
UNKNOWN_TOKENS = re.compile(r"(?:<unk>|\ufffd)+", flags=re.IGNORECASE)
SPACES = re.compile(r"[ \t]{2,}")
# A speaker tag also starts a sentence: a tagged transcript usually has no
# sentence punctuation, and one long "sentence" would let the refusal pattern
# span a whole meeting.
SENTENCE_BREAKS = re.compile(r"(?<=[.!?])\s+|\s+(?=<spk:\d+>)")
REFUSAL_SPAN_WORDS = 15
REFUSAL = re.compile(
    r"\b(?:I'?m sorry|I am sorry|I apologi[sz]e|I (?:cannot|can'?t|am unable|'m unable)|"
    r"unable to|unfortunately)\b[,;:]?(?:\s+\S+){0,"
    + str(REFUSAL_SPAN_WORDS)
    + r"}?\s+(?:transcri\w*|audio|recording|spoken content|no speech)\b",
    flags=re.IGNORECASE,
)
# A refusal is a tail: it may hold at most this share of the words, unless the
# whole answer is short enough to be the refusal alone.
REFUSAL_TAIL_SHARE = 0.2
REFUSAL_SHORT_ANSWER_WORDS = 40


def strip_refusal_tail(text: str) -> str:
    """Drop the first refusal sentence and everything after it, when that is a tail."""
    sentences = SENTENCE_BREAKS.split(text)
    for index, sentence in enumerate(sentences):
        if REFUSAL.search(sentence):
            kept = " ".join(sentences[:index]).strip()
            total = len(text.split())
            dropped = total - len(kept.split())
            if (
                total <= REFUSAL_SHORT_ANSWER_WORDS
                or dropped <= REFUSAL_TAIL_SHARE * total
            ):
                return kept
            else:
                return text
        else:
            pass
    return text


def clean_transcript(text: str) -> str:
    text = THINK_MARKERS.sub(" ", text)
    text = UNKNOWN_TOKENS.sub(" ", text)
    text = strip_refusal_tail(SPACES.sub(" ", text).strip())
    return SPACES.sub(" ", text).strip()


__all__ = ["clean_transcript", "strip_refusal_tail"]
