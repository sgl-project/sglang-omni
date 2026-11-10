# SPDX-License-Identifier: Apache-2.0
"""Speaker tags, diarization segments, and one tagged-decode retry.

A speaker-tag prompt contains at least two ``<spk:N>`` tokens. A bare
``<spk:0>`` does not. Tags in the transcript name the word's speaker, and each
tag is matched to one Sortformer slot by activity. When the transcript has no
tag, the word takes the Sortformer slot with the highest mean activity over
its span.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

TAG_MARKER = "<spk:"
_TAG = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)
_TAG_SHAPE = re.compile(r"<spk:([^<>\s]*)>", flags=re.IGNORECASE)
RETRY_MIN_WORDS = 200

PAD_ONSET = 0.229
PAD_OFFSET = 0.079
MIN_SPEECH = 0.511
MIN_GAP = 0.296
ONSET = 0.5
OFFSET = 0.5


def speaker_tag_request(prompt: str | None) -> bool:
    """True when the prompt asks for more than one speaker tag."""
    return len(_TAG.findall(prompt or "")) >= 2


def strip_speaker_tags(transcript: str) -> str:
    return " ".join(_TAG_SHAPE.sub(" ", transcript).split())


def parse_speaker_tags(transcript: str) -> list[int | None]:
    """Speaker number in force for each whitespace word, or None before the first tag."""
    parts = _TAG_SHAPE.split(transcript)
    tagged = any(part.isdigit() for part in parts[1::2])
    speaker: int | None = 0 if not tagged else None
    words: list[int | None] = []
    for index, part in enumerate(parts):
        if index % 2:
            if part.isdigit():
                speaker = int(part)
            else:
                pass
            continue
        else:
            pass
        for _word in part.split():
            words.append(speaker)
    return words


def decide_tagged_retry(prompt: str | None, transcript: str | None) -> bool:
    """Retry one long tag-prompt decode that came back without a tag."""
    text = transcript or ""
    if not speaker_tag_request(prompt):
        return False
    else:
        pass
    if TAG_MARKER in text:
        return False
    else:
        pass
    return len(text.split()) >= RETRY_MIN_WORDS


def choose_tagged_transcript(first: str, second: str | None) -> str:
    """Keep the retry only when it actually contains a speaker tag."""
    if isinstance(second, str) and TAG_MARKER in second and second.strip():
        return second
    else:
        return first


def diarization_segments(
    activity: list[list[float]],
    frame_seconds: float,
    duration_s: float,
) -> list[dict[str, float | int]]:
    """Hysteresis, then the Speech pad / min-speech / min-gap post-process."""
    if frame_seconds <= 0 or not activity:
        return []
    else:
        pass
    speakers = len(activity[0]) if activity else 0
    raw: list[tuple[int, float, float]] = []
    for speaker in range(speakers):
        speaking = False
        start = 0.0
        for index, frame in enumerate(activity):
            probability = float(frame[speaker]) if speaker < len(frame) else 0.0
            if not speaking and probability >= ONSET:
                speaking = True
                start = index * frame_seconds
            elif speaking and probability < OFFSET:
                speaking = False
                raw.append((speaker, start, index * frame_seconds))
            else:
                pass
        if speaking:
            raw.append((speaker, start, len(activity) * frame_seconds))
        else:
            pass
    return [
        {"speaker": speaker, "start": start, "end": end}
        for speaker, start, end in postprocess_segments(raw, duration_s)
    ]


def postprocess_segments(
    segments: list[tuple[int, float, float]], duration_s: float
) -> list[tuple[int, float, float]]:
    by_speaker: dict[int, list[tuple[float, float]]] = {}
    for speaker, start, end in segments:
        start = max(0.0, start - PAD_ONSET)
        end = min(duration_s, end + PAD_OFFSET) if duration_s > 0 else end + PAD_OFFSET
        if end > start:
            by_speaker.setdefault(speaker, []).append((start, end))
        else:
            pass
    output: list[tuple[int, float, float]] = []
    for speaker, intervals in by_speaker.items():
        merged = merge_intervals(intervals)
        filtered = drop_short_intervals(merged)
        filtered = fill_short_gaps(filtered)
        for start, end in filtered:
            output.append((speaker, round(start, 3), round(end, 3)))
    output.sort(key=lambda item: (item[1], item[2], item[0]))
    return output


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def drop_short_intervals(
    intervals: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    return [(start, end) for start, end in intervals if end - start >= MIN_SPEECH]


def fill_short_gaps(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(intervals) < 2:
        return intervals
    else:
        pass
    output = [intervals[0]]
    for start, end in intervals[1:]:
        if start - output[-1][1] < MIN_GAP:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


def attribute_speakers(
    words: list[dict[str, Any]],
    transcript: str,
    prompt: str | None,
    activity: list[list[float]] | None,
    frame_seconds: float,
) -> list[dict[str, Any]]:
    """Add speaker fields when the prompt asked for speaker tags."""
    if not speaker_tag_request(prompt) or not words:
        return words
    else:
        pass
    tags = parse_speaker_tags(transcript)
    tagged = TAG_MARKER in transcript
    if tagged and len(tags) != len(words):
        tags = [None] * len(words)
        tagged = False
    else:
        pass
    mapping: dict[int, int | None] = {}
    if tagged:
        frames = len(activity) if activity else 0
        spans: dict[int, list[tuple[int, int]]] = {}
        for word, tag in zip(words, tags, strict=True):
            if tag is not None:
                spans.setdefault(tag, []).append(
                    word_frame_span(word, frame_seconds, frames)
                )
            else:
                pass
        mapping = map_tags_to_slots(spans, activity)
    else:
        pass
    attributed: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        entry = dict(word)
        if tagged:
            tag = tags[index]
            label = "unknown" if tag is None else f"spk:{tag}"
            slot = None if tag is None else mapping.get(tag)
        else:
            slot = overlap_slot(word, activity, frame_seconds)
            label = "unknown" if slot is None else f"spk:{slot}"
        entry["speaker"] = label
        if slot is not None:
            entry["speaker_slot"] = slot
            confidence = slot_confidence(word, activity, frame_seconds, slot)
            if confidence is not None:
                entry["speaker_confidence"] = confidence
            else:
                pass
        else:
            pass
        attributed.append(entry)
    return attributed


def map_tags_to_slots(
    tag_spans: dict[int, list[tuple[int, int]]],
    activity: list[list[float]] | None,
) -> dict[int, int | None]:
    """One Sortformer slot per transcript tag, by mean log activity over its words.

    The assignment is one-to-one and maximizes the total score. With more tags
    than slots, the tags with the most word frames are assigned and the rest
    stay unassigned.
    """
    tags = sorted(tag_spans)
    slots = len(activity[0]) if activity else 0
    if not tags or slots == 0:
        return {tag: None for tag in tags}
    else:
        pass
    unassigned: list[int] = []
    if len(tags) > slots:
        mass = {
            tag: sum(max(0, last - first + 1) for first, last in tag_spans[tag])
            for tag in tags
        }
        ranked = sorted(tags, key=lambda tag: (-mass[tag], tag))
        tags, unassigned = ranked[:slots], ranked[slots:]
    else:
        pass
    log_activity = np.log(np.maximum(np.asarray(activity, dtype=np.float64), 1e-6))
    scores: list[list[float]] = []
    for tag in tags:
        frames = [
            np.arange(first, last + 1)
            for first, last in tag_spans[tag]
            if last >= first
        ]
        if frames:
            scores.append(log_activity[np.concatenate(frames)].mean(axis=0).tolist())
        else:
            scores.append([0.0] * slots)
    # Exact maximum-weight assignment over subsets of used slots; Sortformer has few.
    states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for row in scores:
        next_states: dict[int, tuple[float, list[int]]] = {}
        for used, (total, chosen) in states.items():
            for slot, score in enumerate(row):
                mask = used | (1 << slot)
                if mask == used:
                    continue
                else:
                    pass
                candidate = (total + score, [*chosen, slot])
                if mask not in next_states or candidate[0] > next_states[mask][0]:
                    next_states[mask] = candidate
                else:
                    pass
        states = next_states
    best = max(states.values(), key=lambda state: state[0])[1]
    mapping: dict[int, int | None] = dict(zip(tags, best, strict=True))
    mapping.update({tag: None for tag in unassigned})
    return mapping


def word_frame_span(
    word: dict[str, Any], frame_seconds: float, frames: int
) -> tuple[int, int]:
    if frame_seconds <= 0 or frames < 1:
        return 0, -1
    else:
        pass
    first = min(frames - 1, max(0, int(float(word["start"]) / frame_seconds)))
    last = min(frames - 1, max(first, int(float(word["end"]) / frame_seconds) - 1))
    return first, last


def overlap_slot(
    word: dict[str, Any],
    activity: list[list[float]] | None,
    frame_seconds: float,
) -> int | None:
    if not activity:
        return None
    else:
        pass
    first, last = word_frame_span(word, frame_seconds, len(activity))
    if last < first:
        return None
    else:
        pass
    speakers = len(activity[0])
    best_slot = 0
    best_score = -1.0
    for slot in range(speakers):
        total = sum(float(activity[frame][slot]) for frame in range(first, last + 1))
        if total > best_score:
            best_score = total
            best_slot = slot
        else:
            pass
    return best_slot


def slot_confidence(
    word: dict[str, Any],
    activity: list[list[float]] | None,
    frame_seconds: float,
    slot: int,
) -> float | None:
    if not activity or slot >= len(activity[0]):
        return None
    else:
        pass
    first, last = word_frame_span(word, frame_seconds, len(activity))
    if last < first:
        return None
    else:
        pass
    total = sum(float(activity[frame][slot]) for frame in range(first, last + 1))
    return round(total / float(last - first + 1), 3)


def activity_rows(evidence: dict[str, Any] | None) -> list[list[float]] | None:
    if not evidence:
        return None
    else:
        pass
    activity = evidence.get("speaker_activity")
    if activity is None:
        return None
    else:
        pass
    if hasattr(activity, "tolist"):
        rows = activity.tolist()
    else:
        rows = activity
    if not isinstance(rows, list):
        return None
    else:
        return rows
