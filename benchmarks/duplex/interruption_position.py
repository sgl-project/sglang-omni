# SPDX-License-Identifier: Apache-2.0
"""Interruption-position trials, a count cut by a spoken question about where it stopped, and their scores."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, TypeAdapter

from benchmarks.duplex.minicpmo_scores import (
    COUNT_TARGET,
    NUMBER_WORDS,
    TENS_WORDS,
    counted_to,
    format_quantiles,
)
from benchmarks.duplex.minicpmo_trials import COUNT_QUESTION, Placement, Trial
from benchmarks.duplex.oracle_models import TraceRecord, WireEvent
from benchmarks.duplex.playback import ReplyAudioChunk, played_sample_count
from benchmarks.duplex.profiles import PROFILES, ProfileName
from benchmarks.duplex.v15_audio import AUDIO_DELTA, TRANSCRIPT_DELTAS

PROFILE: ProfileName = "turn-based"
OUTPUT_SAMPLE_RATE = PROFILES[PROFILE].output_sample_rate
PCM16_BYTES = 2
PCM16_FULL_SCALE = 32768
TRIAL_S = 30.0
# note (Jisen Li): The count question ends at 3.8 s, so the position question barges in a few seconds into the counting reply.
POSITION_QUESTION_S = 9.0
POSITION_QUESTION_OFFSETS_MS = (0, 250, 500, 750)
# note (Jisen Li): The cascade playground player's JITTER_MS.
START_BUFFER_MS = 300.0
# note (Jisen Li): Whisper ends each word where the next one starts, so a word's speech is taken as its 20 ms frames above -40 dBFS, the clips' own trim rule.
SPEECH_FRAME_S = 0.02
SPEECH_DBFS = -40.0

# note (Jisen Li): A truncating client tells the server what it played of a cancelled reply, as the playground does; a silent one leaves the cut to the server.
ClientMode = Literal["truncating", "silent"]
CLIENT_MODES: tuple[ClientMode, ...] = ("truncating", "silent")
PositionStatus = Literal["scored", "no count", "not interrupted"]
AnswerVerdict = Literal["heard", "partly heard", "ahead", "behind", "no number"]
ANSWER_VERDICTS: tuple[AnswerVerdict, ...] = (
    "heard",
    "partly heard",
    "ahead",
    "behind",
    "no number",
)

TRIALS = tuple(
    Trial(
        name=f"position_o{offset_ms:03d}",
        duration_s=TRIAL_S,
        placements=(
            COUNT_QUESTION,
            Placement(
                clip_name="position_question",
                start_s=POSITION_QUESTION_S + offset_ms / 1000,
                role="question",
            ),
        ),
    )
    for offset_ms in POSITION_QUESTION_OFFSETS_MS
)


class TranscribedWord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word: str
    start_s: float
    end_s: float


TRANSCRIBED_WORDS = TypeAdapter(list[TranscribedWord])


@dataclass(frozen=True, kw_only=True)
class ProgressPoint:
    text_end: int
    audio_ms: float


@dataclass(kw_only=True)
class RecordedReply:
    response_id: str
    item_id: str | None = None
    text: str = ""
    audio: bytearray = field(default_factory=bytearray)
    chunks: list[ReplyAudioChunk] = field(default_factory=list)
    progress: list[ProgressPoint] = field(default_factory=list)
    kept_text_ends: list[int] = field(default_factory=list)
    status: str | None = None
    done_s: float | None = None
    truncated_ms: float | None = None


@dataclass(frozen=True, kw_only=True)
class PositionScore:
    status: PositionStatus
    played_s: float | None = None
    delivered_s: float | None = None
    heard_count: int | None = None
    delivered_count: int | None = None
    progress_count: int | None = None
    kept_count: int | None = None
    answer_number: int | None = None
    is_mid_word: bool | None = None
    is_progress_behind: bool | None = None


def recorded_replies(trace_path: Path) -> list[RecordedReply]:
    """The responses of one recorded turn-based session, in the order they were created."""
    replies_by_response: dict[str, RecordedReply] = {}
    for line in trace_path.read_text().splitlines():
        trace_record = TraceRecord.model_validate(json.loads(line))
        event = WireEvent.model_validate(trace_record.event)
        if trace_record.direction == "send":
            if event.type == "conversation.item.truncate":
                (truncated_reply,) = (
                    reply
                    for reply in replies_by_response.values()
                    if reply.item_id == event.item_id
                )
                truncated_reply.truncated_ms = event.audio_end_ms
            else:
                pass
        elif trace_record.direction != "receive":
            pass
        elif event.type == "response.created":
            assert event.response is not None
            replies_by_response[event.response.id] = RecordedReply(
                response_id=event.response.id
            )
        elif event.type == "response.done":
            assert event.response is not None
            reply = replies_by_response[event.response.id]
            reply.status = event.response.status
            reply.done_s = trace_record.time_s
        elif event.type == AUDIO_DELTA:
            assert event.response_id is not None and event.delta is not None
            reply = replies_by_response[event.response_id]
            audio = base64.b64decode(event.delta)
            reply.item_id = event.item_id
            reply.audio.extend(audio)
            reply.chunks.append(
                ReplyAudioChunk(
                    received_s=trace_record.time_s,
                    sample_count=len(audio) // PCM16_BYTES,
                )
            )
        elif event.type in TRANSCRIPT_DELTAS:
            assert event.response_id is not None and event.delta is not None
            replies_by_response[event.response_id].text += event.delta
        elif event.type == "sglang.response.progress":
            assert (
                event.response_id is not None
                and event.text_end is not None
                and event.audio_ms is not None
            )
            replies_by_response[event.response_id].progress.append(
                ProgressPoint(text_end=event.text_end, audio_ms=event.audio_ms)
            )
        elif event.type == "sglang.response.heard":
            assert event.response_id is not None and event.text_end is not None
            replies_by_response[event.response_id].kept_text_ends.append(event.text_end)
        else:
            pass
    return list(replies_by_response.values())


def counting_reply(replies: list[RecordedReply]) -> RecordedReply | None:
    """The first reply that counts at least to one."""
    return next((reply for reply in replies if counted_to(reply.text) > 0), None)


def first_number(transcript: str) -> int | None:
    """The first number said; one and thirty count only when no other number is, as in "the last one" or "counting to thirty"."""
    tokens = re.findall(r"[a-z]+|\d+", transcript.lower().replace("-", " "))
    numbers: list[int] = []
    for index, token in enumerate(tokens):
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token.isdigit():
            numbers.append(int(token))
        elif token in TENS_WORDS:
            ones_value = NUMBER_WORDS.get(next_token, 0)
            numbers.append(TENS_WORDS[token] + (ones_value if ones_value < 10 else 0))
        elif token in NUMBER_WORDS:
            numbers.append(NUMBER_WORDS[token])
        else:
            pass
    other_numbers = [number for number in numbers if number not in (1, COUNT_TARGET)]
    if other_numbers:
        return other_numbers[0]
    elif numbers:
        return numbers[0]
    else:
        return None


def voiced_words(
    words: list[TranscribedWord], reply_samples: npt.NDArray[np.int16]
) -> list[TranscribedWord]:
    """Each word trimmed to its speech in the reply audio; a word without speech is dropped."""
    frame_samples = round(SPEECH_FRAME_S * OUTPUT_SAMPLE_RATE)
    speech_rms = PCM16_FULL_SCALE * 10 ** (SPEECH_DBFS / 20)
    trimmed_words = []
    for word in words:
        voiced_frames = [
            frame_index
            for frame_index in range(
                round(word.start_s / SPEECH_FRAME_S), round(word.end_s / SPEECH_FRAME_S)
            )
            if np.sqrt(
                np.mean(
                    reply_samples[
                        frame_index * frame_samples : (frame_index + 1) * frame_samples
                    ].astype(np.float64)
                    ** 2
                )
            )
            > speech_rms
        ]
        if voiced_frames:
            trimmed_words.append(
                TranscribedWord(
                    word=word.word,
                    start_s=voiced_frames[0] * SPEECH_FRAME_S,
                    end_s=(voiced_frames[-1] + 1) * SPEECH_FRAME_S,
                )
            )
        else:
            pass
    return trimmed_words


def score_run(
    replies: list[RecordedReply],
    words: list[TranscribedWord] | None,
    *,
    start_buffer_ms: float,
) -> PositionScore:
    """Score the counting reply's cut and the next completed reply's answer; words are Whisper's for the counting reply's audio."""
    cut_reply = counting_reply(replies)
    if cut_reply is None:
        return PositionScore(status="no count")
    elif cut_reply.status != "cancelled":
        return PositionScore(status="not interrupted")
    else:
        pass
    assert cut_reply.done_s is not None
    played_samples = played_sample_count(
        cut_reply.chunks,
        sample_rate=OUTPUT_SAMPLE_RATE,
        start_buffer_ms=start_buffer_ms,
        at_s=cut_reply.done_s,
    )
    if (
        cut_reply.truncated_ms is not None
        and cut_reply.truncated_ms != played_samples * 1000 // OUTPUT_SAMPLE_RATE
    ):
        raise ValueError(
            f"{cut_reply.response_id} was truncated at {cut_reply.truncated_ms} ms, not where a {start_buffer_ms} ms start buffer had played"
        )
    else:
        pass
    played_s = played_samples / OUTPUT_SAMPLE_RATE
    delivered_s = len(cut_reply.audio) / PCM16_BYTES / OUTPUT_SAMPLE_RATE
    progress_text_ends = [
        point.text_end
        for point in cut_reply.progress
        if point.audio_ms <= played_s * 1000
    ]
    answer = next(
        (
            reply
            for reply in replies[replies.index(cut_reply) + 1 :]
            if reply.status == "completed" and reply.text
        ),
        None,
    )
    speech_words = (
        voiced_words(words, np.frombuffer(bytes(cut_reply.audio), dtype="<i2"))
        if words is not None
        else None
    )
    return PositionScore(
        status="scored",
        played_s=played_s,
        delivered_s=delivered_s,
        heard_count=(
            counted_to(
                " ".join(word.word for word in speech_words if word.end_s <= played_s)
            )
            if speech_words is not None
            else None
        ),
        delivered_count=(
            counted_to(" ".join(word.word for word in speech_words))
            if speech_words is not None
            else None
        ),
        progress_count=(
            counted_to(cut_reply.text[: max(progress_text_ends)])
            if progress_text_ends
            else None
        ),
        kept_count=(
            counted_to(cut_reply.text[: cut_reply.kept_text_ends[-1]])
            if cut_reply.kept_text_ends
            else None
        ),
        answer_number=first_number(answer.text) if answer is not None else None,
        is_mid_word=(
            any(word.start_s < played_s < word.end_s for word in speech_words)
            if speech_words is not None
            else None
        ),
        is_progress_behind=(
            max(point.audio_ms for point in cut_reply.progress) < played_s * 1000
            if cut_reply.progress
            else None
        ),
    )


def answer_verdict(score: PositionScore) -> AnswerVerdict:
    """Compare the answer with the last number heard whole; the number cut inside a word may also be named."""
    assert score.heard_count is not None
    if score.answer_number is None:
        return "no number"
    elif score.answer_number == score.heard_count:
        return "heard"
    elif score.answer_number == score.heard_count + 1 and score.is_mid_word:
        return "partly heard"
    elif score.answer_number > score.heard_count:
        return "ahead"
    else:
        return "behind"


def comparison_cells(
    counts: list[int | None], heard_counts: list[int]
) -> tuple[str, str, str]:
    """Equal to, ahead of and behind the heard count, among the runs that have a count."""
    pairs = [
        (count, heard_count)
        for count, heard_count in zip(counts, heard_counts)
        if count is not None
    ]
    if pairs:
        return (
            f"{sum(count == heard_count for count, heard_count in pairs)}/{len(pairs)}",
            f"{sum(count > heard_count for count, heard_count in pairs)}/{len(pairs)}",
            f"{sum(count < heard_count for count, heard_count in pairs)}/{len(pairs)}",
        )
    else:
        return ("", "", "")


def position_tables(
    scores_by_mode: dict[ClientMode, list[PositionScore]],
) -> str:
    """History and answer tables over the transcribed runs whose counting reply was cut."""
    history_rows = [
        "| Client | Cut runs | Cut inside a word | Progress behind the cut | Kept = heard | Kept ahead | Kept behind | Progress-timed = heard | Unheard numbers delivered p50 / p90 / max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    answer_rows = [
        "| Client | Cut runs | "
        + " | ".join(f"Answer {verdict}" for verdict in ANSWER_VERDICTS)
        + " |",
        "| --- | --- | " + " | ".join("---" for _ in ANSWER_VERDICTS) + " |",
    ]
    for client_mode, scores in scores_by_mode.items():
        cut_scores = [
            score
            for score in scores
            if score.status == "scored" and score.heard_count is not None
        ]
        heard_counts = [score.heard_count for score in cut_scores]
        behind_scores = [
            score for score in cut_scores if score.is_progress_behind is not None
        ]
        kept_equal, kept_ahead, kept_behind = comparison_cells(
            [score.kept_count for score in cut_scores], heard_counts
        )
        progress_equal, _, _ = comparison_cells(
            [score.progress_count for score in cut_scores], heard_counts
        )
        history_rows.append(
            f"| {client_mode} | {len(cut_scores)} "
            f"| {sum(bool(score.is_mid_word) for score in cut_scores)} "
            f"| {f'{sum(bool(score.is_progress_behind) for score in behind_scores)}/{len(behind_scores)}' if behind_scores else ''} "
            f"| {kept_equal} | {kept_ahead} | {kept_behind} | {progress_equal} "
            f"| {format_quantiles([float(score.delivered_count - score.heard_count) for score in cut_scores])} |"
        )
        verdicts = [answer_verdict(score) for score in cut_scores]
        answer_rows.append(
            f"| {client_mode} | {len(cut_scores)} | "
            + " | ".join(str(verdicts.count(verdict)) for verdict in ANSWER_VERDICTS)
            + " |"
        )
    return "\n".join([*history_rows, "", *answer_rows])
