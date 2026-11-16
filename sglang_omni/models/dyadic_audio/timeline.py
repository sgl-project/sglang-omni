# SPDX-License-Identifier: Apache-2.0
"""The user's and the agent's audio of one Session on its input timeline, read back a window at a time at 16 kHz.

The timeline counts samples at the Session input's 16 kHz from the Session's
first input. Two kinds of track hold audio on it, each at its own rate:

- a `TimedTrack` holds audio at the stamps it arrives with: the user's audio,
  or the agent's audio of a full-duplex model that answers every input unit
  with one output unit on the same clock, where silence is the agent not
  speaking;
- a `ReplyTrack` holds the agent's replies of a turn-based speech node, which
  arrive faster than real time and without a place on the timeline; the node
  places each reply by its own rule, and an interrupt cuts the replies of
  its turn and drops their later audio.

Windows of audio at another rate are resampled with the filter context of the
whole signal on both sides, so consecutive windows join as one resampling of
the whole track would.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

from sglang_omni.proto.streams import SESSION_INPUT_SAMPLE_RATE_HZ

TIMELINE_RATE_HZ = SESSION_INPUT_SAMPLE_RATE_HZ
PCM16_SCALE = 32768
# scipy's resample_poly filter reaches 10 * max(up, down) samples of the upsampled signal to each side.
RESAMPLE_FILTER_HALF_WIDTH = 10


def pcm16_samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / PCM16_SCALE


def timeline_sample(t_ms: float) -> int:
    return round(t_ms * TIMELINE_RATE_HZ / 1000)


class Resampler:
    """Audio at one rate as timeline samples at 16 kHz."""

    def __init__(self, rate_hz: int) -> None:
        divisor = math.gcd(TIMELINE_RATE_HZ, rate_hz)
        self.rate_hz = rate_hz
        self.up = TIMELINE_RATE_HZ // divisor
        self.down = rate_hz // divisor
        self.context = RESAMPLE_FILTER_HALF_WIDTH * max(self.up, self.down)

    def timeline_samples(self, native: int) -> int:
        """The timeline samples `native` samples at the track's rate last."""
        return native * self.up // self.down

    def native_samples(self, timeline: int) -> int:
        """The samples at the track's rate in `timeline` timeline samples."""
        return timeline * self.down // self.up

    def window(self, audio: np.ndarray, low: int, high: int) -> np.ndarray:
        """Timeline samples [low, high) of audio at the track's rate whose first sample is timeline sample 0, with filter context on both sides, so windows join as one resampling of the whole audio would."""
        if self.up == self.down:
            return audio[low:high]
        else:
            first = max(low - self.context, 0) // self.up * self.up
        native = audio[
            first
            * self.down
            // self.up : math.ceil((high + self.context) * self.down / self.up)
        ]
        return resample_poly(native, self.up, self.down)[
            low - first : high - first
        ].astype(np.float32)


class TimedTrack:
    """Audio at the stamps it arrives with, kept at its own rate from a little before the oldest sample a later window reads; silence where none arrived, and a later chunk over an earlier one where they overlap."""

    def __init__(self, rate_hz: int = TIMELINE_RATE_HZ) -> None:
        self.resampler = Resampler(rate_hz)
        # The sample at the track's rate audio[0] holds; a multiple of resampler.down, so it falls on a timeline sample.
        self.offset = 0
        self.audio = np.zeros(0, dtype=np.float32)
        self.native_end = 0
        self.is_started = False
        self.is_ended = False

    @property
    def end(self) -> int:
        """The timeline sample after the latest sample that arrived."""
        return self.resampler.timeline_samples(self.native_end)

    @property
    def nbytes(self) -> int:
        return self.audio.nbytes

    def append(self, t_start_ms: float, samples: np.ndarray, *, eos: bool) -> None:
        native = round(t_start_ms * self.resampler.rate_hz / 1000)
        if not self.is_started:
            self.offset = native // self.resampler.down * self.resampler.down
            self.is_started = True
        else:
            pass
        stop = native + samples.size
        missing = stop - self.offset - self.audio.size
        if missing > 0:
            self.audio = np.concatenate(
                [self.audio, np.zeros(missing, dtype=np.float32)]
            )
        else:
            pass
        low = max(native, self.offset)
        if stop > low:
            self.audio[low - self.offset : stop - self.offset] = samples[low - native :]
        else:
            pass
        self.native_end = max(self.native_end, stop)
        self.is_ended = self.is_ended or eos

    def window(self, start: int, length: int) -> np.ndarray:
        """Timeline samples [start, start + length) at 16 kHz, silence where none arrived."""
        window = np.zeros(length, dtype=np.float32)
        origin = self.resampler.timeline_samples(self.offset)
        low = max(start, origin)
        high = min(
            start + length,
            origin + self.resampler.timeline_samples(self.audio.size),
        )
        if high > low:
            window[low - start : high - start] = self.resampler.window(
                self.audio, low - origin, high - origin
            )
        else:
            pass
        return window

    def drop_before(self, sample: int) -> None:
        """Forget the audio no window from timeline sample `sample` on reads."""
        keep = (
            self.resampler.native_samples(max(sample - self.resampler.context, 0))
            // self.resampler.down
            * self.resampler.down
        )
        if keep > self.offset:
            self.audio = self.audio[keep - self.offset :].copy()
            self.offset = keep
        else:
            pass


@dataclass(kw_only=True)
class Reply:
    """The agent's audio of one turn at the speech rate; start is the timeline sample its first sample plays at, None until placed."""

    turn: int | None
    audio: np.ndarray
    start: int | None = None
    is_done: bool = False


@dataclass(kw_only=True)
class ReplyTrack:
    """The agent's replies of a turn-based speech node, in the order they arrived."""

    resampler: Resampler
    replies: list[Reply] = field(default_factory=list)
    # Turns an interrupt cut, whose later audio is dropped; None when an interrupt named no turn.
    interrupted_turns: set[int | None] = field(default_factory=set)

    @classmethod
    def at_rate(cls, rate_hz: int) -> ReplyTrack:
        return cls(resampler=Resampler(rate_hz))

    @property
    def nbytes(self) -> int:
        return sum(reply.audio.nbytes for reply in self.replies)

    def length(self, reply: Reply) -> int:
        """The timeline samples the reply's audio lasts."""
        return self.resampler.timeline_samples(reply.audio.size)

    def end(self, reply: Reply) -> int:
        """The timeline sample after a placed reply's last sample."""
        assert reply.start is not None
        return reply.start + self.length(reply)

    def placed_end(self) -> int:
        return max(
            (self.end(reply) for reply in self.replies if reply.start is not None),
            default=0,
        )

    def append_following(
        self, turn: int | None, samples: np.ndarray, *, cursor: int, boundary: int
    ) -> None:
        """Place audio as it arrives. A reply's first audio starts at the cursor, or at the first multiple of `boundary` after the previous reply when that is later. Its later audio follows on, unless the cursor is already past the reply's end, where it then starts."""
        last = self.replies[-1] if self.replies else None
        if turn in self.interrupted_turns:
            pass
        elif last is not None and last.turn == turn and self.end(last) >= cursor:
            last.audio = np.concatenate([last.audio, samples])
        elif last is not None and last.turn == turn:
            self.replies.append(Reply(turn=turn, start=cursor, audio=samples.copy()))
        else:
            previous_end = 0 if last is None else self.end(last)
            start = max(cursor, -(-previous_end // boundary) * boundary)
            self.replies.append(Reply(turn=turn, start=start, audio=samples.copy()))

    def append_pending(self, turn: int | None, samples: np.ndarray) -> None:
        """Add audio to its turn's open reply, or open one, for `place` to place."""
        reply = next(
            (
                reply
                for reply in self.replies
                if reply.turn == turn and not reply.is_done
            ),
            None,
        )
        if turn in self.interrupted_turns:
            pass
        elif reply is None:
            self.replies.append(Reply(turn=turn, audio=samples.copy()))
        else:
            reply.audio = np.concatenate([reply.audio, samples])

    def finish(self, turn: int | None) -> None:
        """The turn's reply has all its audio."""
        for reply in self.replies:
            if reply.turn == turn:
                reply.is_done = True
            else:
                pass

    def place(self, *, cursor: int, boundary: int, min_length: int) -> None:
        """Start each unplaced reply, in arrival order, at the first multiple of `boundary` at or after the cursor and the previous reply's end, once `min_length` timeline samples of it arrived or it is done."""
        end = cursor
        for reply in self.replies:
            length = self.length(reply)
            if reply.start is not None:
                end = max(end, reply.start + length)
            elif length >= min_length or reply.is_done:
                reply.start = math.ceil(end / boundary) * boundary
                end = reply.start + length
            else:
                break

    def interrupt(self, turn: int | None, *, at: int) -> None:
        """Cut the replies of `turn`, or every reply when it is None, at timeline sample `at`: a placed one keeps what plays before it, an unplaced one is dropped, and their later audio is dropped."""
        kept = []
        for reply in self.replies:
            if turn is not None and reply.turn != turn:
                kept.append(reply)
            elif reply.start is None:
                self.interrupted_turns.add(reply.turn)
            else:
                reply.audio = reply.audio[
                    : self.resampler.native_samples(max(at - reply.start, 0))
                ]
                reply.is_done = True
                self.interrupted_turns.add(reply.turn)
                kept.append(reply)
        self.interrupted_turns.add(turn)
        self.replies = kept

    def drop_ended(self, before: int) -> None:
        """Forget the placed replies that end by `before`, but the last, where the next one is placed after."""
        self.replies = [
            reply
            for reply in self.replies
            if reply.start is None
            or self.end(reply) > before
            or reply is self.replies[-1]
        ]

    def drop_finished(self, before: int) -> None:
        """Forget the done, placed replies that end by `before`."""
        self.replies = [
            reply
            for reply in self.replies
            if not reply.is_done or reply.start is None or self.end(reply) > before
        ]

    def window(self, start: int, length: int) -> np.ndarray:
        """Timeline samples [start, start + length) of the placed replies at 16 kHz, silence where none plays."""
        window = np.zeros(length, dtype=np.float32)
        for reply in self.replies:
            if reply.start is None:
                continue
            else:
                low = max(start, reply.start) - reply.start
                high = min(start + length, self.end(reply)) - reply.start
            if high > low:
                offset = reply.start + low - start
                window[offset : offset + high - low] = self.resampler.window(
                    reply.audio, low, high
                )
            else:
                pass
        return window

    def turn_at(self, start: int, length: int) -> int | None:
        """The turn of the first placed reply playing in [start, start + length), None when none plays."""
        return next(
            (
                reply.turn
                for reply in self.replies
                if reply.start is not None
                and reply.start < start + length
                and self.end(reply) > start
            ),
            None,
        )


@dataclass(kw_only=True)
class DyadicAudio:
    """One Session's two parties: the user's audio, and the agent's audio as turn replies or as a stream of units on the input clock."""

    user: TimedTrack
    replies: ReplyTrack
    units: TimedTrack
    # The speech node streams the agent's audio as units from the Session's start, so a window waits for them before the first one arrives too.
    reply_stream: bool = False

    @classmethod
    def at_rates(
        cls,
        *,
        user_rate_hz: int = TIMELINE_RATE_HZ,
        speech_rate_hz: int = TIMELINE_RATE_HZ,
        reply_stream: bool = False,
    ) -> DyadicAudio:
        return cls(
            user=TimedTrack(user_rate_hz),
            replies=ReplyTrack.at_rate(speech_rate_hz),
            units=TimedTrack(speech_rate_hz),
            reply_stream=reply_stream,
        )

    @property
    def nbytes(self) -> int:
        return self.user.nbytes + self.replies.nbytes + self.units.nbytes

    def speech_window(self, start: int, length: int) -> np.ndarray:
        """The agent's audio of [start, start + length) at 16 kHz: its placed replies, or its units."""
        if self.units.is_started:
            return self.units.window(start, length)
        else:
            return self.replies.window(start, length)

    def speech_end(self) -> int:
        """The timeline sample after the agent's last placed sample."""
        return max(self.replies.placed_end(), self.units.end)

    def awaits_units(self, until: int) -> bool:
        """Whether a window up to timeline sample `until` still waits for the agent's units: they stream and have not reached it. Before the first unit arrives a window waits only with `reply_stream`; without it nothing waits then, as a full-duplex speech node answers the Session's first input unit well before the user's audio covers a window."""
        return (
            (self.units.is_started or self.reply_stream)
            and not self.units.is_ended
            and self.units.end < until
        )


__all__ = [
    "DyadicAudio",
    "PCM16_SCALE",
    "Reply",
    "ReplyTrack",
    "Resampler",
    "TimedTrack",
    "pcm16_samples",
    "timeline_sample",
]
