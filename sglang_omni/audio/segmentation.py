# SPDX-License-Identifier: Apache-2.0
"""Utterance boundaries in a Session's mixed audio and control stream."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    PCM16_BYTES_PER_SAMPLE,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    control_type,
)

# note (Junnan Li): Retain audio so late speech.started events can recover the utterance onset.
AUDIO_LOOKBACK_MS = 2000
OPEN_UTTERANCE = "open"
FINISH_UTTERANCE = "finish"
EMPTY_UTTERANCE = "empty"


class UtteranceSegmenter:
    """Places utterance boundaries in a recognizer's mixed input stream.

    Until a speech.started arrives, audio opens an utterance and only commit
    or the end of input closes it. After one, the recognizer is gated:
    speech.started opens an utterance with the held audio from its t_ms on,
    speech.stopped closes it, and audio between utterances is only held.
    """

    def __init__(
        self,
        *,
        uses_speech_events: bool = False,
        should_crop_audio: bool = False,
    ) -> None:
        self.uses_speech_events: bool = uses_speech_events
        self.should_crop_audio: bool = should_crop_audio
        self.is_open: bool = False
        self.pending_audio: deque[TimedChunk] = deque()
        self.speech_start_ms: float | None = None

    def step(self, chunk: TimedChunk) -> list[TimedChunk | str]:
        event_type = control_type(chunk)
        if chunk.modality == AUDIO and not chunk.payload:
            return []
        elif chunk.modality == AUDIO and self.is_open and self.should_crop_audio:
            return self.utterance_audio(chunk)
        elif chunk.modality == AUDIO and self.is_open:
            return [chunk]
        elif chunk.modality == AUDIO and not self.uses_speech_events:
            self.is_open = True
            return [OPEN_UTTERANCE, chunk]
        elif chunk.modality == AUDIO:
            self.hold(chunk)
            return []
        elif event_type == SPEECH_STARTED and self.is_open:
            self.uses_speech_events = True
            return []
        elif event_type == SPEECH_STARTED:
            self.uses_speech_events = True
            self.is_open = True
            self.speech_start_ms = float(chunk.payload["t_ms"])
            pending_audio = [
                cropped_audio
                for audio_chunk in self.pending_audio
                for cropped_audio in self.utterance_audio(audio_chunk)
            ]
            self.pending_audio.clear()
            return [OPEN_UTTERANCE, *pending_audio]
        elif event_type in (SPEECH_STOPPED, COMMIT) and self.is_open:
            self.is_open = False
            self.speech_start_ms = None
            return [FINISH_UTTERANCE]
        elif event_type == COMMIT:
            return [EMPTY_UTTERANCE]
        else:
            return []

    def finish(self) -> list[TimedChunk | str]:
        if self.is_open:
            self.is_open = False
            return [FINISH_UTTERANCE]
        else:
            return []

    def utterance_audio(self, chunk: TimedChunk) -> list[TimedChunk]:
        if self.speech_start_ms is None or chunk.t_start_ms >= self.speech_start_ms:
            return [chunk]
        elif chunk.t_start_ms + chunk.duration_ms <= self.speech_start_ms:
            return []
        elif not self.should_crop_audio:
            return [chunk]
        else:
            skipped_sample_count = round(
                (self.speech_start_ms - chunk.t_start_ms)
                * SESSION_INPUT_SAMPLE_RATE_HZ
                / 1000
            )
            skipped_duration_ms = (
                skipped_sample_count * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ
            )
            return [
                replace(
                    chunk,
                    t_start_ms=chunk.t_start_ms + skipped_duration_ms,
                    duration_ms=chunk.duration_ms - skipped_duration_ms,
                    payload=chunk.payload[
                        skipped_sample_count * PCM16_BYTES_PER_SAMPLE :
                    ],
                )
            ]

    def hold(self, chunk: TimedChunk) -> None:
        self.pending_audio.append(chunk)
        horizon_ms = chunk.t_start_ms + chunk.duration_ms - AUDIO_LOOKBACK_MS
        while (
            self.pending_audio[0].t_start_ms + self.pending_audio[0].duration_ms
            <= horizon_ms
        ):
            self.pending_audio.popleft()
