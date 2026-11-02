# SPDX-License-Identifier: Apache-2.0
"""Utterance boundaries in a Session's mixed audio and control stream."""

from __future__ import annotations

from collections import deque

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
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

    def __init__(self) -> None:
        self.uses_speech_events = False
        self.is_open = False
        self.pending_audio: deque[TimedChunk] = deque()

    def step(self, chunk: TimedChunk) -> list[TimedChunk | str]:
        event_type = control_type(chunk)
        if chunk.modality == AUDIO and not chunk.payload:
            return []
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
            start_ms = float(chunk.payload["t_ms"])
            pending_audio = [
                c for c in self.pending_audio if c.t_start_ms + c.duration_ms > start_ms
            ]
            self.pending_audio.clear()
            return [OPEN_UTTERANCE, *pending_audio]
        elif event_type in (SPEECH_STOPPED, COMMIT) and self.is_open:
            self.is_open = False
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

    def hold(self, chunk: TimedChunk) -> None:
        self.pending_audio.append(chunk)
        horizon_ms = chunk.t_start_ms + chunk.duration_ms - AUDIO_LOOKBACK_MS
        while (
            self.pending_audio[0].t_start_ms + self.pending_audio[0].duration_ms
            <= horizon_ms
        ):
            self.pending_audio.popleft()
