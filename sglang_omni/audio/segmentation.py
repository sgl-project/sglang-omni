# SPDX-License-Identifier: Apache-2.0
"""Utterance boundaries in a Session's mixed audio and control stream."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    PCM16,
    SEGMENTATION_MODE,
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

    def __init__(self) -> None:
        self.uses_speech_events = False
        self.is_open = False
        self.pending_audio: deque[TimedChunk] = deque()
        self.timeline_origin_ms: float | None = None

    def step(self, chunk: TimedChunk) -> list[TimedChunk | str]:
        event_type = control_type(chunk)
        if chunk.modality == AUDIO and not chunk.payload:
            return []
        elif event_type == SEGMENTATION_MODE:
            enabled = chunk.payload.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError("segmentation mode needs a boolean enabled field")
            else:
                pass
            self.uses_speech_events = enabled
            if not enabled:
                self.pending_audio.clear()
            else:
                pass
            return []
        elif chunk.modality == AUDIO and self.is_open:
            return [chunk]
        elif chunk.modality == AUDIO and not self.uses_speech_events:
            self.set_timeline_origin(chunk.t_start_ms)
            self.is_open = True
            return [OPEN_UTTERANCE, chunk]
        elif chunk.modality == AUDIO:
            self.set_timeline_origin(chunk.t_start_ms)
            self.hold(chunk)
            return []
        elif event_type == SPEECH_STARTED and self.is_open:
            self.uses_speech_events = True
            return []
        elif event_type == SPEECH_STARTED:
            self.uses_speech_events = True
            self.is_open = True
            start_ms = float(chunk.payload["t_ms"])
            self.set_timeline_origin(start_ms)
            start_sample = self.sample_at_time_ms(start_ms)
            pending_audio = []
            for pending_chunk in self.pending_audio:
                selected_chunk = self.slice_audio_from_sample(
                    pending_chunk, start_sample
                )
                if selected_chunk is not None:
                    pending_audio.append(selected_chunk)
                else:
                    pass
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

    def set_timeline_origin(self, time_ms: float) -> None:
        if self.timeline_origin_ms is None:
            self.timeline_origin_ms = time_ms
        else:
            pass

    def sample_at_time_ms(self, time_ms: float) -> int:
        if self.timeline_origin_ms is None:
            self.set_timeline_origin(time_ms)
        else:
            pass
        assert self.timeline_origin_ms is not None
        return round(
            (time_ms - self.timeline_origin_ms) * SESSION_INPUT_SAMPLE_RATE_HZ / 1000
        )

    def time_at_sample(self, sample_offset: int) -> float:
        assert self.timeline_origin_ms is not None
        return (
            self.timeline_origin_ms
            + sample_offset * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ
        )

    def slice_audio_from_sample(
        self, chunk: TimedChunk, start_sample: int
    ) -> TimedChunk | None:
        if chunk.format != PCM16 or not isinstance(chunk.payload, bytes):
            return chunk
        elif len(chunk.payload) % 2:
            raise ValueError("PCM16 byte length must contain whole samples")
        else:
            pass
        chunk_start_sample = self.sample_at_time_ms(chunk.t_start_ms)
        chunk_sample_count = len(chunk.payload) // 2
        chunk_end_sample = chunk_start_sample + chunk_sample_count
        slice_start_sample = max(chunk_start_sample, start_sample)
        if slice_start_sample >= chunk_end_sample:
            return None
        else:
            pass
        payload_offset = (slice_start_sample - chunk_start_sample) * 2
        payload = chunk.payload[payload_offset:]
        sample_count = len(payload) // 2
        return replace(
            chunk,
            t_start_ms=self.time_at_sample(slice_start_sample),
            duration_ms=sample_count * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ,
            payload=payload,
        )
