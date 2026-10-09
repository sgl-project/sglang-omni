# SPDX-License-Identifier: Apache-2.0
"""Realtime transcription over one socket: segments refreshed on a cadence, then finalized."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import threading
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sglang_omni_mlx.qwen3_asr.audio import peak_is_silent
from sglang_omni_mlx.qwen3_asr.transcriber import (
    TranscriptionOptions,
    TranscriptionResult,
    normalize_language,
)
from sglang_omni_mlx.qwen3_asr.worker import TranscriptionWorker
from sglang_omni_mlx.transcription import TranscriptionCancelled
from sglang_omni_mlx.wav import PCM16_FULL_SCALE, SAMPLE_RATE

logger = logging.getLogger(__name__)

# A refresh re-decodes the whole segment; after two refreshes it continues from
# the shown text, minus its last tokens, which may still change.
PREFIX_AFTER_REFRESH_COUNT = 2
PREFIX_ROLLBACK_TOKEN_COUNT = 5
SILENT_PEAK = 1e-3
UNSPACED_SCRIPT_RANGES = (
    (0x0E00, 0x0EFF),
    (0x1000, 0x109F),
    (0x1780, 0x17FF),
    (0x2E80, 0x303F),
    (0x3040, 0x30FF),
    (0x3400, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
    (0x20000, 0x2FA1F),
)


@dataclass(frozen=True, kw_only=True)
class RealtimeSettings:
    decode_interval_samples: int
    first_decode_samples: int
    max_segment_samples: int


class EventSender(Protocol):
    async def __call__(self, event: dict[str, object]) -> None: ...


@dataclass(kw_only=True)
class Segment:
    segment_id: int
    start_sample: int
    next_refresh_sample: int
    language: str | None
    decode_count: int = 0
    transcript: str = ""
    last_text: str = ""


def is_spaced_script(character: str) -> bool:
    code_point = ord(character)
    return not character.isspace() and not any(
        low <= code_point <= high for low, high in UNSPACED_SCRIPT_RANGES
    )


def join_transcript_parts(parts: Iterable[str]) -> str:
    """Join segment texts, with a space only between two spaced scripts."""
    joined = ""
    for part in (part.strip() for part in parts):
        if not part:
            continue
        elif joined and is_spaced_script(joined[-1]) and is_spaced_script(part[0]):
            joined += " " + part
        else:
            joined += part
    return joined


class RealtimeSession:
    """Manual-turn session: audio becomes one segment, cut every max segment length."""

    def __init__(
        self,
        *,
        worker: TranscriptionWorker,
        settings: RealtimeSettings,
        send: EventSender,
    ) -> None:
        self.worker = worker
        self.settings = settings
        self.send_event = send
        self.language: str | None = None
        self.samples = np.zeros(0, dtype=np.float32)
        self.buffer_start_sample = 0
        self.segment: Segment | None = None
        self.next_segment_id = 0
        self.committed: list[tuple[int, str]] = []
        self.event_index = 0
        self.decode_lock = asyncio.Lock()
        self.refresh_task: asyncio.Task[None] | None = None
        self.cancel_flag = threading.Event()

    @property
    def end_sample(self) -> int:
        return self.buffer_start_sample + len(self.samples)

    async def send(self, event: dict[str, object]) -> None:
        self.event_index += 1
        await self.send_event(
            {
                **event,
                "event_id": f"evt_{uuid.uuid4().hex}",
                "event_index": self.event_index,
            }
        )

    async def send_error(self, error_type: str, code: str, message: str) -> None:
        await self.send(
            {
                "type": "error",
                "error": {"type": error_type, "code": code, "message": message},
            }
        )

    async def handle(self, message: Mapping[str, object]) -> bool:
        """Apply one client event; False once the session has completed."""
        event_type = message.get("type")
        if event_type == "session.update":
            session = message.get("session")
            if (
                not isinstance(session, Mapping)
                or session.get("turn_detection") is not None
            ):
                await self.send_error(
                    "invalid_request_error",
                    "unsupported_session",
                    "Only manual turns are supported.",
                )
            else:
                language = session.get("language")
                self.language = (
                    normalize_language(language) if isinstance(language, str) else None
                )
                await self.send({"type": "transcription_session.updated"})
            return True
        elif event_type == "input_audio_buffer.append":
            await self.append(message.get("audio"))
            return True
        elif event_type == "input_audio_buffer.commit":
            await self.finalize_through(self.end_sample)
            return True
        elif event_type == "transcription.done":
            await self.finalize_through(self.end_sample)
            await self.send(
                {
                    "type": "transcription.completed",
                    "text": join_transcript_parts(
                        text for _, text in sorted(self.committed)
                    ),
                }
            )
            return False
        else:
            await self.send_error(
                "invalid_request_error", "invalid_event", "Unknown event type."
            )
            return True

    async def append(self, audio: object) -> None:
        try:
            pcm = (
                base64.b64decode(audio, validate=False)
                if isinstance(audio, str)
                else b""
            )
        except (ValueError, binascii.Error):
            pcm = b""
        if not pcm or len(pcm) % 2:
            await self.send_error(
                "invalid_request_error", "invalid_audio", "Audio must be base64 PCM16."
            )
            return
        else:
            pass
        start_sample = self.end_sample
        self.samples = np.concatenate(
            [
                self.samples,
                np.frombuffer(pcm, dtype="<i2").astype(np.float32) / PCM16_FULL_SCALE,
            ]
        )
        if self.segment is None:
            self.start_segment(start_sample)
        else:
            pass
        overflow = (
            self.end_sample - self.segment.start_sample
        ) // self.settings.max_segment_samples
        if overflow > 0:
            cut_sample = (
                self.segment.start_sample + overflow * self.settings.max_segment_samples
            )
            await self.finalize_through(cut_sample)
            self.start_segment(cut_sample)
        else:
            pass
        if self.refresh_task is None or self.refresh_task.done():
            self.refresh_task = asyncio.create_task(self.refresh())
            self.refresh_task.add_done_callback(log_refresh_failure)
        else:
            pass

    def start_segment(self, start_sample: int) -> None:
        self.segment = Segment(
            segment_id=self.next_segment_id,
            start_sample=start_sample,
            next_refresh_sample=start_sample + self.settings.first_decode_samples,
            language=self.language,
        )
        self.next_segment_id += 1

    def segment_samples(self, segment: Segment, end_sample: int) -> np.ndarray:
        return self.samples[
            segment.start_sample
            - self.buffer_start_sample : end_sample
            - self.buffer_start_sample
        ]

    async def refresh(self) -> None:
        """Re-decode the open segment whenever enough new audio has arrived."""
        while True:
            segment = self.segment
            if segment is None or self.end_sample < segment.next_refresh_sample:
                return
            else:
                pass
            end_sample = self.end_sample
            samples = self.segment_samples(segment, end_sample)
            silent = peak_is_silent(samples, SILENT_PEAK)
            # Leading silence keeps the early first decode for the first audible audio.
            if not (silent and segment.decode_count == 0):
                segment.next_refresh_sample = (
                    end_sample + self.settings.decode_interval_samples
                )
            else:
                pass
            if silent:
                return
            else:
                pass
            async with self.decode_lock:
                if self.segment is not segment:
                    return
                else:
                    pass
                text = await self.decode(segment, samples)
            if (
                text is not None
                and self.segment is segment
                and text != segment.last_text
            ):
                segment.last_text = text
                await self.send(
                    {
                        "type": "transcription.segment",
                        "segment_id": segment.segment_id,
                        "text": text,
                        "is_final": False,
                    }
                )
            else:
                pass

    async def decode(self, segment: Segment, samples: np.ndarray) -> str | None:
        use_prefix = (
            segment.decode_count >= PREFIX_AFTER_REFRESH_COUNT
            and bool(segment.transcript)
            and segment.language is not None
        )
        prefix_ids, prefix_text = (
            self.worker.transcriber.retained_prefix(
                segment.transcript, PREFIX_ROLLBACK_TOKEN_COUNT
            )
            if use_prefix
            else ((), "")
        )
        segment.decode_count += 1
        try:
            result: TranscriptionResult = await self.worker.transcribe(
                samples,
                TranscriptionOptions(
                    language=segment.language,
                    prefix_token_ids=prefix_ids,
                    prefix_text=prefix_text,
                ),
                self.cancel_flag,
            )
        except TranscriptionCancelled:
            return None
        except Exception as error:
            # Any failure ends this decode with an error event the client acts
            # on; the type alone is logged, never the audio or text.
            logger.error(f"realtime decode failed: {type(error).__name__}")
            await self.send_error(
                "server_error", "transcription_failed", "Transcription failed."
            )
            return None
        if result.language:
            segment.language = result.language
        else:
            pass
        segment.transcript = result.text
        return result.text

    async def finalize_through(self, end_sample: int) -> None:
        """Final-decode the open segment up to end_sample, cutting at the segment limit."""
        while self.segment is not None and end_sample > self.segment.start_sample:
            segment = self.segment
            cut_sample = min(
                end_sample, segment.start_sample + self.settings.max_segment_samples
            )
            async with self.decode_lock:
                self.segment = None
                samples = self.segment_samples(segment, cut_sample)
                if peak_is_silent(samples, SILENT_PEAK):
                    text = ""
                else:
                    text = await self.decode(segment, samples)
            # A failed or cancelled decode has already been reported (or the
            # client left); its segment is dropped rather than committed empty.
            if text is not None:
                self.committed.append((segment.segment_id, text))
                await self.send(
                    {
                        "type": "transcription.segment",
                        "segment_id": segment.segment_id,
                        "text": text,
                        "is_final": True,
                    }
                )
            else:
                pass
            self.samples = self.samples[cut_sample - self.buffer_start_sample :]
            self.buffer_start_sample = cut_sample
            if cut_sample < end_sample:
                self.start_segment(cut_sample)
            else:
                pass

    def close(self) -> None:
        """The client left: stop any decode in flight."""
        self.cancel_flag.set()


def log_refresh_failure(task: asyncio.Task[None]) -> None:
    """A refresh that died (say, on a closed socket) is logged by type only."""
    if not task.cancelled() and task.exception() is not None:
        logger.error(f"realtime refresh failed: {type(task.exception()).__name__}")
    else:
        pass


def realtime_settings(
    decode_interval_ms: int, first_decode_ms: int, max_segment_seconds: float
) -> RealtimeSettings:
    return RealtimeSettings(
        decode_interval_samples=decode_interval_ms * SAMPLE_RATE // 1000,
        first_decode_samples=first_decode_ms * SAMPLE_RATE // 1000,
        max_segment_samples=int(max_segment_seconds * SAMPLE_RATE),
    )
