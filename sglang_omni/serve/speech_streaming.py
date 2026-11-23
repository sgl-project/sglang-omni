# SPDX-License-Identifier: Apache-2.0
"""Speech audio streaming, event framing, and terminal usage."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import anyio
import numpy as np
import soxr

from sglang_omni.client.audio import (
    DEFAULT_SAMPLE_RATE,
    apply_speed,
    encode_pcm,
    select_audio_delta,
)
from sglang_omni.client.types import GenerateChunk
from sglang_omni.serve.protocol import SpeechStreamOutcome
from sglang_omni.serve.speech_errors import speech_generation_error
from sglang_omni.serve.streaming import ClosableStreamingResponse

logger = logging.getLogger(__name__)
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_CHANNELS = 1
STREAMING_WAV_SIZE = 0xFFFFFFFF


def create_streaming_wav_header(sample_rate: int) -> bytes:
    """Describe a mono PCM stream whose final size is not known yet."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        STREAMING_WAV_SIZE,
        b"WAVE",
        b"fmt ",
        16,
        1,
        PCM_CHANNELS,
        sample_rate,
        sample_rate * PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES,
        PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES,
        PCM_SAMPLE_WIDTH_BYTES * 8,
        b"data",
        STREAMING_WAV_SIZE,
    )


@dataclass(kw_only=True)
class SpeechAudioStream:
    """Encode mono PCM while preserving resampling state across audio chunks."""

    output_sample_rate: int | None
    speed: float
    emitted_samples: int = field(default=0, init=False)
    source_sample_rate: int | None = field(default=None, init=False)
    resampler: soxr.ResampleStream | None = field(default=None, init=False)
    is_finished: bool = field(default=False, init=False)

    @property
    def sample_rate(self) -> int:
        return self.output_sample_rate or self.source_sample_rate or DEFAULT_SAMPLE_RATE

    def push(self, chunk: GenerateChunk) -> bytes | None:
        assert not self.is_finished
        source_sample_rate = chunk.sample_rate or DEFAULT_SAMPLE_RATE
        assert source_sample_rate > 0
        if self.source_sample_rate is None:
            self.source_sample_rate = source_sample_rate
            if self.sample_rate != source_sample_rate:
                self.resampler = soxr.ResampleStream(
                    source_sample_rate, self.sample_rate, num_channels=1
                )
            else:
                pass
        elif source_sample_rate != self.source_sample_rate:
            raise RuntimeError(
                f"Speech stream sample rate changed from {self.source_sample_rate} to {source_sample_rate}"
            )
        else:
            pass
        audio, self.emitted_samples = select_audio_delta(
            chunk.audio_data,
            emitted_samples=self.emitted_samples,
            is_terminal=chunk.finish_reason is not None,
        )
        if audio is None:
            return None
        else:
            pass
        if self.speed != 1.0:
            audio, _ = apply_speed(audio, self.speed, source_sample_rate)
        else:
            pass
        if self.resampler is not None:
            audio = self.resampler.resample_chunk(audio)
        else:
            pass
        return encode_pcm(audio, self.sample_rate) or None

    def finish(self) -> bytes | None:
        if self.is_finished:
            return None
        else:
            self.is_finished = True
        if self.resampler is None:
            return None
        else:
            audio = self.resampler.resample_chunk(
                np.empty(0, dtype=np.float32), last=True
            )
            return encode_pcm(audio, self.sample_rate) or None


def create_speech_sse_response(
    pcm_chunks: AsyncGenerator[bytes, None],
    *,
    stream_outcome: SpeechStreamOutcome,
    sample_rate: int,
    response_format: Literal["pcm", "wav"],
    enable_speech_sse_named_events: bool,
    headers: dict[str, str],
) -> ClosableStreamingResponse:
    """Wrap PCM chunks in SSE events with terminal metadata."""

    def sse_frame(event_type: str, event_json: str) -> str:
        # note (Xin Gao): existing clients listen for unnamed "message" events.
        if enable_speech_sse_named_events:
            return f"event: {event_type}\ndata: {event_json}\n\n"
        else:
            return f"data: {event_json}\n\n"

    async def stream_events() -> AsyncIterator[str]:
        emitted_audio = False
        duration_seconds = 0.0
        try:
            async for pcm_bytes in pcm_chunks:
                audio_packets: tuple[bytes, ...]
                if not emitted_audio and response_format == "wav":
                    audio_packets = (
                        create_streaming_wav_header(sample_rate),
                        pcm_bytes,
                    )
                else:
                    audio_packets = (pcm_bytes,)

                duration_seconds += len(pcm_bytes) / (
                    PCM_SAMPLE_WIDTH_BYTES * PCM_CHANNELS * sample_rate
                )
                for audio_packet in audio_packets:
                    delta = {
                        "type": "speech.audio.delta",
                        "audio": base64.b64encode(audio_packet).decode("ascii"),
                        "response_format": response_format,
                    }
                    emitted_audio = True
                    yield sse_frame(delta["type"], json.dumps(delta))

            final_usage = stream_outcome.usage
            usage: dict[str, int | None] | None
            if final_usage is None:
                usage = None
            else:
                input_tokens = (
                    final_usage.speech_input_tokens
                    if final_usage.speech_input_tokens is not None
                    else final_usage.prompt_tokens
                )
                output_tokens = (
                    final_usage.speech_output_tokens
                    if final_usage.speech_output_tokens is not None
                    else final_usage.completion_tokens
                )
                total_tokens = (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else final_usage.total_tokens
                )
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            done = {
                "type": "speech.audio.done",
                "duration": round(duration_seconds, 3),
                "usage": usage,
                "finish_reason": stream_outcome.finish_reason,
            }
            if stream_outcome.sglext is not None:
                done["sglext"] = stream_outcome.sglext.model_dump()
            else:
                pass
            yield sse_frame(done["type"], json.dumps(done))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                f"Speech SSE stream failed for request {stream_outcome.request_id}"
            )
            failure = speech_generation_error(exc)
            error: dict[str, str | int | bool | None] = {
                "message": failure.message,
                "type": failure.error_type,
                "param": failure.param,
                "code": (
                    failure.code if failure.code is not None else failure.status_code
                ),
            }
            if emitted_audio:
                error.update(partial_audio=True, action="discard")
            else:
                pass
            error_type = (
                "speech.audio.error" if enable_speech_sse_named_events else "error"
            )
            payload = {"type": error_type, "error": error}
            yield sse_frame(error_type, json.dumps(payload))
        finally:
            with anyio.CancelScope(shield=True):
                await pcm_chunks.aclose()

    return ClosableStreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers=headers,
    )
