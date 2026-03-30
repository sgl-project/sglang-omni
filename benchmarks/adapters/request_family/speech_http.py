from __future__ import annotations

import base64

import aiohttp

from benchmarks.core.types import PerRequestResult

from .base import RequestFamilyAdapter, apply_usage, parse_sse_event, wav_duration


class SpeechHTTPAdapter(RequestFamilyAdapter):
    family_name = "speech_http"
    endpoint_path = "/v1/audio/speech"

    async def _handle_non_stream_response(
        self,
        response: aiohttp.ClientResponse,
        result: PerRequestResult,
    ) -> None:
        audio_bytes = await response.read()
        result.audio_duration_s = wav_duration(audio_bytes)
        result.success = result.audio_duration_s > 0
        if not result.success:
            result.error = (
                f"Empty or invalid audio response ({len(audio_bytes)} bytes)"
            )
            return

        result.audio_bytes = audio_bytes

        usage = {
            "prompt_tokens": response.headers.get("X-Prompt-Tokens"),
            "completion_tokens": response.headers.get("X-Completion-Tokens"),
            "engine_time_s": response.headers.get("X-Engine-Time"),
        }
        apply_usage(result, usage)

    async def _handle_stream_response(
        self,
        response: aiohttp.ClientResponse,
        result: PerRequestResult,
    ) -> None:
        total_audio_duration = 0.0
        usage: dict | None = None
        buffer = bytearray()

        async for chunk in response.content.iter_any():
            buffer.extend(chunk)
            while b"\n" in buffer:
                idx = buffer.index(b"\n")
                raw_line = bytes(buffer[:idx])
                del buffer[:idx + 1]
                line = raw_line.decode("utf-8", errors="replace").strip()
                total_audio_duration, usage = _process_sse_line(
                    line, total_audio_duration, usage
                )
        if buffer.strip():
            line = bytes(buffer).decode("utf-8", errors="replace").strip()
            total_audio_duration, usage = _process_sse_line(
                line, total_audio_duration, usage
            )

        result.audio_duration_s = total_audio_duration
        result.success = total_audio_duration > 0
        if not result.success:
            result.error = "No audio chunks received from streaming speech response"
            return
        apply_usage(result, usage)


def _process_sse_line(
    line: str,
    total_duration: float,
    usage: dict | None,
) -> tuple[float, dict | None]:
    event = parse_sse_event(line)
    if event is None:
        return total_duration, usage
    audio = event.get("audio")
    if isinstance(audio, dict) and audio.get("data"):
        total_duration += wav_duration(base64.b64decode(audio["data"]))
    event_usage = event.get("usage")
    if isinstance(event_usage, dict):
        usage = event_usage
    return total_duration, usage
