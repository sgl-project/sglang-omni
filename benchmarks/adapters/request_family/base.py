from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from benchmarks.core.types import PerRequestResult, PreparedRequest

WAV_HEADER_SIZE = 44
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"


class RequestFamilyAdapter(ABC):
    family_name: str
    endpoint_path: str

    async def execute(
        self,
        *,
        session: aiohttp.ClientSession,
        base_url: str,
        request: PreparedRequest,
    ) -> PerRequestResult:
        result = PerRequestResult(
            request_id=request.request_id,
            input_preview=request.input_preview,
        )
        start_time = time.perf_counter()
        url = f"{base_url}{self.endpoint_path}"

        try:
            async with session.post(url, json=request.payload) as response:
                if response.status != 200:
                    body_text = await response.text()
                    result.error = f"HTTP {response.status}: {body_text}"
                elif request.payload.get("stream"):
                    await self._handle_stream_response(response, result)
                else:
                    await self._handle_non_stream_response(response, result)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time

        if result.audio_duration_s and result.audio_duration_s > 0:
            result.rtf = result.latency_s / result.audio_duration_s
        return result

    @abstractmethod
    async def _handle_stream_response(
        self,
        response: aiohttp.ClientResponse,
        result: PerRequestResult,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def _handle_non_stream_response(
        self,
        response: aiohttp.ClientResponse,
        result: PerRequestResult,
    ) -> None:
        raise NotImplementedError


def get_request_family_adapter(name: str) -> RequestFamilyAdapter:
    from .speech_http import SpeechHTTPAdapter

    adapters: dict[str, RequestFamilyAdapter] = {
        "speech_http": SpeechHTTPAdapter(),
    }
    if name not in adapters:
        raise ValueError(f"Unknown request family: {name}")
    return adapters[name]


def wav_duration(wav_bytes: bytes) -> float:
    if len(wav_bytes) <= WAV_HEADER_SIZE:
        return 0.0
    sample_rate = struct.unpack_from("<I", wav_bytes, 24)[0]
    num_channels = struct.unpack_from("<H", wav_bytes, 22)[0]
    bits_per_sample = struct.unpack_from("<H", wav_bytes, 34)[0]
    if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
        return 0.0
    bytes_per_sample = num_channels * bits_per_sample // 8
    pcm_size = len(wav_bytes) - WAV_HEADER_SIZE
    return pcm_size / (sample_rate * bytes_per_sample)


def decode_wav_audio(audio_b64: str) -> bytes:
    return base64.b64decode(audio_b64)


def parse_sse_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None
    return json.loads(line[len(SSE_DATA_PREFIX):])


def apply_usage(
    result: PerRequestResult,
    usage: dict[str, Any] | None,
) -> None:
    if not usage:
        return
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    engine_time_s = usage.get("engine_time_s")
    if prompt_tokens is not None:
        result.prompt_tokens = int(prompt_tokens)
    if completion_tokens is not None:
        result.completion_tokens = int(completion_tokens)
    if engine_time_s is not None:
        result.engine_time_s = float(engine_time_s)
    if result.completion_tokens and result.engine_time_s and result.engine_time_s > 0:
        result.tok_per_s = result.completion_tokens / result.engine_time_s
