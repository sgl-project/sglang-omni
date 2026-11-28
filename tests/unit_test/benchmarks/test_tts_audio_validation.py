# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for TTS benchmark audio response validation."""

from __future__ import annotations

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from benchmarks.tts_serving.audio_validation import validate_pcm_chunk
from benchmarks.tts_serving.http_client import run_http_scenario
from benchmarks.tts_serving.scenarios import Scenario
from benchmarks.tts_serving.spec import BenchmarkSpec


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sample_rate_hz", "duration_ms", "expected_success"),
    [(8000, 60, True), (16000, 60, True), (24000, 50, True), (48000, 40, False)],
)
async def test_http_pcm_minimum_duration_uses_response_sample_rate(
    sample_rate_hz: int, duration_ms: int, expected_success: bool
) -> None:
    """The benchmark misclassified PCM duration when the response was not 24 kHz."""
    pcm_samples = b"\x01\x00" * (sample_rate_hz * duration_ms // 1000)

    async def audio_response(request: web.Request) -> web.Response:
        assert (await request.json())["response_format"] == "pcm"
        return web.Response(
            body=pcm_samples,
            content_type="audio/pcm",
            headers={"X-Sample-Rate": str(sample_rate_hz)},
        )

    application = web.Application()
    application.router.add_post("/v1/audio/speech", audio_response)
    async with TestServer(application) as server:
        spec = BenchmarkSpec(base_url=str(server.make_url("")), model_name="fixture")
        scenario = Scenario(
            id="pcm-duration",
            endpoint="speech",
            category="audio",
            stage_id="pcm",
            capability_key="speech.pcm",
            payload={
                "model": "fixture",
                "input": "hello",
                "voice": "default",
                "response_format": "pcm",
            },
        )
        async with aiohttp.ClientSession() as session:
            result = await run_http_scenario(session, spec, scenario)

    assert result.success is expected_success
    assert result.http_status == 200
    if expected_success:
        assert result.audio_duration_s == pytest.approx(duration_ms / 1000)
    else:
        assert result.status == "invalid_audio_response"
        assert "minimum generated-audio duration" in result.error


@pytest.mark.parametrize("sample_rate_hz", [8000, 16000, 24000, 48000])
def test_pcm_stream_chunk_does_not_require_a_full_response_duration(
    sample_rate_hz: int,
) -> None:
    """The response duration limit must not reject a one-sample streaming chunk."""
    validation = validate_pcm_chunk(b"\x01\x00", sample_rate=sample_rate_hz)

    assert validation.ok
    assert validation.duration_s == pytest.approx(1 / sample_rate_hz)
