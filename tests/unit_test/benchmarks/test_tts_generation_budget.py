# SPDX-License-Identifier: Apache-2.0
"""Budget probes must preserve evidence and reject truncation hidden by HTTP 200."""

import asyncio
import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf

from benchmarks.eval.benchmark_tts_generation_budget import (
    GenerationProbeCase,
    run_generation_probe,
)
from sglang_omni.serve.protocol import CreateSpeechRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_probe_pairs_seeds_and_records_truncated_audio(
    tmp_path: Path, stream: bool, finish_reason: str
) -> None:
    requests: list[dict[str, object]] = []
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, np.zeros(2400), 24000, format="WAV", subtype="PCM_16")

    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(
                200,
                content=bytes(4800) if stream else audio_buffer.getvalue(),
                headers={
                    "X-Request-Id": "speech-1",
                    "X-SGLang-Omni-Speech-Id": "speech-1",
                    "X-SGLang-Omni-Worker": "worker-1",
                    "X-Sample-Rate": "24000",
                    "X-Finish-Reason": (
                        "stop" if "max_new_tokens" in body else finish_reason
                    ),
                    "X-Completion-Tokens": "2",
                },
            )
        else:
            assert request.headers["x-sglang-omni-route-worker"] == "worker-1"
            return httpx.Response(
                200,
                json={
                    "request_id": "speech-1",
                    "finish_reason": (
                        "stop" if "max_new_tokens" in requests[-1] else finish_reason
                    ),
                    "usage": {"completion_tokens": 2},
                },
            )

    output_dir = tmp_path / "probe"
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused", model="test", voice="Vivian"),
            [GenerationProbeCase(id="short", input="Hello.", language="English")],
            range(7, 9),
            stream,
            output_dir,
        )
    assert [request["seed"] for request in requests] == [7, 7, 8, 8]
    assert [request.get("max_new_tokens") for request in requests] == [
        2048,
        None,
        2048,
        None,
    ]
    assert all(request["input"] == "Hello." for request in requests)
    assert len(list(output_dir.glob("*.wav"))) == 4
    recorded = [
        json.loads(line)
        for line in (output_dir / "results.jsonl").read_text().splitlines()
    ]
    assert len(recorded) == 4
    for result in results:
        assert result.audio_seconds == pytest.approx(0.1)
        assert result.completion_frames == 2
        assert result.audio_sha256
        assert sum(result.chunk_bytes) > 0
        if result.policy == "text" and finish_reason == "length":
            assert (
                "Text budget truncated a naturally completed baseline" in result.errors
            )
            assert "Normal-speech case did not finish naturally" in result.errors
        else:
            assert not result.errors


@pytest.mark.asyncio
async def test_probe_retains_audio_when_outcome_lookup_fails(tmp_path: Path) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(
                200,
                content=bytes(4800),
                headers={
                    "X-Sample-Rate": "24000",
                    "X-SGLang-Omni-Speech-Id": "speech-1",
                },
            )
        else:
            return httpx.Response(404)

    output_dir = tmp_path / "probe"
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hello.", language="English")],
            range(1),
            True,
            output_dir,
            include_baseline=False,
        )
    assert results[0].audio_seconds == pytest.approx(0.1)
    assert any("404" in error for error in results[0].errors)
    assert "Missing completion frame count" in results[0].errors
    assert (output_dir / "00000.pcm").read_bytes() == bytes(4800)
    assert (output_dir / "00000.wav").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("header_value", [None, "invalid"])
async def test_probe_retains_audio_when_metadata_is_invalid(
    tmp_path: Path, stream: bool, header_value: str | None
) -> None:
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, np.zeros(2400), 24000, format="WAV", subtype="PCM_16")
    audio_bytes = bytes(4800) if stream else audio_buffer.getvalue()
    headers = {"X-Finish-Reason": "stop", "X-SGLang-Omni-Speech-Id": "speech-1"}
    if header_value is not None:
        headers["X-Sample-Rate" if stream else "X-Completion-Tokens"] = header_value
    else:
        pass

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(200, content=audio_bytes, headers=headers)

    output_dir = tmp_path / "probe"
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hi.", language="English")],
            range(1),
            stream,
            output_dir,
            include_baseline=False,
        )
    result = results[0]
    assert result.errors
    assert result.audio_seconds == pytest.approx(0.1)
    assert sum(result.chunk_bytes) == len(audio_bytes)
    assert (output_dir / ("00000.pcm" if stream else "00000.wav")).read_bytes() == (
        audio_bytes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_probe_timeout_closes_response_and_preserves_audio(
    tmp_path: Path, stream: bool
) -> None:
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, np.zeros(2400), 24000, format="WAV", subtype="PCM_16")
    audio_bytes = bytes(4800) if stream else audio_buffer.getvalue()

    class StalledAudioStream(httpx.AsyncByteStream):
        closed: bool = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield audio_bytes
            await asyncio.Event().wait()

        async def aclose(self) -> None:
            self.closed = True

    audio_stream = StalledAudioStream()

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            stream=audio_stream,
            headers={
                "X-Sample-Rate": "24000",
                "X-SGLang-Omni-Speech-Id": "speech-1",
                "X-Finish-Reason": "stop",
                "X-Completion-Tokens": "2",
            },
        )

    output_dir = tmp_path / "probe"
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hi.", language="English")],
            range(1),
            stream,
            output_dir,
            include_baseline=False,
            request_timeout_seconds=0.1,
        )
    assert audio_stream.closed
    assert results[0].audio_seconds == pytest.approx(0.1)
    recorded = json.loads((output_dir / "results.jsonl").read_text())
    assert "Speech request exceeded its total time limit" in recorded["errors"]
    assert (output_dir / ("00000.pcm" if stream else "00000.wav")).read_bytes() == (
        audio_bytes
    )


@pytest.mark.asyncio
async def test_probe_closes_a_stream_that_exceeds_the_budget(tmp_path: Path) -> None:
    class OversizedAudioStream(httpx.AsyncByteStream):
        closed: bool = False
        chunks_read: int = 0

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for _ in range(3):
                self.chunks_read += 1
                yield bytes(300000)

        async def aclose(self) -> None:
            self.closed = True

    audio_stream = OversizedAudioStream()

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            stream=audio_stream,
            headers={
                "X-Sample-Rate": "24000",
                "X-SGLang-Omni-Speech-Id": "speech-1",
            },
        )

    output_dir = tmp_path / "probe"
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hi.", language="English")],
            range(1),
            True,
            output_dir,
            include_baseline=False,
        )
    assert audio_stream.closed
    assert audio_stream.chunks_read == 2
    assert any(
        "exceeded its codec frame budget" in error for error in results[0].errors
    )
    assert len((output_dir / "00000.pcm").read_bytes()) == 600000


@pytest.mark.asyncio
async def test_probe_rejects_a_stage_override_that_would_disable_the_comparison(
    tmp_path: Path,
) -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="Remove stage_params"):
            await run_generation_probe(
                client,
                "http://test/v1",
                CreateSpeechRequest(
                    input="unused",
                    stage_params={"tts_engine": {"max_new_tokens": 2048}},
                ),
                [GenerationProbeCase(id="short", input="Hi.", language="English")],
                range(1),
                True,
                tmp_path / "probe",
            )
    assert not (tmp_path / "probe").exists()
