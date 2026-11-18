# SPDX-License-Identifier: Apache-2.0
"""Budget probes must preserve evidence and reject truncation hidden by HTTP 200."""

import asyncio
import io
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, Mock

import aiohttp
import httpx
import numpy as np
import pytest
import soundfile as sf

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_tts_generation_budget
from benchmarks.eval.benchmark_tts_generation_budget import (
    CODEC_EOS_ERROR,
    SEEDTTS_RUNAWAY_SAMPLE_IDS,
    GenerationProbeCase,
    codec_eos_request_ids,
    generation_budget_failures,
    run_generation_probe,
    seedtts_probe_cases,
)
from benchmarks.metrics.performance import build_speed_results
from benchmarks.tasks.tts import make_tts_send_fn
from sglang_omni.serve.protocol import CreateSpeechRequest, SpeechReference
from tests.test_model.tts_ci_eos import (
    TtsCiRequestOutcome,
    assert_expected_codec_eos_failures,
    record_codec_eos_failures,
)
from tests.utils import MetricCheckCollector


@pytest.mark.parametrize("include_traceback", [False, True])
def test_ci_speed_eos_budget_requires_matching_worker_evidence(
    tmp_path: Path, include_traceback: bool
) -> None:
    """An interrupted stream only counts as missing EOS for its own worker request ID."""
    worker_log = tmp_path / "worker.log"
    prefix = "worker ERROR: "
    traceback = (
        'Traceback (most recent call last):\n  File "request_builders.py", line 1\n'
        if include_traceback
        else ""
    )
    worker_log.write_text(
        f"{prefix}OmniScheduler terminal output handling failed for request speech-other\n"
        f"{traceback}RuntimeError: vocoder failed\n"
        f"{prefix}OmniScheduler terminal output handling failed for request speech-eos\n"
        f"{traceback}RuntimeError: {CODEC_EOS_ERROR}\n",
        encoding="utf-8",
    )
    requests: list[TtsCiRequestOutcome] = [
        {
            "id": sample,
            "is_success": False,
            "speech_stream_interrupted": True,
            "speech_outcome_id": request_id,
            "error": "incomplete chunked read",
        }
        for sample, request_id in (("eos", "speech-eos"), ("other", "speech-other"))
    ]
    assert codec_eos_request_ids([worker_log]) == {"speech-eos"}
    record_codec_eos_failures(requests, [worker_log])
    saved = requests
    assert saved[0]["codec_eos_missing"] and saved[0]["is_success"] is False
    assert "codec_eos_missing" not in saved[1]

    allowed = MetricCheckCollector()
    assert_expected_codec_eos_failures(
        saved[:1], max_codec_eos_failures=1, label="latency", collector=allowed
    )
    allowed.assert_all()

    rejected = MetricCheckCollector()
    assert_expected_codec_eos_failures(
        saved, max_codec_eos_failures=0, label="latency", collector=rejected
    )
    assert len(rejected.failures) == 2
    assert any(
        "unexpected failures ['other']" in failure for failure in rejected.failures
    )
    assert any("(1 > 0)" in failure for failure in rejected.failures)


@pytest.mark.asyncio
async def test_speed_artifact_retains_request_identity_after_raw_stream_failure() -> (
    None
):
    """The benchmark must keep the worker's ID even when reading PCM fails."""

    async def iter_pcm_chunks() -> AsyncIterator[tuple[bytes, bool]]:
        yield bytes(8), True
        raise aiohttp.ClientPayloadError("incomplete chunked read")

    session = MagicMock(spec=aiohttp.ClientSession)
    session.post.return_value.__aenter__.return_value = MagicMock(
        status=200,
        headers={"Content-Type": "audio/pcm", "X-SGLang-Omni-Speech-Id": "speech-eos"},
        content=Mock(iter_chunks=iter_pcm_chunks),
    )
    send = make_tts_send_fn("test", "http://test/v1/audio/speech", stream=True)
    result = await send(
        session,
        SampleInput(sample_id="sample", ref_text="", ref_audio="", target_text="Hi."),
    )
    row = build_speed_results([result], {}, {})["per_request"][0]
    assert row["id"] == "sample"
    assert row["is_success"] is False
    assert row["speech_stream_interrupted"] is True
    assert row["speech_outcome_id"] == "speech-eos"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("finish_reason", ["stop", "length"])
async def test_probe_pairs_seeds_and_rejects_incomplete_audio(
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
        assert bool(result.errors) == (result.finish_reason == "length")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("baseline", "text", "expected_failure"),
    [
        ("codec_eos", "codec_eos", False),
        ("stop", "codec_eos", True),
        ("codec_eos", "stop", False),
        ("length", "stop", True),
        ("other", "other", True),
    ],
)
async def test_paired_probe_rejects_new_cutoffs_and_unknown_failures(
    tmp_path: Path,
    stream: bool,
    baseline: Literal["stop", "codec_eos", "length", "other"],
    text: Literal["stop", "codec_eos", "other"],
    expected_failure: bool,
) -> None:
    """Only matched missing-EOS failures may pass the paired-budget regression gate."""

    class InterruptedAudio(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield bytes(4800)
            raise httpx.RemoteProtocolError("incomplete chunked read")

    audio = io.BytesIO()
    sf.write(audio, np.zeros(2400), 24000, format="WAV", subtype="PCM_16")
    finish_reasons: dict[str, str] = {}
    request_ids = {"speech-1-baseline", "speech-1-text"}

    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            request_id = request.url.path.rsplit("/", 1)[1]
            return httpx.Response(
                200,
                json={
                    "request_id": request_id,
                    "finish_reason": finish_reasons[request_id],
                    "usage": {"completion_tokens": 2},
                },
            )
        else:
            payload = json.loads(request.content)
        policy = "baseline" if "max_new_tokens" in payload else "text"
        request_id = f"speech-{payload['seed']}-{policy}"
        finish_reason = (
            "stop"
            if payload["seed"] == 0
            else baseline if policy == "baseline" else text
        )
        finish_reasons[request_id] = finish_reason
        headers = {
            "X-Sample-Rate": "24000",
            "X-SGLang-Omni-Speech-Id": request_id,
            "X-Finish-Reason": finish_reason,
            "X-Completion-Tokens": "2",
        }
        if finish_reason == "codec_eos" and stream:
            return httpx.Response(200, headers=headers, stream=InterruptedAudio())
        elif finish_reason in ("codec_eos", "other"):
            return httpx.Response(
                500,
                json={
                    "error": {
                        "type": "server_error",
                        "message": (
                            CODEC_EOS_ERROR
                            if finish_reason == "codec_eos"
                            else "vocoder failed"
                        ),
                    }
                },
            )
        else:
            return httpx.Response(
                200,
                headers=headers,
                content=bytes(4800) if stream else audio.getvalue(),
            )

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hi.", language="English")],
            range(2),
            stream,
            tmp_path / "probe",
        )
    assert bool(generation_budget_failures(results, request_ids)) == expected_failure
    recorded = [
        json.loads(line)
        for line in (tmp_path / "probe" / "results.jsonl").read_text().splitlines()
    ]
    assert bool(recorded[2]["errors"]) == (baseline != "stop")
    assert bool(recorded[3]["errors"]) == (text != "stop")
    if baseline == text == "codec_eos":
        assert generation_budget_failures(results[2:], request_ids) == [
            "No natural completion with the default budget for ['short']"
        ]
        assert bool(generation_budget_failures(results, set())) == stream
    else:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_probe_keeps_only_the_audio_of_failed_pairs(
    tmp_path: Path, stream: bool
) -> None:
    bodies: list[dict[str, object]] = []
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, np.zeros(2400), 24000, format="WAV", subtype="PCM_16")
    outcomes = {0: (("stop", 40), ("stop", 40)), 1: (("stop", 200), ("length", 125))}

    def outcome() -> tuple[str, int]:
        baseline, text = outcomes[int(bodies[-1]["seed"])]
        return baseline if "max_new_tokens" in bodies[-1] else text

    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            bodies.append(json.loads(request.content))
            reason, frames = outcome()
            return httpx.Response(
                200,
                content=bytes(4800) if stream else audio_buffer.getvalue(),
                headers={
                    "X-Sample-Rate": "24000",
                    "X-SGLang-Omni-Speech-Id": "speech-1",
                    "X-Finish-Reason": reason,
                    "X-Completion-Tokens": str(frames),
                },
            )
        reason, frames = outcome()
        return httpx.Response(
            200,
            json={
                "request_id": "speech-1",
                "finish_reason": reason,
                "usage": {"completion_tokens": frames},
            },
        )

    output_dir = tmp_path / "probe"
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hi.", language="English")],
            range(2),
            stream,
            output_dir,
            keep_passing_audio=False,
        )
    assert [bool(result.errors) for result in results] == [False, False, False, True]
    kept = sorted(path.name for path in output_dir.iterdir())
    if stream:
        assert kept == [
            "00002.pcm",
            "00002.wav",
            "00003.pcm",
            "00003.wav",
            "results.jsonl",
        ]
    else:
        assert kept == ["00002.wav", "00003.wav", "results.jsonl"]
    assert len((output_dir / "results.jsonl").read_text().splitlines()) == 4


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
        if request.method == "POST":
            return httpx.Response(200, content=audio_bytes, headers=headers)
        return httpx.Response(
            200,
            json={
                "request_id": "speech-1",
                "finish_reason": "stop",
                "usage": {"completion_tokens": 1},
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
        )
    result = results[0]
    assert result.errors
    if stream:
        assert result.audio_seconds == 0
    else:
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
async def test_probe_closes_a_stream_past_its_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        benchmark_tts_generation_budget, "MAX_PROBE_RESPONSE_BYTES", 500000
    )

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
    assert any("probe byte limit" in error for error in results[0].errors)
    assert len((output_dir / "00000.pcm").read_bytes()) == 600000


@pytest.mark.asyncio
async def test_probe_flags_generation_beyond_only_the_requests_own_limit(
    tmp_path: Path,
) -> None:
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
            return httpx.Response(
                200,
                json={
                    "request_id": "speech-1",
                    "finish_reason": "stop",
                    "usage": {"completion_tokens": 3000},
                },
            )

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hi.", language="English")],
            range(1),
            True,
            tmp_path / "probe",
        )
    assert [result.policy for result in results] == ["baseline", "text"]
    assert results[0].errors == ["Generation exceeded the request's 2048-frame limit"]
    assert results[1].errors == []


@pytest.mark.asyncio
async def test_probe_reports_wrong_stream_rate_without_budget_overrun(
    tmp_path: Path,
) -> None:
    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=bytes(600000),
            headers={"X-Sample-Rate": "48000", "X-SGLang-Omni-Speech-Id": "speech-1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        results = await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(input="unused"),
            [GenerationProbeCase(id="short", input="Hi.", language="English")],
            range(1),
            True,
            tmp_path / "probe",
            include_baseline=False,
        )
    assert any("Expected 24000 Hz, got 48000" in error for error in results[0].errors)
    assert not any(
        "exceeded its codec frame budget" in error for error in results[0].errors
    )
    assert results[0].audio_seconds == pytest.approx(600000 / 2 / 48000)


@pytest.mark.asyncio
async def test_probe_uses_each_cases_reference_over_template_reference(
    tmp_path: Path,
) -> None:
    bodies: list[dict[str, object]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=bytes(4800),
            headers={"X-Finish-Reason": "stop", "X-Completion-Tokens": "2"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        await run_generation_probe(
            client,
            "http://test/v1",
            CreateSpeechRequest(
                input="unused",
                references=[
                    SpeechReference(audio_path="template.wav", text="template")
                ],
            ),
            [
                GenerationProbeCase(
                    id="override",
                    input="Hello.",
                    language="English",
                    references=[SpeechReference(audio_path="case.wav", text="case")],
                ),
                GenerationProbeCase(id="template", input="Hello.", language="English"),
            ],
            range(1),
            False,
            tmp_path / "probe",
            include_baseline=False,
        )
    assert [body["references"] for body in bodies] == [
        [{"audio_path": "case.wav", "text": "case"}],
        [{"audio_path": "template.wav", "text": "template"}],
    ]


def test_seedtts_cases_keep_their_reference_audio_and_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load(source: str, *, split: str) -> list[SampleInput]:
        assert split == "en"
        return [
            SampleInput("a-b", "reference text", "/tmp/reference.wav", "Target."),
            SampleInput("unused", "unused", "/tmp/unused.wav", "Unused."),
        ]

    monkeypatch.setattr(benchmark_tts_generation_budget, "load_seedtts_samples", load)
    cases = seedtts_probe_cases(["a-b"], "en")
    assert cases[0].input == "Target."
    assert cases[0].references == [
        SpeechReference(audio_path="/tmp/reference.wav", text="reference text")
    ]
    with pytest.raises(ValueError, match="missing-id"):
        seedtts_probe_cases(["a-b", "missing-id"], "en")
    with pytest.raises(ValueError, match="no samples"):
        seedtts_probe_cases([], "en")


def test_seedtts_runaway_prompts_are_distinct_ids() -> None:
    assert SEEDTTS_RUNAWAY_SAMPLE_IDS
    assert len(set(SEEDTTS_RUNAWAY_SAMPLE_IDS)) == len(SEEDTTS_RUNAWAY_SAMPLE_IDS)
    assert all(
        re.fullmatch(r"common_voice_en_\d+-common_voice_en_\d+", item)
        for item in SEEDTTS_RUNAWAY_SAMPLE_IDS
    )


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
