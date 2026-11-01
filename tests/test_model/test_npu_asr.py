# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR NPU SeedTTS accuracy and serving regressions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import time
import unicodedata
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import pytest
import requests
from jiwer import cer

from benchmarks.eval.benchmark_asr_seedtts import run_asr_seedtts_once
from benchmarks.tasks.asr import DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
from tests.test_model.asr_ci_config import select_asr_ci_preset
from tests.utils import MetricCheckCollector, assert_wer_results
from tests.utils.npu_ci import load_full_seedtts_samples, managed_npu_server

AUDIO_DOWNLOAD_TIMEOUT_SECONDS = 60
ASR_CONCURRENCY = DEFAULT_ASR_TRANSCRIBE_CONCURRENCY
AsrTestContext = tuple[str, str, dict[str, dict[str, str | float]], Path]

pytestmark = pytest.mark.skipif(
    os.environ.get("OMNI_RUN_NPU_TESTS") != "1",
    reason="Set OMNI_RUN_NPU_TESTS=1 with a reserved NPU and local model",
)


def normalize(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum()
    )


def load_cases(path: Path, cache_dir: Path) -> dict[str, dict[str, str | float]]:
    cases = json.loads(path.read_text())
    assert set(cases) == {"English", "Chinese"}, "Provide both language fixtures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for case in cases.values():
        assert normalize(case["text"]), "A reference transcript is required"
        assert isinstance(case["max_cer"], (int, float))
        assert 0 <= case["max_cer"] < 1, "Set a calibrated CER ceiling below 1"
        audio = (cache_dir / case["audio"]).resolve()
        cached = audio.is_file()
        if cached:
            audio_bytes = audio.read_bytes()
        else:
            with requests.get(
                case["url"], timeout=AUDIO_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                response.raise_for_status()
                audio_bytes = response.content
        assert (
            hashlib.sha256(audio_bytes).hexdigest() == case["sha256"]
        ), f"Audio checksum mismatch: {audio}"
        if not cached:
            audio.write_bytes(audio_bytes)
        else:
            pass
        case["audio"] = str(audio)
    return cases


def stream_text(lines: Iterable[bytes | str]) -> str:
    deltas = []
    final = None
    ended = False
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        else:
            pass
        if not line.startswith("data:"):
            continue
        else:
            pass
        assert not ended, "Unexpected event after [DONE]"
        data = line[5:].strip()
        if data == "[DONE]":
            ended = True
            continue
        else:
            pass
        event = json.loads(data)
        if event["type"] == "transcript.text.delta":
            assert final is None, "Delta after final transcript"
            deltas.append(event["delta"])
        elif event["type"] == "transcript.text.done":
            assert final is None, "Duplicate final transcript"
            final = event["text"]
        else:
            raise AssertionError(f"Unexpected transcription event: {event}")
    assert ended and final is not None, "Incomplete transcription stream"
    if deltas:
        assert "".join(deltas) == final, "Deltas differ from final transcript"
    else:
        pass
    return final


@pytest.fixture(scope="module")
def asr_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AsrTestContext]:
    cases = load_cases(
        Path(os.environ["OMNI_NPU_ASR_CASES"]),
        Path(tempfile.gettempdir()) / "sglang-omni-npu-asr",
    )
    output = Path(
        os.environ.get("OMNI_NPU_ASR_OUTPUT", str(tmp_path_factory.mktemp("npu-asr")))
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2))
    with managed_npu_server("ASR", output) as server:
        yield server.base_url, server.model_path, cases, output


def request_transcription(
    server: AsrTestContext,
    language: Literal["English", "Chinese"],
    case_name: str,
    stream: bool = False,
) -> None:
    base_url, model_id, cases, output = server
    case = cases[language]
    start = time.monotonic()
    with requests.Session() as session, open(case["audio"], "rb") as audio:
        session.trust_env = False
        with session.post(
            base_url + "/v1/audio/transcriptions",
            files={"file": (Path(case["audio"]).name, audio, "audio/wav")},
            data={
                "model": model_id,
                "language": language,
                "response_format": "json",
                "stream": str(stream).lower(),
            },
            stream=stream,
            timeout=(10, 300),
        ) as response:
            assert response.status_code == 200, response.text[:2000]
            if stream:
                assert "text/event-stream" in response.headers["content-type"]
                # Persist events before asserting completeness, including failures.
                lines = list(response.iter_lines())
                (output / f"{case_name}.sse").write_bytes(b"\n".join(lines))
                text = stream_text(lines)
            else:
                result = response.json()
                (output / f"{case_name}-response.json").write_text(json.dumps(result))
                text = result["text"]
    actual = normalize(text)
    error_rate = cer(normalize(case["text"]), actual)
    (output / f"{case_name}.json").write_text(
        json.dumps(
            {"text": text, "cer": error_rate, "elapsed_s": time.monotonic() - start},
            ensure_ascii=False,
            indent=2,
        )
    )
    assert actual, "Empty transcript"
    assert error_rate <= case["max_cer"], f"CER {error_rate}: {text!r}"


def test_stream_transcription(asr_server: AsrTestContext) -> None:
    request_transcription(asr_server, "English", "stream", stream=True)


def test_concurrent_transcriptions(asr_server: AsrTestContext) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                request_transcription, asr_server, language, f"concurrent-{language}"
            )
            for language in ("English", "Chinese")
        ]
        for future in futures:
            future.result()


def test_transcription_after_rejection(asr_server: AsrTestContext) -> None:
    base_url, model_id, _, _ = asr_server
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(
            base_url + "/v1/audio/transcriptions",
            data={"model": model_id},
            timeout=10,
        )
        assert response.status_code in {400, 422}, response.text
    request_transcription(asr_server, "English", "after-rejection")


@pytest.mark.parametrize("language", ["en", "zh"])
def test_full_seedtts(
    asr_server: AsrTestContext,
    language: Literal["en", "zh"],
) -> None:
    base_url, model_id, _, output = asr_server
    samples = load_full_seedtts_samples(language)
    port = urlsplit(base_url).port
    assert port is not None
    results = asyncio.run(
        run_asr_seedtts_once(
            samples,
            host="127.0.0.1",
            port=port,
            model_path=model_id,
            lang=language,
            concurrency=ASR_CONCURRENCY,
            warmup=ASR_CONCURRENCY * 2,
        )
    )
    (output / f"asr_seedtts_{language}_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    summary = results["summary"]
    assert summary["total_samples"] == summary["evaluated"] == len(samples), summary
    assert all(sample["hyp_norm"].strip() for sample in results["per_sample"])
    thresholds = select_asr_ci_preset("qwen3")[1].thresholds
    corpus_wer_max = (
        thresholds.en_corpus_wer_max
        if language == "en"
        else thresholds.zh_corpus_wer_max
    )
    sample_wer_max = (
        thresholds.en_sample_wer_max
        if language == "en"
        else thresholds.zh_sample_wer_max
    )
    checks = MetricCheckCollector(f"NPU ASR SeedTTS {language}")
    assert_wer_results(results, corpus_wer_max, sample_wer_max, collector=checks)
    checks.assert_all()
