import json

import numpy as np
import pytest
import soundfile as sf

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.restage_quality import evaluate_tts_quality


@pytest.mark.asyncio
async def test_tts_quality_gates_the_corpus_not_single_clips(tmp_path):
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.sin(np.arange(16000) * 0.1).astype(np.float32) * 0.1, 16000)
    results = [
        RequestResult(request_id=f"ok{i}", is_success=True, wav_path=str(wav))
        for i in range(4)
    ] + [
        RequestResult(request_id="wrong", is_success=True, wav_path=str(wav)),
        RequestResult(
            request_id="missing", is_success=True, wav_path=str(tmp_path / "absent.wav")
        ),
    ]
    long_text = "the quick brown fox jumps over the lazy dog near the river bank"

    async def transcribe(samples):
        assert {s.sample_id for s in samples} == {"ok0", "ok1", "ok2", "ok3", "wrong"}
        return [
            RequestResult(request_id=s.sample_id, is_success=True, text=long_text)
            for s in samples
            if s.sample_id != "wrong"
        ] + [
            RequestResult(
                request_id="wrong", is_success=True, text="the footwalk was long"
            )
        ]

    targets = {r.request_id: long_text for r in results}
    targets["wrong"] = "the foot walk was long"
    report = await evaluate_tts_quality(
        results,
        targets=targets,
        transcribe=transcribe,
        output=tmp_path / "quality-detail.json",
        lang="en",
        max_wer=0.2,
    )
    # Note (Jiaxin Deng): one short clip with two word errors must not sink a
    # trial whose corpus is otherwise exact.
    assert report.verdicts == {
        "ok0": True,
        "ok1": True,
        "ok2": True,
        "ok3": True,
        "wrong": True,
        "missing": False,
    }
    assert report.corpus_pass is True
    assert 0 < report.corpus_wer < 0.2
    detail = json.loads((tmp_path / "quality-detail.json").read_text())
    assert 0.2 < detail["requests"]["wrong"]["wer"] <= 0.5
    assert detail["corpus_pass"] is True
    assert detail["evaluated"] == 5


@pytest.mark.asyncio
async def test_corpus_wer_above_the_limit_fails_the_gate(tmp_path):
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.full(16000, 0.1), 16000)
    results = [RequestResult(request_id="a", is_success=True, wav_path=str(wav))]

    async def transcribe(samples):
        return [RequestResult(request_id="a", is_success=True, text="something else")]

    report = await evaluate_tts_quality(
        results,
        targets={"a": "hello world"},
        transcribe=transcribe,
        output=tmp_path / "quality.json",
        lang="en",
        max_wer=0.2,
    )
    assert report.verdicts == {"a": False}
    assert report.corpus_pass is False and report.corpus_wer is None


@pytest.mark.asyncio
async def test_silent_audio_and_missing_transcripts_do_not_pass(tmp_path):
    wav = tmp_path / "silent.wav"
    sf.write(wav, np.zeros(16000), 16000)

    async def transcribe(samples):
        assert samples == []
        return []

    report = await evaluate_tts_quality(
        [RequestResult(request_id="a", is_success=True, wav_path=str(wav))],
        targets={"a": "hello"},
        transcribe=transcribe,
        output=tmp_path / "quality.json",
        lang="en",
        max_wer=0.2,
    )
    assert report.verdicts == {"a": False}
    assert report.corpus_pass is False and report.corpus_wer is None


@pytest.mark.asyncio
async def test_valid_audio_with_missing_or_failed_asr_is_not_quality_success(tmp_path):
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.full(16000, 0.1), 16000)
    results = [
        RequestResult(request_id=key, is_success=True, wav_path=str(wav))
        for key in ("missing", "failed")
    ]

    async def transcribe(samples):
        return [RequestResult(request_id="failed", error="ASR unavailable")]

    report = await evaluate_tts_quality(
        results,
        targets={"missing": "hello", "failed": "hello"},
        transcribe=transcribe,
        output=tmp_path / "quality.json",
        lang="en",
        max_wer=0.2,
    )
    assert report.verdicts == {"missing": False, "failed": False}
    detail = json.loads((tmp_path / "quality.json").read_text())
    assert detail["requests"]["missing"]["error"] == "Missing ASR result"
    assert detail["requests"]["failed"]["error"] == "ASR unavailable"


@pytest.mark.asyncio
async def test_runaway_generation_fails_its_request_not_the_corpus(tmp_path):
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.full(16000, 0.1), 16000)
    results = [
        RequestResult(request_id=k, is_success=True, wav_path=str(wav))
        for k in ("good", "runaway")
    ]

    async def transcribe(samples):
        return [
            RequestResult(request_id="good", is_success=True, text="hello world"),
            RequestResult(request_id="runaway", is_success=True, text="hello " * 40),
        ]

    report = await evaluate_tts_quality(
        results,
        targets={"good": "hello world", "runaway": "hello world"},
        transcribe=transcribe,
        output=tmp_path / "quality.json",
        lang="en",
        max_wer=0.2,
    )
    assert report.verdicts == {"good": True, "runaway": False}
    assert report.corpus_pass is True and report.corpus_wer == 0
    assert json.loads((tmp_path / "quality.json").read_text())["runaway"] == 1
