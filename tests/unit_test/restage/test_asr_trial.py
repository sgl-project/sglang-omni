import json

import pytest

from benchmarks.benchmarker import restage_asr
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
async def test_asr_scores_reference_audio_transcript_and_preserves_failures(
    tmp_path, monkeypatch
):
    warmup = SampleInput("warm", "different audio", "/warm.wav", "")

    async def trial(**kwargs):
        assert kwargs["warmup_sample"] is warmup
        kwargs["destination"].mkdir()
        kwargs["send_factory"]("http://localhost:18000", tmp_path)
        return await kwargs["quality"](
            [
                RequestResult(request_id="a", is_success=True, text="hello world"),
                RequestResult(
                    request_id="b",
                    is_success=False,
                    text="hello world",
                    error="HTTP 500",
                ),
                RequestResult(request_id="c", is_success=True, text="wrong words"),
            ]
        )

    send_calls = []
    monkeypatch.setattr(restage_asr, "execute_trial", trial)
    monkeypatch.setattr(
        restage_asr, "make_asr_send_fn", lambda *a, **kw: send_calls.append((a, kw))
    )
    result = await restage_asr.execute_asr_trial(
        config_path=tmp_path / "asr.yaml",
        model_path="asr-checkpoint",
        samples=[
            SampleInput(k, "hello world", "/clip.wav", "unrelated TTS target")
            for k in ("a", "b", "c")
        ],
        slo=SLO(max_latency_s=2, max_rtf=1),
        rate=1,
        destination=tmp_path / "trial",
        port=18000,
        lang="en",
        max_wer=0.2,
        warmup_sample=warmup,
    )
    assert result.verdicts == {"a": True, "b": False, "c": False}
    assert result.corpus_pass is True and result.corpus_wer == 0
    assert send_calls == [
        (
            ("asr-checkpoint", "http://localhost:18000/v1/audio/transcriptions"),
            {"lang": "en", "stream": False},
        )
    ]
    details = json.loads((tmp_path / "trial/quality-detail.json").read_text())
    assert details["requests"]["a"]["target_text"] == "hello world"
    assert details["requests"]["b"]["error"] == "HTTP 500"

    workload = json.loads((tmp_path / "trial/workload.json").read_text())
    assert workload["warmup_sample"]["sample_id"] == "warm"


@pytest.mark.asyncio
@pytest.mark.parametrize("slo", [SLO(max_ttfa_s=1), SLO(max_underrun_s=1)])
async def test_asr_rejects_audio_output_slo_before_launch(tmp_path, slo):
    with pytest.raises(ValueError, match="audio-output"):
        await restage_asr.execute_asr_trial(
            config_path=tmp_path / "asr.yaml",
            model_path="asr",
            samples=[SampleInput("a", "hello", "/clip.wav", "")],
            slo=slo,
            rate=1,
            destination=tmp_path / "trial",
            port=18000,
            lang="en",
            max_wer=0.2,
        )
    assert not (tmp_path / "trial").exists()


@pytest.mark.asyncio
async def test_asr_repeated_corpus_retains_quality_and_source_identity(
    tmp_path, monkeypatch
):
    originals = [
        SampleInput("a", "hello world", "/a.wav", ""),
        SampleInput("b", "good morning", "/b.wav", ""),
    ]
    warmup = SampleInput("warm", "warmup", "/warm.wav", "")

    async def trial(**kwargs):
        samples = kwargs["samples"]
        assert len(samples) == len({s.sample_id for s in samples}) == 6
        assert [s.ref_audio for s in samples] == ["/a.wav", "/b.wav"] * 3
        assert kwargs["warmup_sample"] is warmup
        kwargs["send_factory"]("http://localhost:18000", tmp_path)
        return await kwargs["quality"](
            [
                RequestResult(
                    request_id=s.sample_id,
                    is_success=True,
                    text="incorrect" if i == 4 else s.ref_text,
                )
                for i, s in enumerate(samples)
            ]
        )

    monkeypatch.setattr(restage_asr, "execute_trial", trial)
    monkeypatch.setattr(restage_asr, "make_asr_send_fn", lambda *a, **kw: None)
    result = await restage_asr.execute_asr_trial(
        config_path=tmp_path / "asr.yaml",
        model_path="asr",
        samples=originals,
        slo=SLO(max_latency_s=2),
        rate=64,
        destination=tmp_path,
        port=18000,
        lang="en",
        max_wer=0.2,
        warmup_sample=warmup,
        corpus_repeats=3,
    )
    assert len(result.verdicts) == 6 and sum(result.verdicts.values()) == 5
    assert result.verdicts["restage-repeat-2-0"] is False
    assert result.corpus_pass is True and result.corpus_wer == 0
    workload = json.loads((tmp_path / "workload.json").read_text())
    assert workload["corpus_repeats"] == 3
    assert list(workload["request_sources"].values()) == ["a", "b"] * 3
    assert set(workload["request_sources"]) == set(result.verdicts)
    assert [s.sample_id for s in originals] == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
async def test_asr_invalid_corpus_repeats_rejected_before_launch(tmp_path, repeats):
    with pytest.raises(ValueError, match="corpus_repeats"):
        await restage_asr.execute_asr_trial(
            config_path=tmp_path / "asr.yaml",
            model_path="asr",
            samples=[SampleInput("a", "hello", "/a.wav", "")],
            slo=SLO(max_latency_s=2),
            rate=1,
            destination=tmp_path / "trial",
            port=18000,
            lang="en",
            max_wer=0.2,
            corpus_repeats=repeats,
        )
    assert not (tmp_path / "trial").exists()
