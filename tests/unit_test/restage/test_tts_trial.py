from contextlib import contextmanager

import numpy as np
import pytest
import soundfile as sf

from benchmarks.benchmarker import restage_tts
from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.seedtts import SampleInput
from sglang_omni.restage.evaluation import SLO


@pytest.mark.asyncio
async def test_tts_trial_launches_asr_only_for_quality(tmp_path, monkeypatch):
    warmup = SampleInput("warm", "", "", "different text")
    events = []
    wav = tmp_path / "speech.wav"
    sf.write(wav, np.full(16000, 0.1), 16000)

    @contextmanager
    def server(**kwargs):
        assert kwargs["model_path"] == "asr-checkpoint"
        assert kwargs["server_config"] == str((tmp_path / "asr.yaml").resolve())
        assert kwargs["wait_for_gpu_release"] is False
        events.append("asr-start")
        try:
            yield
        finally:
            events.append("asr-stop")

    async def transcribe(samples, **kwargs):
        assert events[-1] == "asr-start"
        return [
            RequestResult(request_id=s.sample_id, is_success=True, text="hello")
            for s in samples
        ], 0.1

    async def trial(**kwargs):
        events.append("tts-start")
        kwargs["destination"].mkdir()
        kwargs["send_factory"]("http://localhost:18000", tmp_path)
        events.append("tts-stop")
        assert kwargs["warmup_sample"] is warmup
        return await kwargs["quality"](
            [RequestResult(request_id="a", is_success=True, wav_path=str(wav))]
        )

    monkeypatch.setattr(restage_tts, "managed_omni_server", server)
    monkeypatch.setattr(restage_tts, "run_asr_transcription", transcribe)
    monkeypatch.setattr(restage_tts, "execute_trial", trial)
    send_calls = []
    monkeypatch.setattr(
        restage_tts, "make_tts_send_fn", lambda *a, **kw: send_calls.append((a, kw))
    )
    result = await restage_tts.execute_tts_trial(
        config_path=tmp_path / "tts.yaml",
        model_path="tts-checkpoint",
        asr_config_path=tmp_path / "asr.yaml",
        asr_model_path="asr-checkpoint",
        samples=[SampleInput("a", "", "", "hello")],
        slo=SLO(max_latency_s=2, max_underrun_s=0.1),
        rate=1,
        destination=tmp_path / "trial",
        port=18000,
        lang="en",
        max_wer=0.2,
        warmup_sample=warmup,
        sender_options={"stream": True},
    )
    assert result.verdicts == {"a": True} and result.corpus_pass is True
    assert events == ["tts-start", "tts-stop", "asr-start", "asr-stop"]
    assert (tmp_path / "trial/quality-detail.json").exists()
    assert send_calls[0][0] == (
        "tts-checkpoint",
        "http://localhost:18000/v1/audio/speech",
    )
    assert send_calls[0][1]["stream"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("slo", [SLO(max_ttfa_s=1), SLO(max_underrun_s=0)])
async def test_tts_requires_streaming_for_audio_output_constraints(tmp_path, slo):
    with pytest.raises(ValueError, match="stream"):
        await restage_tts.execute_tts_trial(
            config_path=tmp_path / "tts.yaml",
            model_path="tts",
            asr_config_path=tmp_path / "asr.yaml",
            asr_model_path="asr",
            samples=[SampleInput("a", "", "", "hello")],
            slo=slo,
            rate=1,
            destination=tmp_path / "trial",
            port=18000,
            lang="en",
            max_wer=0.2,
            sender_options={"stream": False},
        )
    assert not (tmp_path / "trial").exists()
