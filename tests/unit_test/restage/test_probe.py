import json

import pytest

from benchmarks.benchmarker.restage_probe import _fit, load_calibration_spec
from benchmarks.dataset.seedtts import SampleInput

STAGES = {
    "tts_engine": {"weights_gib": 3.4, "kv_bytes_per_token": 20480},
    "vocoder": {"weights_gib": 1.0},
}


def single(latency, engine, audio=4.0, ttfp=0.2):
    return {
        "is_success": True,
        "latency_s": latency,
        "engine_time_s": engine,
        "audio_duration_s": audio,
        "audio_ttfp_s": ttfp,
        "text_ttft_s": None,
    }


def saturation(concurrency, throughput, rtf, successful=8):
    return {
        "concurrency": concurrency,
        "requests": 8,
        "successful": successful,
        "errors": [],
        "elapsed_s": 10.0,
        "throughput_audio_s_per_s": throughput,
        "rtf_p99": rtf,
    }


def fit(evidence, **overrides):
    fields = dict(
        stages=STAGES,
        model_path="m",
        gpu_name="g",
        gpu_mem_gib=80.0,
        slo_rtf=1.0,
        context_tokens=None,
        samples=[SampleInput("a", "", "", "twelve chars")],
    )
    fields.update(overrides)
    return _fit(evidence, **fields)


def test_fit_splits_the_pipeline_throughput_by_engine_share():
    evidence = {
        "single": [single(1.0, 0.6), single(1.0, 0.7), single(1.0, 0.65)],
        "saturation": [
            saturation(4, 10.0, 0.6),
            saturation(8, 14.0, 0.9),
            saturation(16, 15.0, 1.4),
        ],
    }
    constants = fit(evidence)
    assert constants.pipeline_throughput == 14.0
    assert (
        "c=8" in constants.pipeline_provenance
        and "inside SLO" in constants.pipeline_provenance
    )
    engine, vocoder = constants.stages["tts_engine"], constants.stages["vocoder"]
    assert engine.throughput == pytest.approx(14.0 / 0.65)
    assert vocoder.throughput == pytest.approx(14.0 / 0.35)
    assert 1 / (1 / engine.throughput + 1 / vocoder.throughput) == pytest.approx(14.0)
    assert engine.delta_s == pytest.approx(0.2) and vocoder.delta_s == 0.0
    assert engine.kv_bytes_per_token == 20480 and vocoder.kv_bytes_per_token is None
    assert constants.audio_seconds == 4.0
    assert constants.context_tokens == 32
    assert all(
        row.provenance.startswith("PREDICTED") for row in constants.stages.values()
    )


def test_fit_flags_an_unbracketed_or_infeasible_boundary():
    passing = {
        "single": [single(1.0, 0.5)],
        "saturation": [saturation(4, 10.0, 0.5), saturation(8, 12.0, 0.8)],
    }
    assert "not bracketed" in fit(passing).pipeline_provenance
    failing = {
        "single": [single(1.0, 0.5)],
        "saturation": [saturation(4, 10.0, 1.5), saturation(8, 12.0, 2.0)],
    }
    constants = fit(failing)
    assert constants.pipeline_throughput == 10.0
    assert "no probe met the SLO" in constants.pipeline_provenance


def test_fit_without_engine_time_leaves_tails_non_binding():
    evidence = {"single": [single(1.0, 0.0)], "saturation": [saturation(4, 10.0, 0.5)]}
    constants = fit(evidence)
    assert constants.stages["tts_engine"].throughput == 10.0
    assert constants.stages["vocoder"].throughput is None
    assert constants.stages["vocoder"].provenance.startswith("PRIOR")


def test_fit_requires_one_kv_stage_and_successful_probes():
    evidence = {"single": [single(1.0, 0.5)], "saturation": [saturation(4, 10.0, 0.5)]}
    with pytest.raises(ValueError, match="kv_bytes_per_token"):
        fit(evidence, stages={"a": {"weights_gib": 1.0}})
    with pytest.raises(ValueError, match="single-request"):
        fit({"single": [{**single(1.0, 0.5), "is_success": False}], "saturation": []})


def test_calibration_spec_resolves_audio_relative_to_the_spec(tmp_path):
    audio = tmp_path / "ref.wav"
    audio.write_bytes(b"x")
    spec = tmp_path / "calibrate.json"
    spec.write_text(
        json.dumps(
            {
                "model_path": "m",
                "stages": STAGES,
                "gpu": {"name": "H200", "mem_gib": 140},
                "samples": [
                    {
                        "sample_id": "a",
                        "ref_text": "r",
                        "ref_audio": "ref.wav",
                        "target_text": "t",
                    }
                ],
            }
        )
    )
    options = load_calibration_spec(spec)
    assert options["samples"][0].ref_audio == str(audio)
    assert options["gpu"] == ("H200", 140.0)
    assert options["concurrencies"] == (4, 8, 16)
    spec.write_text(
        json.dumps(
            {
                "model_path": "m",
                "stages": STAGES,
                "samples": [
                    {
                        "sample_id": "a",
                        "ref_text": "r",
                        "ref_audio": "missing.wav",
                        "target_text": "t",
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="does not exist"):
        load_calibration_spec(spec)


def test_specs_accept_a_seedtts_source(tmp_path, monkeypatch):
    from benchmarks.benchmarker import restage_probe

    def fake_loader(source, max_samples=None, *, split="en", revision=None):
        assert (
            source == "org/seed-tts-eval-arrow" and max_samples == 2 and split == "zh"
        )
        return [SampleInput("s1", "r", str(tmp_path / "a.wav"), "t")]

    monkeypatch.setattr("benchmarks.dataset.seedtts.load_seedtts_samples", fake_loader)
    (tmp_path / "a.wav").write_bytes(b"x")
    spec = tmp_path / "calibrate.json"
    spec.write_text(
        json.dumps(
            {
                "model_path": "m",
                "stages": STAGES,
                "gpu": {"name": "H200", "mem_gib": 140},
                "samples": {
                    "source": "org/seed-tts-eval-arrow",
                    "max_samples": 2,
                    "split": "zh",
                },
            }
        )
    )
    options = restage_probe.load_calibration_spec(spec)
    assert [s.sample_id for s in options["samples"]] == ["s1"]


def test_fit_excludes_non_binding_stages_from_the_share_split():
    evidence = {
        "single": [single(1.0, 0.5)],
        "saturation": [saturation(4, 10.0, 0.5)],
    }
    stages = {**STAGES, "preprocessing": {"weights_gib": 0.5, "binding": False}}
    constants = fit(evidence, stages=stages)
    assert constants.stages["vocoder"].throughput == pytest.approx(20.0)
    assert constants.stages["preprocessing"].throughput is None
    assert constants.stages["preprocessing"].weights_gib == 0.5


def test_fit_reads_the_engine_share_from_the_plain_pass_and_rejects_failed_points():
    evidence = {
        "single": [single(1.0, 0.0)],
        "single_plain": [single(1.0, 0.5), single(1.0, 0.7)],
        "saturation": [
            saturation(4, 10.0, 0.5),
            saturation(8, 14.0, 0.8),
            saturation(16, 15.0, 0.9, successful=6),
        ],
    }
    constants = fit(evidence)
    assert constants.pipeline_throughput == 14.0
    assert "c=8" in constants.pipeline_provenance
    assert constants.stages["tts_engine"].throughput == pytest.approx(14.0 / 0.6)


def test_fit_keeps_the_best_feasible_throughput_not_the_largest_concurrency():
    evidence = {
        "single": [single(1.0, 0.5)],
        "saturation": [
            saturation(16, 65.0, 0.05),
            saturation(64, 53.0, 0.18),
            saturation(128, 52.0, 0.38),
        ],
    }
    constants = fit(evidence)
    assert constants.pipeline_throughput == 65.0
    assert "c=16" in constants.pipeline_provenance
    assert "not bracketed" in constants.pipeline_provenance
