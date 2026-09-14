# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
import wave

import pytest

from benchmarks.eval import benchmark_omni_seedtts as benchmark


@pytest.fixture
def warmup_config(tmp_path):
    measured = tmp_path / "measured.lst"
    measured.write_text("measured|reference|measured.wav|Measured text\n")
    warmup = tmp_path / "warmup.lst"
    warmup.write_text(
        "warm-a|reference A|a.wav|Warmup text A\n"
        "warm-b|reference B|b.wav|Warmup text B\n"
    )
    config = benchmark.OmniSeedttsBenchmarkConfig(
        model="test",
        meta=str(measured),
        warmup=2,
        warmup_meta=str(warmup),
        max_concurrency=2,
        stream=True,
        voice_clone=True,
        disable_tqdm=True,
        output_dir=str(tmp_path / "results"),
    )
    return config


def _wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setparams((1, 2, 16000, 160, "NONE", "not compressed"))
        handle.writeframes(b"\x01\x00" * 160)
    return output.getvalue()


@pytest.mark.asyncio
async def test_distinct_warmup_finishes_before_measured_requests(
    monkeypatch, tmp_path, warmup_config
):
    entered, completed = [], []
    both_started = asyncio.Event()

    async def generate(self, session, url, model, sample, lang, **kwargs):
        entered.append(sample.sample_id)
        if sample.sample_id.startswith("warm-"):
            if len(entered) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=2)
        else:
            assert set(completed) == {"warm-a", "warm-b"}
        assert kwargs["stream"] and kwargs["voice_clone"]
        completed.append(sample.sample_id)
        return _wav(), 16000, {}

    monkeypatch.setattr(benchmark.VoiceCloneOmni, "generate_speech", generate)
    result = await benchmark.run_omni_seedtts_benchmark(warmup_config)

    assert set(entered[:2]) == {"warm-a", "warm-b"}
    assert entered[2:] == ["measured"]
    assert len(result["per_request"]) == 1
    assert result["config"]["warmup_meta"] == warmup_config.warmup_meta
    output = tmp_path / "results"
    assert {path.name for path in (output / "audio").iterdir()} == {"measured.wav"}
    assert {path.name for path in (output / "warmup" / "audio").iterdir()} == {
        "warm-a.wav",
        "warm-b.wav",
    }
    report = json.loads((output / "warmup" / "results.json").read_text())
    assert report["completed"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [asyncio.TimeoutError, ValueError])
async def test_failed_warmup_is_recorded_and_prevents_measurement(
    monkeypatch, tmp_path, warmup_config, failure
):
    seen = []

    async def generate(self, session, url, model, sample, lang, **kwargs):
        seen.append(sample.sample_id)
        raise failure("warmup failed")

    monkeypatch.setattr(benchmark.VoiceCloneOmni, "generate_speech", generate)
    with pytest.raises(RuntimeError, match="warmup.*0/2"):
        await benchmark.run_omni_seedtts_benchmark(warmup_config)
    assert set(seen) == {"warm-a", "warm-b"}
    report = json.loads((tmp_path / "results/warmup/results.json").read_text())
    assert report["completed"] == 0
    assert all(not result["is_success"] for result in report["results"])


@pytest.mark.asyncio
async def test_zero_warmup_does_not_load_separate_dataset(monkeypatch, warmup_config):
    warmup_config.warmup = 0
    warmup_config.warmup_meta = "does-not-exist.lst"
    seen = []

    async def generate(self, session, url, model, sample, lang, **kwargs):
        seen.append(sample.sample_id)
        return _wav(), 16000, {}

    monkeypatch.setattr(benchmark.VoiceCloneOmni, "generate_speech", generate)
    await benchmark.run_omni_seedtts_benchmark(warmup_config)
    assert seen == ["measured"]


@pytest.mark.asyncio
async def test_separate_warmup_requires_enough_samples(warmup_config):
    warmup_config.warmup = 3
    with pytest.raises(ValueError, match="3.*2"):
        await benchmark.run_omni_seedtts_benchmark(warmup_config)


def test_cli_accepts_separate_warmup_dataset():
    args = benchmark._build_arg_parser().parse_args(
        ["--meta", "measured.lst", "--warmup-meta", "warmup.lst", "--warmup", "2"]
    )
    config = benchmark._config_from_args(args)
    assert config.warmup_meta == "warmup.lst" and config.warmup == 2
