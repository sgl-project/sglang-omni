# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_omni_seedtts as benchmark
from benchmarks.tasks.tts import (
    ReferenceAudioField,
    TalkerSamplingParams,
    VoiceCloneOmni,
)


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


def wav():
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
        return wav(), 16000, {}

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
        return wav(), 16000, {}

    monkeypatch.setattr(benchmark.VoiceCloneOmni, "generate_speech", generate)
    await benchmark.run_omni_seedtts_benchmark(warmup_config)
    assert seen == ["measured"]


@pytest.mark.asyncio
async def test_separate_warmup_requires_enough_samples(warmup_config):
    warmup_config.warmup = 3
    with pytest.raises(ValueError, match="3.*2"):
        await benchmark.run_omni_seedtts_benchmark(warmup_config)


def test_cli_accepts_separate_warmup_dataset():
    args = benchmark._build_arg_parser().parse_args(  # noqa: leading-underscore  # production name
        ["--meta", "measured.lst", "--warmup-meta", "warmup.lst", "--warmup", "2"]
    )
    config = benchmark._config_from_args(
        args
    )  # noqa: leading-underscore  # production name
    assert config.warmup_meta == "warmup.lst" and config.warmup == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference_audio_field, voice_clone",
    [(None, True), ("audio.ref_audio", True), ("audio.ref_audio", False)],
)
async def test_speaker_reference_transport_reaches_warmup_and_measured_requests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    warmup_config: benchmark.OmniSeedttsBenchmarkConfig,
    reference_audio_field: str | None,
    voice_clone: bool,
) -> None:
    warmup_config.stream = False
    warmup_config.voice_clone = voice_clone
    if reference_audio_field is not None:
        warmup_config.reference_audio_field = reference_audio_field
    reference = wav()
    if voice_clone:
        for filename in ("a.wav", "b.wav", "measured.wav"):
            (tmp_path / filename).write_bytes(reference)

    response = AsyncMock()
    response.status = 200
    response.json.return_value = {
        "choices": [{"message": {"audio": {"data": base64.b64encode(wav()).decode()}}}]
    }
    request = AsyncMock()
    request.__aenter__.return_value = response
    post = Mock(return_value=request)
    monkeypatch.setattr(benchmark.aiohttp.ClientSession, "post", post)

    results = await benchmark.run_omni_seedtts_benchmark(warmup_config)

    assert post.call_count == 3
    for call in post.call_args_list:
        payload = call.kwargs["json"]
        if voice_clone and reference_audio_field is None:
            assert payload["audios"][0].endswith(".wav")
            assert payload["audio"] == {"format": "wav"}
            assert "Listen to the audio above" in payload["messages"][-1]["content"]
        elif voice_clone:
            assert "audios" not in payload
            prefix, encoded = payload["audio"]["ref_audio"].split(",", 1)
            assert prefix == "data:audio/wav;base64"
            assert base64.b64decode(encoded, validate=True) == reference
            assert (
                "Please read the following text out loud"
                in payload["messages"][-1]["content"]
            )
        else:
            assert "audios" not in payload
            assert payload["audio"] == {"format": "wav"}
    assert results["config"]["reference_audio_field"] == (
        reference_audio_field or "audios"
    )
    assert (tmp_path / "results/audio/measured.wav").read_bytes() == wav()


def test_cli_selects_explicit_speaker_reference_transport() -> None:
    args = benchmark._build_arg_parser().parse_args(  # noqa: leading-underscore  # production name
        ["--voice-clone", "--reference-audio-field", "audio.ref_audio"]
    )
    config = benchmark._config_from_args(
        args
    )  # noqa: leading-underscore  # production name
    assert config.voice_clone
    assert config.reference_audio_field == "audio.ref_audio"


@dataclass(kw_only=True)
class ForwardedSpeech:
    sample_id: str
    seed: int | None
    reference_audio_data: str | None
    talker_params: TalkerSamplingParams | None


class RecordedGenerateSpeech(Protocol):
    async def __call__(
        self,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        lang: str,
        *,
        speaker: str = ...,
        max_tokens: int | None = ...,
        temperature: float = ...,
        seed: int | None = ...,
        voice_clone: bool = ...,
        stream: bool = ...,
        system_prompt: str | None = ...,
        chunk_times_out: list[float] | None = ...,
        text_first_time_holder: list[float] | None = ...,
        reference_audio_field: ReferenceAudioField = ...,
        reference_audio_data: str | None = ...,
        talker_params: TalkerSamplingParams | None = ...,
    ) -> tuple[bytes, float, dict[str, int]]: ...


def recording_generate_speech(
    forwarded: list[ForwardedSpeech],
) -> RecordedGenerateSpeech:
    async def generate_speech(
        self: VoiceCloneOmni,
        session: aiohttp.ClientSession,
        api_url: str,
        model_name: str,
        sample: SampleInput,
        lang: str,
        *,
        speaker: str = "Ethan",
        max_tokens: int | None = None,
        temperature: float = 0.7,
        seed: int | None = None,
        voice_clone: bool = False,
        stream: bool = False,
        system_prompt: str | None = None,
        chunk_times_out: list[float] | None = None,
        text_first_time_holder: list[float] | None = None,
        reference_audio_field: ReferenceAudioField = "audios",
        reference_audio_data: str | None = None,
        talker_params: TalkerSamplingParams | None = None,
    ) -> tuple[bytes, float, dict[str, int]]:
        forwarded.append(
            ForwardedSpeech(
                sample_id=sample.sample_id,
                seed=seed,
                reference_audio_data=reference_audio_data,
                talker_params=talker_params,
            )
        )
        return wav(), 0.0, {}

    return generate_speech


@pytest.mark.asyncio
async def test_inline_reference_is_preloaded_and_seed_forwarded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(wav())
    meta = tmp_path / "measured.lst"
    meta.write_text(f"measured|reference|{reference}|Measured text\n")
    config = benchmark.OmniSeedttsBenchmarkConfig(
        model="test",
        meta=str(meta),
        warmup=0,
        max_concurrency=1,
        voice_clone=True,
        reference_audio_field="audio.ref_audio",
        seed=7,
        disable_tqdm=True,
        output_dir=str(tmp_path / "results"),
    )
    expected_reference = "data:audio/wav;base64," + base64.b64encode(wav()).decode(
        "ascii"
    )
    forwarded: list[ForwardedSpeech] = []

    monkeypatch.setattr(
        benchmark.VoiceCloneOmni,
        "generate_speech",
        recording_generate_speech(forwarded),
    )
    benchmark_results = await benchmark.run_omni_seedtts_benchmark(config)

    assert forwarded == [
        ForwardedSpeech(
            sample_id="measured",
            seed=7,
            reference_audio_data=expected_reference,
            talker_params={},
        )
    ]
    assert benchmark_results["config"]["seed"] == 7
    assert benchmark_results["config"]["temperature"] == 0.7


@pytest.mark.asyncio
async def test_inline_reference_requires_preloaded_data() -> None:
    sample = SampleInput(
        sample_id="missing-reference",
        ref_text="reference transcript",
        ref_audio="missing.wav",
        target_text="target transcript",
    )
    async with aiohttp.ClientSession() as session:
        with pytest.raises(ValueError, match="preloaded reference audio"):
            await VoiceCloneOmni().generate_speech(
                session,
                "http://localhost",
                "test",
                sample,
                "en",
                voice_clone=True,
                reference_audio_field="audio.ref_audio",
            )


def test_cli_accepts_seed() -> None:
    args = benchmark._build_arg_parser().parse_args(
        ["--seed", "3"]
    )  # noqa: leading-underscore  # production name
    assert (
        benchmark._config_from_args(args).seed == 3
    )  # noqa: leading-underscore  # production name


def test_cli_parses_talker_and_sweep_flags() -> None:
    args = benchmark._build_arg_parser().parse_args(  # noqa: leading-underscore  # production name
        [
            "--talker-temperature",
            "0.6",
            "--talker-top-k",
            "10",
            "--concurrencies",
            "1,16",
            "--repeats",
            "2",
            "--generate-only",
        ]
    )
    config = benchmark._config_from_args(
        args
    )  # noqa: leading-underscore  # production name
    assert (config.talker_temperature, config.talker_top_k) == (0.6, 10)
    assert config.talker_top_p is None and config.talker_repetition_penalty is None
    assert args.concurrencies == [1, 16] and args.repeats == 2


def test_sweep_requires_generate_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["benchmark_omni_seedtts", "--concurrencies", "1"])
    with pytest.raises(SystemExit):
        benchmark.main()


def test_results_config_records_talker_params_and_optional_fingerprint() -> None:
    config = benchmark.OmniSeedttsBenchmarkConfig(
        model="test", meta="measured.lst", talker_top_p=0.9
    )
    results_config = (
        benchmark._build_results_config(  # noqa: leading-underscore  # production name
            config, base_url="http://localhost:8000"
        )
    )
    assert results_config["talker_top_p"] == 0.9
    assert results_config["talker_top_k"] is None
    assert "environment_fingerprint" not in results_config

    config.environment_fingerprint = {"client": {"git": {}}, "server": {}}
    results_config = (
        benchmark._build_results_config(  # noqa: leading-underscore  # production name
            config, base_url="http://localhost:8000"
        )
    )
    assert results_config["environment_fingerprint"]["server"] == {}


@pytest.mark.asyncio
async def test_talker_params_are_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    warmup_config: benchmark.OmniSeedttsBenchmarkConfig,
) -> None:
    warmup_config.warmup = 0
    warmup_config.talker_temperature = 0.6
    forwarded: list[ForwardedSpeech] = []

    monkeypatch.setattr(
        benchmark.VoiceCloneOmni,
        "generate_speech",
        recording_generate_speech(forwarded),
    )
    benchmark_results = await benchmark.run_omni_seedtts_benchmark(warmup_config)
    assert [call.talker_params for call in forwarded] == [{"talker_temperature": 0.6}]
    assert benchmark_results["config"]["talker_temperature"] == 0.6


def test_sweep_writes_per_level_runs_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    warmup_config: benchmark.OmniSeedttsBenchmarkConfig,
) -> None:
    warmup_config.warmup = 0
    warmup_config.output_dir = str(tmp_path / "sweep")

    monkeypatch.setattr(
        benchmark.VoiceCloneOmni,
        "generate_speech",
        recording_generate_speech([]),
    )
    sweep = benchmark.run_sweep(warmup_config, [1, 2], 2)

    assert sweep["config"]["concurrencies"] == [1, 2]
    assert [level["concurrency"] for level in sweep["results"]] == [1, 2]
    for level in sweep["results"]:
        assert level["repeats"] == 2 and level["failed_requests"] == 0
        assert level["throughput_qps"]["n"] == 2
        assert level["throughput_qps"]["min"] <= level["throughput_qps"]["mean"]
        assert [row["repeat"] for row in level["per_repeat"]] == [1, 2]
    run_dirs = {path.name for path in (tmp_path / "sweep").iterdir()}
    assert run_dirs == {"c1_r1", "c1_r2", "c2_r1", "c2_r2", "sweep.json"}


def test_sweep_rejects_non_positive_repeats(
    warmup_config: benchmark.OmniSeedttsBenchmarkConfig,
) -> None:
    with pytest.raises(ValueError, match="repeats must be positive"):
        benchmark.run_sweep(warmup_config, [1], 0)


def test_sweep_rejects_empty_concurrencies(
    warmup_config: benchmark.OmniSeedttsBenchmarkConfig,
) -> None:
    with pytest.raises(ValueError, match="concurrencies must not be empty"):
        benchmark.run_sweep(warmup_config, [], 1)
