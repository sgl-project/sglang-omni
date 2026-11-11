# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark entry point: model profiles, server lifecycle, WER filter, CLI configuration, concurrency sweeps, sustained overshoot and mixture loads."""

import argparse
import asyncio
import io
import json
import socket
import sys
import threading
import wave
from collections.abc import AsyncIterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import BinaryIO
from unittest.mock import MagicMock

import pytest
import requests
from aiohttp import web

from benchmarks.benchmarker.data import FinishReason
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval import benchmark_tts_seedtts as tts
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    _build_arg_parser,
    _build_results_config,
    _config_from_args,
    _parse_args,
    _parse_concurrencies,
    _validate_args,
    parse_mixture_loads,
    plan_sustained_overshoot,
    run_tts_concurrency_sweep,
    run_tts_mixture_sweep,
    run_tts_sustained_overshoot,
)
from benchmarks.metrics.mixture import MixtureLoad
from benchmarks.metrics.wer import (
    SampleOutput,
    calculate_asr_speed_metrics,
    calculate_wer_metrics,
)
from benchmarks.tasks import asr
from benchmarks.tasks.tts import (
    _build_tts_payload,
    make_tts_send_fn,
    stream_outcome_collector,
)
from sglang_omni.admission import QueueFullError
from tests.utils import QWEN3_ASR_WER_CONCURRENCY, assert_wer_partitioned

SEEDTTS_SAMPLE = SampleInput(
    sample_id="sample-1",
    ref_text="reference",
    ref_audio="ref.wav",
    target_text="hello world",
)


@pytest.mark.parametrize(
    "model, is_auk",
    [
        ("tencent/AuK", True),
        ("tencent/AuK-Flash", True),
        ("tencent/AuK@revision", True),
        ("/ckpt/auk-flash", True),
        ("fishaudio/s2-pro", False),
    ],
)
def test_cli_defaults_follow_checkpoint_name(monkeypatch, model, is_auk):
    monkeypatch.setattr(sys, "argv", ["benchmark", "--model", model])
    args, profile = tts._parse_args(
        tts._build_arg_parser()
    )  # noqa: leading-underscore  # production name
    config = tts._config_from_args(args)  # noqa: leading-underscore  # production name
    assert profile.forward_sglang_engine is not is_auk
    if is_auk:
        assert config.concurrency == config.warmup == 1
        assert config.seed == 1234
        assert config.output_dir == "results/auk_seedtts"
    else:
        assert profile.argument_defaults == {}


@pytest.mark.parametrize("model", ["tencent/AuK", "fishaudio/s2-pro"])
def test_evaluation_releases_tts_server_before_starting_asr(monkeypatch, model):
    events = []
    servers = []

    @contextmanager
    def server(**kwargs):
        servers.append(kwargs)
        events.append("start")
        yield
        events.append("stop")

    async def generate(config):
        assert config.port == 18280
        assert config.max_samples == 2
        assert config.model == model
        events.append("generate")

    def transcribe(config, **kwargs):
        assert kwargs["asr_router_port"] == 18280
        events.append("transcribe")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            model,
            "--port",
            "18280",
            "--max-samples",
            "2",
        ],
    )
    monkeypatch.setattr(tts, "managed_omni_server", server)
    monkeypatch.setattr(tts, "benchmark", generate)
    monkeypatch.setattr(tts, "run_tts_seedtts_transcribe", transcribe)
    tts.main()

    assert events == ["start", "generate", "stop", "start", "transcribe", "stop"]
    assert servers[0]["model_path"] == model
    assert servers[1]["model_path"] != model
    if model == "tencent/AuK":
        assert "max_running_requests" not in servers[0]
        assert "cuda_graph_max_bs" not in servers[0]
        assert servers[0]["server_config"] is None
    else:
        assert servers[0]["max_running_requests"] == 64
        assert servers[0]["cuda_graph_max_bs"] == 64


def test_filtered_wer_mean_keeps_exactly_50_percent_and_excludes_failures():
    metrics = calculate_wer_metrics(
        [
            SampleOutput(is_success=True, wer=0, hits=10),
            SampleOutput(is_success=True, wer=0.5, hits=1, deletions=1),
            SampleOutput(is_success=True, wer=0.75, hits=1, deletions=3),
            SampleOutput(is_success=False),
        ],
        "en",
    )
    assert metrics["wer_below_50_per_sample_mean"] == 0.25
    assert metrics["wer_below_50_corpus"] == pytest.approx(1 / 12)
    assert metrics["n_above_50_pct_wer"] == 1
    assert metrics["evaluated"] == 3
    assert metrics["skipped"] == 1


def test_explicit_cli_overrides_model_profile_defaults(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "tencent/AuK",
            "--max-concurrency",
            "3",
            "--warmup",
            "0",
            "--seed",
            "7",
            "--output-dir",
            "custom-results",
            "--server-config",
            "custom.yaml",
            "--max-new-tokens",
            "512",
        ],
    )

    args, _ = tts._parse_args(
        tts._build_arg_parser()
    )  # noqa: leading-underscore  # production name
    config = tts._config_from_args(args)  # noqa: leading-underscore  # production name
    assert config.concurrency == 3
    assert config.warmup == 0
    assert config.seed == 7
    assert config.output_dir == "custom-results"
    assert config.server_config == "custom.yaml"
    assert tts.resolve_max_new_tokens(config) == 512


@pytest.mark.parametrize(
    "model, max_new_tokens",
    [
        ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", None),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", None),
    ],
)
def test_max_new_tokens_default_applies_without_cli(model, max_new_tokens):
    # note (Yucheng Hu): TTS CI builds the config directly, so the per-model
    # default has to resolve at request-build time, not only in _parse_args.
    config = tts.TtsSeedttsBenchmarkConfig(model=model, meta="meta.lst")
    payload = _build_tts_payload(
        SEEDTTS_SAMPLE,
        model,
        **tts._build_generation_kwargs(
            config
        ),  # noqa: leading-underscore  # production name
    )
    assert payload.get("max_new_tokens") == max_new_tokens


def test_stream_send_fn_records_the_ids_the_outcome_collector_needs():
    async def iter_pcm_chunks() -> AsyncIterator[tuple[bytes, bool]]:
        yield bytes(8), True

    session = MagicMock()
    session.post.return_value.__aenter__.return_value = MagicMock(
        status=200,
        headers={
            "Content-Type": "audio/pcm",
            "X-Request-Id": "correlation-1",
            "X-SGLang-Omni-Speech-Id": "speech-1",
            "X-SGLang-Omni-Worker": "worker-b",
            "x-sample-rate": "4",
            "x-channels": "1",
            "x-bit-depth": "16",
        },
        content=MagicMock(iter_chunks=iter_pcm_chunks),
    )
    send_fn = make_tts_send_fn(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "http://host/v1/audio/speech",
        stream=True,
    )

    result = asyncio.run(send_fn(session, SEEDTTS_SAMPLE))

    assert result.is_success
    assert result.speech_outcome_id == "speech-1"
    assert result.server_worker_id == "worker-b"

    session.get.assert_not_called()


@pytest.mark.asyncio
async def test_slow_outcomes_do_not_occupy_generation_connections() -> None:
    lookup_started = asyncio.Event()
    generations_finished = asyncio.Event()
    release_lookups = asyncio.Event()
    completed = 0
    request_count = 120
    lookup_ids: list[str] = []

    async def speech(request: web.Request) -> web.Response:
        nonlocal completed
        if completed:
            await lookup_started.wait()
        else:
            pass
        completed += 1
        if completed == request_count:
            generations_finished.set()
        else:
            pass
        return web.Response(
            body=bytes(8),
            content_type="audio/pcm",
            headers={
                "X-SGLang-Omni-Speech-Id": f"speech#{completed}?%",
                "X-SGLang-Omni-Worker": "worker-b",
                "X-Sample-Rate": "4",
                "X-Channels": "1",
                "X-Bit-Depth": "16",
            },
        )

    async def outcome(request: web.Request) -> web.Response:
        lookup_ids.append(request.match_info["speech_id"])
        assert request.headers["x-sglang-omni-route-worker"] == "worker-b"
        lookup_started.set()
        await release_lookups.wait()
        return web.json_response(
            {
                "finish_reason": "length",
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 120,
                    "engine_time_s": 4.8,
                },
            }
        )

    application = web.Application()
    application.router.add_post("/v1/audio/speech", speech)
    application.router.add_get("/v1/audio/speech/{speech_id}", outcome)
    server = web.AppRunner(application)
    await server.setup()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.setblocking(False)
        api_url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1/audio/speech"
        await web.SockSite(server, listener).start()
        try:
            runner = BenchmarkRunner(
                RunConfig(max_concurrency=1, warmup=0, disable_tqdm=True)
            )
            send = make_tts_send_fn(
                "tts", api_url, stream=True, no_ref_audio=True, no_ref_text=True
            )
            async with stream_outcome_collector(api_url) as collect:
                task = asyncio.create_task(
                    runner.run(
                        [SEEDTTS_SAMPLE] * request_count, send, after_send=collect
                    )
                )
                try:
                    await asyncio.wait_for(generations_finished.wait(), timeout=3)
                    assert not task.done()
                finally:
                    release_lookups.set()
                    results = await asyncio.wait_for(task, timeout=5)
            assert all(result.is_success for result in results)
            assert all(
                result.finish_reason is FinishReason.LENGTH for result in results
            )
            assert all(result.completion_tokens == 120 for result in results)
            assert all(result.tok_per_s == pytest.approx(25.0) for result in results)
            assert set(lookup_ids) == {
                f"speech#{index}?%" for index in range(1, request_count + 1)
            }
        finally:
            await server.cleanup()


def test_wer_fanout_preserves_all_twenty_samples_at_long_audio_admission_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # note (wenyao): routing can send every request to one four-slot worker.
    slots = threading.BoundedSemaphore(4)
    cohort = threading.Barrier(min(QWEN3_ASR_WER_CONCURRENCY, 20))
    uploaded: list[str] = []

    def post(
        url: str, *, files: dict[str, tuple[str, BinaryIO, str]], **kwargs: object
    ) -> requests.Response:
        admitted = slots.acquire(blocking=False)
        try:
            # note (wenyao): all requests attempt admission before slots reopen.
            cohort.wait(timeout=5)
            uploaded.append(files["file"][0])
            response = requests.Response()
            response.url = url
            response.status_code = 200 if admitted else 503
            response._content = json.dumps(  # noqa: leading-underscore  # upstream name
                {"text": "hello world"}
                if admitted
                else {
                    "detail": "Too many long-audio transcriptions in flight "
                    "(limit 4); retry later"
                }
            ).encode()
            return response
        finally:
            if admitted:
                slots.release()

    monkeypatch.setattr(asr.requests, "post", post)
    records: list[dict[str, str | bool | int]] = []
    for index in range(20):
        path = tmp_path / f"sample-{index}.wav"
        path.write_bytes(b"saved audio for mocked transcription service")
        records.append(
            {
                "sample_id": f"sample-{index}",
                "raw_response": "hello world",
                "is_success": True,
                "wav_path": str(path),
                "audio_duration_s": 31,
            }
        )
    result = asr.compute_text_audio_consistency_from_records(
        records,
        "en",
        "cuda:0",
        asr_router_port=12345,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
    )

    assert len(uploaded) == len(set(uploaded)) == 20
    assert result["summary"]["evaluated"] == 20
    assert result["summary"]["skipped"] == 0
    assert_wer_partitioned(result, max_wer_below_50_corpus=0, max_n_above_50=0)
    assert asr.DEFAULT_ASR_TRANSCRIBE_CONCURRENCY == 32


def config_from_cli(*args: str) -> TtsSeedttsBenchmarkConfig:
    parser = _build_arg_parser()
    parsed = parser.parse_args(list(args))
    _validate_args(parser, parsed)
    return _config_from_args(parsed)


def test_seedtts_benchmark_batch_args_default_to_64() -> None:
    config = config_from_cli()

    assert config.max_running_requests == 64
    assert config.cuda_graph_max_bs == 64
    assert config.max_queued_requests is None

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 64
    assert results_config["cuda_graph_max_bs"] == 64
    assert results_config["max_queued_requests"] is None


def test_seedtts_benchmark_batch_args_are_independent() -> None:
    config = config_from_cli(
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "128",
        "--max-queued-requests",
        "16",
    )

    assert config.max_running_requests == 32
    assert config.cuda_graph_max_bs == 128
    assert config.max_queued_requests == 16

    results_config = _build_results_config(
        config,
        base_url="http://localhost:8000",
    )
    assert results_config["max_running_requests"] == 32
    assert results_config["cuda_graph_max_bs"] == 128
    assert results_config["max_queued_requests"] == 16


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ([], False),
        (["--stream"], True),
        (["--stream", "--no-collect-stream-outcomes"], False),
    ],
)
def test_outcome_collection_control_is_preserved_in_results(
    arguments: list[str],
    expected: bool,
) -> None:
    config = config_from_cli(*arguments)
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["collect_stream_outcomes"] is expected


def test_seedtts_benchmark_records_quantization() -> None:
    config = config_from_cli("--quantization", "fp8")
    assert config.quantization == "fp8"
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["quantization"] == "fp8"


def test_seedtts_benchmark_quantization_defaults_to_none() -> None:
    config = config_from_cli()
    assert config.quantization is None
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["quantization"] is None


def test_results_config_records_sampling_and_omits_fingerprint() -> None:
    config = config_from_cli(
        "--temperature",
        "0.2",
        "--top-p",
        "0.9",
        "--top-k",
        "10",
        "--repetition-penalty",
        "1.1",
        "--seed",
        "3",
    )
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["temperature"] == 0.2
    assert results_config["top_p"] == 0.9
    assert results_config["top_k"] == 10
    assert results_config["repetition_penalty"] == 1.1
    assert results_config["seed"] == 3
    assert "environment_fingerprint" not in results_config

    unset = _build_results_config(config_from_cli(), base_url="http://localhost:8000")
    assert unset["temperature"] is None
    assert unset["top_p"] is None
    assert unset["top_k"] is None
    assert unset["repetition_penalty"] is None
    assert unset["seed"] is None


@pytest.mark.parametrize(
    "model, max_new_tokens",
    [
        ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", None),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", None),
    ],
)
def test_results_config_records_the_resolved_max_new_tokens(
    model: str, max_new_tokens: int | None
) -> None:
    results_config = _build_results_config(
        config_from_cli("--model", model), base_url="http://localhost:8000"
    )
    assert results_config["max_new_tokens"] == max_new_tokens


def test_results_config_includes_fingerprint_when_set() -> None:
    config = config_from_cli()
    config.environment_fingerprint = {"client": {"git": {}}, "server": {"url": "u"}}
    results_config = _build_results_config(config, base_url="http://localhost:8000")
    assert results_config["environment_fingerprint"]["server"]["url"] == "u"


def test_parse_concurrencies() -> None:
    assert _parse_concurrencies("16,32,48,64") == [16, 32, 48, 64]


def test_plan_sustained_overshoot() -> None:
    plan = plan_sustained_overshoot(
        max_running_requests=32,
        max_queued_requests=16,
        duration_s=10,
    )
    assert plan.capacity == 48
    assert plan.concurrency == 0
    assert plan.request_rate == 96
    assert plan.request_count == 960
    assert plan.overshoot_ratio == 2.0
    with pytest.raises(ValueError, match="request_rate > admission capacity"):
        plan_sustained_overshoot(
            max_running_requests=32,
            max_queued_requests=16,
            duration_s=10,
            request_rate=48,
        )


def make_sample(sample_id: str) -> SampleInput:
    return SampleInput(
        sample_id=sample_id, ref_text="r", ref_audio="r.wav", target_text="t"
    )


def stub_seedtts_benchmark(monkeypatch, run_fn, *, samples=None) -> None:
    monkeypatch.setattr(
        "benchmarks.eval.benchmark_tts_seedtts.run_tts_seedtts_benchmark",
        run_fn,
    )
    monkeypatch.setattr(
        "benchmarks.eval.benchmark_tts_seedtts.print_speed_summary",
        lambda *args, **kwargs: None,
    )
    if samples is not None:
        monkeypatch.setattr(
            "benchmarks.eval.benchmark_tts_seedtts._load_benchmark_samples",
            lambda config: samples,
        )


@pytest.mark.parametrize(
    "argv, needle",
    [
        (
            [
                "--generate-only",
                "--sustained-overshoot",
                "--max-running-requests",
                "32",
            ],
            "max-queued-requests",
        ),
        (
            [
                "--generate-only",
                "--sustained-overshoot",
                "--max-queued-requests",
                "16",
                "--concurrencies",
                "16,32",
            ],
            "cannot be combined",
        ),
        (["--repeats", "3"], "require --generate-only"),
        (["--generate-only", "--repeats", "0"], "must be positive"),
    ],
)
def test_cli_rejects_invalid_overshoot(
    capsys: pytest.CaptureFixture[str], argv: list[str], needle: str
) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        _validate_args(parser, args)
    assert exc.value.code == 2
    assert needle in capsys.readouterr().err


@pytest.mark.asyncio
async def test_concurrency_sweep_writes_per_point_output_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_output_dirs: list[str] = []

    async def fake_run(config: TtsSeedttsBenchmarkConfig, **_kwargs) -> dict:
        seen_output_dirs.append(config.output_dir)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(config.output_dir) / "speed_results.json").write_text(
            "{}", encoding="utf-8"
        )
        return {
            "summary": {
                "completed_requests": 1,
                "failed_requests": 0,
                "latency_p95_s": 0.1,
                "audio_ttfp_p95_s": 0.05,
            }
        }

    stub_seedtts_benchmark(monkeypatch, fake_run)

    config = config_from_cli("--output-dir", str(tmp_path))
    payload = await run_tts_concurrency_sweep(config, [16, 32])
    expected = [str(tmp_path / "c16"), str(tmp_path / "c32")]
    assert seen_output_dirs == expected
    assert [row["per_repeat"][0]["output_dir"] for row in payload["rows"]] == expected
    assert [row["repeats"] for row in payload["rows"]] == [1, 1]
    assert payload["rows"][0]["latency_p95_s"] == {
        "mean": 0.1,
        "min": 0.1,
        "max": 0.1,
        "n": 1,
    }
    assert (tmp_path / "c16" / "speed_results.json").is_file()
    assert (tmp_path / "c32" / "speed_results.json").is_file()
    assert payload["repeats"] == 1


@pytest.mark.asyncio
async def test_concurrency_sweep_repeats_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(config: TtsSeedttsBenchmarkConfig, **_kwargs) -> dict:
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        latency = 3.0 if config.output_dir.endswith("_r2") else 1.0
        return {
            "summary": {
                "completed_requests": 1,
                "failed_requests": 0,
                "latency_p95_s": latency,
                "throughput_qps": 4.0,
            }
        }

    stub_seedtts_benchmark(monkeypatch, fake_run)
    config = config_from_cli("--output-dir", str(tmp_path), "--generate-only")
    payload = await run_tts_concurrency_sweep(config, [16], repeats=2)
    row = payload["rows"][0]
    assert [item["output_dir"] for item in row["per_repeat"]] == [
        str(tmp_path / "c16_r1"),
        str(tmp_path / "c16_r2"),
    ]
    assert row["latency_p95_s"] == {"mean": 2.0, "min": 1.0, "max": 3.0, "n": 2}
    assert row["throughput_qps"] == {"mean": 4.0, "min": 4.0, "max": 4.0, "n": 2}
    assert row["latency_mean_s"]["n"] == 0
    assert [item["repeat"] for item in row["per_repeat"]] == [1, 2]


@pytest.mark.asyncio
async def test_concurrency_sweep_rejects_non_positive_repeats(tmp_path: Path) -> None:
    config = config_from_cli("--output-dir", str(tmp_path))
    with pytest.raises(ValueError, match="repeats must be positive"):
        await run_tts_concurrency_sweep(config, [1], repeats=0)


@pytest.mark.asyncio
async def test_sustained_overshoot_uses_open_loop_and_unique_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[TtsSeedttsBenchmarkConfig] = []
    seen_samples: list[list[str]] = []

    async def fake_run(config: TtsSeedttsBenchmarkConfig, **kwargs) -> dict:
        seen.append(config)
        seen_samples.append([sample.sample_id for sample in kwargs["samples"]])
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        return {
            "summary": {
                "completed_requests": 1,
                "failed_requests": 1,
                "latency_p95_s": 0.4,
                "audio_ttfp_p95_s": 0.05,
            },
            "per_request": [
                {
                    "is_success": True,
                    "error": None,
                    "audio_ttfp_s": 0.05,
                    "latency_s": 0.4,
                },
                {
                    "is_success": False,
                    "error": f"HTTP 503: {QueueFullError.MESSAGE}",
                    "audio_ttfp_s": None,
                    "latency_s": 0.01,
                },
                {
                    "is_success": False,
                    "error": "HTTP 500: boom",
                    "audio_ttfp_s": None,
                    "latency_s": 1.0,
                },
            ],
            "config": {"concurrency": config.concurrency},
        }

    stub_seedtts_benchmark(monkeypatch, fake_run, samples=[make_sample("s0")])

    payload = await run_tts_sustained_overshoot(
        config_from_cli(
            "--output-dir",
            str(tmp_path),
            "--generate-only",
            "--sustained-overshoot",
            "--max-running-requests",
            "32",
            "--max-queued-requests",
            "16",
            "--overshoot-duration-s",
            "1",
            "--request-rate",
            "64",
        )
    )

    assert seen[0].concurrency == 0
    assert seen[0].request_rate == 64
    assert seen[0].output_dir == str(tmp_path / "overshoot")
    assert payload["outcomes"] == {
        "total_requests": 3,
        "success": 1,
        "queue_full": 1,
        "other_failed": 1,
        "success_ttfa_p95_s": 0.05,
        "queue_full_latency_p95_s": 0.01,
    }
    assert len(set(seen_samples[0])) == 64
    assert (tmp_path / "overshoot" / "sustained_overshoot.json").is_file()


def test_asr_speed_counts_scored_and_unscorable_transcriptions() -> None:
    """Scored and unscorable transcriptions both contribute to speed, unlike request failures."""
    outputs = [
        SampleOutput(
            is_transcribed=True,
            is_success=True,
            asr_latency_s=0.25,
            audio_duration_s=1.0,
        ),
        SampleOutput(is_transcribed=True, asr_latency_s=0.75, audio_duration_s=3.0),
        SampleOutput(error="Transcription failed", asr_latency_s=9.0),
    ]
    speed = calculate_asr_speed_metrics(outputs, wall_time_s=2.0)
    wer = calculate_wer_metrics(outputs, "en")
    assert speed["evaluated"] == 2
    assert speed["failed_requests"] == 1
    assert speed["asr_throughput_samples_per_s"] == 1.0
    assert speed["asr_latency_mean_s"] == 0.5
    assert speed["asr_latency_p95_s"] == pytest.approx(0.725)
    assert speed["asr_audio_processed_s"] == 4.0
    assert speed["asr_rtf_mean"] == 0.25
    assert wer["evaluated"] == 1
    assert wer["skipped"] == 2


def test_parse_mixture_loads_keeps_the_given_order() -> None:
    assert parse_mixture_loads("48+16, 16+16,48+0,0+16") == [
        MixtureLoad(non_streaming_clients=48, streaming_clients=16),
        MixtureLoad(non_streaming_clients=16, streaming_clients=16),
        MixtureLoad(non_streaming_clients=48, streaming_clients=0),
        MixtureLoad(non_streaming_clients=0, streaming_clients=16),
    ]


@pytest.mark.parametrize(
    "value", ["48", "48+", "+16", "-1+2", "0+0", "a+b", "48+16,48+16", ""]
)
def test_parse_mixture_loads_rejects_malformed_loads(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_mixture_loads(value)


@pytest.mark.parametrize(
    "argv, message",
    [
        (
            ["--mixture", "48+16", "--generate-only", "--concurrencies", "16,32"],
            "--mixture cannot be combined with --concurrencies",
        ),
        (
            ["--mixture", "48+16", "--concurrency", "8"],
            "--mixture cannot be combined with --concurrency/--max-concurrency",
        ),
        (
            ["--model", "tencent/AuK", "--mixture", "4+2"],
            "the model profile sets concurrency and warmup",
        ),
        (
            ["--mixture", "48+16", "--stream"],
            "--mixture cannot be combined with --stream",
        ),
        (
            ["--mixture", "48+16", "--request-rate", "4"],
            "--mixture cannot be combined with --request-rate",
        ),
        (
            ["--mixture", "48+16", "--arrival-seed", "3"],
            "--mixture cannot be combined with --arrival-seed",
        ),
        (
            [
                "--mixture",
                "48+16",
                "--generate-only",
                "--sustained-overshoot",
                "--max-queued-requests",
                "16",
            ],
            "--mixture cannot be combined with --sustained-overshoot",
        ),
        (
            ["--mixture", "48+16", "--subtalker-dosample-ratio", "0.5"],
            "--mixture cannot be combined with --subtalker-dosample-ratio",
        ),
        (
            ["--mixture", "48+16", "--warmup", "2"],
            "--mixture cannot be combined with --warmup other than 0",
        ),
        (
            ["--mixture", "48+16", "--transcribe-only"],
            "--mixture cannot be combined with --transcribe-only",
        ),
        (
            ["--mixture", "48+16", "--similarity-only"],
            "--mixture cannot be combined with --similarity-only",
        ),
        (
            ["--mixture", "48+16", "--utmos-only"],
            "--mixture cannot be combined with --utmos-only",
        ),
        (
            ["--mixture", "48+16,48+0"],
            "--mixture with more than one load requires --generate-only",
        ),
        (
            [
                "--mixture",
                "48+16",
                "--max-running-requests",
                "32",
                "--max-queued-requests",
                "16",
            ],
            "--mixture above --max-running-requests plus --max-queued-requests "
            "needs --warmup 0",
        ),
        (["--mixture-duration-s", "0"], "--mixture-duration-s must be positive"),
        (["--mixture-settle-s", "-1"], "--mixture-settle-s must be at least 0"),
    ],
)
def test_cli_rejects_invalid_mixture(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])
    parser = _build_arg_parser()
    args, _ = _parse_args(parser)
    with pytest.raises(SystemExit) as exc:
        _validate_args(parser, args)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_cli_accepts_a_mixture_above_capacity_without_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--mixture",
            "48+16",
            "--max-running-requests",
            "32",
            "--max-queued-requests",
            "16",
            "--warmup",
            "0",
        ],
    )
    parser = _build_arg_parser()
    args, _ = _parse_args(parser)
    _validate_args(parser, args)


def test_mixture_config_records_the_load_and_keeps_the_legacy_concurrency() -> None:
    legacy = config_from_cli()
    assert legacy.concurrency == tts.DEFAULT_TTS_BENCHMARK_CONCURRENCY
    assert _build_results_config(legacy, base_url="http://h")["mixture"] is None

    single = config_from_cli("--mixture", "48+16")
    results_config = _build_results_config(single, base_url="http://h")
    assert single.mixture == MixtureLoad(non_streaming_clients=48, streaming_clients=16)
    assert (results_config["concurrency"], results_config["warmup"]) == (64, 64)
    assert results_config["stream"] is None
    assert results_config["collect_stream_outcomes"] is True
    assert results_config["mixture"] == {
        "non_streaming_clients": 48,
        "streaming_clients": 16,
        "settle_s": 10.0,
        "duration_s": 120.0,
    }
    assert config_from_cli("--mixture", "48+16,48+0", "--generate-only").mixture is None

    load = MixtureLoad(non_streaming_clients=48, streaming_clients=16)
    for warmup, expected_warmup in ((None, 64), (0, 0)):
        direct = TtsSeedttsBenchmarkConfig(
            model="tts", meta="unused", mixture=load, warmup=warmup
        )
        recorded = _build_results_config(direct, base_url="http://h")
        assert (recorded["concurrency"], recorded["warmup"]) == (64, expected_warmup)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repeats, point_names, streaming_audio",
    [
        (1, ["ns48_s16", "ns48_s0"], {"mean": 50.0, "min": 50.0, "max": 50.0, "n": 1}),
        (
            2,
            ["ns48_s16_r1", "ns48_s16_r2", "ns48_s0_r1", "ns48_s0_r2"],
            {"mean": 75.0, "min": 50.0, "max": 100.0, "n": 2},
        ),
    ],
)
async def test_mixture_sweep_writes_per_load_output_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repeats: int,
    point_names: list[str],
    streaming_audio: dict[str, float | int],
) -> None:
    seen: list[TtsSeedttsBenchmarkConfig] = []

    async def fake_run(config: TtsSeedttsBenchmarkConfig, **_kwargs) -> dict:
        seen.append(config)
        audio_scale = 2.0 if config.output_dir.endswith("_r2") else 1.0
        classes = list(config.mixture.clients_by_class)
        return {
            "summary": {"completed_requests": 4, "failed_requests": 0},
            "summary_by_class": {
                request_class: {"audio_throughput_s_per_s": 50.0 * audio_scale}
                for request_class in classes
            },
            "mixture": {
                "outside_window_failed_requests": {
                    request_class: 1 for request_class in classes
                }
            },
        }

    stub_seedtts_benchmark(monkeypatch, fake_run)
    monkeypatch.setattr(tts, "print_mixture_summary", lambda *args, **kwargs: None)
    loads = parse_mixture_loads("48+16,48+0")
    config = config_from_cli(
        "--output-dir",
        str(tmp_path),
        "--generate-only",
        "--mixture",
        "48+16,48+0",
        "--mixture-settle-s",
        "5",
        "--mixture-duration-s",
        "60",
    )

    await run_tts_mixture_sweep(config, loads, repeats=repeats)

    expected_dirs = [str(tmp_path / name) for name in point_names]
    assert [point.output_dir for point in seen] == expected_dirs
    assert [(point.mixture, point.concurrency) for point in seen] == [
        (load, load.total_clients) for load in loads for _ in range(repeats)
    ]
    written = json.loads((tmp_path / "mixture_sweep.json").read_text())
    assert written["loads"] == ["ns48_s16", "ns48_s0"]
    leg_config = written["config"]
    assert leg_config["mixture"] == {
        "loads": ["ns48_s16", "ns48_s0"],
        "settle_s": 5.0,
        "duration_s": 60.0,
    }
    assert {leg_config[name] for name in ("stream", "concurrency", "warmup")} == {None}
    assert leg_config["collect_stream_outcomes"] is True
    mixture_row, single_row = written["rows"]
    assert mixture_row["category"] == "mixture"
    assert single_row["category"] == "non_streaming"
    assert set(single_row["summary_by_class"]) == {"non_streaming"}
    assert (
        mixture_row["summary_by_class"]["streaming"]["audio_throughput_s_per_s"]
        == streaming_audio
    )
    assert mixture_row["failed_requests"] == 2 * repeats
    per_repeat_dirs = [item["output_dir"] for item in mixture_row["per_repeat"]]
    assert per_repeat_dirs == expected_dirs[:repeats]


@pytest.mark.asyncio
async def test_mixture_run_against_a_fake_server_tags_both_classes(
    tmp_path: Path,
) -> None:
    """Each class reaches the server in its own mode and every request is saved."""
    stream_flags: list[bool] = []
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(bytes(1600))

    async def speech(request: web.Request) -> web.Response:
        payload = await request.json()
        stream_flags.append(bool(payload.get("stream")))
        await asyncio.sleep(0.01)
        if payload.get("stream"):
            return web.Response(
                body=bytes(1600),
                content_type="audio/pcm",
                headers={
                    "X-SGLang-Omni-Speech-Id": "speech-1",
                    "X-Sample-Rate": "8000",
                    "X-Channels": "1",
                    "X-Bit-Depth": "16",
                },
            )
        else:
            return web.Response(body=wav_buffer.getvalue(), content_type="audio/wav")

    async def outcome(request: web.Request) -> web.Response:
        return web.Response(status=404)

    application = web.Application()
    application.router.add_post("/v1/audio/speech", speech)
    application.router.add_get("/v1/audio/speech/{speech_id}", outcome)
    server = web.AppRunner(application)
    await server.setup()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.setblocking(False)
        base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        await web.SockSite(server, listener).start()
        try:
            config = TtsSeedttsBenchmarkConfig(
                model="tts",
                meta="unused",
                base_url=base_url,
                output_dir=str(tmp_path),
                concurrency=3,
                mixture=MixtureLoad(non_streaming_clients=2, streaming_clients=1),
                mixture_duration_s=0.2,
                mixture_settle_s=0.05,
            )
            results = await tts.run_tts_seedtts_benchmark(
                config, samples=[make_sample(f"s{index}") for index in range(4)]
            )
        finally:
            await server.cleanup()

    per_request = results["per_request"]
    assert set(results["summary_by_class"]) == {"non_streaming", "streaming"}
    assert {entry["request_class"] for entry in per_request} == {
        "non_streaming",
        "streaming",
    }
    for entry in per_request:
        is_stream = entry["request_class"] == "streaming"
        assert entry["id"].split("#")[1][0] == ("s" if is_stream else "n")
        assert (entry["audio_ttfp_s"] is not None) is is_stream
    assert set(stream_flags) == {True, False}
    generated = json.loads((tmp_path / "generated.json").read_text())
    assert {entry["sample_id"] for entry in generated} == {
        entry["id"] for entry in per_request
    }
    saved = json.loads((tmp_path / "speed_results.json").read_text())
    assert saved["mixture"]["load"] == "ns2_s1"
    assert saved["config"]["mixture"]["streaming_clients"] == 1


@pytest.mark.asyncio
async def test_mixture_config_rejects_settings_the_runner_ignores() -> None:
    """A config built without the CLI used to run a mixture load with these dropped."""
    config = TtsSeedttsBenchmarkConfig(
        model="tts",
        meta="unused",
        mixture=MixtureLoad(non_streaming_clients=2, streaming_clients=1),
        warmup=5,
        stream=True,
        request_rate=4.0,
        arrival_seed=3,
        subtalker_dosample_ratio=0.5,
    )
    with pytest.raises(ValueError) as error:
        await tts.run_tts_seedtts_benchmark(config)
    for name in (
        "warmup other than 0",
        "stream",
        "request_rate",
        "arrival_seed",
        "subtalker_dosample_ratio",
    ):
        assert name in str(error.value)


@pytest.mark.parametrize(
    "saved_config, expected_mode, expected_concurrency",
    [
        (
            {
                "stream": None,
                "collect_stream_outcomes": True,
                "concurrency": 6,
                "mixture": {
                    "non_streaming_clients": 4,
                    "streaming_clients": 2,
                    "settle_s": 10.0,
                    "duration_s": 120.0,
                },
            },
            "mixture",
            6,
        ),
        ({"stream": False}, "non-streaming", tts.DEFAULT_TTS_BENCHMARK_CONCURRENCY),
    ],
)
def test_a_later_transcribe_keeps_the_generation_mode_of_the_saved_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    saved_config: dict,
    expected_mode: str,
    expected_concurrency: int,
) -> None:
    """--transcribe-only after a mixture run used to record non streaming at concurrency 16."""
    (tmp_path / "speed_results.json").write_text(
        json.dumps({"summary": {}, "config": saved_config}), encoding="utf-8"
    )
    calls: list[dict] = []
    monkeypatch.setattr(
        tts, "run_seedtts_transcribe", lambda config, **kwargs: calls.append(kwargs)
    )

    tts.run_tts_seedtts_transcribe(
        config_from_cli("--transcribe-only", "--output-dir", str(tmp_path)),
        asr_router_port=8000,
    )

    assert calls[0]["generation_mode"] == expected_mode
    assert calls[0]["wer_config"]["concurrency"] == expected_concurrency
    assert calls[0]["wer_config"]["mixture"] == saved_config.get("mixture")


@pytest.mark.parametrize(
    "argv, expected_events, expected_points, expected_transcribed",
    [
        (
            ["--mixture", "4+2"],
            ["start", "generate", "stop", "start", "transcribe", "stop"],
            [(".", "ns4_s2")],
            ["ns4_s2"],
        ),
        (
            ["--mixture", "4+2", "--repeats", "2", "--generate-only"],
            ["start", "generate", "generate", "stop"],
            [("ns4_s2_r1", "ns4_s2"), ("ns4_s2_r2", "ns4_s2")],
            [],
        ),
        (
            ["--mixture", "4+2,0+2", "--generate-only"],
            ["start", "generate", "generate", "stop"],
            [("ns4_s2", "ns4_s2"), ("ns0_s2", "ns0_s2")],
            [],
        ),
    ],
)
def test_main_sends_one_mixture_load_to_wer_and_several_to_the_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected_events: list[str],
    expected_points: list[tuple[str, str]],
    expected_transcribed: list[str],
) -> None:
    """One load generates and then scores WER after the TTS server stops, while repeats or several loads run the generate only sweep."""
    events: list[str] = []
    generated: list[TtsSeedttsBenchmarkConfig] = []
    transcribed: list[TtsSeedttsBenchmarkConfig] = []

    @contextmanager
    def server(**kwargs):
        events.append("start")
        yield
        events.append("stop")

    async def generate(config: TtsSeedttsBenchmarkConfig, **kwargs) -> dict:
        events.append("generate")
        generated.append(config)
        classes = list(config.mixture.clients_by_class)
        return {
            "summary": {"failed_requests": 0},
            "summary_by_class": {request_class: {} for request_class in classes},
            "mixture": {
                "outside_window_failed_requests": {
                    request_class: 0 for request_class in classes
                }
            },
        }

    def transcribe(config: TtsSeedttsBenchmarkConfig, **kwargs) -> None:
        events.append("transcribe")
        transcribed.append(config)

    monkeypatch.setattr(
        sys, "argv", ["benchmark", "--output-dir", str(tmp_path), *argv]
    )
    monkeypatch.setattr(tts, "managed_omni_server", server)
    monkeypatch.setattr(tts, "run_tts_seedtts_benchmark", generate)
    monkeypatch.setattr(tts, "run_tts_seedtts_transcribe", transcribe)
    monkeypatch.setattr(tts, "print_mixture_summary", lambda *args, **kwargs: None)
    tts.main()

    points = [
        (Path(point.output_dir).relative_to(tmp_path).as_posix(), point.mixture.name)
        for point in generated
    ]
    assert events == expected_events
    assert points == expected_points
    assert [config.mixture.name for config in transcribed] == expected_transcribed
    assert (tmp_path / "mixture_sweep.json").is_file() == (not expected_transcribed)


@pytest.mark.parametrize(
    ("sample_ids", "extra_argv", "expected_ids"),
    [
        (("row0#n0", "row1#s0", "row0#n2"), (), ("row0#n0", "row1#s0", "row0#n2")),
        (
            ("row2#s0", "row0#n0", "row1#s1", "row0#n3"),
            ("--max-samples", "2"),
            ("row0#n0", "row1#s1", "row0#n3"),
        ),
    ],
)
def test_similarity_after_a_mixture_run_scores_each_repeat_against_its_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_ids: tuple[str, ...],
    extra_argv: tuple[str, ...],
    expected_ids: tuple[str, ...],
) -> None:
    """--similarity-only on a mixture output used to skip every suffixed id, and --max-samples kept the first completions instead of the selected rows."""
    reference_paths = {
        name: tmp_path / f"{name}.ref.wav" for name in ("row0", "row1", "row2")
    }
    for reference_path in reference_paths.values():
        reference_path.write_bytes(b"reference")
    generated = []
    for sample_id in sample_ids:
        wav_path = tmp_path / f"{sample_id}.wav"
        wav_path.write_bytes(b"generated")
        generated.append(
            {"sample_id": sample_id, "wav_path": str(wav_path), "is_success": True}
        )
    (tmp_path / "generated.json").write_text(json.dumps(generated), encoding="utf-8")
    scored_pairs: list[tuple[str, str]] = []

    class FakeSpeakerSimilarity:
        def __init__(self, **kwargs: object) -> None:
            pass

        def score_batch(
            self, ref_audio_paths: list[str], wav_paths: list[str]
        ) -> list[float]:
            scored_pairs.extend(zip(ref_audio_paths, wav_paths))
            return [0.5] * len(wav_paths)

    monkeypatch.setattr(
        "benchmarks.tasks.tts.load_seedtts_samples",
        lambda meta, max_samples, split: [
            SampleInput(
                sample_id=name, ref_text="r", ref_audio=str(path), target_text="t"
            )
            for name, path in list(reference_paths.items())[:max_samples]
        ],
    )
    monkeypatch.setattr(
        "benchmarks.tasks.tts.ensure_speaker_similarity_assets",
        lambda **kwargs: SimpleNamespace(finetune_checkpoint="ft", wavlm_base="wavlm"),
    )
    monkeypatch.setattr(
        "benchmarks.tasks.tts.WavLMSpeakerSimilarity", FakeSpeakerSimilarity
    )
    config = config_from_cli(
        "--similarity-only",
        "--output-dir",
        str(tmp_path),
        "--device",
        "cpu",
        *extra_argv,
    )

    result = tts.run_seedtts_similarity(config)

    assert result["summary"]["evaluated"] == len(expected_ids)
    assert result["summary"]["total_samples"] == len(expected_ids)
    assert sorted(scored_pairs) == sorted(
        (
            str(reference_paths[sample_id.rsplit("#", 1)[0]]),
            str(tmp_path / f"{sample_id}.wav"),
        )
        for sample_id in expected_ids
    )
