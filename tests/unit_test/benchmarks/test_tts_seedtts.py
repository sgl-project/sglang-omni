# SPDX-License-Identifier: Apache-2.0
"""SeedTTS benchmark entry-point: model profiles, server lifecycle, WER filter, CLI
configuration, concurrency sweeps and sustained overshoot."""

import asyncio
import json
import socket
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import contextmanager
from pathlib import Path
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
    _parse_concurrencies,
    _validate_args,
    plan_sustained_overshoot,
    run_tts_concurrency_sweep,
    run_tts_sustained_overshoot,
)
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
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
