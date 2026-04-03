# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks and voice-clone WER thresholds CI for S2-Pro as a representative of TTS models.

Usage:
    pytest tests/test_model/test_s2pro_tts_ci.py -s -x
    pytest tests/test_model/test_s2pro_tts_ci.py -s -x --concurrency 8
    pytest tests/test_model/test_s2pro_tts_ci.py -s -x --concurrency all

Author:
    chenyang zhao https://github.com/zhaochenyang20
    Raitsh P https://github.com/Ratish1
    Jingwen Guo https://github.com/JingwenGu0829
    Yuan Luo https://github.com/yuan-luo
    Yitong Guan https://github.com/minleminzui

The benchmark supports one selected concurrency per test run. Use `--concurrency 8`
in CI, run without the flag to use concurrency 1, or pass `--concurrency all`
to sweep all supported concurrency values locally.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_tts_speed import (
    TtsSpeedBenchmarkConfig,
    run_tts_speed_benchmark,
)
from tests.utils import find_free_port, start_server, stop_server

PER_REQUEST_STORE: dict[str, list[dict]] = {}
SPEED_OUTPUT_DIRS: dict[str, dict[int, str]] = {"non_stream": {}, "stream": {}}

S2PRO_MODEL_PATH = "fishaudio/s2-pro"
S2PRO_CONFIG_PATH = "examples/configs/s2pro_tts.yaml"

STARTUP_TIMEOUT = 600
BENCHMARK_TIMEOUT = 600
WER_TIMEOUT = 600
DATASET_CACHE_ENV = "SGLANG_SEEDTTS50_DIR"
STREAMING_BENCHMARK_MAX_SAMPLES = 16

# Thresholds reference: https://github.com/sgl-project/sglang-omni/pull/242
# Note (chenyang): the RTF thresholds also includes the reference audio
# processing time.

# Note (Ratish, Chenyang): We evalute the performance of S2-Pro CI on our H20
# CI machines and compute the thresholds based on the results.

# The following are the P95 thresholds compute from previous runs.

# concurrency 1: throughput>=0.13, tok/s>=82.5, latency_mean<=7.6, rtf_mean<=2.03
# concurrency 2: throughput>=0.25, tok/s>=78.4, latency_mean<=7.9, rtf_mean<=2.10
# concurrency 4: throughput>=0.47, tok/s>=75.3, latency_mean<=8.3, rtf_mean<=2.21
# concurrency 8: throughput>=0.80, tok/s>=67.7, latency_mean<=9.1, rtf_mean<=2.43
# concurrency 16: throughput>=1.17, tok/s>=60.7, latency_mean<=11.2, rtf_mean<=3.01

VC_WER_MAX_CORPUS = 0.025
VC_WER_MAX_PER_SAMPLE = 0.40
VC_STREAM_WER_MAX_CORPUS = 0.025
VC_STREAM_WER_MAX_PER_SAMPLE = 0.40


# TODO (Ratish, Chenyang): The precomputed performance thresholds compute directly with
# the CI machines still varies from CI actual results. We should figure out the difference
# and update the thresholds accordingly.

# Note (Chenyang): Only thresholds for concurrency 8 are dedicatedly tuned, others may not
# pass the CI.

VC_NON_STREAM_THRESHOLDS = {
    1: {
        "throughput_qps_min": 0.11,
        "tok_per_s_agg_min": 74.2,
        "latency_mean_s_max": 8.4,
        "rtf_mean_max": 2.24,
    },
    2: {
        "throughput_qps_min": 0.22,
        "tok_per_s_agg_min": 70.5,
        "latency_mean_s_max": 8.7,
        "rtf_mean_max": 2.31,
    },
    4: {
        "throughput_qps_min": 0.42,
        "tok_per_s_agg_min": 67.7,
        "latency_mean_s_max": 9.2,
        "rtf_mean_max": 2.44,
    },
    8: {
        "throughput_qps_min": 0.68,
        "tok_per_s_agg_min": 60.9,
        "latency_mean_s_max": 10.1,
        "rtf_mean_max": 2.68,
    },
    16: {
        "throughput_qps_min": 1.05,
        "tok_per_s_agg_min": 54.6,
        "latency_mean_s_max": 12.4,
        "rtf_mean_max": 3.32,
    },
}

# Pre-slack H20 reference values from 5 repeated runs before extra CI slack:
# concurrency 1: throughput>=0.09, tok/s>=21.0, latency_mean<=10.8, rtf_mean<=2.60
# concurrency 2: throughput>=0.15, tok/s>=14.7, latency_mean<=13.3, rtf_mean<=3.20
# concurrency 4: throughput>=0.23, tok/s>=13.7, latency_mean<=15.7, rtf_mean<=4.08
# concurrency 8: throughput>=0.32, tok/s>=9.3, latency_mean<=22.7, rtf_mean<=5.89
# concurrency 16: throughput>=0.27, tok/s>=2.9, latency_mean<=47.7, rtf_mean<=12.02

VC_STREAM_THRESHOLDS = {
    1: {
        "throughput_qps_min": 0.08,
        "tok_per_s_agg_min": 18.9,
        "latency_mean_s_max": 11.9,
        "rtf_mean_max": 2.86,
    },
    2: {
        "throughput_qps_min": 0.13,
        "tok_per_s_agg_min": 13.2,
        "latency_mean_s_max": 14.7,
        "rtf_mean_max": 3.52,
    },
    4: {
        "throughput_qps_min": 0.20,
        "tok_per_s_agg_min": 12.3,
        "latency_mean_s_max": 17.3,
        "rtf_mean_max": 4.49,
    },
    8: {
        "throughput_qps_min": 0.18,
        "tok_per_s_agg_min": 6.0,
        "latency_mean_s_max": 28.0,
        "rtf_mean_max": 6.48,
    },
    16: {
        "throughput_qps_min": 0.24,
        "tok_per_s_agg_min": 2.6,
        "latency_mean_s_max": 52.5,
        "rtf_mean_max": 13.23,
    },
}

WER_SCRIPT = str(
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "eval"
    / "voice_clone_s2pro_wer.py"
)


def _run_benchmark(
    port: int,
    testset: str,
    output_dir: str,
    *,
    concurrency: int,
    max_samples: int | None = None,
    stream: bool = False,
) -> dict:
    benchmark_config = TtsSpeedBenchmarkConfig(
        model=S2PRO_MODEL_PATH,
        port=port,
        testset=testset,
        output_dir=output_dir,
        concurrency=concurrency,
        max_samples=max_samples,
        save_audio=True,
        stream=stream,
    )
    speed_results = asyncio.run(run_tts_speed_benchmark(benchmark_config))
    assert (
        "summary" in speed_results
    ), f"Missing 'summary' key in results. Keys: {list(speed_results.keys())}"
    assert (
        "per_request" in speed_results
    ), f"Missing 'per_request' key in results. Keys: {list(speed_results.keys())}"
    return speed_results


def _no_proxy_env() -> dict[str, str]:
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}


def _run_wer_transcribe(
    meta_path: str,
    output_dir: str,
    *,
    stream: bool = False,
    lang: str = "en",
    device: str = "cuda:0",
) -> dict:
    """Transcribe saved audio and compute WER in CI."""
    cmd = [
        sys.executable,
        WER_SCRIPT,
        "--transcribe-only",
        "--meta",
        meta_path,
        "--output-dir",
        output_dir,
        "--model",
        S2PRO_MODEL_PATH,
        "--lang",
        lang,
        "--device",
        device,
    ]
    if stream:
        cmd.append("--stream")

    result = subprocess.run(
        cmd,
        text=True,
        timeout=WER_TIMEOUT,
        env=_no_proxy_env(),
    )
    assert result.returncode == 0, f"WER transcribe failed (rc={result.returncode})"

    results_path = Path(output_dir) / "wer_results.json"
    assert results_path.exists(), f"WER results file not found: {results_path}"

    with open(results_path) as f:
        wer_results = json.load(f)
    assert (
        "summary" in wer_results
    ), f"Missing 'summary' key in WER results. Keys: {list(wer_results.keys())}"
    assert (
        "per_sample" in wer_results
    ), f"Missing 'per_sample' key in WER results. Keys: {list(wer_results.keys())}"

    summary = wer_results["summary"]
    if summary.get("skipped", 0) > 0:
        print(
            f"\n[WER DIAGNOSTIC] {summary['skipped']}/{summary['total_samples']} "
            "samples skipped."
        )
        for sample in wer_results["per_sample"]:
            if not sample.get("is_success", True):
                print(f"  FAILED sample {sample['id']}: {sample.get('error')}")

    return wer_results


def _dataset_cache_dir() -> Path:
    override_dir = os.environ.get(DATASET_CACHE_ENV)
    if override_dir:
        return Path(override_dir).expanduser()
    raise ValueError(f"{DATASET_CACHE_ENV} is not set")


def _cleanup_generated_audio() -> None:
    for output_dirs in SPEED_OUTPUT_DIRS.values():
        for output_dir in output_dirs.values():
            audio_dir = Path(output_dir) / "audio"
            if audio_dir.exists():
                shutil.rmtree(audio_dir)


def _print_stage(stage: str, mode: str, concurrency: int, details: str = "") -> None:
    message = f"\n[Stage] {stage} benchmark | mode={mode} | concurrency={concurrency}"
    if details:
        message += f" | {details}"
    print(message)


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = (
        _dataset_cache_dir()
        if os.environ.get(DATASET_CACHE_ENV)
        else tmp_path_factory.mktemp("seed_tts_eval") / "data"
    )
    download_dataset(DATASETS["seedtts-50"], str(root), quiet=True)
    return root


@pytest.fixture(scope="module", autouse=True)
def cleanup_generated_audio_fixture():
    yield
    _cleanup_generated_audio()


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the s2-pro server and wait until healthy."""
    port = find_free_port()
    log_file = tmp_path_factory.mktemp("server_logs") / "server.log"
    proc = start_server(S2PRO_MODEL_PATH, S2PRO_CONFIG_PATH, log_file, port)
    proc.port = port
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def wer_input_dirs(server_process: subprocess.Popen) -> dict[str, dict[int, str]]:
    """Reuse saved benchmark audio for WER after freeing the TTS server GPU."""
    stop_server(server_process)
    for mode in ("non_stream", "stream"):
        for concurrency, output_dir in SPEED_OUTPUT_DIRS[mode].items():
            generated_path = Path(output_dir) / "generated.json"
            assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return SPEED_OUTPUT_DIRS


def _assert_summary_metrics(summary: dict) -> None:
    """Verify summary-level sanity invariants that must hold for every run."""
    assert (
        summary["failed_requests"] == 0
    ), f"Expected 0 failed requests, got {summary['failed_requests']}"
    assert (
        summary["audio_duration_mean_s"] > 0
    ), f"Expected positive audio duration, got {summary['audio_duration_mean_s']}"
    assert (
        summary.get("gen_tokens_mean", 0) > 0
    ), f"Expected positive gen_tokens_mean, got {summary.get('gen_tokens_mean', 0)}"
    assert (
        summary.get("prompt_tokens_mean", 0) > 0
    ), f"Expected positive prompt_tokens_mean, got {summary.get('prompt_tokens_mean', 0)}"


def _assert_per_request_fields(per_request: list[dict]) -> None:
    """Verify every request has valid audio, prompt_tokens, and completion_tokens."""
    for req in per_request:
        rid = req["id"]
        assert req["is_success"], f"Request {rid} failed: {req.get('error')}"
        assert (
            req["audio_duration_s"] is not None and req["audio_duration_s"] > 0
        ), f"Request {rid}: audio_duration_s={req['audio_duration_s']}, expected > 0"
        assert (
            req["prompt_tokens"] is not None and req["prompt_tokens"] > 0
        ), f"Request {rid}: prompt_tokens={req['prompt_tokens']}, expected > 0"
        assert (
            req["completion_tokens"] is not None and req["completion_tokens"] > 0
        ), f"Request {rid}: completion_tokens={req['completion_tokens']}, expected > 0"


def _assert_streaming_consistency(
    non_stream_requests: list[dict],
    stream_requests: list[dict],
    *,
    total_completion_token_rtol: float = 0.12,
    median_completion_token_rtol: float = 0.20,
    total_audio_duration_rtol: float = 0.12,
) -> None:
    """Assert stable invariants on the shared request subset."""
    ns_by_id = {r["id"]: r for r in non_stream_requests}
    st_by_id = {r["id"]: r for r in stream_requests}
    common_ids = sorted(set(ns_by_id) & set(st_by_id))
    assert common_ids, "No overlapping request IDs between non-stream and stream runs"
    assert set(st_by_id).issubset(set(ns_by_id)), (
        "Streaming requests must be a subset of non-streaming requests: "
        f"non_stream={sorted(ns_by_id)}, stream={sorted(st_by_id)}"
    )
    assert len(st_by_id) == STREAMING_BENCHMARK_MAX_SAMPLES, (
        f"Expected {STREAMING_BENCHMARK_MAX_SAMPLES} streaming requests, "
        f"got {len(st_by_id)}"
    )

    ns_completion_tokens: list[int] = []
    st_completion_tokens: list[int] = []
    ns_audio_duration_total = 0.0
    st_audio_duration_total = 0.0

    for rid in common_ids:
        ns, st = ns_by_id[rid], st_by_id[rid]
        assert ns["prompt_tokens"] == st["prompt_tokens"], (
            f"Request {rid}: prompt_tokens mismatch — "
            f"non_stream={ns['prompt_tokens']}, stream={st['prompt_tokens']}"
        )
        ns_completion_tokens.append(ns["completion_tokens"])
        st_completion_tokens.append(st["completion_tokens"])
        ns_audio_duration_total += ns["audio_duration_s"]
        st_audio_duration_total += st["audio_duration_s"]

    ns_completion_total = sum(ns_completion_tokens)
    st_completion_total = sum(st_completion_tokens)
    max_completion_total = max(ns_completion_total, st_completion_total)
    assert abs(ns_completion_total - st_completion_total) <= (
        total_completion_token_rtol * max_completion_total
    ), (
        "Total completion_tokens differ too much — "
        f"non_stream={ns_completion_total}, stream={st_completion_total} "
        f"(rtol={total_completion_token_rtol})"
    )

    ns_completion_median = sorted(ns_completion_tokens)[len(ns_completion_tokens) // 2]
    st_completion_median = sorted(st_completion_tokens)[len(st_completion_tokens) // 2]
    max_completion_median = max(ns_completion_median, st_completion_median)
    assert abs(ns_completion_median - st_completion_median) <= (
        median_completion_token_rtol * max_completion_median
    ), (
        "Median completion_tokens differ too much — "
        f"non_stream={ns_completion_median}, stream={st_completion_median} "
        f"(rtol={median_completion_token_rtol})"
    )

    max_audio_duration_total = max(ns_audio_duration_total, st_audio_duration_total)
    assert abs(ns_audio_duration_total - st_audio_duration_total) <= (
        total_audio_duration_rtol * max_audio_duration_total
    ), (
        "Total audio_duration_s differs too much — "
        f"non_stream={ns_audio_duration_total}, stream={st_audio_duration_total} "
        f"(rtol={total_audio_duration_rtol})"
    )


def _assert_wer_results(
    results: dict,
    max_corpus_wer: float,
    max_per_sample_wer: float,
) -> None:
    summary = results["summary"]
    per_sample = results["per_sample"]

    failed_details = [
        f"  sample {s['id']}: {s.get('error')}"
        for s in per_sample
        if not s.get("is_success", True)
    ]
    assert summary["evaluated"] == summary["total_samples"], (
        f"Only {summary['evaluated']}/{summary['total_samples']} samples evaluated, "
        f"{summary['skipped']} skipped.\n"
        f"Per-sample errors:\n" + "\n".join(failed_details)
    )

    assert summary["wer_corpus"] <= max_corpus_wer, (
        f"Corpus WER {summary['wer_corpus']:.4f} ({summary['wer_corpus'] * 100:.2f}%) "
        f"> threshold {max_corpus_wer} ({max_corpus_wer * 100:.0f}%)"
    )

    assert summary["n_above_50_pct_wer"] == 0, (
        f"{summary['n_above_50_pct_wer']} samples have >50% WER — "
        f"expected 0 catastrophic failures"
    )

    for sample in per_sample:
        assert sample[
            "is_success"
        ], f"Sample {sample['id']} failed: {sample.get('error')}"
        if sample["wer"] is not None:
            assert (
                sample["wer"] <= max_per_sample_wer
            ), f"Sample {sample['id']} WER {sample['wer']:.4f} > {max_per_sample_wer}"


def _assert_speed_thresholds(summary: dict, thresholds: dict, concurrency: int) -> None:
    thresholds = thresholds[concurrency]
    assert summary["throughput_qps"] >= thresholds["throughput_qps_min"], (
        f"throughput_qps {summary['throughput_qps']} < "
        f"{thresholds['throughput_qps_min']} at concurrency {concurrency}"
    )
    assert summary["tok_per_s_agg"] >= thresholds["tok_per_s_agg_min"], (
        f"tok_per_s_agg {summary['tok_per_s_agg']} < "
        f"{thresholds['tok_per_s_agg_min']} at concurrency {concurrency}"
    )
    assert summary["latency_mean_s"] <= thresholds["latency_mean_s_max"], (
        f"latency_mean_s {summary['latency_mean_s']} > "
        f"{thresholds['latency_mean_s_max']} at concurrency {concurrency}"
    )
    assert summary["rtf_mean"] <= thresholds["rtf_mean_max"], (
        f"rtf_mean {summary['rtf_mean']} > "
        f"{thresholds['rtf_mean_max']} at concurrency {concurrency}"
    )


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
    selected_s2pro_tts_concurrencies: tuple[int, ...],
) -> None:
    print(
        "\n[S2 Pro benchmark] selected concurrency: "
        f"{selected_s2pro_tts_concurrencies}"
    )
    for concurrency in selected_s2pro_tts_concurrencies:
        _print_stage("TTS speed", "non-streaming", concurrency, "generate WAVs for WER")
        output_dir = str(tmp_path / f"vc_nonstream_c{concurrency}")
        results = _run_benchmark(
            server_process.port,
            str(dataset_dir / "en" / "meta.lst"),
            output_dir,
            concurrency=concurrency,
        )
        summary, per_request = results["summary"], results["per_request"]
        _assert_summary_metrics(summary)
        _assert_per_request_fields(per_request)
        PER_REQUEST_STORE["vc_nonstream"] = per_request
        SPEED_OUTPUT_DIRS["non_stream"][concurrency] = output_dir
        _assert_speed_thresholds(summary, VC_NON_STREAM_THRESHOLDS, concurrency)


@pytest.mark.benchmark
def test_voice_cloning_streaming(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
    selected_s2pro_tts_concurrencies: tuple[int, ...],
) -> None:
    for concurrency in selected_s2pro_tts_concurrencies:
        _print_stage(
            "TTS speed",
            "streaming",
            concurrency,
            f"max_samples={STREAMING_BENCHMARK_MAX_SAMPLES} | generate WAVs for WER",
        )
        output_dir = str(tmp_path / f"vc_stream_c{concurrency}")
        results = _run_benchmark(
            server_process.port,
            str(dataset_dir / "en" / "meta.lst"),
            output_dir,
            concurrency=concurrency,
            max_samples=STREAMING_BENCHMARK_MAX_SAMPLES,
            stream=True,
        )
        summary, per_request = results["summary"], results["per_request"]
        _assert_summary_metrics(summary)
        _assert_per_request_fields(per_request)
        PER_REQUEST_STORE["vc_stream"] = per_request
        SPEED_OUTPUT_DIRS["stream"][concurrency] = output_dir
        _assert_speed_thresholds(summary, VC_STREAM_THRESHOLDS, concurrency)


@pytest.mark.benchmark
def test_voice_cloning_streaming_consistency() -> None:
    ns = PER_REQUEST_STORE.get("vc_nonstream")
    st = PER_REQUEST_STORE.get("vc_stream")
    assert ns is not None, "vc_nonstream results missing"
    assert st is not None, "vc_stream results missing"
    _assert_streaming_consistency(ns, st)


@pytest.mark.benchmark
def test_voice_cloning_wer(
    wer_input_dirs: dict[str, dict[int, str]],
    dataset_dir: Path,
    selected_s2pro_tts_concurrencies: tuple[int, ...],
) -> None:
    for concurrency in selected_s2pro_tts_concurrencies:
        _print_stage(
            "WER",
            "non-streaming",
            concurrency,
            "transcribe speed-stage WAVs",
        )
        results = _run_wer_transcribe(
            str(dataset_dir / "en" / "meta.lst"),
            wer_input_dirs["non_stream"][concurrency],
        )
        _assert_wer_results(results, VC_WER_MAX_CORPUS, VC_WER_MAX_PER_SAMPLE)


@pytest.mark.benchmark
def test_voice_cloning_streaming_wer(
    wer_input_dirs: dict[str, dict[int, str]],
    dataset_dir: Path,
    selected_s2pro_tts_concurrencies: tuple[int, ...],
) -> None:
    for concurrency in selected_s2pro_tts_concurrencies:
        _print_stage(
            "WER",
            "streaming",
            concurrency,
            f"transcribe {STREAMING_BENCHMARK_MAX_SAMPLES} speed-stage WAVs",
        )
        results = _run_wer_transcribe(
            str(dataset_dir / "en" / "meta.lst"),
            wer_input_dirs["stream"][concurrency],
            stream=True,
        )
        _assert_wer_results(
            results, VC_STREAM_WER_MAX_CORPUS, VC_STREAM_WER_MAX_PER_SAMPLE
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
