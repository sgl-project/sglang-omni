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

The benchmark supports one selected concurrency per test run. Use --concurrency 8
in CI, run without the flag to use concurrency 1, or pass --concurrency all
to sweep all supported concurrency values locally.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
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

# Note (Chenyang): The streaming mode evaluation is only run at first 16.

STREAMING_BENCHMARK_MAX_SAMPLES = 16

# Thresholds reference: https://github.com/sgl-project/sglang-omni/pull/242
# Note (chenyang): the RTF thresholds also includes the reference audio
# processing time.

# Note (Ratish, Chenyang): We evalute the performance of S2-Pro CI on our H20
# CI machines and compute the thresholds based on the results.

# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics (throughput, tok/s): threshold = P95 × slack_higher
# Lower-is-better metrics (latency, rtf): threshold = P95 × slack_lower

THRESHOLD_SLACK_HIGHER = 0.75
THRESHOLD_SLACK_LOWER = 1.25

VC_WER_MAX_CORPUS = 0.025
VC_WER_MAX_PER_SAMPLE = 0.6
VC_STREAM_WER_MAX_CORPUS = 0.025
VC_STREAM_WER_MAX_PER_SAMPLE = 0.6

# Note (Chenyang): Only thresholds for concurrency 8 are dedicatedly tuned, others
# may not pass the CI.

_VC_NON_STREAM_P95 = {
    1: {
        "throughput_qps": 0.13,
        "tok_per_s_agg": 82.5,
        "latency_mean_s": 7.6,
        "rtf_mean": 2.03,
    },
    2: {
        "throughput_qps": 0.25,
        "tok_per_s_agg": 78.4,
        "latency_mean_s": 7.9,
        "rtf_mean": 2.10,
    },
    4: {
        "throughput_qps": 0.47,
        "tok_per_s_agg": 75.3,
        "latency_mean_s": 8.3,
        "rtf_mean": 2.21,
    },
    8: {
        "throughput_qps": 0.80,
        "tok_per_s_agg": 67.7,
        "latency_mean_s": 9.1,
        "rtf_mean": 2.43,
    },
    16: {
        "throughput_qps": 1.17,
        "tok_per_s_agg": 60.7,
        "latency_mean_s": 11.2,
        "rtf_mean": 3.01,
    },
}

_VC_STREAM_P95 = {
    1: {
        "throughput_qps": 0.09,
        "tok_per_s_agg": 21.0,
        "latency_mean_s": 10.8,
        "rtf_mean": 2.60,
    },
    2: {
        "throughput_qps": 0.15,
        "tok_per_s_agg": 14.7,
        "latency_mean_s": 13.3,
        "rtf_mean": 3.20,
    },
    4: {
        "throughput_qps": 0.23,
        "tok_per_s_agg": 13.7,
        "latency_mean_s": 15.7,
        "rtf_mean": 4.08,
    },
    8: {
        "throughput_qps": 0.32,
        "tok_per_s_agg": 9.3,
        "latency_mean_s": 22.7,
        "rtf_mean": 5.89,
    },
    16: {
        "throughput_qps": 0.27,
        "tok_per_s_agg": 2.9,
        "latency_mean_s": 47.7,
        "rtf_mean": 12.02,
    },
}


def _apply_slack(
    p95: dict[int, dict[str, float]],
    slack_higher: float = THRESHOLD_SLACK_HIGHER,
    slack_lower: float = THRESHOLD_SLACK_LOWER,
) -> dict[int, dict[str, float]]:
    """Derive CI thresholds from P95 references with uniform slack."""
    result: dict[int, dict[str, float]] = {}
    for conc, m in p95.items():
        result[conc] = {
            "throughput_qps_min": round(m["throughput_qps"] * slack_higher, 2),
            "tok_per_s_agg_min": round(m["tok_per_s_agg"] * slack_higher, 1),
            "latency_mean_s_max": round(m["latency_mean_s"] * slack_lower, 1),
            "rtf_mean_max": round(m["rtf_mean"] * slack_lower, 2),
        }
    return result


VC_NON_STREAM_THRESHOLDS = _apply_slack(_VC_NON_STREAM_P95)
VC_STREAM_THRESHOLDS = _apply_slack(_VC_STREAM_P95)

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


def _print_stage(stage: str, mode: str, concurrency: int, details: str = "") -> None:
    message = f"\n[Stage] {stage} benchmark | mode={mode} | concurrency={concurrency}"
    if details:
        message += f" | {details}"
    print(message)


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    override_dir = os.environ.get(DATASET_CACHE_ENV)
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        root = tmp_path_factory.mktemp("seed_tts_eval") / "data"
    download_dataset(DATASETS["seedtts-50"], str(root), quiet=True)
    return root


@pytest.fixture(scope="module", autouse=True)
def cleanup_generated_audio_fixture():
    yield
    for output_dirs in SPEED_OUTPUT_DIRS.values():
        for output_dir in output_dirs.values():
            audio_dir = Path(output_dir) / "audio"
            if audio_dir.exists():
                shutil.rmtree(audio_dir)


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

    ns_completion_median = statistics.median(ns_completion_tokens)
    st_completion_median = statistics.median(st_completion_tokens)
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
    level_thresholds = thresholds[concurrency]
    assert summary["throughput_qps"] >= level_thresholds["throughput_qps_min"], (
        f"throughput_qps {summary['throughput_qps']} < "
        f"{level_thresholds['throughput_qps_min']} at concurrency {concurrency}"
    )
    assert summary["tok_per_s_agg"] >= level_thresholds["tok_per_s_agg_min"], (
        f"tok_per_s_agg {summary['tok_per_s_agg']} < "
        f"{level_thresholds['tok_per_s_agg_min']} at concurrency {concurrency}"
    )
    assert summary["latency_mean_s"] <= level_thresholds["latency_mean_s_max"], (
        f"latency_mean_s {summary['latency_mean_s']} > "
        f"{level_thresholds['latency_mean_s_max']} at concurrency {concurrency}"
    )
    assert summary["rtf_mean"] <= level_thresholds["rtf_mean_max"], (
        f"rtf_mean {summary['rtf_mean']} > "
        f"{level_thresholds['rtf_mean_max']} at concurrency {concurrency}"
    )


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
    selected_s2pro_tts_concurrencies: tuple[int, ...],
) -> None:
    print(
        f"\n[S2 Pro benchmark] selected concurrency: {selected_s2pro_tts_concurrencies}"
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
        PER_REQUEST_STORE[f"vc_nonstream_c{concurrency}"] = per_request
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
        PER_REQUEST_STORE[f"vc_stream_c{concurrency}"] = per_request
        SPEED_OUTPUT_DIRS["stream"][concurrency] = output_dir
        _assert_speed_thresholds(summary, VC_STREAM_THRESHOLDS, concurrency)


@pytest.mark.benchmark
def test_voice_cloning_streaming_consistency(
    selected_s2pro_tts_concurrencies: tuple[int, ...],
) -> None:
    for concurrency in selected_s2pro_tts_concurrencies:
        ns = PER_REQUEST_STORE.get(f"vc_nonstream_c{concurrency}")
        st = PER_REQUEST_STORE.get(f"vc_stream_c{concurrency}")
        assert ns is not None, f"vc_nonstream_c{concurrency} results missing"
        assert st is not None, f"vc_stream_c{concurrency} results missing"
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
