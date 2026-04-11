# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks and voice-clone WER thresholds CI for S2-Pro as a representative of TTS models.

Usage:
    pytest tests/test_model/test_s2pro_tts_ci.py -s -x
    pytest tests/test_model/test_s2pro_tts_ci.py -s -x --concurrency 8
    pytest tests/test_model/test_s2pro_tts_ci.py -s -x --concurrency all

Author:
    Chenyang Zhao https://github.com/zhaochenyang20
    Raitsh P https://github.com/Ratish1
    Jingwen Guo https://github.com/JingwenGu0829
    Yuan Luo https://github.com/yuan-luo
    Yitong Guan https://github.com/minleminzui
    Xuesong Ye https://github.com/yxs

The benchmark supports one selected concurrency per test run. Use --concurrency 8
in CI, run without the flag to use concurrency 1, or pass --concurrency all
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
from tests.utils import (
    apply_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_streaming_consistency,
    assert_summary_metrics,
    assert_wer_results,
    find_free_port,
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)

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

VC_WER_MAX_CORPUS = 0.015
VC_WER_MAX_PER_SAMPLE = 0.5
VC_STREAM_WER_MAX_CORPUS = 0.015
VC_STREAM_WER_MAX_PER_SAMPLE = 0.5

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
        "throughput_qps": 0.31,
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


VC_NON_STREAM_THRESHOLDS = apply_slack(
    _VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
VC_STREAM_THRESHOLDS = apply_slack(
    _VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)

WER_SCRIPT = str(
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "eval"
    / "voice_clone_tts_wer.py"
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
        env=no_proxy_env(),
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
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        S2PRO_MODEL_PATH,
        "--config",
        S2PRO_CONFIG_PATH,
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port)
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
        assert_summary_metrics(summary)
        assert_per_request_fields(per_request)
        PER_REQUEST_STORE[f"vc_nonstream_c{concurrency}"] = per_request
        SPEED_OUTPUT_DIRS["non_stream"][concurrency] = output_dir
        assert_speed_thresholds(summary, VC_NON_STREAM_THRESHOLDS, concurrency)


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
        assert_summary_metrics(summary)
        assert_per_request_fields(per_request)
        PER_REQUEST_STORE[f"vc_stream_c{concurrency}"] = per_request
        SPEED_OUTPUT_DIRS["stream"][concurrency] = output_dir
        assert_speed_thresholds(summary, VC_STREAM_THRESHOLDS, concurrency)


@pytest.mark.benchmark
def test_voice_cloning_streaming_consistency(
    selected_s2pro_tts_concurrencies: tuple[int, ...],
) -> None:
    for concurrency in selected_s2pro_tts_concurrencies:
        ns = PER_REQUEST_STORE.get(f"vc_nonstream_c{concurrency}")
        st = PER_REQUEST_STORE.get(f"vc_stream_c{concurrency}")
        assert ns is not None, f"vc_nonstream_c{concurrency} results missing"
        assert st is not None, f"vc_stream_c{concurrency} results missing"
        assert_streaming_consistency(
            ns, st, expected_stream_count=STREAMING_BENCHMARK_MAX_SAMPLES
        )


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
        assert_wer_results(results, VC_WER_MAX_CORPUS, VC_WER_MAX_PER_SAMPLE)


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
        assert_wer_results(
            results, VC_STREAM_WER_MAX_CORPUS, VC_STREAM_WER_MAX_PER_SAMPLE
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
