# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks and voice-clone WER CI for Qwen3-Omni.

Usage:
    pytest tests/test_model/test_qwen3_omni_tts_ci.py -s -x

TODO (Jingwen, Chenyang): Support streaming for audio output
and concurrency of vocoder.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_omni_tts_speed import (
    OmniTtsSpeedBenchmarkConfig,
    run_omni_tts_speed_benchmark,
)
from tests.utils import (
    apply_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_results,
    find_free_port,
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# TODO(Chenyang): Currently we only run concurrency=1 and a small dataset
# (seedtts-mini, 10 samples). Support higher concurrency and larger datasets
# once the Qwen3-Omni pipeline is optimized for concurrent requests.

CONCURRENCY = 1
MAX_SAMPLES = 10

STARTUP_TIMEOUT = 900
WER_TIMEOUT = 600

# Note (Chenyang): P95 values measured on H20 CI machines with concurrency=1,
# seedtts-mini dataset (5 samples). Update these when hardware or model changes.

_VC_NON_STREAM_P95 = {
    1: {
        "throughput_qps": 0.17,
        "tok_per_s_agg": 2.3,
        "latency_mean_s": 6.0,
        "rtf_mean": 2.0,
    },
}


# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics (throughput): threshold = P95 x slack_higher
# Lower-is-better metrics (latency, rtf): threshold = P95 x slack_lower

THRESHOLD_SLACK_HIGHER = 0.75
THRESHOLD_SLACK_LOWER = 1.25

VC_NON_STREAM_THRESHOLDS = apply_slack(
    _VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)

VC_WER_MAX_CORPUS = 0.06
VC_WER_MAX_PER_SAMPLE = 0.30

WER_SCRIPT = str(
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "eval"
    / "voice_clone_omni_wer.py"
)


def _run_benchmark(
    port: int,
    testset: str,
    output_dir: str,
) -> dict:
    config = OmniTtsSpeedBenchmarkConfig(
        model="qwen3-omni",
        port=port,
        testset=testset,
        output_dir=output_dir,
        max_samples=MAX_SAMPLES,
    )
    speed_results = asyncio.run(run_omni_tts_speed_benchmark(config))
    assert (
        "summary" in speed_results
    ), f"Missing 'summary' key in results. Keys: {list(speed_results.keys())}"
    assert (
        "per_request" in speed_results
    ), f"Missing 'per_request' key in results. Keys: {list(speed_results.keys())}"
    return speed_results


def _run_wer_generate(
    port: int,
    meta_path: str,
    output_dir: str,
    voice_clone: bool = True,
) -> None:
    """Generate TTS audio in CI."""
    cmd = [
        sys.executable,
        WER_SCRIPT,
        "--generate-only",
        "--meta",
        meta_path,
        "--output-dir",
        output_dir,
        "--model",
        "qwen3-omni",
        "--port",
        str(port),
        "--max-samples",
        str(MAX_SAMPLES),
    ]
    if voice_clone:
        cmd.append("--voice-clone")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=WER_TIMEOUT,
        env=no_proxy_env(),
    )
    assert result.returncode == 0, (
        f"WER generate failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _run_wer_transcribe(
    meta_path: str,
    output_dir: str,
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
        "qwen3-omni",
        "--lang",
        lang,
        "--device",
        device,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=WER_TIMEOUT,
        env=no_proxy_env(),
    )
    assert result.returncode == 0, (
        f"WER transcribe failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

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
            f"samples skipped.\nSubprocess stderr:\n{result.stderr}"
        )
        for sample in wer_results["per_sample"]:
            if not sample.get("is_success", True):
                print(f"  FAILED sample {sample['id']}: {sample.get('error')}")

    return wer_results


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("seed_tts_eval") / "data"
    download_dataset(DATASETS["seedtts-mini"], str(root))
    return root


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server and wait until healthy."""
    port = find_free_port()
    log_file = tmp_path_factory.mktemp("server_logs") / "server.log"
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_speech_server.py",
        "--model-path",
        MODEL_PATH,
        "--gpu-thinker",
        "0",
        "--gpu-talker",
        "1",
        "--gpu-code-predictor",
        "1",
        "--gpu-code2wav",
        "1",
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    proc.port = port
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def wer_audio_dir(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Generate WER audio in CI, then kill server to free GPU for Whisper."""
    tmp = tmp_path_factory.mktemp("wer")
    meta = str(dataset_dir / "en" / "meta.lst")

    _run_wer_generate(
        server_process.port,
        meta,
        str(tmp / "vc"),
        voice_clone=True,
    )

    stop_server(server_process)

    return str(tmp / "vc")


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    results = _run_benchmark(
        server_process.port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "vc_nonstream"),
    )
    summary, per_request = results["summary"], results["per_request"]
    assert_summary_metrics(summary)
    assert_per_request_fields(per_request)
    assert_speed_thresholds(summary, VC_NON_STREAM_THRESHOLDS, CONCURRENCY)


@pytest.mark.benchmark
def test_voice_cloning_wer(
    wer_audio_dir: str,
    dataset_dir: Path,
) -> None:
    results = _run_wer_transcribe(
        str(dataset_dir / "en" / "meta.lst"),
        wer_audio_dir,
    )
    assert_wer_results(results, VC_WER_MAX_CORPUS, VC_WER_MAX_PER_SAMPLE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
