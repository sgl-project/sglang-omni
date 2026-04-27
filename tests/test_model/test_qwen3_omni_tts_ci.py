# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks and voice-clone WER CI for Qwen3-Omni.

Usage:
    pytest tests/test_model/test_qwen3_omni_tts_ci.py -s -x

"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)
from sglang_omni.utils import find_available_port
from tests.utils import (
    apply_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_partitioned,
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONCURRENCY = 8
MAX_SAMPLES = 50
DATASET_CACHE_ENV = "SGLANG_SEEDTTS50_DIR"

STARTUP_TIMEOUT = 300
WER_TIMEOUT = 600

# Threshold reference: https://github.com/sgl-project/sglang-omni/pull/337#issuecomment-4321089804
VC_WER_BELOW_50_CORPUS_MAX = 0.03
VC_N_ABOVE_50_MAX = 1

_VC_NON_STREAM_P95 = {
    8: {
        "throughput_qps": 0.523,
        "tok_per_s_agg": 1.00,
        "latency_mean_s": 14.709,
        "rtf_mean": 4.2570,
    },
}


# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics (throughput): threshold = P95 x slack_higher
# Lower-is-better metrics (latency, rtf): threshold = P95 x slack_lower

VC_NON_STREAM_THRESHOLDS = apply_slack(_VC_NON_STREAM_P95)


def _run_benchmark(
    port: int,
    meta: str,
    output_dir: str,
) -> dict:
    config = OmniSeedttsBenchmarkConfig(
        model="qwen3-omni",
        port=port,
        meta=meta,
        output_dir=output_dir,
        max_samples=MAX_SAMPLES,
        max_concurrency=CONCURRENCY,
        voice_clone=True,
    )
    speed_results = asyncio.run(run_omni_seedtts_benchmark(config))
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
    lang: str = "en",
    device: str = "cuda:0",
) -> dict:
    """Transcribe saved audio and compute WER in CI.

    note (Chenyang): We invoke the benchmark as ``python -m
    benchmarks.eval.benchmark_omni_seedtts`` rather than via a direct file
    path so the ``benchmarks`` package is discovered via PEP 420 namespace
    lookup from the project root (which PYTHONPATH guarantees below).
    """
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
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

    env = no_proxy_env()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT)
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=WER_TIMEOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
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
    override_dir = os.environ.get(DATASET_CACHE_ENV)
    if override_dir:
        root = Path(override_dir).expanduser()
    else:
        root = tmp_path_factory.mktemp("seed_tts_eval") / "data"
    download_dataset(DATASETS["seedtts-50"], str(root), quiet=True)
    return root


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server and wait until healthy."""
    port = find_available_port()
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
def speed_output_dir(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Run the speed benchmark once and expose the output directory.

    Keeping the benchmark in its own fixture (rather than a test body that
    writes a module-level global) lets the WER stage consume the audio
    without coupling test ordering through mutable globals.
    """
    output_dir = str(tmp_path_factory.mktemp("vc_nonstream"))
    results = _run_benchmark(
        server_process.port,
        str(dataset_dir / "en" / "meta.lst"),
        output_dir,
    )
    summary, per_request = results["summary"], results["per_request"]
    assert_summary_metrics(summary)
    assert_per_request_fields(per_request)
    assert_speed_thresholds(summary, VC_NON_STREAM_THRESHOLDS, CONCURRENCY)
    return output_dir


@pytest.fixture(scope="module")
def wer_audio_dir(
    server_process: subprocess.Popen,
    speed_output_dir: str,
) -> str:
    """Reuse speed-benchmark audio for WER after freeing the TTS server GPU.

    ``stop_server`` is called here (in addition to the ``server_process``
    teardown) so Whisper-large-v3 can claim the GPU memory before the
    transcribe subprocess runs. ``stop_server`` is idempotent so the later
    teardown call is a safe no-op.
    """
    stop_server(server_process)
    generated_path = Path(speed_output_dir) / "generated.json"
    assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return speed_output_dir


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(speed_output_dir: str) -> None:
    """Smoke check: the speed-benchmark fixture asserts metrics/thresholds."""
    assert Path(speed_output_dir).is_dir()


@pytest.mark.benchmark
def test_voice_cloning_wer(
    wer_audio_dir: str,
    dataset_dir: Path,
) -> None:
    results = _run_wer_transcribe(
        str(dataset_dir / "en" / "meta.lst"),
        wer_audio_dir,
    )
    assert_wer_partitioned(
        results,
        max_wer_below_50_corpus=VC_WER_BELOW_50_CORPUS_MAX,
        max_n_above_50=VC_N_ABOVE_50_MAX,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
