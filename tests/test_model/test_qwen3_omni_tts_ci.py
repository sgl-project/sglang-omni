# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks, concurrency validation, and voice-clone WER CI for Qwen3-Omni.

Usage:
    pytest tests/test_model/test_qwen3_omni_tts_ci.py -s -x

Author:
    Jingwen Guo https://github.com/JingwenGu0829
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.prepare import DATASETS, download_dataset
from tests.test_model.helpers import disable_proxy
from tests.utils import find_free_port

PER_REQUEST_STORE: dict[str, list[dict]] = {}

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MAX_SAMPLES = 10

STARTUP_TIMEOUT = 900
BENCHMARK_TIMEOUT = 600
WER_TIMEOUT = 600

# ---------------------------------------------------------------------------
# Speed thresholds
# ---------------------------------------------------------------------------
# Qwen3-Omni uses /v1/chat/completions (non-streaming only for now).
# Baseline measured on H200: VC RTF ~1.78, Plain RTF ~1.64.

VC_NON_STREAM_MAX_RTF = 3.5
PLAIN_NON_STREAM_MAX_RTF = 3.0

# ---------------------------------------------------------------------------
# WER thresholds
# ---------------------------------------------------------------------------
# Baseline measured on H200 with seedtts-mini (10 samples):
# corpus WER = 3.77%, worst per-sample = 25%.

VC_WER_MAX_CORPUS = 0.06
VC_WER_MAX_PER_SAMPLE = 0.30

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
CONCURRENCY_LEVEL = 4

SPEED_SCRIPT = str(
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "eval"
    / "benchmark_omni_tts_speed.py"
)

WER_SCRIPT = str(
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "eval"
    / "voice_clone_qwen3_omni_wer.py"
)


def _no_proxy_env() -> dict[str, str]:
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}


def _run_benchmark(
    port: int,
    testset: str,
    output_dir: str,
    extra_args: list[str] | None = None,
) -> dict:
    cmd = [
        sys.executable,
        SPEED_SCRIPT,
        "--model",
        "qwen3-omni",
        "--port",
        str(port),
        "--testset",
        testset,
        "--max-samples",
        str(MAX_SAMPLES),
        "--output-dir",
        output_dir,
    ]
    if extra_args:
        cmd.extend(extra_args)

    proc_result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BENCHMARK_TIMEOUT,
        env=_no_proxy_env(),
    )
    assert proc_result.returncode == 0, (
        f"Benchmark failed (rc={proc_result.returncode}).\n"
        f"stdout:\n{proc_result.stdout}\nstderr:\n{proc_result.stderr}"
    )

    results_path = Path(output_dir) / "speed_results.json"
    assert results_path.exists(), f"Results file not found: {results_path}"

    with open(results_path) as f:
        speed_results = json.load(f)
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
        env=_no_proxy_env(),
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
        env=_no_proxy_env(),
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


def _kill_server(proc: subprocess.Popen) -> None:
    """Kill the server process group, tolerating already-dead processes."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, ChildProcessError):
        pass
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except (ProcessLookupError, ChildProcessError):
            pass


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("seed_tts_eval") / "data"
    download_dataset(DATASETS["seedtts-mini"], str(root))
    return root


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the Qwen3-Omni speech server and wait until healthy."""
    port = find_free_port()
    log_dir = tmp_path_factory.mktemp("server_logs")
    log_file = log_dir / "server.log"
    with open(log_file, "w") as log_handle:
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

        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        proc.port = port

        api_base = f"http://localhost:{port}"
        try:
            with disable_proxy():
                wait_for_service(
                    api_base,
                    timeout=STARTUP_TIMEOUT,
                    server_process=proc,
                    server_log_file=log_file,
                    health_body_contains="healthy",
                )
        except TimeoutError:
            _kill_server(proc)
            server_log = log_file.read_text()
            pytest.fail(
                f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n"
                f"{server_log}"
            )
        except RuntimeError as exc:
            pytest.fail(str(exc))

        yield proc

        _kill_server(proc)


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

    _kill_server(server_process)

    return str(tmp / "vc")


def _assert_summary_metrics(summary: dict) -> None:
    """Verify summary-level sanity invariants that must hold for every run."""
    assert (
        summary["failed_requests"] == 0
    ), f"Expected 0 failed requests, got {summary['failed_requests']}"
    assert (
        summary["audio_duration_mean_s"] > 0
    ), f"Expected positive audio duration, got {summary['audio_duration_mean_s']}"


def _assert_per_request_fields(per_request: list[dict]) -> None:
    """Verify every request has valid audio."""
    for req in per_request:
        rid = req["id"]
        assert req["is_success"], f"Request {rid} failed: {req.get('error')}"
        assert (
            req["audio_duration_s"] is not None and req["audio_duration_s"] > 0
        ), f"Request {rid}: audio_duration_s={req['audio_duration_s']}, expected > 0"


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
            assert sample["wer"] <= max_per_sample_wer, (
                f"Sample {sample['id']} WER {sample['wer']:.4f} "
                f"> {max_per_sample_wer}"
            )


# ---------------------------------------------------------------------------
# Speed tests
# ---------------------------------------------------------------------------


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
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    PER_REQUEST_STORE["vc_nonstream"] = per_request
    assert (
        summary["rtf_mean"] <= VC_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {VC_NON_STREAM_MAX_RTF}"


@pytest.mark.benchmark
def test_plain_tts_non_streaming(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    results = _run_benchmark(
        server_process.port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "plain_nonstream"),
        ["--no-ref-audio"],
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    PER_REQUEST_STORE["plain_nonstream"] = per_request
    assert (
        summary["rtf_mean"] <= PLAIN_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {PLAIN_NON_STREAM_MAX_RTF}"


# ---------------------------------------------------------------------------
# Concurrency tests — validates the fix for issue #229
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_concurrency_no_crash(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    """Verify concurrent requests complete without CUDA crash (issue #229)."""
    results = _run_benchmark(
        server_process.port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "vc_concurrent"),
        ["--max-concurrency", str(CONCURRENCY_LEVEL)],
    )
    summary, per_request = results["summary"], results["per_request"]
    assert summary["failed_requests"] == 0, (
        f"Concurrency={CONCURRENCY_LEVEL}: {summary['failed_requests']} requests failed. "
        f"See https://github.com/sgl-project/sglang-omni/issues/229"
    )
    _assert_per_request_fields(per_request)


# ---------------------------------------------------------------------------
# WER tests
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_voice_cloning_wer(
    wer_audio_dir: str,
    dataset_dir: Path,
) -> None:
    results = _run_wer_transcribe(
        str(dataset_dir / "en" / "meta.lst"),
        wer_audio_dir,
    )
    _assert_wer_results(results, VC_WER_MAX_CORPUS, VC_WER_MAX_PER_SAMPLE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
