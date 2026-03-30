# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks and voice-clone WER thresholds CI for S2-Pro as a representative of TTS models.

Usage:
    pytest tests/test_model/test_s2pro_tts_ci.py -s -x

Author:
    chenyang zhao https://github.com/zhaochenyang20
    Jingwen Guo https://github.com/JingwenGu0829
    Yuan Luo https://github.com/yuan-luo
    Yitong Guan https://github.com/minleminzui
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

MODEL_PATH = "fishaudio/s2-pro"
CONFIG_PATH = "examples/configs/s2pro_tts.yaml"
MAX_SAMPLES = 10

STARTUP_TIMEOUT = 600
BENCHMARK_TIMEOUT = 600
WER_TIMEOUT = 600

# Thresholds reference: https://github.com/sgl-project/sglang-omni/issues/193
# Note (chenyang): the RTF thresholds also includes the reference audio
# processing time. The Plain text RTF is far less than 1.0.

VC_NON_STREAM_MIN_TOK_PER_S = 80
VC_NON_STREAM_MAX_RTF = 2.85
VC_STREAM_MAX_LATENCY_S = 12.5
VC_STREAM_MIN_THROUGHPUT_QPS = 0.08
PLAIN_NON_STREAM_MIN_TOK_PER_S = 80
PLAIN_NON_STREAM_MAX_RTF = 0.35
PLAIN_STREAM_MAX_LATENCY_S = 4.0
PLAIN_STREAM_MIN_THROUGHPUT_QPS = 0.25


# TODO (Chenyang): Current WER thresholds is computed over mini set, which can not
# capture the accuracy regression fixed by https://github.com/sgl-project/sglang-omni/pull/217
# We shall have more strict rules that can let #217 pass but let commit 8deddef fail.

VC_WER_MAX_CORPUS = 0.0
VC_WER_MAX_PER_SAMPLE = 0.0
VC_STREAM_WER_MAX_CORPUS = 0.0
VC_STREAM_WER_MAX_PER_SAMPLE = 0.0


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
    extra_args: list[str] | None = None,
) -> dict:
    cmd = [
        sys.executable,
        str(
            Path(__file__).resolve().parents[2]
            / "benchmarks"
            / "eval"
            / "benchmark_tts_speed.py"
        ),
        "--model",
        MODEL_PATH,
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

    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    env = {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}

    proc_result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=BENCHMARK_TIMEOUT,
        env=env,
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


def _no_proxy_env() -> dict[str, str]:
    proxy_keys = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    return {k: v for k, v in os.environ.items() if k.lower() not in proxy_keys}


def _run_wer_generate(
    port: int,
    meta_path: str,
    output_dir: str,
    stream: bool = False,
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
        MODEL_PATH,
        "--port",
        str(port),
        "--max-samples",
        str(MAX_SAMPLES),
    ]
    if stream:
        cmd.append("--stream")

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
        MODEL_PATH,
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
    """Start the s2-pro server and wait until healthy."""
    port = find_free_port()
    log_dir = tmp_path_factory.mktemp("server_logs")
    log_file = log_dir / "server.log"
    with open(log_file, "w") as log_handle:
        cmd = [
            sys.executable,
            "-m",
            "sglang_omni.cli.cli",
            "serve",
            "--model-path",
            MODEL_PATH,
            "--config",
            CONFIG_PATH,
            "--port",
            str(port),
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
                f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n{server_log}"
            )
        except RuntimeError as exc:
            pytest.fail(str(exc))

        yield proc

        _kill_server(proc)


@pytest.fixture(scope="module")
def wer_audio_dirs(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Generate WER audio in CI, then kill server in order to prevent CI OOM."""
    tmp = tmp_path_factory.mktemp("wer")
    meta = str(dataset_dir / "en" / "meta.lst")

    _run_wer_generate(server_process.port, meta, str(tmp / "non_stream"))
    _run_wer_generate(server_process.port, meta, str(tmp / "stream"), stream=True)

    _kill_server(server_process)

    return {"non_stream": str(tmp / "non_stream"), "stream": str(tmp / "stream")}


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
    completion_token_rtol: float = 0.10,
    audio_duration_rtol: float = 0.12,
) -> None:
    """Assert per-request metrics are close between streaming and non-streaming."""
    ns_by_id = {r["id"]: r for r in non_stream_requests}
    st_by_id = {r["id"]: r for r in stream_requests}
    assert set(ns_by_id) == set(st_by_id), (
        f"Request ID mismatch: "
        f"non_stream={sorted(ns_by_id)}, stream={sorted(st_by_id)}"
    )
    for rid in sorted(ns_by_id):
        ns, st = ns_by_id[rid], st_by_id[rid]

        ns_ct, st_ct = ns["completion_tokens"], st["completion_tokens"]
        max_ct = max(ns_ct, st_ct)
        assert abs(ns_ct - st_ct) <= completion_token_rtol * max_ct, (
            f"Request {rid}: completion_tokens differ too much — "
            f"non_stream={ns_ct}, stream={st_ct} "
            f"(rtol={completion_token_rtol})"
        )

        assert ns["prompt_tokens"] == st["prompt_tokens"], (
            f"Request {rid}: prompt_tokens mismatch — "
            f"non_stream={ns['prompt_tokens']}, stream={st['prompt_tokens']}"
        )

        ns_ad, st_ad = ns["audio_duration_s"], st["audio_duration_s"]
        max_ad = max(ns_ad, st_ad)
        assert abs(ns_ad - st_ad) <= audio_duration_rtol * max_ad, (
            f"Request {rid}: audio_duration_s differ too much — "
            f"non_stream={ns_ad}, stream={st_ad} "
            f"(rtol={audio_duration_rtol})"
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
            assert sample["wer"] <= max_per_sample_wer, (
                f"Sample {sample['id']} WER {sample['wer']:.4f} "
                f"> {max_per_sample_wer}"
            )


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
        summary["tok_per_s_agg"] >= VC_NON_STREAM_MIN_TOK_PER_S
    ), f"tok_per_s_agg {summary['tok_per_s_agg']} < {VC_NON_STREAM_MIN_TOK_PER_S}"
    assert (
        summary["rtf_mean"] <= VC_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {VC_NON_STREAM_MAX_RTF}"


@pytest.mark.benchmark
def test_voice_cloning_streaming(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    results = _run_benchmark(
        server_process.port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "vc_stream"),
        ["--stream"],
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    PER_REQUEST_STORE["vc_stream"] = per_request
    assert (
        summary["latency_mean_s"] <= VC_STREAM_MAX_LATENCY_S
    ), f"latency_mean_s {summary['latency_mean_s']} > {VC_STREAM_MAX_LATENCY_S}"
    assert (
        summary["throughput_qps"] >= VC_STREAM_MIN_THROUGHPUT_QPS
    ), f"throughput_qps {summary['throughput_qps']} < {VC_STREAM_MIN_THROUGHPUT_QPS}"


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
        summary["tok_per_s_agg"] >= PLAIN_NON_STREAM_MIN_TOK_PER_S
    ), f"tok_per_s_agg {summary['tok_per_s_agg']} < {PLAIN_NON_STREAM_MIN_TOK_PER_S}"
    assert (
        summary["rtf_mean"] <= PLAIN_NON_STREAM_MAX_RTF
    ), f"rtf_mean {summary['rtf_mean']} > {PLAIN_NON_STREAM_MAX_RTF}"


@pytest.mark.benchmark
def test_plain_tts_streaming(
    server_process: subprocess.Popen,
    dataset_dir: Path,
    tmp_path: Path,
) -> None:
    results = _run_benchmark(
        server_process.port,
        str(dataset_dir / "en" / "meta.lst"),
        str(tmp_path / "plain_stream"),
        ["--no-ref-audio", "--stream"],
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_summary_metrics(summary)
    _assert_per_request_fields(per_request)
    PER_REQUEST_STORE["plain_stream"] = per_request
    assert (
        summary["latency_mean_s"] <= PLAIN_STREAM_MAX_LATENCY_S
    ), f"latency_mean_s {summary['latency_mean_s']} > {PLAIN_STREAM_MAX_LATENCY_S}"
    assert (
        summary["throughput_qps"] >= PLAIN_STREAM_MIN_THROUGHPUT_QPS
    ), f"throughput_qps {summary['throughput_qps']} < {PLAIN_STREAM_MIN_THROUGHPUT_QPS}"


@pytest.mark.benchmark
def test_voice_cloning_streaming_consistency() -> None:
    ns = PER_REQUEST_STORE.get("vc_nonstream")
    st = PER_REQUEST_STORE.get("vc_stream")
    assert ns is not None, "vc_nonstream results missing"
    assert st is not None, "vc_stream results missing"
    _assert_streaming_consistency(ns, st)


@pytest.mark.benchmark
def test_plain_tts_streaming_consistency() -> None:
    ns = PER_REQUEST_STORE.get("plain_nonstream")
    st = PER_REQUEST_STORE.get("plain_stream")
    assert ns is not None, "plain_nonstream results missing"
    assert st is not None, "plain_stream results missing"
    _assert_streaming_consistency(ns, st)


@pytest.mark.benchmark
def test_voice_cloning_wer(
    wer_audio_dirs: dict[str, str],
    dataset_dir: Path,
) -> None:
    results = _run_wer_transcribe(
        str(dataset_dir / "en" / "meta.lst"),
        wer_audio_dirs["non_stream"],
    )
    _assert_wer_results(results, VC_WER_MAX_CORPUS, VC_WER_MAX_PER_SAMPLE)


@pytest.mark.benchmark
def test_voice_cloning_streaming_wer(
    wer_audio_dirs: dict[str, str],
    dataset_dir: Path,
) -> None:
    results = _run_wer_transcribe(
        str(dataset_dir / "en" / "meta.lst"),
        wer_audio_dirs["stream"],
    )
    _assert_wer_results(results, VC_STREAM_WER_MAX_CORPUS, VC_STREAM_WER_MAX_PER_SAMPLE)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
