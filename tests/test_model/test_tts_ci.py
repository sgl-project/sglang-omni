# SPDX-License-Identifier: Apache-2.0
"""TTS SeedTTS CI benchmark.

The workflow runs this file in two GPU stages:

1. full SeedTTS EN non-stream generation + WER
2. full SeedTTS EN SSE streaming generation + WER

Each generation stage starts two TTS workers behind the managed router, matching
the online-serving topology used by the other Omni CI stages. WER runs after the
router is stopped so the ASR transcription can reuse the GPU memory.
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
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
)
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    assert_workers_served_requests_since,
    launch_managed_router,
    print_router_diagnostics,
    router_get_json,
)
from tests.utils import (
    MetricCheckCollector,
    apply_slack,
    apply_wer_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_partitioned,
    no_proxy_env,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TTS_MODEL_PATH = os.environ.get(
    "TTS_MODEL_PATH", "boson-sglang/higgs-audio-v3-tts-4b-base"
)
TTS_CONFIG_PATH = os.environ.get(
    "TTS_CONFIG_PATH", "examples/configs/higgs_tts.yaml"
)
TTS_REF_FORMAT = "references"
TTS_STAGE_OUTPUT_ROOT_ENV = "TTS_STAGE_OUTPUT_ROOT"
TTS_CONCURRENCY = 16
TTS_SEED = 601

STARTUP_TIMEOUT = 300
BENCHMARK_TIMEOUT = 900
WER_TIMEOUT = 900

SPEED_OUTPUT_DIRS: dict[str, dict[int, str]] = {"non_stream": {}, "stream": {}}

_TTS_NON_STREAM_REFERENCE = {
    TTS_CONCURRENCY: {
        "throughput_qps": 9.104,
        "output_tok_per_req_s": 112.9,
        "latency_mean_s": 1.749,
        "rtf_mean": 0.425,
    }
}
_TTS_STREAM_REFERENCE = {
    TTS_CONCURRENCY: {
        "throughput_qps": 7.0,
        "output_tok_per_req_s": 80.0,
        "latency_mean_s": 2.5,
        "rtf_mean": 0.8,
    }
}

TTS_THRESHOLD_SLACK_HIGHER = 0.35
TTS_THRESHOLD_SLACK_LOWER = 3.0
TTS_NON_STREAM_THRESHOLDS = apply_slack(
    _TTS_NON_STREAM_REFERENCE,
    TTS_THRESHOLD_SLACK_HIGHER,
    TTS_THRESHOLD_SLACK_LOWER,
)
TTS_STREAM_THRESHOLDS = apply_slack(
    _TTS_STREAM_REFERENCE,
    TTS_THRESHOLD_SLACK_HIGHER,
    TTS_THRESHOLD_SLACK_LOWER,
)

TTS_WER_BELOW_50_CORPUS_MAX = 0.0136
TTS_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(TTS_WER_BELOW_50_CORPUS_MAX)
TTS_WER_N_ABOVE_50_MAX = 3


@pytest.fixture(scope="module")
def dataset_repo() -> str:
    repo_id = DATASETS["seedtts"]
    download_dataset(repo_id, quiet=True)
    return repo_id


@pytest.fixture(scope="module")
def router_server(tmp_path_factory: pytest.TempPathFactory):
    """Start two TTS workers behind the router and wait until healthy."""
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=TTS_MODEL_PATH,
        model_name=TTS_MODEL_PATH,
        worker_extra_args=f"--config {TTS_CONFIG_PATH}",
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="tts_router_logs",
    ) as router:
        yield router


@pytest.fixture(scope="module")
def wer_input_dirs(router_server: ManagedRouterHandle) -> dict[str, dict[int, str]]:
    """Free the TTS router workers before starting ASR transcription."""
    router_server.stop()
    for output_dirs in SPEED_OUTPUT_DIRS.values():
        for output_dir in output_dirs.values():
            generated_path = Path(output_dir) / "generated.json"
            assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return SPEED_OUTPUT_DIRS


def _run_benchmark(
    port: int,
    meta: str,
    output_dir: str,
    *,
    stream: bool = False,
) -> dict:
    benchmark_config = TtsSeedttsBenchmarkConfig(
        model=TTS_MODEL_PATH,
        port=port,
        meta=meta,
        output_dir=output_dir,
        concurrency=TTS_CONCURRENCY,
        stream=stream,
        ref_format=TTS_REF_FORMAT,
        seed=TTS_SEED,
        disable_tqdm=True,
    )
    return asyncio.run(
        asyncio.wait_for(
            run_tts_seedtts_benchmark(benchmark_config),
            timeout=BENCHMARK_TIMEOUT,
        )
    )


def _run_wer_transcribe(
    meta: str,
    output_dir: str,
    *,
    stream: bool = False,
    device: str = "cuda:0",
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--transcribe-only",
        "--meta",
        meta,
        "--output-dir",
        output_dir,
        "--model",
        TTS_MODEL_PATH,
        "--ref-format",
        TTS_REF_FORMAT,
        "--lang",
        "en",
        "--device",
        device,
    ]
    if stream:
        cmd.append("--stream")

    env = no_proxy_env()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(PROJECT_ROOT)
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
    with results_path.open(encoding="utf-8") as f:
        return json.load(f)


def _resolve_stage_output_dir(tmp_path: Path, output_dir_name: str) -> str:
    output_root = os.environ.get(TTS_STAGE_OUTPUT_ROOT_ENV)
    if output_root:
        output_dir = Path(output_root) / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    return str(tmp_path / output_dir_name)


def _validate_speed_results(
    results: dict,
    *,
    mode: str,
    output_dir: str,
) -> None:
    checks = MetricCheckCollector(f"TTS {mode} speed")
    summary = results.get("summary", {})
    per_request = results.get("per_request", [])
    assert_summary_metrics(summary, collector=checks)
    assert_per_request_fields(per_request, collector=checks)
    thresholds = (
        TTS_STREAM_THRESHOLDS if mode == "stream" else TTS_NON_STREAM_THRESHOLDS
    )
    assert_speed_thresholds(
        summary,
        thresholds,
        TTS_CONCURRENCY,
        collector=checks,
    )
    SPEED_OUTPUT_DIRS[mode][TTS_CONCURRENCY] = output_dir
    checks.assert_all()


def _validate_wer_results(results: dict, *, mode: str) -> None:
    checks = MetricCheckCollector(f"TTS {mode} WER")
    assert_wer_partitioned(
        results,
        max_wer_below_50_corpus=TTS_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=TTS_WER_N_ABOVE_50_MAX,
        collector=checks,
    )
    checks.assert_all()


def _assert_stage_used_all_router_workers(
    *,
    router_server: ManagedRouterHandle,
    before_workers: dict,
    results: dict,
    label: str,
    collector: MetricCheckCollector,
) -> None:
    collector.check_assertion(
        f"{label} router worker traffic",
        assert_workers_served_requests_since,
        port=router_server.port,
        before_snapshot=before_workers,
        label=label,
        min_total_requests=results["summary"]["completed_requests"],
    )


@pytest.mark.benchmark
def test_tts_non_streaming_generation(
    router_server: ManagedRouterHandle,
    dataset_repo: str,
    tmp_path: Path,
) -> None:
    output_dir = _resolve_stage_output_dir(
        tmp_path, f"tts_nonstream_c{TTS_CONCURRENCY}"
    )
    before_workers = router_get_json(router_server.port, "/workers")
    try:
        results = _run_benchmark(router_server.port, dataset_repo, output_dir)
        checks = MetricCheckCollector("TTS non-stream router traffic")
        _assert_stage_used_all_router_workers(
            router_server=router_server,
            before_workers=before_workers,
            results=results,
            label=f"TTS non-stream c{TTS_CONCURRENCY}",
            collector=checks,
        )
        checks.assert_all()
    except Exception:
        print_router_diagnostics(router_server)
        raise
    _validate_speed_results(results, mode="non_stream", output_dir=output_dir)


@pytest.mark.benchmark
def test_tts_streaming_generation(
    router_server: ManagedRouterHandle,
    dataset_repo: str,
    tmp_path: Path,
) -> None:
    output_dir = _resolve_stage_output_dir(
        tmp_path, f"tts_stream_c{TTS_CONCURRENCY}"
    )
    before_workers = router_get_json(router_server.port, "/workers")
    try:
        results = _run_benchmark(
            router_server.port,
            dataset_repo,
            output_dir,
            stream=True,
        )
        checks = MetricCheckCollector("TTS stream router traffic")
        _assert_stage_used_all_router_workers(
            router_server=router_server,
            before_workers=before_workers,
            results=results,
            label=f"TTS stream c{TTS_CONCURRENCY}",
            collector=checks,
        )
        checks.assert_all()
    except Exception:
        print_router_diagnostics(router_server)
        raise
    _validate_speed_results(results, mode="stream", output_dir=output_dir)


# Keep both generation tests before WER so full-file calibration does not stop
# the router before streaming audio has been generated.
@pytest.mark.benchmark
def test_tts_non_streaming_wer(
    wer_input_dirs: dict[str, dict[int, str]],
    dataset_repo: str,
) -> None:
    output_dir = wer_input_dirs["non_stream"][TTS_CONCURRENCY]
    results = _run_wer_transcribe(dataset_repo, output_dir)
    _validate_wer_results(results, mode="non_stream")


@pytest.mark.benchmark
def test_tts_streaming_wer(
    wer_input_dirs: dict[str, dict[int, str]],
    dataset_repo: str,
) -> None:
    output_dir = wer_input_dirs["stream"][TTS_CONCURRENCY]
    results = _run_wer_transcribe(dataset_repo, output_dir, stream=True)
    _validate_wer_results(results, mode="stream")
