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
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    assert_workers_served_requests,
    print_log_tail,
    print_router_diagnostics,
    print_worker_snapshot,
    router_get_json,
)
from tests.utils import (
    apply_slack,
    apply_wer_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_partitioned,
    no_proxy_env,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONCURRENCY = 16
MAX_SAMPLES = 50
DATASET_CACHE_ENV = "SGLANG_SEEDTTS50_DIR"

WER_TIMEOUT = 600

VC_WER_BELOW_50_CORPUS_MAX = 0.014184397163120567
VC_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(VC_WER_BELOW_50_CORPUS_MAX)
VC_N_ABOVE_50_MAX = 0

# Note (Chenyang): The thresholds for the throughput_qps of tests/test_model/test_qwen3_omni_tts_ci.py
# are the most unstable metrics, so I drop it a lot.

_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 5.865,
        "tok_per_s_agg": 5.8,
        "latency_mean_s": 2.536,
        "rtf_mean": 0.8369,
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

    note (Chenyang): We invoke the benchmark as python -m
    benchmarks.eval.benchmark_omni_seedtts rather than via a direct file
    path so the benchmarks package is discovered via PEP 420 namespace
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


@dataclass
class _SpeedArtifacts:
    """Outputs from the voice-clone speed benchmark.

    Speed-threshold assertions are deliberately NOT made here so that a
    speed miss does not cascade-skip the WER fixture chain. The speed
    test asserts; the WER test reuses only ``output_dir``.
    """

    output_dir: str
    summary: dict
    per_request: list


@pytest.fixture(scope="module")
def speed_artifacts(
    qwen3_omni_router_server: ManagedRouterHandle,
    dataset_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> _SpeedArtifacts:
    """Run the speed benchmark once and expose its artifacts."""
    output_dir = str(tmp_path_factory.mktemp("vc_nonstream"))
    try:
        workers = router_get_json(qwen3_omni_router_server.port, "/workers")
        print_worker_snapshot("initial /workers snapshot", workers)
        assert workers["total_workers"] == 2
        assert workers["healthy_workers"] == 2
        assert workers["routable_workers"] == 2

        models = router_get_json(qwen3_omni_router_server.port, "/v1/models")
        assert {card["id"] for card in models["data"]} == {"qwen3-omni"}

        results = _run_benchmark(
            qwen3_omni_router_server.port,
            str(dataset_dir / "en" / "meta.lst"),
            output_dir,
        )
    except Exception:
        print_router_diagnostics(qwen3_omni_router_server)
        raise
    return _SpeedArtifacts(
        output_dir=output_dir,
        summary=results["summary"],
        per_request=results["per_request"],
    )


@pytest.fixture(scope="module")
def wer_audio_dir(
    qwen3_omni_router_server: ManagedRouterHandle,
    speed_artifacts: _SpeedArtifacts,
) -> str:
    """Reuse speed-benchmark audio for WER after freeing the TTS server GPU."""
    qwen3_omni_router_server.stop()
    generated_path = Path(speed_artifacts.output_dir) / "generated.json"
    assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return speed_artifacts.output_dir


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    qwen3_omni_router_server: ManagedRouterHandle,
    speed_artifacts: _SpeedArtifacts,
) -> None:
    """Print speed summary and assert metrics meet thresholds."""
    try:
        print_speed_summary(
            speed_artifacts.summary,
            "qwen3-omni",
            CONCURRENCY,
            title="TTS Voice-Clone Speed",
        )
        assert_summary_metrics(speed_artifacts.summary)
        assert_per_request_fields(speed_artifacts.per_request)
        assert_speed_thresholds(
            speed_artifacts.summary, VC_NON_STREAM_THRESHOLDS, CONCURRENCY
        )
        assert Path(speed_artifacts.output_dir).is_dir()

        final_workers = router_get_json(qwen3_omni_router_server.port, "/workers")
        print_worker_snapshot("final /workers snapshot", final_workers)
        assert final_workers["routable_workers"] == 2
        assert all(
            worker["active_requests"] == 0 for worker in final_workers["workers"]
        )
        assert_workers_served_requests(
            final_workers,
            min_total_requests=MAX_SAMPLES,
        )
    except Exception:
        print_router_diagnostics(qwen3_omni_router_server)
        raise


@pytest.mark.benchmark
def test_voice_cloning_wer(
    qwen3_omni_router_server: ManagedRouterHandle,
    wer_audio_dir: str,
    dataset_dir: Path,
) -> None:
    results = _run_wer_transcribe(
        str(dataset_dir / "en" / "meta.lst"),
        wer_audio_dir,
    )
    print_wer_summary(results["summary"], "qwen3-omni")
    assert_wer_partitioned(
        results,
        max_wer_below_50_corpus=VC_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=VC_N_ABOVE_50_MAX,
    )
    print_log_tail("router", qwen3_omni_router_server.log_file)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
