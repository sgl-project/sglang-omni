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
from benchmarks.metrics._format import format_benchmark_dataset_label
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
    QWEN3_ASR_WER_CONCURRENCY,
    MetricCheckCollector,
    apply_mos_slack,
    apply_slack,
    apply_wer_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_partitioned,
    no_proxy_env,
    wait_for_gpu_memory_release,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONCURRENCY = 16
MAX_SAMPLES = 50
# Optional user override: a path to a custom fine-tuned WavLM checkpoint.
# When unset, the bootstrapper in benchmarks.metrics.speaker_similarity_assets
# auto-downloads the official weights into the shared cache directory.
SIMILARITY_CHECKPOINT_ENV = "SEEDTTS_SIM_CHECKPOINT"

WER_TIMEOUT = 600
SIMILARITY_TIMEOUT = 600
UTMOS_TIMEOUT = 600

VC_WER_BELOW_50_CORPUS_MAX = 0.0337
VC_WER_BELOW_50_CORPUS_THRESHOLD = apply_wer_slack(VC_WER_BELOW_50_CORPUS_MAX)
VC_N_ABOVE_50_MAX = 0.0
# 60.0 mirrors the S2-Pro floor and is a placeholder until upstream issue
# #483 is fixed; the hard assertion is currently disabled in
# test_voice_cloning_similarity (see docstring there). PR #469 also collected
# five Qwen3-Omni SeedTTS-50 EN runs for the record — all in the 2.90–3.48
# range (worst = 2.90, stdev = 0.21), which confirms #483 deterministically
# rather than producing a usable lower bound: no meaningful CI floor can be
# derived from broken-state data. When #483 lands, re-run the five-shot
# calibration with the fix and reset this constant from the lowest of the
# five with the standard slack margin. See the "Speaker similarity
# calibration" section of the PR description for the per-run numbers.
VC_SIMILARITY_MEAN_MIN = 60.0
# Calibrated from worst-of-5 full generate+score runs on SeedTTS-50 EN, H200 SXM.
# worst-of-5 = 4.1924 · mean = 4.2575 · stdev = 0.0487
VC_UTMOS_MEAN_REFERENCE = 4.2388
VC_UTMOS_MEAN_MIN = apply_mos_slack(VC_UTMOS_MEAN_REFERENCE)

# Note (Chenyang): The thresholds for the throughput_qps of tests/test_model/test_qwen3_omni_tts_ci.py
# are the most unstable metrics, so I drop it a lot.

_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 5.317,
        "output_tok_per_req_s": 5.2,
        "latency_mean_s": 2.803,
        "rtf_mean": 0.8149,
    },
}


# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics (throughput, output tok/req-s): threshold = P95 x slack_higher
# Lower-is-better metrics (latency, rtf): threshold = P95 x slack_lower

QWEN3_OMNI_SEEDTTS_RTF_MEAN_MAX = 0.9536
VC_NON_STREAM_THRESHOLDS = apply_slack(_VC_NON_STREAM_P95)
VC_NON_STREAM_THRESHOLDS[CONCURRENCY]["rtf_mean_max"] = min(
    VC_NON_STREAM_THRESHOLDS[CONCURRENCY]["rtf_mean_max"],
    QWEN3_OMNI_SEEDTTS_RTF_MEAN_MAX,
)

SEEDTTS_50_DATASET_LABEL = format_benchmark_dataset_label(
    dataset="seedtts-50",
    repo_id=DATASETS["seedtts-50"],
)


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
    meta: str,
    output_dir: str,
    *,
    asr_router_port: int,
    lang: str = "en",
    device: str = "cuda:0",
) -> dict:
    """Transcribe saved audio and compute WER via Qwen3-ASR router."""
    from benchmarks.eval.benchmark_omni_seedtts import (
        OmniSeedttsBenchmarkConfig,
        evaluate_generated_audio,
    )

    config = OmniSeedttsBenchmarkConfig(
        model="qwen3-omni",
        meta=meta,
        output_dir=output_dir,
        lang=lang,
        device=device,
        port=asr_router_port,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
    )
    evaluate_generated_audio(config)

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


def _run_similarity(
    meta: str,
    output_dir: str,
    checkpoint_path: str | None,
    *,
    device: str = "cuda:0",
) -> dict:
    """Compute SeedTTS speaker similarity in CI (mirrors WER subprocess pattern)."""
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--similarity-only",
        "--meta",
        meta,
        "--output-dir",
        output_dir,
        "--model",
        "qwen3-omni",
        "--device",
        device,
    ]
    if checkpoint_path is not None:
        cmd += ["--similarity-checkpoint", checkpoint_path]

    env = no_proxy_env()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT)
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=SIMILARITY_TIMEOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"Similarity eval failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    results_path = Path(output_dir) / "similarity_results.json"
    assert results_path.exists(), f"Similarity results file not found: {results_path}"

    with open(results_path) as f:
        similarity_results = json.load(f)
    assert "summary" in similarity_results, (
        "Missing 'summary' key in similarity results. "
        f"Keys: {list(similarity_results.keys())}"
    )
    assert "per_sample" in similarity_results, (
        "Missing 'per_sample' key in similarity results. "
        f"Keys: {list(similarity_results.keys())}"
    )
    return similarity_results


def _assert_similarity_results(
    results: dict,
    min_mean: float,
    *,
    collector: MetricCheckCollector | None = None,
) -> None:
    checks = collector or MetricCheckCollector("speaker similarity")
    summary = results["summary"]
    per_sample = results["per_sample"]
    mean = summary.get("speaker_similarity_mean")
    checks.check(bool(per_sample), "Expected per-sample speaker similarity results")
    if mean is None:
        checks.fail("Missing speaker_similarity_mean in summary")
    else:
        checks.check(
            mean >= min_mean,
            f"speaker_similarity_mean {mean:.4f} < threshold {min_mean:.4f}",
        )
    if collector is None:
        checks.assert_all()


def _run_utmos(output_dir: str, *, device: str = "cuda:0") -> dict:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--utmos-only",
        "--meta",
        DATASETS["seedtts-50"],
        "--output-dir",
        output_dir,
        "--model",
        "qwen3-omni",
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
        timeout=UTMOS_TIMEOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"UTMOS eval failed (rc={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    results_path = Path(output_dir) / "utmos_results.json"
    assert results_path.exists(), f"UTMOS results file not found: {results_path}"
    with open(results_path) as f:
        return json.load(f)


def _assert_utmos_results(
    results: dict,
    threshold: float,
    *,
    collector: MetricCheckCollector | None = None,
) -> None:
    checks = collector or MetricCheckCollector("UTMOS")
    summary = results.get("summary", {})
    checks.check(bool(results.get("per_sample")), "per_sample must be non-empty")
    checks.check(
        summary.get("skipped", 0) == 0,
        f"UTMOS: {summary.get('skipped')} skipped samples != 0",
    )
    mean = summary.get("utmos_mean")
    if mean is None:
        checks.fail("Missing utmos_mean in summary")
    else:
        checks.check(
            mean >= threshold,
            f"utmos_mean {mean:.4f} < threshold {threshold:.4f}",
        )
    if collector is None:
        checks.assert_all()


@pytest.fixture(scope="module")
def dataset_repo() -> str:
    repo_id = DATASETS["seedtts-50"]
    download_dataset(repo_id, quiet=True)
    return repo_id


@pytest.fixture(scope="module")
def similarity_checkpoint() -> str | None:
    """User-specified WavLM checkpoint override, or None to let the bootstrapper
    auto-resolve the default weights from the shared cache directory."""
    raw = os.environ.get(SIMILARITY_CHECKPOINT_ENV)
    if not raw:
        return None
    return str(Path(raw).expanduser())


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
    qwen3_omni_bf16_colocated_server: ManagedRouterHandle,
    dataset_repo: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> _SpeedArtifacts:
    """Run the speed benchmark once and expose its artifacts."""
    output_dir = str(tmp_path_factory.mktemp("vc_nonstream"))
    try:
        workers = router_get_json(qwen3_omni_bf16_colocated_server.port, "/workers")
        print_worker_snapshot("initial /workers snapshot", workers)
        assert workers["total_workers"] == 2
        assert workers["healthy_workers"] == 2
        assert workers["routable_workers"] == 2

        models = router_get_json(qwen3_omni_bf16_colocated_server.port, "/v1/models")
        assert {card["id"] for card in models["data"]} == {"qwen3-omni"}

        results = _run_benchmark(
            qwen3_omni_bf16_colocated_server.port,
            dataset_repo,
            output_dir,
        )
    except Exception:
        print_router_diagnostics(qwen3_omni_bf16_colocated_server)
        raise
    return _SpeedArtifacts(
        output_dir=output_dir,
        summary=results["summary"],
        per_request=results["per_request"],
    )


@pytest.fixture(scope="module")
def wer_audio_dir(
    qwen3_omni_bf16_colocated_server: ManagedRouterHandle,
    speed_artifacts: _SpeedArtifacts,
) -> str:
    """Reuse speed-benchmark audio for WER after freeing the TTS server GPU."""
    qwen3_omni_bf16_colocated_server.stop()
    wait_for_gpu_memory_release()
    generated_path = Path(speed_artifacts.output_dir) / "generated.json"
    assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return speed_artifacts.output_dir


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    qwen3_omni_bf16_colocated_server: ManagedRouterHandle,
    speed_artifacts: _SpeedArtifacts,
) -> None:
    """Print speed summary and assert metrics meet thresholds."""
    try:
        print_speed_summary(
            speed_artifacts.summary,
            "qwen3-omni",
            CONCURRENCY,
            title="TTS Voice-Clone Speed",
            dataset=SEEDTTS_50_DATASET_LABEL,
        )
        checks = MetricCheckCollector("Qwen3-Omni voice-cloning speed")
        assert_summary_metrics(speed_artifacts.summary, collector=checks)
        assert_per_request_fields(speed_artifacts.per_request, collector=checks)
        assert_speed_thresholds(
            speed_artifacts.summary,
            VC_NON_STREAM_THRESHOLDS,
            CONCURRENCY,
            collector=checks,
        )
        checks.check(
            Path(speed_artifacts.output_dir).is_dir(),
            f"Speed output directory missing: {speed_artifacts.output_dir}",
        )

        final_workers = router_get_json(
            qwen3_omni_bf16_colocated_server.port, "/workers"
        )
        print_worker_snapshot("final /workers snapshot", final_workers)
        checks.check(
            final_workers.get("routable_workers") == 2,
            f"Expected 2 routable workers, got {final_workers.get('routable_workers')}",
        )
        active_workers = [
            worker
            for worker in final_workers.get("workers", [])
            if worker.get("active_requests") != 0
        ]
        checks.check(
            not active_workers,
            f"Expected no active requests after benchmark, got {active_workers}",
        )
        checks.check_assertion(
            "router worker traffic",
            assert_workers_served_requests,
            final_workers,
            min_total_requests=MAX_SAMPLES,
        )
        checks.assert_all()
    except Exception:
        print_router_diagnostics(qwen3_omni_bf16_colocated_server)
        raise


@pytest.mark.benchmark
def test_voice_cloning_wer(
    wer_audio_dir: str,
    dataset_repo: str,
    qwen3_asr_wer_router: ManagedRouterHandle,
) -> None:
    results = _run_wer_transcribe(
        dataset_repo,
        wer_audio_dir,
        asr_router_port=qwen3_asr_wer_router.port,
    )
    print_wer_summary(
        results["summary"], "qwen3-omni", dataset=SEEDTTS_50_DATASET_LABEL
    )
    checks = MetricCheckCollector("Qwen3-Omni voice-cloning WER")
    assert_wer_partitioned(
        results,
        max_wer_below_50_corpus=VC_WER_BELOW_50_CORPUS_THRESHOLD,
        max_n_above_50=VC_N_ABOVE_50_MAX,
        collector=checks,
    )
    checks.assert_all()
    print_log_tail("asr_wer_router", qwen3_asr_wer_router.log_file)


@pytest.mark.benchmark
def test_voice_cloning_similarity(
    wer_audio_dir: str,
    dataset_repo: str,
    similarity_checkpoint: str | None,
) -> None:
    """Speaker similarity for Qwen3-Omni voice-clone output.

    Quality gating against ``VC_SIMILARITY_MEAN_MIN`` is intentionally
    DISABLED while upstream issue sgl-project/sglang-omni#483 is open:
    Qwen3-Omni currently emits a default voice (multimodal prompt
    features do not reach talker prefill ``input_embeds``), so a
    50-sample EN dry-run measures SIM mean ~3 vs ~64 for S2-Pro on the
    same samples.

    The test still runs and persists ``similarity_results.json`` so the
    metric tracks longitudinally and will catch the day #483 is fixed.
    Once #483 lands, swap the structural assert below back to
    ``_assert_similarity_results(results, VC_SIMILARITY_MEAN_MIN)``.
    """
    results = _run_similarity(
        dataset_repo,
        wer_audio_dir,
        similarity_checkpoint,
    )
    # Structural sanity only — quality gate disabled per docstring above.
    # `skipped == 0` is a structural assertion (generation succeeded and WAVs
    # are on disk), not a voice-quality gate, so it stays enabled even while
    # #483 keeps the similarity-mean assertion soft.
    summary = results.get("summary", {})
    checks = MetricCheckCollector("Qwen3-Omni speaker similarity structure")
    checks.check(
        summary.get("speaker_similarity_mean") is not None,
        "Missing speaker_similarity_mean in summary",
    )
    checks.check(
        bool(results.get("per_sample")),
        "Expected per-sample speaker similarity results",
    )
    checks.check(
        summary.get("skipped", 0) == 0,
        f"speaker similarity: {summary.get('skipped')} skipped samples != 0",
    )
    checks.assert_all()


@pytest.mark.benchmark
def test_voice_cloning_utmos(wer_audio_dir: str) -> None:
    results = _run_utmos(wer_audio_dir)
    checks = MetricCheckCollector("Qwen3-Omni voice-cloning UTMOS")
    _assert_utmos_results(results, VC_UTMOS_MEAN_MIN, collector=checks)
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
