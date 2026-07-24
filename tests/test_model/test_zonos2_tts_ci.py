# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks and voice-clone WER CI for ZONOS2 (Zyphra MoE TTS).

ZONOS2's default pipeline is a single-process colocated (DP1) server, so the
TTS server is launched through ``managed_omni_server`` (one
``sglang_omni.cli serve`` process) rather than the 2-worker router used by
Qwen3-Omni / Higgs. The WER phase reuses the shared Qwen3-ASR router fixture
exactly like the other TTS CI tests.

Usage:
    pytest tests/test_model/test_zonos2_tts_ci.py -s -x

"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
    run_tts_seedtts_transcribe,
)
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
from tests.test_model.omni_router_utils import (
    ManagedRouterHandle,
    _find_available_port_range,
    print_log_tail,
)
from tests.utils import (
    QWEN3_ASR_WER_CONCURRENCY,
    MetricCheckCollector,
    apply_slack,
    assert_per_request_fields,
    assert_speed_thresholds,
    assert_summary_metrics,
    assert_wer_results,
    server_log_file,
    wait_for_gpu_memory_release,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ZONOS2 auto-detects its architecture from params.json (model_type "zonos2"
# -> Zonos2ForCausalLM in sglang_omni/utils/hf.py), so ``serve`` needs only
# --model-path, no --config (mirrors Higgs).
ZONOS2_MODEL_PATH = os.environ.get("ZONOS2_MODEL_PATH", "/data/gaokaiz/zonos2")
# The benchmark echoes this into the request ``model`` field; default it to the
# model path so it matches the served pipeline name.
ZONOS2_MODEL_NAME = os.environ.get("ZONOS2_MODEL_NAME", ZONOS2_MODEL_PATH)

CONCURRENCY = 16
MAX_SAMPLES = 50
STARTUP_TIMEOUT = 600

# WER gate fixed by the task: corpus WER < 2% (wer_corpus is a fraction).
VC_WER_MAX_CORPUS = 0.02

# RTF / throughput speed baseline.
#
# TODO(calibrate on CI GPU): _VC_NON_STREAM_P95 MUST be populated from a real
# ZONOS2 seed-tts-eval-50 EN c=16 run on the target CI GPU before the speed
# gate can enforce a number. Procedure: launch the zonos2 server, run
#   python -m benchmarks.eval.benchmark_tts_seedtts \
#     --meta zhaochenyang20/seed-tts-eval-50-arrow \
#     --model <ZONOS2_MODEL_NAME> --port <port> \
#     --ref-format references --lang en \
#     --max-concurrency 16 --max-samples 50 --output-dir results/zonos2_en
# three times, then set the dict below from the P95/worst of
# speed_results.json['summary'] {throughput_qps, output_tok_per_req_s,
# latency_mean_s, rtf_mean} and let apply_slack derive the thresholds.
# Do NOT copy Qwen3-Omni's 0.8149 / Higgs's 0.335 — different model, different
# GPU. While this is None the speed test skips its threshold assertion (the WER
# test and the structural sanity checks still run).
#
# Measured reference (NOT the gate) from a single dev H100 80GB run on this
# branch (Track B rebased onto ZONOS2 = Track A + FP8-off + loop-guard), c=16,
# seed-tts-eval-50 EN:
#   rtf_mean=0.8212  throughput_qps=5.167  latency_mean_s=2.827
# output_tok_per_req_s is unavailable via --use-existing-server (no per-request
# engine_time_s); the in-harness managed-server path the test uses populates it,
# so calibrate there. WER sanity check (whisper-large-v3, not the Qwen3-ASR
# gate judge): corpus 0.89%, no >50% outliers.
_VC_NON_STREAM_P95: dict[int, dict[str, float]] | None = None

# Optional ZONOS2-specific RTF hard cap (analogue of
# QWEN3_OMNI_SEEDTTS_RTF_MEAN_MAX). Leave None until a measured run justifies a
# value; the apply_slack-derived rtf_mean_max alone is the gate otherwise.
ZONOS2_SEEDTTS_RTF_MEAN_MAX: float | None = None

if _VC_NON_STREAM_P95 is not None:
    VC_NON_STREAM_THRESHOLDS: dict[int, dict[str, float]] | None = apply_slack(
        _VC_NON_STREAM_P95
    )
    if ZONOS2_SEEDTTS_RTF_MEAN_MAX is not None and "rtf_mean_max" in (
        VC_NON_STREAM_THRESHOLDS[CONCURRENCY]
    ):
        VC_NON_STREAM_THRESHOLDS[CONCURRENCY]["rtf_mean_max"] = min(
            VC_NON_STREAM_THRESHOLDS[CONCURRENCY]["rtf_mean_max"],
            ZONOS2_SEEDTTS_RTF_MEAN_MAX,
        )
else:
    VC_NON_STREAM_THRESHOLDS = None


def _run_benchmark(
    port: int,
    meta: str,
    output_dir: str,
) -> dict:
    config = TtsSeedttsBenchmarkConfig(
        model=ZONOS2_MODEL_NAME,
        port=port,
        meta=meta,
        output_dir=output_dir,
        max_samples=MAX_SAMPLES,
        concurrency=CONCURRENCY,
        voice_clone=True,
        ref_format="references",
    )
    speed_results = asyncio.run(run_tts_seedtts_benchmark(config))
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
    config = TtsSeedttsBenchmarkConfig(
        model=ZONOS2_MODEL_NAME,
        meta=meta,
        output_dir=output_dir,
        lang=lang,
        device=device,
        concurrency=CONCURRENCY,
        asr_concurrency=QWEN3_ASR_WER_CONCURRENCY,
    )
    run_tts_seedtts_transcribe(config, asr_router_port=asr_router_port)

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


@pytest.fixture(scope="module")
def dataset_repo() -> str:
    repo_id = DATASETS["seedtts-50"]
    download_dataset(repo_id, quiet=True)
    return repo_id


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
def zonos2_server(tmp_path_factory: pytest.TempPathFactory):
    """Start a single colocated ZONOS2 server and yield its port."""
    port = _find_available_port_range(1)
    log_file = server_log_file(tmp_path_factory, "zonos2_server_logs")
    with managed_omni_server(
        model_path=ZONOS2_MODEL_PATH,
        port=port,
        host="127.0.0.1",
        log_file=log_file,
        timeout=STARTUP_TIMEOUT,
        wait_for_gpu_release=True,
    ):
        yield port


@pytest.fixture(scope="module")
def speed_artifacts(
    zonos2_server: int,
    dataset_repo: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> _SpeedArtifacts:
    """Run the speed benchmark once and expose its artifacts."""
    output_dir = str(tmp_path_factory.mktemp("vc_nonstream"))
    results = _run_benchmark(zonos2_server, dataset_repo, output_dir)
    return _SpeedArtifacts(
        output_dir=output_dir,
        summary=results["summary"],
        per_request=results["per_request"],
    )


@pytest.fixture(scope="module")
def wer_audio_dir(speed_artifacts: _SpeedArtifacts) -> str:
    """Reuse speed-benchmark audio for WER after freeing the TTS server GPU."""
    # zonos2_server (depended on by speed_artifacts) is module-scoped; it tears
    # down and frees the GPU after the last test that needs it. Force a cleanup
    # here so the Qwen3-ASR WER router has the GPU before it starts.
    wait_for_gpu_memory_release()
    generated_path = Path(speed_artifacts.output_dir) / "generated.json"
    assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return speed_artifacts.output_dir


@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    speed_artifacts: _SpeedArtifacts,
) -> None:
    """Print speed summary and assert metrics meet thresholds."""
    print_speed_summary(
        speed_artifacts.summary,
        ZONOS2_MODEL_NAME,
        CONCURRENCY,
        title="ZONOS2 TTS Voice-Clone Speed",
    )
    checks = MetricCheckCollector("ZONOS2 voice-cloning speed")
    assert_summary_metrics(speed_artifacts.summary, collector=checks)
    assert_per_request_fields(speed_artifacts.per_request, collector=checks)
    if VC_NON_STREAM_THRESHOLDS is not None:
        assert_speed_thresholds(
            speed_artifacts.summary,
            VC_NON_STREAM_THRESHOLDS,
            CONCURRENCY,
            collector=checks,
        )
    else:
        print(
            "\n[ZONOS2 speed] RTF/throughput gate is a placeholder "
            "(_VC_NON_STREAM_P95 is uncalibrated); enforcing structural "
            "checks only. See TODO(calibrate on CI GPU) in this module."
        )
    checks.check(
        Path(speed_artifacts.output_dir).is_dir(),
        f"Speed output directory missing: {speed_artifacts.output_dir}",
    )
    checks.assert_all()


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
    print_wer_summary(results["summary"], ZONOS2_MODEL_NAME)
    checks = MetricCheckCollector("ZONOS2 voice-cloning WER")
    assert_wer_results(results, VC_WER_MAX_CORPUS, collector=checks)
    checks.assert_all()
    print_log_tail("asr_wer_router", qwen3_asr_wer_router.log_file)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
