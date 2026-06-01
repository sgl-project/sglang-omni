# SPDX-License-Identifier: Apache-2.0
"""Speed benchmarks and voice-clone WER thresholds CI for TTS models.

Usage:
    pytest tests/test_model/test_tts_ci.py -s -x
    pytest tests/test_model/test_tts_ci.py -s -x --concurrency 16
    pytest tests/test_model/test_tts_ci.py -s -x --concurrency 16 \
        --tts-stage tts-stage-1-nonstream
    pytest tests/test_model/test_tts_ci.py -s -x --concurrency all

Author:
    Chenyang Zhao https://github.com/zhaochenyang20
    Raitsh P https://github.com/Ratish1
    Jingwen Guo https://github.com/JingwenGu0829
    Yuan Luo https://github.com/yuan-luo
    Yitong Guan https://github.com/minleminzui
    Xuesong Ye https://github.com/yxs

The benchmark supports one selected concurrency per test run. Use --concurrency 16
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
from typing import Literal

import pytest

pytest_plugins = ["tests.test_model.omni_whisper_wer_utils"]

from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
)
from tests.test_model.conftest import (
    TTS_STAGE_CONSISTENCY,
    TTS_STAGE_NONSTREAM,
    TTS_STAGE_STREAM,
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
    apply_mos_slack,
    apply_slack,
    apply_wer_slack,
    assert_speed_thresholds,
    assert_streaming_consistency,
    assert_wer_results,
    no_proxy_env,
    wait_for_gpu_memory_release,
)

PER_REQUEST_STORE: dict[str, list[dict]] = {}
SPEED_OUTPUT_DIRS: dict[str, dict[int, str]] = {"non_stream": {}, "stream": {}}

TTS_MODEL_PATH = os.environ.get(
    "TTS_MODEL_PATH", "boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999"
)

STARTUP_TIMEOUT = 180
BENCHMARK_TIMEOUT = 600
WER_TIMEOUT = 600
SIMILARITY_TIMEOUT = 600
UTMOS_TIMEOUT = 600

SIMILARITY_CHECKPOINT_ENV = "SEEDTTS_SIM_CHECKPOINT"
TTS_STAGE_OUTPUT_ROOT_ENV = "TTS_STAGE_OUTPUT_ROOT"
TTS_STAGE1_SPEED_RESULTS_DIR_ENV = "TTS_STAGE1_SPEED_RESULTS_DIR"
TTS_STAGE2_SPEED_RESULTS_DIR_ENV = "TTS_STAGE2_SPEED_RESULTS_DIR"
TTS_SIMILARITY_MAX_SAMPLES_ENV = "TTS_SIMILARITY_MAX_SAMPLES"

SEEDTTS_EN_FULLSET_SAMPLES = 1088
STREAMING_BENCHMARK_MAX_SAMPLES: int | None = None
TTS_SIMILARITY_MAX_SAMPLES = 50

# Note (chenyang): the RTF thresholds also includes the reference audio
# processing time.

# Note (Ratish, Chenyang): We evalute the performance of TTS CI on our H20
# CI machines and compute the thresholds based on the results.

# Slack factors applied to P95 reference values to derive CI thresholds.
# Higher-is-better metrics (throughput, output tok/req-s): threshold = P95 × slack_higher
# Lower-is-better metrics (latency, rtf): threshold = P95 × slack_lower

THRESHOLD_SLACK_HIGHER = 0.75
THRESHOLD_SLACK_LOWER = 1.25
VC_WER_MAX_CORPUS = 0.0104
VC_WER_CORPUS_THRESHOLD = apply_wer_slack(VC_WER_MAX_CORPUS)
VC_STREAM_WER_MAX_CORPUS = 0.0098
VC_STREAM_WER_CORPUS_THRESHOLD = apply_wer_slack(VC_STREAM_WER_MAX_CORPUS)

VC_SIMILARITY_MEAN_MIN = 66.18289001464844
# Calibrated from worst-of-5 full generate+score runs on SeedTTS-50 EN, H200 SXM.
# worst-of-5 = 4.1538 · mean = 4.1618 · stdev = 0.0079
VC_UTMOS_MEAN_REFERENCE = 4.1538
VC_UTMOS_MEAN_MIN = apply_mos_slack(VC_UTMOS_MEAN_REFERENCE)

# Note (Chenyang): Only thresholds for the CI concurrency are dedicatedly tuned,
# others may not pass the CI.

_VC_NON_STREAM_P95 = {
    16: {
        "throughput_qps": 11.558,
        "output_tok_per_req_s": 119.9,
        "latency_mean_s": 1.372,
        "rtf_mean": 0.335,
    }
}

_VC_STREAM_P95 = {
    16: {
        "throughput_qps": 11.535,
        "output_tok_per_req_s": 109.7,
        "latency_mean_s": 1.381,
        "rtf_mean": 0.3174,
    }
}


VC_NON_STREAM_THRESHOLDS = apply_slack(
    _VC_NON_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)
VC_STREAM_THRESHOLDS = apply_slack(
    _VC_STREAM_P95, THRESHOLD_SLACK_HIGHER, THRESHOLD_SLACK_LOWER
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WER_MODULE = "benchmarks.eval.benchmark_tts_seedtts"


def _validate_speed_results_keys(speed_results: dict) -> None:
    assert (
        "summary" in speed_results
    ), f"Missing 'summary' key in results. Keys: {list(speed_results.keys())}"
    assert (
        "per_request" in speed_results
    ), f"Missing 'per_request' key in results. Keys: {list(speed_results.keys())}"


def _run_benchmark(
    port: int,
    testset: str,
    output_dir: str,
    *,
    concurrency: int,
    max_samples: int | None = None,
    stream: bool = False,
) -> dict:
    benchmark_config = TtsSeedttsBenchmarkConfig(
        model=TTS_MODEL_PATH,
        port=port,
        meta=testset,
        output_dir=output_dir,
        concurrency=concurrency,
        max_samples=max_samples,
        stream=stream,
    )
    speed_results = asyncio.run(run_tts_seedtts_benchmark(benchmark_config))
    _validate_speed_results_keys(speed_results)
    return speed_results


def _run_wer_transcribe(
    meta: str,
    output_dir: str,
    *,
    whisper_router_port: int,
    stream: bool = False,
    lang: str = "en",
    device: str = "cuda:0",
) -> dict:
    """Transcribe saved audio and compute WER via Omni Whisper router."""
    from benchmarks.eval.benchmark_tts_seedtts import (
        TtsSeedttsBenchmarkConfig,
        run_tts_seedtts_transcribe,
    )

    config = TtsSeedttsBenchmarkConfig(
        model=TTS_MODEL_PATH,
        meta=meta,
        output_dir=output_dir,
        lang=lang,
        device=device,
        stream=stream,
    )
    run_tts_seedtts_transcribe(
        config,
        whisper_router_port=whisper_router_port,
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
    max_samples: int | None = None,
) -> dict:
    """Compute SeedTTS speaker similarity in CI."""
    wait_for_gpu_memory_release()

    cmd = [
        sys.executable,
        "-m",
        WER_MODULE,
        "--similarity-only",
        "--meta",
        meta,
        "--output-dir",
        output_dir,
        "--model",
        TTS_MODEL_PATH,
        "--device",
        device,
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    if checkpoint_path is not None:
        cmd += ["--similarity-checkpoint", checkpoint_path]

    env = no_proxy_env()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(PROJECT_ROOT)
    )

    result = subprocess.run(
        cmd,
        text=True,
        timeout=SIMILARITY_TIMEOUT,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"Similarity eval failed (rc={result.returncode})"

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
    checks.check(bool(per_sample), "Expected per-sample speaker similarity results")
    checks.check(
        summary.get("skipped", 0) == 0,
        f"speaker similarity: {summary.get('skipped')} skipped samples != 0",
    )
    mean = summary.get("speaker_similarity_mean")
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
        WER_MODULE,
        "--utmos-only",
        "--meta",
        DATASETS["seedtts-50"],
        "--output-dir",
        output_dir,
        "--model",
        TTS_MODEL_PATH,
        "--device",
        device,
    ]
    env = no_proxy_env()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing_pp}" if existing_pp else str(PROJECT_ROOT)
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


def _load_speed_results(results_path: Path) -> dict:
    assert results_path.exists(), f"Speed results file not found: {results_path}"
    with open(results_path) as f:
        speed_results = json.load(f)
    _validate_speed_results_keys(speed_results)
    return speed_results


def _assert_tts_speed_result_integrity(
    summary: dict,
    per_request: list[dict],
    *,
    label: str,
    collector: MetricCheckCollector,
) -> None:
    failed_rows = [
        request for request in per_request if request.get("is_success") is not True
    ]
    failed_requests = summary.get("failed_requests")
    completed_requests = summary.get("completed_requests")
    collector.check(
        isinstance(failed_requests, int),
        f"{label}: failed_requests must be an int, got {failed_requests}",
    )
    collector.check(
        isinstance(completed_requests, int),
        f"{label}: completed_requests must be an int, got {completed_requests}",
    )
    if isinstance(failed_requests, int):
        collector.check(
            failed_requests == len(failed_rows),
            f"{label}: summary failed_requests={failed_requests}, "
            f"per_request failures={len(failed_rows)}",
        )
    if isinstance(failed_requests, int) and isinstance(completed_requests, int):
        collector.check(
            completed_requests + failed_requests == len(per_request),
            f"{label}: completed_requests + failed_requests = "
            f"{completed_requests + failed_requests}, per_request={len(per_request)}",
        )

    audio_duration_mean_s = summary.get("audio_duration_mean_s")
    collector.check(
        audio_duration_mean_s is not None and audio_duration_mean_s > 0,
        f"{label}: expected positive audio_duration_mean_s, "
        f"got {audio_duration_mean_s}",
    )
    output_tokens_mean = summary.get("output_tokens_mean", 0)
    collector.check(
        output_tokens_mean > 0,
        f"{label}: expected positive output_tokens_mean, got {output_tokens_mean}",
    )
    prompt_tokens_mean = summary.get("prompt_tokens_mean", 0)
    collector.check(
        prompt_tokens_mean > 0,
        f"{label}: expected positive prompt_tokens_mean, got {prompt_tokens_mean}",
    )

    for request in per_request:
        request_id = request.get("id", "<missing id>")
        if request.get("is_success") is not True:
            collector.check(
                bool(request.get("error")),
                f"{label}: failed request {request_id} is missing error detail",
            )
            continue
        audio_duration_s = request.get("audio_duration_s")
        collector.check(
            audio_duration_s is not None and audio_duration_s > 0,
            f"{label}: request {request_id} audio_duration_s={audio_duration_s}, "
            "expected > 0",
        )
        prompt_tokens = request.get("prompt_tokens")
        completion_tokens = request.get("completion_tokens")
        collector.check(
            prompt_tokens is not None and prompt_tokens > 0,
            f"{label}: request {request_id} prompt_tokens={prompt_tokens}, "
            "expected > 0",
        )
        collector.check(
            completion_tokens is not None and completion_tokens > 0,
            f"{label}: request {request_id} completion_tokens={completion_tokens}, "
            "expected > 0",
        )


def _store_consistency_inputs(
    *,
    mode: Literal["non_stream", "stream"],
    concurrency: int,
    output_dir: str,
    results: dict,
    collector: MetricCheckCollector | None = None,
) -> None:
    checks = collector or MetricCheckCollector(
        f"TTS {mode} speed results at concurrency {concurrency}"
    )
    summary, per_request = results["summary"], results["per_request"]
    _assert_tts_speed_result_integrity(
        summary,
        per_request,
        label=f"TTS {mode} c{concurrency}",
        collector=checks,
    )
    if mode == "non_stream":
        assert_speed_thresholds(
            summary, VC_NON_STREAM_THRESHOLDS, concurrency, collector=checks
        )
        store_key = f"vc_nonstream_c{concurrency}"
    else:
        assert_speed_thresholds(
            summary, VC_STREAM_THRESHOLDS, concurrency, collector=checks
        )
        store_key = f"vc_stream_c{concurrency}"
    PER_REQUEST_STORE[store_key] = per_request
    SPEED_OUTPUT_DIRS[mode][concurrency] = output_dir
    if collector is None:
        checks.assert_all()


def _assert_stage_used_all_router_workers(
    *,
    router_server: ManagedRouterHandle,
    before_workers: dict,
    results: dict,
    label: str,
    collector: MetricCheckCollector | None = None,
) -> None:
    kwargs = {
        "port": router_server.port,
        "before_snapshot": before_workers,
        "label": label,
        "min_total_requests": results["summary"]["completed_requests"],
    }
    if collector is None:
        assert_workers_served_requests_since(**kwargs)
    else:
        collector.check_assertion(
            f"{label} router worker traffic",
            assert_workers_served_requests_since,
            **kwargs,
        )


def _find_downloaded_speed_results(
    artifact_root: str,
    output_dir_name: str,
) -> tuple[str, dict]:
    root = Path(artifact_root)
    matches = sorted(root.rglob(f"{output_dir_name}/speed_results.json"))
    assert (
        matches
    ), f"Downloaded speed results not found under {artifact_root}: {output_dir_name}"
    results_path = matches[0]
    return str(results_path.parent), _load_speed_results(results_path)


def _load_consistency_artifact_inputs(
    selected_tts_concurrencies: tuple[int, ...],
) -> bool:
    non_stream_results_root = os.environ.get(TTS_STAGE1_SPEED_RESULTS_DIR_ENV)
    stream_results_root = os.environ.get(TTS_STAGE2_SPEED_RESULTS_DIR_ENV)
    if not (non_stream_results_root and stream_results_root):
        return False

    for concurrency in selected_tts_concurrencies:
        non_stream_output_dir, non_stream_results = _find_downloaded_speed_results(
            non_stream_results_root, f"vc_nonstream_c{concurrency}"
        )
        stream_output_dir, stream_results = _find_downloaded_speed_results(
            stream_results_root, f"vc_stream_c{concurrency}"
        )
        _store_consistency_inputs(
            mode="non_stream",
            concurrency=concurrency,
            output_dir=non_stream_output_dir,
            results=non_stream_results,
        )
        _store_consistency_inputs(
            mode="stream",
            concurrency=concurrency,
            output_dir=stream_output_dir,
            results=stream_results,
        )
    return True


def _generate_consistency_inputs(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    selected_tts_concurrencies: tuple[int, ...],
) -> None:
    router_server = request.getfixturevalue("router_server")
    dataset_repo = request.getfixturevalue("dataset_repo")
    output_root = tmp_path_factory.mktemp("tts_consistency")
    for concurrency in selected_tts_concurrencies:
        non_stream_key = f"vc_nonstream_c{concurrency}"
        stream_key = f"vc_stream_c{concurrency}"

        if non_stream_key not in PER_REQUEST_STORE:
            output_dir = str(output_root / f"vc_nonstream_c{concurrency}")
            results = _run_benchmark(
                router_server.port,
                dataset_repo,
                output_dir,
                concurrency=concurrency,
            )
            _store_consistency_inputs(
                mode="non_stream",
                concurrency=concurrency,
                output_dir=output_dir,
                results=results,
            )

        if stream_key not in PER_REQUEST_STORE:
            output_dir = str(output_root / f"vc_stream_c{concurrency}")
            results = _run_benchmark(
                router_server.port,
                dataset_repo,
                output_dir,
                concurrency=concurrency,
                max_samples=STREAMING_BENCHMARK_MAX_SAMPLES,
                stream=True,
            )
            _store_consistency_inputs(
                mode="stream",
                concurrency=concurrency,
                output_dir=output_dir,
                results=results,
            )


def _resolve_stage_output_dir(tmp_path: Path, output_dir_name: str) -> str:
    output_root = os.environ.get(TTS_STAGE_OUTPUT_ROOT_ENV)
    if output_root:
        output_dir = Path(output_root) / output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    return str(tmp_path / output_dir_name)


def _print_stage(stage: str, mode: str, concurrency: int, details: str = "") -> None:
    message = f"\n[Stage] {stage} benchmark | mode={mode} | concurrency={concurrency}"
    if details:
        message += f" | {details}"
    print(message)


def _sample_scope_label(max_samples: int | None) -> str:
    if max_samples is None:
        return "full SeedTTS EN set"
    return f"max_samples={max_samples}"


def _assert_full_seedtts_en_speed_results(
    results: dict,
    *,
    label: str,
    collector: MetricCheckCollector,
) -> None:
    per_request = results.get("per_request") or []
    collector.check(
        len(per_request) == SEEDTTS_EN_FULLSET_SAMPLES,
        f"{label} generated {len(per_request)}/{SEEDTTS_EN_FULLSET_SAMPLES} "
        "SeedTTS EN samples",
    )


def _assert_full_seedtts_en_wer_results(
    results: dict,
    *,
    label: str,
    collector: MetricCheckCollector,
) -> None:
    summary = results.get("summary") or {}
    total_samples = summary.get("total_samples")
    collector.check(
        total_samples == SEEDTTS_EN_FULLSET_SAMPLES,
        f"{label} WER total_samples={total_samples}, expected "
        f"{SEEDTTS_EN_FULLSET_SAMPLES}",
    )


@pytest.fixture(scope="module")
def dataset_repo() -> str:
    repo_id = DATASETS["seedtts"]
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


@pytest.fixture(scope="module", autouse=True)
def cleanup_generated_audio_fixture():
    yield
    for output_dirs in SPEED_OUTPUT_DIRS.values():
        for output_dir in output_dirs.values():
            audio_dir = Path(output_dir) / "audio"
            if audio_dir.exists():
                shutil.rmtree(audio_dir)


@pytest.fixture(scope="module")
def router_server(tmp_path_factory: pytest.TempPathFactory):
    """Start two TTS workers behind the router and wait until healthy."""
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=TTS_MODEL_PATH,
        model_name=TTS_MODEL_PATH,
        worker_extra_args="",
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="tts_router_logs",
    ) as router:
        yield router


@pytest.fixture(scope="module")
def consistency_stage_inputs(
    selected_tts_ci_stage: str,
    tmp_path_factory: pytest.TempPathFactory,
    selected_tts_concurrencies: tuple[int, ...],
    request: pytest.FixtureRequest,
) -> None:
    if selected_tts_ci_stage != TTS_STAGE_CONSISTENCY:
        return

    if _load_consistency_artifact_inputs(selected_tts_concurrencies):
        return

    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise AssertionError(
            "Stage 3 requires downloaded stage 1/2 speed artifacts when running in CI."
        )

    _generate_consistency_inputs(
        request,
        tmp_path_factory,
        selected_tts_concurrencies,
    )


@pytest.fixture(scope="module")
def wer_input_dirs(
    router_server: ManagedRouterHandle,
) -> dict[str, dict[int, str]]:
    """Reuse saved benchmark audio for WER after freeing the TTS server GPU."""
    router_server.stop()
    wait_for_gpu_memory_release()

    for output_dirs in SPEED_OUTPUT_DIRS.values():
        for output_dir in output_dirs.values():
            generated_path = Path(output_dir) / "generated.json"
            assert generated_path.exists(), f"WER metadata missing: {generated_path}"
    return SPEED_OUTPUT_DIRS


@pytest.mark.tts_stage(TTS_STAGE_NONSTREAM)
@pytest.mark.benchmark
def test_voice_cloning_non_streaming(
    router_server: ManagedRouterHandle,
    dataset_repo: str,
    tmp_path: Path,
    selected_tts_concurrencies: tuple[int, ...],
) -> None:
    print(f"\n[TTS benchmark] selected concurrency: {selected_tts_concurrencies}")
    for concurrency in selected_tts_concurrencies:
        _print_stage("TTS speed", "non-streaming", concurrency, "generate WAVs for WER")
        output_dir = _resolve_stage_output_dir(tmp_path, f"vc_nonstream_c{concurrency}")
        before_workers = router_get_json(router_server.port, "/workers")
        try:
            results = _run_benchmark(
                router_server.port,
                dataset_repo,
                output_dir,
                concurrency=concurrency,
            )
            checks = MetricCheckCollector(f"TTS non-streaming benchmark c{concurrency}")
            _assert_full_seedtts_en_speed_results(
                results,
                label=f"TTS non-stream c{concurrency}",
                collector=checks,
            )
            _assert_stage_used_all_router_workers(
                router_server=router_server,
                before_workers=before_workers,
                results=results,
                label=f"TTS non-stream c{concurrency}",
                collector=checks,
            )
        except Exception:
            print_router_diagnostics(router_server)
            raise
        _store_consistency_inputs(
            mode="non_stream",
            concurrency=concurrency,
            output_dir=output_dir,
            results=results,
            collector=checks,
        )
        checks.assert_all()


@pytest.mark.tts_stage(TTS_STAGE_STREAM)
@pytest.mark.benchmark
def test_voice_cloning_streaming(
    router_server: ManagedRouterHandle,
    dataset_repo: str,
    tmp_path: Path,
    selected_tts_concurrencies: tuple[int, ...],
) -> None:
    for concurrency in selected_tts_concurrencies:
        _print_stage(
            "TTS speed",
            "streaming",
            concurrency,
            f"{_sample_scope_label(STREAMING_BENCHMARK_MAX_SAMPLES)} | "
            "generate WAVs for WER",
        )
        output_dir = _resolve_stage_output_dir(tmp_path, f"vc_stream_c{concurrency}")
        before_workers = router_get_json(router_server.port, "/workers")
        try:
            results = _run_benchmark(
                router_server.port,
                dataset_repo,
                output_dir,
                concurrency=concurrency,
                max_samples=STREAMING_BENCHMARK_MAX_SAMPLES,
                stream=True,
            )
            checks = MetricCheckCollector(f"TTS streaming benchmark c{concurrency}")
            _assert_full_seedtts_en_speed_results(
                results,
                label=f"TTS stream c{concurrency}",
                collector=checks,
            )
            _assert_stage_used_all_router_workers(
                router_server=router_server,
                before_workers=before_workers,
                results=results,
                label=f"TTS stream c{concurrency}",
                collector=checks,
            )
        except Exception:
            print_router_diagnostics(router_server)
            raise
        _store_consistency_inputs(
            mode="stream",
            concurrency=concurrency,
            output_dir=output_dir,
            results=results,
            collector=checks,
        )
        checks.assert_all()


@pytest.mark.tts_stage(TTS_STAGE_CONSISTENCY)
@pytest.mark.benchmark
def test_voice_cloning_streaming_consistency(
    consistency_stage_inputs: None,
    selected_tts_concurrencies: tuple[int, ...],
) -> None:
    checks = MetricCheckCollector("TTS streaming consistency")
    for concurrency in selected_tts_concurrencies:
        ns = PER_REQUEST_STORE.get(f"vc_nonstream_c{concurrency}")
        st = PER_REQUEST_STORE.get(f"vc_stream_c{concurrency}")
        if ns is None:
            checks.fail(f"vc_nonstream_c{concurrency} results missing")
        if st is None:
            checks.fail(f"vc_stream_c{concurrency} results missing")
        if ns is None or st is None:
            continue
        assert_streaming_consistency(
            ns,
            st,
            expected_stream_count=len(ns),
            max_failed_requests=0,
            collector=checks,
        )
    checks.assert_all()


@pytest.mark.tts_stage(TTS_STAGE_NONSTREAM)
@pytest.mark.benchmark
def test_voice_cloning_wer(
    wer_input_dirs: dict[str, dict[int, str]],
    dataset_repo: str,
    selected_tts_concurrencies: tuple[int, ...],
    omni_whisper_wer_router: ManagedRouterHandle,
) -> None:
    checks = MetricCheckCollector("TTS non-streaming WER")
    for concurrency in selected_tts_concurrencies:
        _print_stage(
            "WER",
            "non-streaming",
            concurrency,
            "transcribe speed-stage WAVs",
        )
        results = _run_wer_transcribe(
            dataset_repo,
            wer_input_dirs["non_stream"][concurrency],
            whisper_router_port=omni_whisper_wer_router.port,
        )
        _assert_full_seedtts_en_wer_results(
            results,
            label=f"TTS non-stream c{concurrency}",
            collector=checks,
        )
        assert_wer_results(
            results,
            VC_WER_CORPUS_THRESHOLD,
            collector=checks,
        )
    checks.assert_all()


@pytest.mark.tts_stage(TTS_STAGE_NONSTREAM)
@pytest.mark.benchmark
def test_voice_cloning_similarity(
    wer_input_dirs: dict[str, dict[int, str]],
    dataset_repo: str,
    similarity_checkpoint: str | None,
    selected_tts_concurrencies: tuple[int, ...],
) -> None:
    checks = MetricCheckCollector("TTS non-streaming speaker similarity")
    for concurrency in selected_tts_concurrencies:
        _print_stage(
            "SIM",
            "non-streaming",
            concurrency,
            "score speed-stage WAVs",
        )
        results = _run_similarity(
            dataset_repo,
            wer_input_dirs["non_stream"][concurrency],
            similarity_checkpoint,
            max_samples=TTS_SIMILARITY_MAX_SAMPLES,
        )
        _assert_similarity_results(results, VC_SIMILARITY_MEAN_MIN, collector=checks)
    checks.assert_all()


@pytest.mark.tts_stage(TTS_STAGE_NONSTREAM)
@pytest.mark.benchmark
def test_voice_cloning_utmos(
    wer_input_dirs: dict[str, dict[int, str]],
    selected_tts_concurrencies: tuple[int, ...],
) -> None:
    checks = MetricCheckCollector("TTS non-streaming UTMOS")
    for concurrency in selected_tts_concurrencies:
        _print_stage("UTMOS", "non-streaming", concurrency, "score speed-stage WAVs")
        results = _run_utmos(wer_input_dirs["non_stream"][concurrency])
        _assert_utmos_results(results, VC_UTMOS_MEAN_MIN, collector=checks)
    checks.assert_all()


@pytest.mark.tts_stage(TTS_STAGE_STREAM)
@pytest.mark.benchmark
def test_voice_cloning_streaming_wer(
    wer_input_dirs: dict[str, dict[int, str]],
    dataset_repo: str,
    selected_tts_concurrencies: tuple[int, ...],
    omni_whisper_wer_router: ManagedRouterHandle,
) -> None:
    checks = MetricCheckCollector("TTS streaming WER")
    for concurrency in selected_tts_concurrencies:
        _print_stage(
            "WER",
            "streaming",
            concurrency,
            f"transcribe {_sample_scope_label(STREAMING_BENCHMARK_MAX_SAMPLES)} "
            "speed-stage WAVs",
        )
        results = _run_wer_transcribe(
            dataset_repo,
            wer_input_dirs["stream"][concurrency],
            stream=True,
            whisper_router_port=omni_whisper_wer_router.port,
        )
        _assert_full_seedtts_en_wer_results(
            results,
            label=f"TTS stream c{concurrency}",
            collector=checks,
        )
        assert_wer_results(
            results,
            VC_STREAM_WER_CORPUS_THRESHOLD,
            collector=checks,
        )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
