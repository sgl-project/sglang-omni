# SPDX-License-Identifier: Apache-2.0
"""Streaming first-audio latency for the Qwen3-TTS CI arms.

One TTS worker behind the router takes open-loop Poisson arrivals at each point
of the preset's ``latency`` table, and the stage gates the median first
playable latency against the calibrated references in ``tts_ci_config.py``.

It is a module of its own so that its worker is the only server on the GPU:
the stages in ``test_tts_ci.py`` keep a two-worker router alive for the whole
of their module.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS, download_dataset
from sglang_omni.models.qwen3_tts.config import Qwen3TTSPipelineConfig
from tests.test_model.omni_router_utils import (
    CiRouterTopology,
    ManagedRouterHandle,
    launch_managed_router,
    print_router_diagnostics,
    router_get_json,
)
from tests.test_model.test_tts_ci import (
    PRESET,
    STARTUP_TIMEOUT,
    TTS_CI_PRESET,
    TTS_MODEL_PATH,
    TTS_WORKER_EXTRA_ARGS,
    assert_stage_used_all_router_workers,
    print_stage,
    resolve_stage_output_dir,
    run_benchmark,
)
from tests.utils import (
    MetricCheckCollector,
    assert_open_loop_latency_results,
    print_latency_point,
)

# note (luojiaxuan): every run offers the same Poisson arrival sequence, so run
# to run spread comes from the server and not from a different offered load.
_ARRIVAL_SEED = 0


@pytest.fixture(scope="module")
def dataset_repo() -> str:
    repo_id = DATASETS["seedtts"]
    download_dataset(repo_id, quiet=True)
    return repo_id


@pytest.fixture(scope="module")
def single_worker_router_server(tmp_path_factory: pytest.TempPathFactory):
    """One TTS worker behind the router, for latency at a fixed offered load.

    Two workers would split 20 rps into 10 rps each, which is not the operating
    point the first-audio work is measured at.
    """
    with launch_managed_router(
        tmp_path_factory=tmp_path_factory,
        model_path=TTS_MODEL_PATH,
        model_name=TTS_MODEL_PATH,
        worker_extra_args=f"{TTS_WORKER_EXTRA_ARGS} {PRESET.worker_extra_args}".strip(),
        router_topology=CiRouterTopology.TTS,
        num_workers=1,
        num_gpus_per_worker=PRESET.num_gpus_per_worker,
        wait_timeout=STARTUP_TIMEOUT,
        log_prefix="tts_latency_router_logs",
        named_voice=not PRESET.voice_clone,
    ) as router:
        yield router


def worker_admission_cap() -> int:
    """The worker's ``max_running_requests``, which bounds the client's slot cap."""
    match = re.search(
        r"--tts_engine\.engine\.max_running_requests (\d+)",
        f"{TTS_WORKER_EXTRA_ARGS} {PRESET.worker_extra_args}",
    )
    if match is not None:
        return int(match.group(1))
    return Qwen3TTSPipelineConfig.generation_admission_defaults()[
        "max_running_requests"
    ]


@pytest.mark.benchmark
def test_streaming_first_audio_latency(
    single_worker_router_server: ManagedRouterHandle,
    dataset_repo: str,
    tmp_path: Path,
) -> None:
    latency = TTS_CI_PRESET.latency
    if latency is None:
        pytest.skip(f"preset {PRESET.model_path} has no latency points")
    client_cap = worker_admission_cap()
    checks = MetricCheckCollector("TTS streaming first-audio latency")
    for point in latency.points:
        label = f"TTS latency stream {point.request_rate:g} rps"
        print_stage(
            "TTS latency", "streaming", client_cap, f"{point.request_rate:g} rps"
        )
        output_dir = resolve_stage_output_dir(
            tmp_path, f"vc_latency_r{point.request_rate:g}"
        )
        before_workers = router_get_json(
            single_worker_router_server.port, "/diagnostics"
        )
        try:
            results = run_benchmark(
                single_worker_router_server.port,
                dataset_repo,
                output_dir,
                concurrency=client_cap,
                max_samples=point.samples,
                stream=True,
                request_rate=point.request_rate,
                arrival_seed=_ARRIVAL_SEED,
            )
        except Exception:
            print_router_diagnostics(single_worker_router_server)
            raise
        print_latency_point(results["summary"], label=label)
        assert_open_loop_latency_results(
            results, samples=point.samples, label=label, collector=checks
        )
        assert_stage_used_all_router_workers(
            router_server=single_worker_router_server,
            before_workers=before_workers,
            results=results,
            label=label,
            collector=checks,
            expected_workers=1,
        )
        if latency.calibrated:
            summary = results["summary"]
            median = summary["audio_ttfp_from_arrival_median_s"]
            checks.check(
                median <= point.ttfp_median_max_s,
                f"{label}: first playable median from arrival {median} s exceeds "
                f"{point.ttfp_median_max_s} s",
            )
    checks.assert_all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
