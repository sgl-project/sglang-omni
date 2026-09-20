# SPDX-License-Identifier: Apache-2.0
"""Model selection and calibration admission for shared Omni CI benchmarks."""

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig
from benchmarks.eval.benchmark_omni_seedtts import OmniSeedttsBenchmarkConfig
from tests.test_model import test_qwen3_omni_mmmu_ci as mmmu_ci
from tests.test_model import test_qwen3_omni_tts_ci as tts_ci
from tests.test_model import test_qwen3_omni_videoamme_talker_tp2_ci as videoamme_ci
from tests.test_model.omni_ci_config import OMNI_CI_PRESETS, select_omni_ci_preset


def test_omni_ci_selection_honors_explicit_model_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNI_CI_MODEL", raising=False)
    assert select_omni_ci_preset()[0] == "qwen3-omni"
    monkeypatch.setenv("OMNI_CI_MODEL", "minicpmo")
    name, preset = select_omni_ci_preset()
    assert name == "minicpmo"
    assert preset.model_path == "openbmb/MiniCPM-o-4_5"
    assert select_omni_ci_preset("qwen3-omni")[0] == "qwen3-omni"
    with pytest.raises(ValueError, match="Unsupported OMNI_CI_MODEL"):
        select_omni_ci_preset("missing-model")


@pytest.mark.parametrize("stage", tuple(OMNI_CI_PRESETS["minicpmo"].thresholds))
def test_uncalibrated_minicpmo_cannot_qualify(stage: str) -> None:
    thresholds = replace(
        OMNI_CI_PRESETS["minicpmo"].thresholds[stage], calibrated=False
    )
    with pytest.raises(AssertionError, match="thresholds are uncalibrated"):
        thresholds.require_calibrated("minicpmo", stage)
    replace(thresholds, calibrated=True).require_calibrated("minicpmo", stage)
    OMNI_CI_PRESETS["qwen3-omni"].thresholds[stage].require_calibrated(
        "qwen3-omni", stage
    )


@pytest.mark.parametrize(
    (
        "model_name",
        "calibrated",
        "failed",
        "traffic_error",
        "accuracy",
        "throughput",
        "expected_error",
    ),
    [
        ("qwen3-omni", True, 0, None, 1.0, 100.0, None),
        ("minicpmo", False, 0, None, 1.0, 100.0, "thresholds are uncalibrated"),
        ("minicpmo", False, 1, None, 1.0, 100.0, "MMMU had 1/50 failed requests"),
        (
            "minicpmo",
            False,
            0,
            "worker 1 served no requests",
            1.0,
            100.0,
            "worker 1 served no requests",
        ),
        ("minicpmo", True, 0, None, 1.0, 100.0, None),
        ("minicpmo", True, 1, None, 1.0, 100.0, "MMMU had 1/50 failed requests"),
        (
            "minicpmo",
            True,
            0,
            "worker 1 served no requests",
            1.0,
            100.0,
            "worker 1 served no requests",
        ),
        ("minicpmo", True, 0, None, 0.5, 100.0, "MMMU accuracy"),
        ("minicpmo", True, 0, None, 1.0, 1.0, "throughput_qps"),
    ],
)
def test_mmmu_collects_selected_model_metrics_before_calibration_gate(
    model_name: str,
    calibrated: bool,
    failed: int,
    traffic_error: str | None,
    accuracy: float,
    throughput: float,
    expected_error: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preset = OMNI_CI_PRESETS[model_name]
    if model_name == "minicpmo":
        # note (wenyao): Synthetic gates keep admission tests independent of calibration.
        thresholds = replace(
            preset.thresholds["mmmu"],
            calibrated=calibrated,
            accuracy=0.6,
            speed={
                16: {
                    "throughput_qps_min": 2.0,
                    "output_tok_per_req_s_min": 10.0,
                    "latency_mean_s_max": 1.0,
                }
            },
        )
        preset = replace(preset, thresholds={**preset.thresholds, "mmmu": thresholds})
    configs: list[MMMUEvalConfig] = []

    async def evaluate(config: MMMUEvalConfig) -> dict[str, object]:
        configs.append(config)
        return {
            "summary": {"accuracy": accuracy, "failed": failed, "total_samples": 50},
            "speed": {
                "throughput_qps": throughput,
                "output_tok_per_req_s": 100.0,
                "latency_mean_s": 0.1,
            },
        }

    print_speed = Mock()
    served = Mock(side_effect=AssertionError(traffic_error) if traffic_error else None)
    monkeypatch.setattr(mmmu_ci, "run_mmmu_eval", evaluate)
    monkeypatch.setattr(mmmu_ci, "print_mmmu_accuracy_summary", Mock())
    monkeypatch.setattr(mmmu_ci, "print_speed_summary", print_speed)
    monkeypatch.setattr(
        mmmu_ci,
        "router_worker_traffic_guard",
        lambda *args, **kwargs: nullcontext(SimpleNamespace(assert_served=served)),
    )
    expected = (
        pytest.raises(AssertionError, match=expected_error)
        if expected_error is not None
        else nullcontext()
    )
    with expected:
        mmmu_ci.test_mmmu_accuracy_and_speed(
            preset, SimpleNamespace(port=8000), tmp_path
        )
    assert len(configs) == 1
    assert configs[0].model == model_name
    assert configs[0].max_concurrency == 16
    assert configs[0].warmup == 2
    print_speed.assert_called_once()
    served.assert_called_once_with(min_total_requests=50)


@pytest.mark.parametrize(
    ("model_name", "reference_field"),
    [("qwen3-omni", "audios"), ("minicpmo", "audio.ref_audio")],
)
def test_seedtts_uses_model_specific_reference_audio(
    model_name: str, reference_field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs: list[OmniSeedttsBenchmarkConfig] = []

    async def benchmark(config: OmniSeedttsBenchmarkConfig) -> dict[str, object]:
        configs.append(config)
        return {"summary": {}, "per_request": []}

    monkeypatch.setattr(tts_ci, "run_omni_seedtts_benchmark", benchmark)
    tts_ci._run_benchmark(OMNI_CI_PRESETS[model_name], 8000, "dataset", "results")
    assert configs[0].model == model_name
    assert configs[0].reference_audio_field == reference_field
    assert configs[0].voice_clone is True
    assert configs[0].stream is False


def test_minicpmo_omits_qwen_specific_kernel_and_tp2_assertions() -> None:
    preset = OMNI_CI_PRESETS["minicpmo"]
    with pytest.raises(pytest.skip.Exception, match="Qwen3-Omni-specific"):
        tts_ci.test_speech_prefill_graph_replays_in_existing_tts_stage(preset, None)
    with pytest.raises(pytest.skip.Exception, match="MiniCPM-o uses DP2"):
        videoamme_ci.test_thinker_tp2_actually_applied(preset, None)
