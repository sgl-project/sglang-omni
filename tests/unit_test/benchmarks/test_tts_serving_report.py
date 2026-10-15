import pytest

from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import (
    build_results_report,
    summarize_output_throughput,
)
from benchmarks.tts_serving.spec import BenchmarkSpec


def scenario_result(
    scenario_id: str,
    *,
    actual_start_s: float | None,
    completed_s: float | None,
    audio_duration_s: float = 0.0,
    completion_tokens: int | None = None,
    success: bool = True,
    status: str = "ok",
    category: str = "traffic",
    cancelled: bool = False,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        endpoint="speech",
        category=category,
        stage_id="stage-1",
        success=success,
        status=status,
        actual_start_s=actual_start_s,
        completed_s=completed_s,
        audio_duration_s=audio_duration_s,
        completion_tokens=completion_tokens,
        was_cancelled=cancelled,
    )


def test_output_throughput_uses_all_traffic_observation_window() -> None:
    results = [
        scenario_result(
            "audio-1",
            actual_start_s=0.0,
            completed_s=3.0,
            audio_duration_s=6.0,
            completion_tokens=30,
        ),
        scenario_result(
            "audio-2",
            actual_start_s=1.0,
            completed_s=5.0,
            audio_duration_s=4.0,
            completion_tokens=20,
        ),
        scenario_result(
            "late-failure", actual_start_s=2.0, completed_s=10.0, success=False
        ),
        scenario_result(
            "capability-probe",
            actual_start_s=-100.0,
            completed_s=100.0,
            audio_duration_s=100.0,
            completion_tokens=100,
            category="capability_probe",
        ),
    ]

    summary = summarize_output_throughput(results)

    assert summary == {
        "observation_wall_time_s": 10.0,
        "audio_throughput_s_per_s": 1.0,
        "completion_tokens_per_s": 5.0,
        "audio_duration_sample_count": 2,
        "completion_token_sample_count": 2,
    }


def test_output_throughput_is_null_without_valid_window_or_token_samples() -> None:
    missing_window = summarize_output_throughput(
        [
            scenario_result(
                "missing",
                actual_start_s=0.0,
                completed_s=None,
                audio_duration_s=4.0,
                completion_tokens=8,
            )
        ]
    )
    assert missing_window == {
        "observation_wall_time_s": None,
        "audio_throughput_s_per_s": None,
        "completion_tokens_per_s": None,
        "audio_duration_sample_count": 1,
        "completion_token_sample_count": 1,
    }

    nonpositive_window = summarize_output_throughput(
        [
            scenario_result(
                "instant",
                actual_start_s=2.0,
                completed_s=2.0,
                audio_duration_s=4.0,
                completion_tokens=8,
            )
        ]
    )
    assert nonpositive_window == {
        "observation_wall_time_s": None,
        "audio_throughput_s_per_s": None,
        "completion_tokens_per_s": None,
        "audio_duration_sample_count": 1,
        "completion_token_sample_count": 1,
    }

    missing_tokens = summarize_output_throughput(
        [
            scenario_result(
                "audio", actual_start_s=0.0, completed_s=2.0, audio_duration_s=4.0
            )
        ]
    )
    assert missing_tokens["observation_wall_time_s"] == 2.0
    assert missing_tokens["audio_throughput_s_per_s"] == 2.0
    assert missing_tokens["completion_tokens_per_s"] is None
    assert missing_tokens["completion_token_sample_count"] == 0


def test_output_throughput_counts_only_successful_non_cancelled_audio() -> None:
    results = [
        scenario_result(
            "expected-error",
            actual_start_s=0.0,
            completed_s=4.0,
            audio_duration_s=20.0,
            success=False,
            status="expected_error",
        ),
        scenario_result(
            "cancelled",
            actual_start_s=1.0,
            completed_s=5.0,
            audio_duration_s=20.0,
            cancelled=True,
        ),
        scenario_result(
            "success",
            actual_start_s=2.0,
            completed_s=6.0,
            audio_duration_s=6.0,
            completion_tokens=12,
        ),
    ]

    summary = summarize_output_throughput(results)

    assert summary["observation_wall_time_s"] == 6.0
    assert summary["audio_throughput_s_per_s"] == 1.0
    assert summary["completion_tokens_per_s"] == 2.0
    assert summary["audio_duration_sample_count"] == 1
    assert summary["completion_token_sample_count"] == 1


def test_output_throughput_distinguishes_censored_audio_from_all_failures() -> None:
    no_audio = summarize_output_throughput(
        [
            scenario_result(
                "empty-audio", actual_start_s=0.0, completed_s=4.0, completion_tokens=3
            )
        ]
    )
    assert no_audio == {
        "observation_wall_time_s": 4.0,
        "audio_throughput_s_per_s": None,
        "completion_tokens_per_s": 0.75,
        "audio_duration_sample_count": 0,
        "completion_token_sample_count": 1,
    }

    all_failed = summarize_output_throughput(
        [
            scenario_result(
                "failure-1", actual_start_s=0.0, completed_s=2.0, success=False
            ),
            scenario_result(
                "failure-2", actual_start_s=1.0, completed_s=5.0, success=False
            ),
        ]
    )
    assert all_failed["observation_wall_time_s"] == 5.0
    assert all_failed["audio_throughput_s_per_s"] == 0.0
    assert all_failed["completion_tokens_per_s"] is None


@pytest.mark.parametrize(
    ("unobserved_start_s", "unobserved_completion_s"),
    [(None, 1.5), (1.0, None)],
)
def test_partial_traffic_timestamps_do_not_invent_throughput_denominator(
    unobserved_start_s: float | None, unobserved_completion_s: float | None
) -> None:
    summary = summarize_output_throughput(
        [
            scenario_result(
                "observed",
                actual_start_s=0.0,
                completed_s=3.0,
                audio_duration_s=4.0,
                completion_tokens=8,
            ),
            scenario_result(
                "unobserved",
                actual_start_s=unobserved_start_s,
                completed_s=unobserved_completion_s,
                audio_duration_s=10.0,
                completion_tokens=20,
            ),
        ]
    )

    assert summary["observation_wall_time_s"] is None
    assert summary["audio_throughput_s_per_s"] is None
    assert summary["completion_tokens_per_s"] is None
    assert summary["audio_duration_sample_count"] == 2
    assert summary["completion_token_sample_count"] == 2


def test_partial_token_usage_does_not_estimate_unreported_output() -> None:
    summary = summarize_output_throughput(
        [
            scenario_result(
                "reported",
                actual_start_s=0.0,
                completed_s=2.0,
                audio_duration_s=2.0,
                completion_tokens=20,
            ),
            scenario_result(
                "unreported",
                actual_start_s=1.0,
                completed_s=4.0,
                audio_duration_s=6.0,
            ),
        ]
    )

    assert summary["observation_wall_time_s"] == 4.0
    assert summary["audio_throughput_s_per_s"] == 2.0
    assert summary["completion_tokens_per_s"] == 5.0
    assert summary["audio_duration_sample_count"] == 2
    assert summary["completion_token_sample_count"] == 1


@pytest.mark.parametrize(
    ("audio_duration_s", "audio_throughput_s_per_s", "audio_duration_sample_count"),
    [
        (0.0, None, 1),
        (-3.0, None, 1),
        (float("nan"), None, 1),
        (float("inf"), None, 1),
        (4.0, 2.0, 2),
    ],
)
def test_report_does_not_treat_censored_successful_audio_as_complete_output(
    audio_duration_s: float,
    audio_throughput_s_per_s: float | None,
    audio_duration_sample_count: int,
) -> None:
    report = build_results_report(
        BenchmarkSpec(base_url="http://127.0.0.1:1", model_name="report-test"),
        [
            scenario_result(
                "observed",
                actual_start_s=0.0,
                completed_s=2.0,
                audio_duration_s=4.0,
                completion_tokens=8,
            ),
            scenario_result(
                "possibly-censored",
                actual_start_s=1.0,
                completed_s=4.0,
                audio_duration_s=audio_duration_s,
                completion_tokens=12,
            ),
        ],
    )
    expected = {
        "observation_wall_time_s": 4.0,
        "audio_throughput_s_per_s": audio_throughput_s_per_s,
        "completion_tokens_per_s": 5.0,
        "audio_duration_sample_count": audio_duration_sample_count,
        "completion_token_sample_count": 2,
    }

    assert report["metrics"]["output_throughput"] == expected
    assert report["metrics"]["by_stage"]["stage-1"]["output_throughput"] == expected
    assert report["metrics"]["by_endpoint"]["speech"]["output_throughput"] == expected


def test_report_observes_gaps_between_stages_without_scaling_output_by_counts() -> None:
    report = build_results_report(
        BenchmarkSpec(base_url="http://127.0.0.1:1", model_name="report-test"),
        [
            ScenarioResult(
                scenario_id="batch",
                endpoint="batch",
                category="traffic",
                stage_id="stage-a",
                success=True,
                status="ok",
                actual_start_s=1000000.0,
                completed_s=1000002.0,
                audio_duration_s=8.0,
                completion_tokens=0,
                batch_size=4,
                load_concurrency=4,
                audio_bytes=384000,
            ),
            ScenarioResult(
                scenario_id="later-speech",
                endpoint="speech",
                category="traffic",
                stage_id="stage-b",
                success=True,
                status="ok",
                actual_start_s=1000010.0,
                completed_s=1000012.0,
                audio_duration_s=4.0,
                completion_tokens=24,
                load_concurrency=1,
                audio_bytes=999999999,
            ),
        ],
    )

    assert report["metrics"]["output_throughput"] == {
        "observation_wall_time_s": 12.0,
        "audio_throughput_s_per_s": 1.0,
        "completion_tokens_per_s": 2.0,
        "audio_duration_sample_count": 2,
        "completion_token_sample_count": 2,
    }
    assert report["metrics"]["by_stage"]["stage-a"]["output_throughput"] == {
        "observation_wall_time_s": 2.0,
        "audio_throughput_s_per_s": 4.0,
        "completion_tokens_per_s": 0.0,
        "audio_duration_sample_count": 1,
        "completion_token_sample_count": 1,
    }
    assert report["metrics"]["by_stage"]["stage-b"]["output_throughput"] == {
        "observation_wall_time_s": 2.0,
        "audio_throughput_s_per_s": 2.0,
        "completion_tokens_per_s": 12.0,
        "audio_duration_sample_count": 1,
        "completion_token_sample_count": 1,
    }
