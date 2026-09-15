"""Performance selection counts every offered request and joint SLOs."""

import pytest

from sglang_omni.restage.evaluation import SLO, Observation, evaluate


def test_failed_and_missing_requests_remain_in_good_fraction():
    requests = [
        Observation("a", 0, 0.5, True, True, first_output_s=0.1),
        Observation("b", 1, 1.5, True, True, first_output_s=1.1),
        Observation("c", 2, None, False, None),
    ]
    result = evaluate(
        requests,
        SLO(max_ttfa_s=0.3, min_good_fraction=0.99),
        expected_requests=4,
        elapsed_s=10,
    )
    assert result.good_fraction == 0.5
    assert result.goodput_qps == 0.2
    assert result.feasible is False
    assert result.missing_requests == 1


def test_latency_and_quality_must_pass_on_the_same_requests():
    requests = [
        Observation("fast_bad", 0, 0.1, True, False),
        Observation("slow_good", 0, 2, True, True),
    ]
    result = evaluate(requests, SLO(max_latency_s=1), expected_requests=2, elapsed_s=2)
    assert result.good_requests == 0
    assert result.completed_qps == 1
    assert not result.feasible


def test_missing_streaming_measurements_cannot_pass_streaming_slo():
    requests = [Observation("nonstream", 0, 0.5, True, True, audio_duration_s=2)]
    result = evaluate(requests, SLO(max_ttfa_s=1), expected_requests=1, elapsed_s=1)
    assert not result.feasible
    assert result.violations["ttfa_missing_or_invalid"] == 1


def test_scheduled_arrival_is_latency_origin_and_rtf_uses_audio_duration():
    request = Observation(
        "a",
        10,
        12,
        True,
        True,
        first_output_s=11.2,
        audio_duration_s=4,
        max_playback_underrun_s=0,
    )
    result = evaluate(
        [request], SLO(max_ttfa_s=1, max_rtf=0.6), expected_requests=1, elapsed_s=2
    )
    assert result.violations == {"ttfa": 1}
    assert not result.feasible


def test_no_requests_or_unmeasured_quality_never_produces_a_winner():
    with pytest.raises(ValueError, match="expected_requests"):
        evaluate([], SLO(max_latency_s=1), expected_requests=0, elapsed_s=1)
    result = evaluate(
        [Observation("a", 0, 0.1, True, None)],
        SLO(max_latency_s=1),
        expected_requests=1,
        elapsed_s=1,
    )
    assert not result.feasible


def test_joint_good_requests_make_feasible_result():
    result = evaluate(
        [
            Observation(
                "a",
                0,
                0.5,
                True,
                True,
                first_output_s=0.1,
                audio_duration_s=2,
                max_playback_underrun_s=0,
            )
        ],
        SLO(max_ttfa_s=0.2, max_rtf=1, max_underrun_s=0),
        expected_requests=1,
        elapsed_s=1,
    )
    assert result.feasible
    assert result.goodput_qps == 1


@pytest.mark.parametrize("first", [float("nan"), float("inf"), -1, 2])
def test_invalid_first_output_timestamps_fail_ttfa(first):
    result = evaluate(
        [Observation("a", 0, 1, True, True, first_output_s=first)],
        SLO(max_ttfa_s=3),
        expected_requests=1,
        elapsed_s=1,
    )
    assert not result.feasible
    assert result.violations["ttfa_missing_or_invalid"] == 1


def test_duplicate_requests_cannot_inflate_goodput():
    item = Observation("same", 0, 0.1, True, True)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate([item, item], SLO(max_latency_s=1), expected_requests=2, elapsed_s=1)


def test_underrun_is_a_joint_constraint():
    result = evaluate(
        [
            Observation(
                "a", 0, 1, True, True, first_output_s=0.1, max_playback_underrun_s=0.2
            )
        ],
        SLO(max_ttfa_s=1, max_underrun_s=0),
        expected_requests=1,
        elapsed_s=1,
    )
    assert not result.feasible
    assert result.violations == {"underrun": 1}


def test_corpus_quality_gate_overrides_a_passing_request_fraction():
    item = Observation("a", 0, 0.1, True, True)
    passing = evaluate([item], SLO(max_latency_s=1), expected_requests=1, elapsed_s=1)
    assert passing.feasible and passing.corpus_quality_pass is None
    gated = evaluate(
        [item],
        SLO(max_latency_s=1),
        expected_requests=1,
        elapsed_s=1,
        corpus_wer=0.5,
        corpus_quality_pass=False,
    )
    assert not gated.feasible
    assert gated.good_requests == 1 and gated.corpus_wer == 0.5
