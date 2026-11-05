# SPDX-License-Identifier: Apache-2.0
"""The interval the two SLOs leave: a continuity floor, a first-audio ceiling, and the wait cap."""
from __future__ import annotations

import pytest

from sglang_omni.models.fishaudio_drama3.prefill_cadence import (
    AUDIO_FRAME_SECONDS,
    PrefillCadence,
    PrefillCadenceConfig,
    critical_frame_interval_s,
    worst_stall_s,
)

DEFAULT_WINDOWS = (4, 10, 12, 32)
FOLLOWUP_WINDOW = 64


def cadence(**overrides: object) -> PrefillCadence:
    return PrefillCadence(config=PrefillCadenceConfig(**overrides))


def test_the_ramp_stalls_by_a_frame_even_at_twice_real_time() -> None:
    """The small first window buys 4 frames of audio while the second takes 10 to make."""
    stall_s = worst_stall_s(
        AUDIO_FRAME_SECONDS / 2, DEFAULT_WINDOWS, FOLLOWUP_WINDOW, 0.006, 120
    )
    assert stall_s == pytest.approx(AUDIO_FRAME_SECONDS, abs=0.002)


def test_a_stream_whose_frames_come_out_fast_enough_never_stalls() -> None:
    stall_s = worst_stall_s(
        AUDIO_FRAME_SECONDS / 8, DEFAULT_WINDOWS, FOLLOWUP_WINDOW, 0.006, 120
    )
    assert stall_s == pytest.approx(0.0, abs=1e-9)


def test_the_critical_interval_is_the_longest_one_inside_the_stall_budget() -> None:
    interval_s = critical_frame_interval_s(0.1, DEFAULT_WINDOWS, FOLLOWUP_WINDOW)
    assert worst_stall_s(interval_s, DEFAULT_WINDOWS, FOLLOWUP_WINDOW, 0.006, 120) <= 0.1
    assert (
        worst_stall_s(interval_s * 1.1, DEFAULT_WINDOWS, FOLLOWUP_WINDOW, 0.006, 120)
        > 0.1
    )


def test_a_looser_stall_budget_tolerates_a_longer_frame_interval() -> None:
    assert critical_frame_interval_s(
        0.4, DEFAULT_WINDOWS, FOLLOWUP_WINDOW
    ) > critical_frame_interval_s(0.1, DEFAULT_WINDOWS, FOLLOWUP_WINDOW)


def test_no_interval_is_chosen_before_a_step_and_a_prefill_are_measured() -> None:
    assert cadence().choose_interval() == 0


def test_the_continuity_floor_is_the_interval_that_hides_the_prefill() -> None:
    policy = cadence(max_stall_s=0.1, continuity_margin_s=0.0)
    # The floor is the shortest interval whose step + prefill / interval fits the critical interval.
    headroom_s = policy.critical_interval_s - 0.012
    assert policy.continuity_interval(0.012, 0.060) == pytest.approx(
        -(-0.060 // headroom_s)
    )


def test_a_step_longer_than_the_critical_interval_asks_for_the_longest_interval() -> None:
    policy = cadence(longest_interval=32)
    assert policy.continuity_interval(policy.critical_interval_s + 0.001, 0.060) == 32


def test_the_first_audio_ceiling_is_what_the_slo_leaves_for_waiting() -> None:
    policy = cadence(
        first_audio_slo_s=0.5, first_window=4, vocoder_margin_s=0.03, longest_interval=64
    )
    # 500 ms - 60 ms prefill - 4 x 12 ms first window - 30 ms vocoder = 362 ms of waiting.
    assert policy.first_audio_interval(0.012, 0.060) == int(0.362 / 0.012)


def test_a_prefill_longer_than_the_first_audio_slo_leaves_no_waiting() -> None:
    policy = cadence(first_audio_slo_s=0.5)
    assert policy.first_audio_interval(0.012, 0.600) == 0


def test_the_chosen_interval_sits_on_the_first_audio_ceiling() -> None:
    policy = cadence(longest_interval=64)
    policy.observe_decode_step(0.012, 48)
    policy.observe_prefill(0.060, 8)
    assert policy.choose_interval() == policy.first_audio_interval(0.012, 0.060)


def test_continuity_wins_when_the_ceiling_falls_below_the_floor() -> None:
    # A step at the critical interval leaves no headroom, so the floor is the longest interval,
    # while a prefill this long leaves almost no room under the first-audio SLO.
    policy = cadence(longest_interval=16, first_audio_slo_s=0.5)
    policy.observe_decode_step(policy.critical_interval_s, 128)
    policy.observe_prefill(0.400, 32)
    assert policy.choose_interval() == 16
    assert policy.conflicts == 1


def test_a_wait_is_held_while_the_running_batch_has_no_room() -> None:
    policy = cadence(release_min_admit=1)
    policy.observe_decode_step(0.012, 48)
    policy.observe_prefill(0.060, 8)
    assert not policy.should_release(waiting_count=4, oldest_wait_s=10.0, free_slots=0)
    assert policy.releases["held_full"] == 1


def test_a_wait_past_the_cap_releases_the_interval() -> None:
    policy = cadence(release_min_admit=1)
    policy.observe_decode_step(0.012, 48)
    policy.observe_prefill(0.060, 8)
    assert policy.wait_cap_s() == pytest.approx(0.5 - 0.060 - 0.048 - 0.03 - 0.012)
    assert policy.should_release(
        waiting_count=4, oldest_wait_s=policy.wait_cap_s(), free_slots=4
    )
    assert not policy.should_release(
        waiting_count=4, oldest_wait_s=policy.wait_cap_s() / 2, free_slots=4
    )


def test_the_report_carries_the_step_and_prefill_tables_and_clears_them() -> None:
    policy = cadence()
    policy.observe_decode_step(0.010, 44)
    policy.observe_decode_step(0.014, 45)
    policy.observe_prefill(0.060, 8)
    report = policy.report()
    assert "decode ms by batch {40: (2, 12.0)}" in report
    assert "prefill ms by cohort {8: (1, 60.0)}" in report
    assert "decode ms by batch {}" in policy.report()


def test_the_longest_only_ablation_ignores_both_bounds_but_keeps_the_wait_cap() -> None:
    policy = cadence(longest_interval=32, is_longest_only=True)
    assert policy.choose_interval() == 32
    policy.observe_decode_step(0.012, 48)
    policy.observe_prefill(0.060, 8)
    assert policy.choose_interval() == 32
    assert policy.wait_cap_s() == pytest.approx(0.5 - 0.060 - 0.048 - 0.03 - 0.012)


def test_the_prefill_fit_reads_the_cohort_a_prefill_will_admit() -> None:
    policy = cadence()
    policy.observe_prefill(0.030, 4)
    policy.observe_prefill(0.058, 14)
    assert policy.prefill_estimate_s(4) == pytest.approx(0.030)
    assert policy.prefill_estimate_s(14) == pytest.approx(0.058)
    assert policy.prefill_estimate_s(24) == pytest.approx(0.086, abs=1e-6)


def test_the_wait_cap_shrinks_when_more_requests_are_waiting() -> None:
    policy = cadence()
    policy.observe_decode_step(0.012, 48)
    policy.observe_prefill(0.030, 4)
    policy.observe_prefill(0.058, 14)
    assert policy.wait_cap_s(4) > policy.wait_cap_s(24)


def test_a_light_load_takes_a_short_interval_and_a_heavy_one_a_long_one() -> None:
    light = cadence(longest_interval=64, is_latency_driven=True)
    heavy = cadence(longest_interval=64, is_latency_driven=True)
    for policy, batch, arrivals in ((light, 8, 8.0), (heavy, 96, 40.0)):
        policy.observe_prefill(0.025, 2)
        policy.observe_prefill(0.060, 16)
        for _ in range(30):
            policy.observe_decode_step(0.0085 if batch == 8 else 0.021, batch)
        policy.observe_arrival(1.0, int(arrivals * policy.config.rate_window_s))
        policy.requests_admitted = 10
        policy.frames_decoded = 840
    assert light.latency_interval(1.0, 0.0085, 0.030) < heavy.latency_interval(1.0, 0.021, 0.060)
    assert light.latency_interval(1.0, 0.0085, 0.030) <= 8


def test_the_modelled_frame_interval_stops_improving_below_the_arrival_crossover() -> None:
    policy = cadence(longest_interval=64, is_latency_driven=True)
    policy.observe_prefill(0.025, 2)
    policy.observe_prefill(0.060, 16)
    light = [policy.modelled_frame_interval_s(level, 0.0085, 9.0) for level in (1, 2, 4)]
    assert light[0] == pytest.approx(light[2], rel=1e-9)
    heavy = [policy.modelled_frame_interval_s(level, 0.021, 40.0) for level in (1, 8, 32)]
    assert heavy[0] > heavy[1] > heavy[2]


def test_the_stall_follows_the_prefills_before_the_window_that_binds() -> None:
    """Not the mean frame interval: the mean barely moves while the stall falls by a factor of four."""
    policy = cadence(longest_interval=64, is_latency_driven=True)
    policy.observe_prefill(0.022, 1)
    policy.observe_prefill(0.087, 24)
    policy.frames_decoded, policy.requests_admitted = 8400, 100
    means = [policy.modelled_frame_interval_s(level, 0.0214, 39.7) for level in (16, 32)]
    stalls = [policy.worst_burst_stall_s(level, 0.0214, 39.7) for level in (16, 32)]
    assert means[0] / means[1] < 1.05
    assert stalls[0] / stalls[1] > 3


def test_the_last_window_is_cut_to_the_frames_a_request_produces() -> None:
    policy = cadence(longest_interval=64, is_latency_driven=True)
    policy.frames_decoded, policy.requests_admitted = 8400, 100
    assert policy.window_ends()[-1] == 84


def test_a_longer_run_of_frames_hides_more_of_the_prefill() -> None:
    policy = cadence(longest_interval=64, is_latency_driven=True)
    policy.observe_prefill(0.025, 2)
    policy.observe_prefill(0.060, 16)
    policy.frames_decoded, policy.requests_admitted = 8400, 100
    stalls = [policy.worst_burst_stall_s(level, 0.0214, 39.7) for level in (2, 4, 8, 16, 32)]
    assert stalls == sorted(stalls, reverse=True)
    assert stalls[0] > 0.4 > stalls[-1]


def test_a_light_load_never_stalls_however_short_the_interval() -> None:
    """Below the crossover the run between prefills is set by the gap between arrivals."""
    policy = cadence(longest_interval=64, is_latency_driven=True)
    policy.observe_prefill(0.022, 1)
    policy.observe_prefill(0.030, 4)
    policy.frames_decoded, policy.requests_admitted = 8400, 100
    assert policy.worst_burst_stall_s(1, 0.0085, 9.2) == pytest.approx(0.0, abs=1e-9)
    assert 1 in policy.smooth_intervals(0.0085, 9.2)


def test_the_rate_comes_from_frames_decoded_not_from_counting_arrivals() -> None:
    """A closed loop finishes requests in clumps, so the arrival count swings and this does not."""
    policy = cadence(longest_interval=64, is_latency_driven=True)
    for index in range(100):
        policy.observe_decode_step(0.0206, 96, index * 0.0206)
    policy.requests_admitted = policy.frames_decoded / 84
    # 96 frames every 20.6 ms over 84 frames a request is about 55 requests a second.
    assert policy.request_rate(99 * 0.0206) == pytest.approx(96 / 0.0206 / 84, rel=0.05)
    policy.observe_arrival(99 * 0.0206, 3)
    assert policy.arrival_rate(99 * 0.0206) < policy.request_rate(99 * 0.0206) / 10


def test_the_interval_chosen_passes_its_own_stall_test() -> None:
    """The stall saw-tooths with the interval, so the search filters rather than bounding."""
    policy = cadence(longest_interval=64, is_latency_driven=True)
    policy.observe_prefill(0.0215, 1)
    policy.observe_prefill(0.0787, 20)
    for _ in range(40):
        policy.observe_decode_step(0.0208, 96, 0.0)
    policy.requests_admitted = policy.frames_decoded / 83
    smooth = policy.smooth_intervals(0.0208, 40.7)
    # The bands: a run of 20 puts two prefills before the window ending at frame 58, a run of 19
    # puts three, so they are 2.4x apart in stall and only one of them passes.
    assert 20 in smooth and 19 not in smooth
    assert policy.worst_burst_stall_s(19, 0.0208, 40.7) > 2 * policy.worst_burst_stall_s(
        20, 0.0208, 40.7
    )
    chosen = policy.latency_interval(1.0, 0.0208, 0.0658)
    assert chosen in smooth or chosen == policy.first_audio_interval(0.0208, 0.0658)
