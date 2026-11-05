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
