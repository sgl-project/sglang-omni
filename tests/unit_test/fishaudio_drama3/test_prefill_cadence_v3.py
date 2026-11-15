# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Drama 3 dynamic prefill cadence (table-driven, v3)."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace
from unittest import mock

import pytest

from sglang_omni.models.fishaudio_drama3.prefill_cadence_v3 import (
    D_TABLE_ASYNC_OFF,
    D_TABLE_ASYNC_ON,
    EXPECTED_COHORT_MAX_TOKENS,
    GLOBAL_MEAN_FRAMES,
    N_MAX,
    T_P_EAGER_INTERCEPT_MS,
    T_P_EAGER_SLOPE_MS_PER_TOKEN,
    T_P_FIXED_MS,
    T_P_GRAPH_SLOPE_MS_PER_TOKEN,
    T_P_SMALL_MAX_TOKENS,
    T_P_SMALL_SLOPE_MS_PER_TOKEN,
    PrefillCadenceV3,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def make_cadence(**kwargs) -> PrefillCadenceV3:
    return PrefillCadenceV3(**kwargs)


class FakeForwardMode:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def is_extend(self) -> bool:
        return self.mode == "extend"

    def is_decode(self) -> bool:
        return self.mode == "decode"


class FakeRunningReq:
    def __init__(self, finished: bool = False) -> None:
        self.is_finished = finished

    def finished(self) -> bool:
        return self.is_finished


class FakeBatch:
    def __init__(
        self,
        mode: str,
        bs: int,
        extend_num_tokens: int = 0,
        *,
        batch_is_full: bool = False,
    ) -> None:
        self.forward_mode = FakeForwardMode(mode)
        self.is_extend_in_batch = mode == "extend"
        self.reqs = [FakeRunningReq() for _ in range(bs)]
        self.extend_num_tokens = extend_num_tokens
        self.batch_is_full = batch_is_full

    def batch_size(self) -> int:
        return len(self.reqs)


class FakeReq:
    def __init__(self, rid: str, prompt_tokens: int, *, cached_tokens: int = 0):
        self.rid = rid
        self.origin_input_ids = [0] * prompt_tokens
        self.prefix_indices = [0] * cached_tokens
        self.time_stats = SimpleNamespace(wait_queue_entry_time=0.0, created_time=0.0)


def make_scheduler(cadence=None, interval: int = 4) -> OmniScheduler:
    sched = object.__new__(OmniScheduler)
    sched.prefill_cadence = cadence
    sched.prefill_decode_interval = interval
    sched._prefill_decode_interval_remaining = (
        0  # noqa: leading-underscore  # production name
    )
    sched.prefill_decode_interval_min_bs = 1
    sched.require_mlp_sync = False
    sched.forward_ct = 0
    sched.processed_tokens_counter = 0
    sched._sched_idled = False  # noqa: leading-underscore  # production name
    sched.max_running_requests = 16
    sched.running_batch = FakeBatch("decode", 16)
    sched.waiting_queue = []
    return sched


def feed_prefill_history(
    cadence: PrefillCadenceV3, tok: int, reqs: int, count: int = 8
) -> None:
    for i in range(count):
        cadence.observe_extend_launch(1000.0 + i, cohort_tokens=tok, cohort_reqs=reqs)


# --- J(N) argmin -------------------------------------------------------------


def test_argmin_matches_closed_form():
    """J(N) = N*D/2 + f*D + f*T_P/N, so argmin_N J ~ sqrt(2*f*T_P/D)."""
    cadence = make_cadence()
    tok, cohort_reqs = 200, 1
    n_star = cadence.compute_interval(
        running_bs=16, cohort_tokens=tok, cohort_reqs=cohort_reqs
    )
    d_ms = cadence.decode_step_ms(16)
    t_p = cadence.prefill_ms(tok)
    f_est = cadence.frames_estimate(tok, cohort_reqs)
    continuous = math.sqrt(2.0 * f_est * t_p / d_ms)
    assert 1 < n_star < N_MAX  # unclamped region
    assert n_star == pytest.approx(continuous, abs=1.0)
    # The scan must return the true discrete argmin.
    costs = {n: cadence.cost(n, d_ms, t_p, f_est) for n in range(1, N_MAX + 1)}
    assert n_star == min(costs, key=costs.get)


def test_argmin_clamps_to_n_max():
    cadence = make_cadence()
    assert (
        cadence.compute_interval(running_bs=4, cohort_tokens=1600, cohort_reqs=4)
        == N_MAX
    )


def test_argmin_monotonic_in_cohort_size():
    cadence = make_cadence()
    small = cadence.compute_interval(running_bs=16, cohort_tokens=100, cohort_reqs=1)
    large = cadence.compute_interval(running_bs=16, cohort_tokens=1600, cohort_reqs=4)
    assert large > small


# --- empty cohort: expected cohort from prefill history -----------------------


def test_empty_cohort_without_history_falls_back_to_static():
    cadence = make_cadence(fallback_interval=4)
    assert cadence.compute_interval(running_bs=16, cohort_tokens=0, cohort_reqs=0) == 4


def test_empty_cohort_uses_expected_cohort_not_n_one():
    """An empty queue at arm time must not collapse to T_P=0 / N=1: the
    median cohort of recent prefills stands in for the next cohort."""
    cadence = make_cadence()
    feed_prefill_history(cadence, tok=800, reqs=8)
    n_empty = cadence.compute_interval(running_bs=16, cohort_tokens=0, cohort_reqs=0)
    n_explicit = cadence.compute_interval(
        running_bs=16, cohort_tokens=800, cohort_reqs=8
    )
    assert n_empty == n_explicit
    assert n_empty > 1


def test_expected_cohort_is_median_of_window():
    cadence = make_cadence()
    for tok in (100, 200, 300, 400, 500, 600, 700, 800, 10000):
        cadence.observe_extend_launch(0.0, cohort_tokens=tok, cohort_reqs=1)
    # Window keeps the last 8: 100 drops out, median of the rest is 550.
    assert cadence.expected_cohort() == (550, 1)


def test_expected_cohort_not_used_when_queue_nonempty():
    cadence = make_cadence()
    feed_prefill_history(cadence, tok=800, reqs=8)
    n_real = cadence.compute_interval(running_bs=16, cohort_tokens=100, cohort_reqs=1)
    n_expected = cadence.compute_interval(running_bs=16, cohort_tokens=0, cohort_reqs=0)
    assert n_real != n_expected


def test_expected_cohort_outlier_median_falls_back_to_static():
    """A corrupt history (median beyond EXPECTED_COHORT_MAX_TOKENS) must not
    pin N at the cap; the arm falls back to the static interval, while a
    median at the cap still stands in."""
    cadence = make_cadence(fallback_interval=4)
    feed_prefill_history(cadence, tok=EXPECTED_COHORT_MAX_TOKENS + 1, reqs=1)
    assert cadence.compute_interval(running_bs=16, cohort_tokens=0, cohort_reqs=0) == 4
    cadence_at_cap = make_cadence(fallback_interval=4)
    feed_prefill_history(cadence_at_cap, tok=EXPECTED_COHORT_MAX_TOKENS, reqs=1)
    assert (
        cadence_at_cap.compute_interval(running_bs=16, cohort_tokens=0, cohort_reqs=0)
        != 4
    )


# --- table lookups ------------------------------------------------------------


def test_d_table_exact_knots_and_interpolation():
    cadence = make_cadence()
    for b, d in D_TABLE_ASYNC_ON:
        assert cadence.decode_step_ms(b) == pytest.approx(d)
    # Midpoint between (8, 6.84) and (16, 8.66).
    assert cadence.decode_step_ms(12) == pytest.approx((6.84 + 8.66) / 2.0)


def test_d_table_out_of_range_is_none():
    cadence = make_cadence()
    assert cadence.decode_step_ms(D_TABLE_ASYNC_ON[0][0] - 1) is None
    assert cadence.decode_step_ms(D_TABLE_ASYNC_ON[-1][0] + 1) is None


def test_d_table_async_off_selected():
    cadence = make_cadence(async_decode=False)
    for b, d in D_TABLE_ASYNC_OFF:
        assert cadence.decode_step_ms(b) == pytest.approx(d)


def test_t_p_small_cohort_fixed_cost():
    """100 tokens must cost one graphed prefill's measured fixed cost, not
    the ~0 the raw negative-intercept OLS produced."""
    cadence = make_cadence()
    assert 15.0 <= cadence.prefill_ms(100) <= 25.0
    assert cadence.prefill_ms(0) == T_P_FIXED_MS


def test_t_p_segments():
    cadence = make_cadence(graph_max_tokens=2048)
    assert cadence.prefill_ms(T_P_SMALL_MAX_TOKENS) == pytest.approx(
        T_P_FIXED_MS + T_P_SMALL_SLOPE_MS_PER_TOKEN * T_P_SMALL_MAX_TOKENS
    )
    assert cadence.prefill_ms(T_P_SMALL_MAX_TOKENS + 1) == pytest.approx(
        T_P_FIXED_MS + T_P_GRAPH_SLOPE_MS_PER_TOKEN * (T_P_SMALL_MAX_TOKENS + 1)
    )
    assert cadence.prefill_ms(2048) == pytest.approx(
        T_P_FIXED_MS + T_P_GRAPH_SLOPE_MS_PER_TOKEN * 2048
    )
    assert cadence.prefill_ms(2049) == pytest.approx(
        T_P_EAGER_INTERCEPT_MS + T_P_EAGER_SLOPE_MS_PER_TOKEN * 2049
    )
    # The graph/eager boundary is a constructor input, not hardcoded.
    cadence1500 = make_cadence(graph_max_tokens=1500)
    assert cadence1500.prefill_ms(1500) == pytest.approx(
        T_P_FIXED_MS + T_P_GRAPH_SLOPE_MS_PER_TOKEN * 1500
    )
    assert cadence1500.prefill_ms(1501) == pytest.approx(
        T_P_EAGER_INTERCEPT_MS + T_P_EAGER_SLOPE_MS_PER_TOKEN * 1501
    )


def test_frames_estimate_fit_vs_language_fallback():
    """f_est is per-request: the cohort total is averaged over its
    requests before the EN fit applies; the fallback is the per-request
    mean, independent of cohort size."""
    en = make_cadence(language="en")
    beta, alpha = 30.105, 0.37155
    assert en.frames_estimate(500, 2) == pytest.approx(beta + alpha * 250)
    zh = make_cadence(language="zh")
    assert zh.frames_estimate(500, 2) == pytest.approx(GLOBAL_MEAN_FRAMES)
    none_lang = make_cadence(language=None)
    assert none_lang.frames_estimate(500, 2) == pytest.approx(GLOBAL_MEAN_FRAMES)


# --- fallback -----------------------------------------------------------------


def test_out_of_range_batch_falls_back_to_static_interval():
    cadence = make_cadence(fallback_interval=4)
    feed_prefill_history(cadence, tok=500, reqs=4)
    assert (
        cadence.compute_interval(running_bs=128, cohort_tokens=500, cohort_reqs=4) == 4
    )
    assert cadence.compute_interval(running_bs=1, cohort_tokens=500, cohort_reqs=4) == 4


def test_missing_table_falls_back():
    cadence = make_cadence(d_table=())
    assert cadence.decode_step_ms(16) is None
    assert (
        cadence.compute_interval(running_bs=16, cohort_tokens=500, cohort_reqs=4)
        == cadence.fallback_interval
    )


# --- age valve (per-request queue age) ----------------------------------------


def test_age_valve_threshold():
    cadence = make_cadence(first_audio_budget_ms=500.0)
    sched = make_scheduler(cadence)
    sched.running_batch = FakeBatch("decode", 8)
    fresh = FakeReq("fresh", 100)
    fresh._coalesce_enqueue_t = time.perf_counter()
    sched.waiting_queue = [fresh]
    assert sched.age_valve_admits() is False
    aged = FakeReq("aged", 100)
    aged._coalesce_enqueue_t = time.perf_counter() - 0.6
    sched.waiting_queue = [aged]
    assert sched.age_valve_admits() is True
    assert cadence.age_valve_intercepts == 1


def test_age_valve_tracks_queue_age_across_holds():
    """A request that keeps waiting across holds still trips the budget."""
    cadence = make_cadence(first_audio_budget_ms=500.0)
    sched = make_scheduler(cadence)
    sched.running_batch = FakeBatch("decode", 8)
    aged = FakeReq("aged", 100)
    aged._coalesce_enqueue_t = time.perf_counter() - 1.2
    sched.waiting_queue = [aged]
    assert sched.age_valve_admits() is True


def test_age_valve_skipped_when_batch_full():
    cadence = make_cadence(first_audio_budget_ms=500.0)
    sched = make_scheduler(cadence)
    aged = FakeReq("aged", 100)
    aged._coalesce_enqueue_t = time.perf_counter() - 1.0
    sched.waiting_queue = [aged]
    sched.running_batch = FakeBatch("decode", 15, batch_is_full=True)
    assert sched.age_valve_admits() is False
    assert cadence.age_valve_intercepts == 0
    # A full batch with a finished row admits again (upstream rule).
    sched.running_batch.reqs[0] = FakeRunningReq(finished=True)
    assert sched.age_valve_admits() is True


def test_age_valve_silent_when_queue_exceeds_allocatable():
    """A deep queue, not the hold, delays first audio: the valve stays shut."""
    cadence = make_cadence(first_audio_budget_ms=500.0)
    sched = make_scheduler(cadence)
    sched.running_batch = FakeBatch("decode", 15)  # one allocatable row
    aged = [FakeReq(f"r{i}", 100) for i in range(4)]
    for req in aged:
        req._coalesce_enqueue_t = time.perf_counter() - 2.0
    sched.waiting_queue = aged
    assert sched.age_valve_admits() is False
    assert cadence.age_valve_intercepts == 0


def test_age_valve_ignores_empty_queue_and_missing_stamp():
    cadence = make_cadence()
    sched = make_scheduler(cadence)
    sched.running_batch = FakeBatch("decode", 8)
    assert sched.age_valve_admits() is False
    sched.waiting_queue = [FakeReq("r", 100)]
    assert sched.age_valve_admits() is False
    assert cadence.age_valve_intercepts == 0


# --- arm logging state ---------------------------------------------------------


def test_arm_logs_prediction_vs_actual_pair():
    cadence = make_cadence()
    cadence.compute_interval(running_bs=16, cohort_tokens=800, cohort_reqs=8)
    predicted = cadence.prev_arm_t_p_ms
    assert predicted == pytest.approx(cadence.prefill_ms(800))
    cadence.observe_extend_launch(1000.0, cohort_tokens=800, cohort_reqs=8)
    cadence.observe_extend_done(1000.250)
    assert cadence.prev_actual_t_p_ms == pytest.approx(250.0)
    # A done without a launch is ignored.
    cadence.observe_extend_done(1001.0)
    assert cadence.prev_actual_t_p_ms == pytest.approx(250.0)


# --- scheduler integration -----------------------------------------------------


def test_arm_passes_running_batch_and_cohort_tokens():
    cadence = mock.Mock(wraps=make_cadence())
    sched = make_scheduler(cadence)
    sched.running_batch = FakeBatch("decode", 8)
    sched.waiting_queue = [
        FakeReq("a", 300, cached_tokens=50),
        FakeReq("b", 200),
    ]
    sched._arm_prefill_decode_interval(
        FakeBatch("extend", 2)
    )  # noqa: leading-underscore  # production name
    _, kwargs = cadence.compute_interval.call_args
    assert kwargs["running_bs"] == 8
    assert kwargs["cohort_tokens"] == 250 + 200
    assert kwargs["cohort_reqs"] == 2
    assert (
        sched._prefill_decode_interval_remaining == sched.prefill_decode_interval
    )  # noqa: leading-underscore  # production name


def test_arm_cohort_capped_by_allocatable_rows():
    cadence = mock.Mock(wraps=make_cadence())
    sched = make_scheduler(cadence)
    sched.running_batch = FakeBatch("decode", 15)
    sched.waiting_queue = [FakeReq(f"r{i}", 100) for i in range(8)]
    sched._arm_prefill_decode_interval(
        FakeBatch("extend", 2)
    )  # noqa: leading-underscore  # production name
    _, kwargs = cadence.compute_interval.call_args
    assert kwargs["cohort_reqs"] == 1
    assert kwargs["cohort_tokens"] == 100


def test_arm_only_after_extend_batch():
    cadence = make_cadence()
    sched = make_scheduler(cadence)
    sched._arm_prefill_decode_interval(
        FakeBatch("decode", 16)
    )  # noqa: leading-underscore  # production name
    assert (
        sched._prefill_decode_interval_remaining == 0
    )  # noqa: leading-underscore  # production name
    assert cadence.arm_count == 0
    sched._arm_prefill_decode_interval(
        FakeBatch("extend", 2)
    )  # noqa: leading-underscore  # production name
    assert (
        sched._prefill_decode_interval_remaining == sched.prefill_decode_interval
    )  # noqa: leading-underscore  # production name
    assert cadence.arm_count == 1


def test_fixed_interval_behavior_unchanged_without_cadence():
    sched = make_scheduler(cadence=None, interval=4)
    sched._arm_prefill_decode_interval(
        FakeBatch("decode", 16)
    )  # noqa: leading-underscore  # production name
    assert (
        sched._prefill_decode_interval_remaining == 0
    )  # noqa: leading-underscore  # production name
    sched._arm_prefill_decode_interval(
        FakeBatch("extend", 16)
    )  # noqa: leading-underscore  # production name
    assert (
        sched._prefill_decode_interval_remaining == 4
    )  # noqa: leading-underscore  # production name
    assert sched.prefill_decode_interval == 4
    # The age valve never fires without a cadence: pure upstream deferral.
    sched.waiting_queue = [FakeReq("old", 100)]
    assert (
        sched._should_defer_prefill(sched.running_batch) is True
    )  # noqa: leading-underscore  # production name
    assert (
        sched._prefill_decode_interval_remaining == 3
    )  # noqa: leading-underscore  # production name
    # Disarmed interval stays disarmed.
    sched0 = make_scheduler(cadence=None, interval=0)
    sched0._arm_prefill_decode_interval(
        FakeBatch("extend", 16)
    )  # noqa: leading-underscore  # production name
    assert (
        sched0._prefill_decode_interval_remaining == 0
    )  # noqa: leading-underscore  # production name


def test_arm_with_none_batch_is_noop():
    cadence = make_cadence()
    sched = make_scheduler(cadence)
    sched._arm_prefill_decode_interval(
        None
    )  # noqa: leading-underscore  # production name
    assert (
        sched._prefill_decode_interval_remaining == 0
    )  # noqa: leading-underscore  # production name
    assert cadence.arm_count == 0


def test_stamp_observes_extend_launch_with_cohort():
    cadence = mock.Mock()
    sched = make_scheduler(cadence)
    sched.stamp_batch_launch(FakeBatch("decode", 16, extend_num_tokens=37))
    assert cadence.observe_extend_launch.call_count == 0
    sched.stamp_batch_launch(FakeBatch("extend", 3, extend_num_tokens=512))
    assert cadence.observe_extend_launch.call_count == 1
    _, kwargs = cadence.observe_extend_launch.call_args
    assert kwargs["cohort_tokens"] == 512
    assert kwargs["cohort_reqs"] == 3
    assert sched.processed_tokens_counter == 37 + 512


def test_stamp_without_cadence_is_pure_upstream():
    sched = make_scheduler(cadence=None)
    batch = FakeBatch("decode", 16, extend_num_tokens=0)
    sched.stamp_batch_launch(batch)
    assert sched.forward_ct == 1
    assert batch.forward_iter == 1
