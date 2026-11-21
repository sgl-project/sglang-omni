# SPDX-License-Identifier: Apache-2.0
"""Dynamic prefill/decode cadence for Drama 3 (table-driven, v3).

    N* = argmin_{N in [1, N_MAX]} J(N)
    J(N) = N*D(B)/2 + f_est*D(B) / (1 - T_P(tok)/(N*D(B) + T_P(tok)))

There are no online estimators: every model quantity comes from offline
calibration fits baked in as constants below, and the only inputs are the
running batch size and the waiting cohort's total uncached tokens at arm
time.  All fits recompute from results/pd-calibration-20261115/ on the
calibration host:

- D(B): piecewise-linear decode-step launch-to-launch time vs running
  batch size.  Mean dt_ms of the samples with b == B in
  d_B{4,8,16,32,64}_{async_on,async_off}/decode_steps.json.  The async-off
  table is fitted alongside but unused by default.
- T_P(tok): three segments.  Small cohorts (tok <= 400) and the rest of
  the graph region (tok <= graph boundary) share the measured fixed cost
  of one graphed prefill, T_P_FIXED_MS = 19.5 ms (the n = 1 short rows of
  tp_async_on/tp_cohorts.jsonl average 20.95 ms at 100 tokens), with
  per-token slopes anchored through the single-prefill calibration rows:
  0.0145 ms/tok from the n = 1 short rows (100 tok -> 20.95 ms mean) and
  0.0133 ms/tok from the n = 1 long rows (687 tok -> 28.65 ms mean).
  Single-prefill rows are the right basis because the scheduler admits a
  cohort as one coalesced extend batch; the multi-request rows chain
  several prefills and would overstate a small cohort (the raw OLS
  intercept came out negative and drove T_P to ~0).  Eager segment
  (tok > boundary): F_E + A_E*tok, OLS over the long rows with x > 2048.
  The boundary follows the server's cuda_graph_max_bs_prefill rather than
  a hardcoded 2048.  The eager segment is fitted from one sample per
  realized prefill batch (the multi-request rows share one forward, so the
  cost is the longest span once, not the sum of spans).
- f_est = F_BETA + F_ALPHA*(tok/cohort_reqs): frames one request of the
  cohort will decode, OLS of completion_tokens vs prompt_tokens over the
  EN seed-tts-eval results.csv rows (lang en, 256 samples, pooled across
  the dynN-sweepB-20261115 cells).  The fit is per-request, so the
  cohort's mean text length feeds it; languages without a fit fall back
  to GLOBAL_MEAN_FRAMES (the per-request mean).

Empty-cohort arms (nothing waiting right after an extend) substitute an
expected cohort: the median cohort_tokens (and request count) of the last
EXPECTED_COHORT_WINDOW realized prefill batches.  With no prefill history
yet — or a median beyond EXPECTED_COHORT_MAX_TOKENS, which no plausible
cohort reaches — the arm falls back to the static interval.

The age valve compares only hold-induced wait — time elapsed since the
last arm — against the budget, not a request's total queue age; the
scheduler additionally skips the valve while the running batch admits no
prefill (batch full).
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import deque

logger = logging.getLogger(__name__)

DEFAULT_FIRST_AUDIO_BUDGET_MS = 500.0
DEFAULT_FALLBACK_INTERVAL = 4
N_MAX = 32
EXPECTED_COHORT_WINDOW = 8
# A median expected cohort above this is a corrupt history, not a real
# workload; the arm falls back to the static interval rather than let the
# estimate pin N at the cap.
EXPECTED_COHORT_MAX_TOKENS = 100_000

# Mean dt_ms at b == B, d_B{4,8,16,32,64}_async_on/decode_steps.json.
D_TABLE_ASYNC_ON = (
    (4, 5.46),
    (8, 6.84),
    (16, 8.66),
    (32, 11.65),
    (64, 15.22),
)
# Mean dt_ms at b == B, d_B{4,8,16,32,64}_async_off/decode_steps.json.
D_TABLE_ASYNC_OFF = (
    (4, 6.29),
    (8, 7.64),
    (16, 9.56),
    (32, 12.66),
    (64, 16.35),
)

# Fixed cost of one graphed prefill, measured by the n = 1 short rows of
# tp_async_on/tp_cohorts.jsonl (20.81 / 20.97 / 21.08 ms at 100 tokens).
T_P_FIXED_MS = 19.5
# tok <= 400: slope anchored at T_P_FIXED_MS through the n = 1 short rows.
T_P_SMALL_MAX_TOKENS = 400
T_P_SMALL_SLOPE_MS_PER_TOKEN = 0.0145
# 400 < tok <= graph boundary: slope anchored at T_P_FIXED_MS through the
# n = 1 long rows (28.59 / 28.61 / 28.74 ms at 687 tokens).
T_P_GRAPH_SLOPE_MS_PER_TOKEN = 0.0133
# tok > boundary: one sample per shared forward (span deduped by batch).
T_P_EAGER_INTERCEPT_MS = 67.7
T_P_EAGER_SLOPE_MS_PER_TOKEN = 0.0077

# OLS of completion_tokens vs prompt_tokens, EN seed-tts-eval rows.
F_FIT_BY_LANGUAGE = {"en": (30.105, 0.37155)}
GLOBAL_MEAN_FRAMES = 92.33


class PrefillCadenceV3:
    def __init__(
        self,
        *,
        fallback_interval: int = DEFAULT_FALLBACK_INTERVAL,
        first_audio_budget_ms: float = DEFAULT_FIRST_AUDIO_BUDGET_MS,
        graph_max_tokens: int = 2048,
        async_decode: bool = True,
        language: str | None = "en",
        d_table: tuple[tuple[int, float], ...] | None = None,
        clock=time.perf_counter,
    ) -> None:
        self.fallback_interval = int(fallback_interval)
        self.first_audio_budget_ms = float(first_audio_budget_ms)
        self.graph_max_tokens = int(graph_max_tokens)
        self.language = language
        self.clock = clock
        if d_table is not None:
            self.d_table = tuple(d_table)
        elif async_decode:
            self.d_table = D_TABLE_ASYNC_ON
        else:
            self.d_table = D_TABLE_ASYNC_OFF
        self.arm_count = 0
        self.age_valve_intercepts = 0
        self.intercept_since_last_arm = False
        self.extend_launch_ts: float | None = None
        self.prefill_tokens_history: deque[int] = deque(maxlen=EXPECTED_COHORT_WINDOW)
        self.prefill_reqs_history: deque[int] = deque(maxlen=EXPECTED_COHORT_WINDOW)
        self.prev_arm_t_p_ms: float | None = None
        self.prev_actual_t_p_ms: float | None = None

    def decode_step_ms(self, running_bs: int) -> float | None:
        """Piecewise-linear D(B); None when B falls outside the table."""
        table = self.d_table
        if not table or running_bs < table[0][0] or running_bs > table[-1][0]:
            return None
        else:
            pass
        for (b_lo, d_lo), (b_hi, d_hi) in zip(table, table[1:]):
            if b_lo <= running_bs <= b_hi:
                if b_hi == b_lo:
                    return d_lo
                else:
                    frac = (running_bs - b_lo) / (b_hi - b_lo)
                    return d_lo + frac * (d_hi - d_lo)
            else:
                pass
        return table[-1][1]

    def prefill_ms(self, tok: int) -> float:
        """Three-segment T_P(tok); intercepts are never negative."""
        if tok <= T_P_SMALL_MAX_TOKENS:
            t_p = T_P_FIXED_MS + T_P_SMALL_SLOPE_MS_PER_TOKEN * tok
        elif tok <= self.graph_max_tokens:
            t_p = T_P_FIXED_MS + T_P_GRAPH_SLOPE_MS_PER_TOKEN * tok
        else:
            t_p = T_P_EAGER_INTERCEPT_MS + T_P_EAGER_SLOPE_MS_PER_TOKEN * tok
        return max(t_p, 0.0)

    def frames_estimate(self, tok: int, cohort_reqs: int) -> float:
        """Frames one cohort request will decode; per-language fit or mean."""
        fit = F_FIT_BY_LANGUAGE.get(self.language or "")
        if fit is None or cohort_reqs <= 0:
            return GLOBAL_MEAN_FRAMES
        else:
            beta, alpha = fit
            return beta + alpha * (tok / cohort_reqs)

    @staticmethod
    def cost(n: int, d_ms: float, t_p_ms: float, f_est: float) -> float:
        return n * d_ms / 2.0 + f_est * d_ms / (1.0 - t_p_ms / (n * d_ms + t_p_ms))

    def expected_cohort(self) -> tuple[int, int] | None:
        """Median cohort of the most recent realized prefills; None when the
        median is implausibly large."""
        if not self.prefill_tokens_history:
            return None
        else:
            pass
        tok = int(round(statistics.median(self.prefill_tokens_history)))
        if tok > EXPECTED_COHORT_MAX_TOKENS:
            return None
        else:
            pass
        reqs = max(1, int(round(statistics.median(self.prefill_reqs_history))))
        return tok, reqs

    def compute_interval(
        self,
        *,
        running_bs: int,
        cohort_tokens: int,
        cohort_reqs: int = 0,
    ) -> int:
        d_ms = self.decode_step_ms(running_bs)
        tok = max(int(cohort_tokens), 0)
        reqs = max(int(cohort_reqs), 0)
        expected_cohort = False
        if tok == 0 or reqs == 0:
            expected = self.expected_cohort()
            if expected is not None:
                tok, reqs = expected
                expected_cohort = True
            else:
                # No prefill history yet: no basis for a cohort estimate.
                d_ms = None
        else:
            pass
        if d_ms is None or d_ms <= 0.0:
            interval = self.fallback_interval
            f_est = t_p_ms = None
        else:
            t_p_ms = self.prefill_ms(tok)
            f_est = self.frames_estimate(tok, reqs)
            interval = min(
                range(1, N_MAX + 1),
                key=lambda n: self.cost(n, d_ms, t_p_ms, f_est),
            )
        self.arm_count += 1
        logger.info(
            "prefill_cadence_v3 arm #%d: running_batch=%d cohort_reqs=%d "
            "cohort_tokens=%d expected_cohort=%s f_est=%s D=%s T_P=%s N=%d "
            "age_valve_intercept=%s prev_T_P_pred=%s prev_T_P_actual=%s",
            self.arm_count,
            running_bs,
            reqs,
            tok,
            expected_cohort,
            "n/a" if f_est is None else f"{f_est:.1f}",
            "oob" if d_ms is None else f"{d_ms:.2f}ms",
            "n/a" if t_p_ms is None else f"{t_p_ms:.2f}ms",
            interval,
            self.intercept_since_last_arm,
            (
                "n/a"
                if self.prev_arm_t_p_ms is None
                else f"{self.prev_arm_t_p_ms:.2f}ms"
            ),
            (
                "n/a"
                if self.prev_actual_t_p_ms is None
                else f"{self.prev_actual_t_p_ms:.2f}ms"
            ),
        )
        self.prev_arm_t_p_ms = t_p_ms
        self.intercept_since_last_arm = False
        return interval

    def mark_age_valve_intercept(self) -> None:
        self.age_valve_intercepts += 1
        self.intercept_since_last_arm = True

    def observe_extend_launch(
        self, ts: float, *, cohort_tokens: int, cohort_reqs: int
    ) -> None:
        self.extend_launch_ts = ts
        if cohort_tokens > 0:
            self.prefill_tokens_history.append(int(cohort_tokens))
            self.prefill_reqs_history.append(int(cohort_reqs))
        else:
            pass

    def observe_extend_done(self, ts: float) -> None:
        """Record the realized prefill time for the next arm's log line."""
        if self.extend_launch_ts is None:
            return
        else:
            pass
        self.prev_actual_t_p_ms = (ts - self.extend_launch_ts) * 1000.0
        self.extend_launch_ts = None
