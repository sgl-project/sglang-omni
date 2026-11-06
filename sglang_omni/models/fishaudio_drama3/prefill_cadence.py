# SPDX-License-Identifier: Apache-2.0
"""How many decode steps Drama 3 runs between prefills, from the serving SLOs.

A prefill stops every running stream for one forward, and most of its cost does not grow with the
requests it admits: every active expert is read and the tower runs over the prompt rows. Run
interval decode steps between prefills and a stream's frames come out every

    frame interval = decode step + prefill / interval

so a longer interval buys back decode duty cycle, and what it buys is the fixed part of a prefill,
not the part that grows with the cohort.

Two SLOs bound the interval from opposite sides. A request that arrives mid-run waits for the rest
of the run before its prefill, so the first-audio SLO sets the longest interval. Frames reach a
stream in runs, a run at the decode step and then one prefill, and a window is decoded once its
last frame is in, so the stall a stream sees is set by how many prefills fall before the window
that binds; the stall budget admits some intervals and not others, in bands. Between the two the
policy takes the interval with the shortest modelled wall time of one request: its admission wait
plus its frames at the frame interval the prefill share leaves.

A prefill happens only when a request is waiting, so below the load where one cycle is shorter
than the gap between arrivals a longer interval batches nothing and only adds wait. The rate that
tells the two regimes apart is read from the frames the engine decodes: the same rate in steady
state as counting arrivals, and far smoother under a fixed number of streams.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

AUDIO_FRAME_SECONDS = 2048 / 44100
BATCH_BUCKET = 8
# How far the cohorts seen must spread before the prefill's fixed and per-request parts are told
# apart; below it the fit is not identifiable and the whole prefill is read as fixed.
COHORT_SPREAD = 2


@dataclass(frozen=True, kw_only=True)
class PrefillCadenceConfig:
    """The SLOs the interval follows, and the window schedule it reads them against."""

    longest_interval: int = 64
    first_audio_slo_s: float = 0.5
    max_stall_s: float = 0.1
    # Held back from the critical interval, for the jitter a measured mean does not carry.
    continuity_margin_s: float = 0.008
    # What the vocoder adds to first audio after the engine has the first window.
    vocoder_margin_s: float = 0.03
    first_window: int = 4
    window_ramp: tuple[int, ...] = (10, 12, 32)
    followup_window: int = 64
    # A wait is cut short only when the running batch has room for this many waiting requests: at a
    # full batch, releasing for one freed slot at a time turns every decode step into a prefill.
    release_min_admit: int = 1
    step_smoothing: float = 0.1
    # The ablation: take longest_interval every time and leave first audio to the wait cap alone.
    is_longest_only: bool = False
    # latency: between the two bounds, take the interval with the shortest modelled wall time per
    # request instead of either bound. How far back arrivals and admissions are counted:
    rate_window_s: float = 4.0
    # Held back from the stall budget as a share of it. A margin in ms cannot transfer: 8 ms off a
    # 25.5 ms critical interval is a third of it where a step is 10 ms and all of it where it is 20.
    # The size is calibrated: the modelled worst stall is deterministic at the mean costs while the
    # SLO is a percentile over requests, and on one H100 the model came in 1.25-1.5x under the
    # measured p95 across the grid, so a third of the budget is held back for that dispersion.
    continuity_margin: float = 0.35
    is_latency_driven: bool = False


def worst_stall_s(
    frame_interval_s: float,
    windows: tuple[int, ...],
    followup_window: int,
    vocoder_s: float,
    frame_count: int,
) -> float:
    """Longest playback stall of a stream whose frames arrive every frame_interval_s."""
    window_ends: list[int] = []
    for window in (*windows, *[followup_window] * frame_count):
        if window_ends and window_ends[-1] >= frame_count:
            break
        else:
            window_ends.append(
                min(frame_count, (window_ends[-1] if window_ends else 0) + window)
            )
    playback_end_s = (
        window_ends[0] * frame_interval_s
        + vocoder_s
        + window_ends[0] * AUDIO_FRAME_SECONDS
    )
    worst_s = 0.0
    for previous_end, window_end in zip(window_ends, window_ends[1:]):
        ready_s = window_end * frame_interval_s + vocoder_s
        worst_s = max(worst_s, ready_s - playback_end_s)
        playback_end_s = (
            max(playback_end_s, ready_s)
            + (window_end - previous_end) * AUDIO_FRAME_SECONDS
        )
    return worst_s


def critical_frame_interval_s(
    max_stall_s: float,
    windows: tuple[int, ...],
    followup_window: int,
    vocoder_s: float = 0.006,
    frame_count: int = 120,
) -> float:
    """Longest constant frame interval whose worst stall stays within max_stall_s."""
    shortest_s, longest_s = 0.0, AUDIO_FRAME_SECONDS
    for _ in range(40):
        middle_s = (shortest_s + longest_s) / 2
        if (
            worst_stall_s(middle_s, windows, followup_window, vocoder_s, frame_count)
            <= max_stall_s
        ):
            shortest_s = middle_s
        else:
            longest_s = middle_s
    return shortest_s


@dataclass
class PrefillCadence:
    """Online decode step and prefill time, and the interval the two SLOs leave between them."""

    config: PrefillCadenceConfig
    step_s: float | None = None
    prefill_s: float | None = None
    conflicts: int = 0
    chosen_intervals: collections.Counter[int] = field(
        default_factory=collections.Counter
    )
    releases: collections.Counter[str] = field(default_factory=collections.Counter)
    step_by_batch: dict[int, list[float]] = field(default_factory=dict)
    prefill_by_count: dict[int, list[float]] = field(default_factory=dict)
    arrivals: collections.deque[tuple[float, int]] = field(
        default_factory=collections.deque
    )
    decoded: collections.deque[tuple[float, int]] = field(
        default_factory=collections.deque
    )
    prefill_line: list[float] = field(default_factory=lambda: [0.0] * 5)
    prefill_least_s: float = 0.0
    cohort_seen: tuple[float, float] = (0.0, 0.0)
    frames_decoded: float = 0.0
    requests_admitted: float = 0.0
    chosen_targets: collections.Counter[str] = field(default_factory=collections.Counter)

    def __post_init__(self) -> None:
        config = self.config
        self.critical_interval_s = critical_frame_interval_s(
            config.max_stall_s,
            (config.first_window, *config.window_ramp),
            config.followup_window,
        )

    def observe_decode_step(
        self, step_s: float, batch_size: int, now_s: float = 0.0
    ) -> None:
        self.tally(self.step_by_batch, batch_size // BATCH_BUCKET * BATCH_BUCKET, step_s)
        self.step_s = self.smoothed(self.step_s, step_s)
        self.frames_decoded += batch_size
        self.decoded.append((now_s, batch_size))
        while self.decoded and self.decoded[0][0] < now_s - self.config.rate_window_s:
            self.decoded.popleft()

    def observe_prefill(self, prefill_s: float, request_count: int) -> None:
        self.tally(self.prefill_by_count, request_count, prefill_s)
        self.prefill_s = self.smoothed(self.prefill_s, prefill_s)
        self.requests_admitted += request_count
        self.prefill_least_s = (
            prefill_s if self.prefill_least_s == 0 else min(self.prefill_least_s, prefill_s)
        )
        least, most = self.cohort_seen
        self.cohort_seen = (
            request_count if least == 0 else min(least, request_count),
            max(most, request_count),
        )
        weight, count, total, squares, products = self.prefill_line
        self.prefill_line = [
            weight + 1,
            count + request_count,
            total + prefill_s,
            squares + request_count**2,
            products + request_count * prefill_s,
        ]

    def observe_arrival(self, now_s: float, request_count: int) -> None:
        self.arrivals.append((now_s, request_count))
        while self.arrivals and self.arrivals[0][0] < now_s - self.config.rate_window_s:
            self.arrivals.popleft()

    def arrival_rate(self, now_s: float) -> float:
        recent = sum(
            count
            for time_s, count in self.arrivals
            if time_s >= now_s - self.config.rate_window_s
        )
        return recent / self.config.rate_window_s

    def request_rate(self, now_s: float) -> float:
        """Requests per second, from the frames decoded and the frames a request takes.

        Smoother than counting arrivals. In steady state the two agree, but a fixed number of
        concurrent streams admits and finishes requests in clumps, so an arrival count over a few
        seconds swings by a factor of two, and the continuity floor swings with it.
        """
        frames = self.frames_per_request()
        if len(self.decoded) < 2 or frames <= 0:
            return self.arrival_rate(now_s)
        else:
            span_s = max(self.config.rate_window_s / 4, now_s - self.decoded[0][0])
            return sum(count for _, count in self.decoded) / span_s / frames

    def frames_per_request(self) -> float:
        if self.requests_admitted < 1:
            return 0.0
        else:
            return self.frames_decoded / self.requests_admitted

    def prefill_estimate_s(self, cohort: float) -> float:
        """T_P(n) = fixed + per request * n, fitted on every prefill seen, not smoothed over time.

        The split is only identifiable when the cohorts seen differ. Under the arrival crossover
        every arrival gets its own prefill, so every cohort is 1, and a two-parameter fit on a
        regressor with no spread puts the whole cost on the per-request term: a run at C16 fitted
        `0.3 + 28.6 n` where the same engine under load fitted `20.6 + 7.15 n`. A fixed part of zero
        says the interval can buy nothing, and the policy then takes the shortest interval there is.
        So with no spread, read the prefill as **all** fixed cost, which is the side the measurements
        fall on: the fixed part dominates at the small cohorts this regime produces.
        """
        weight, count, total, squares, products = self.prefill_line
        spread = weight * squares - count**2
        least, most = self.cohort_seen
        if weight < 2 or spread <= 0 or most - least < COHORT_SPREAD:
            return self.prefill_least_s or (self.prefill_s or 0.0)
        else:
            per_request = (weight * products - count * total) / spread
            fixed = (total - per_request * count) / weight
            if fixed < 0:
                # note (0xtoward): a prefill cannot cost less than nothing at an empty cohort. The
                # few large cohorts a window sees arrive when the engine is busiest, so they lever
                # the line below zero: one H200 cell at C64 fitted -1.2 + 5.39 n. Read it as all
                # fixed cost instead, the same reading as no spread at all.
                return self.prefill_least_s or (self.prefill_s or 0.0)
            else:
                return fixed + max(0.0, per_request) * cohort

    def smoothed(self, mean_s: float | None, sample_s: float) -> float:
        if mean_s is None:
            return sample_s
        else:
            weight = self.config.step_smoothing
            return (1 - weight) * mean_s + weight * sample_s

    def tally(self, table: dict[int, list[float]], key: int, seconds: float) -> None:
        row = table.setdefault(key, [0, 0.0])
        row[0] += 1
        row[1] += seconds

    def continuity_interval(self, step_s: float, prefill_s: float) -> int:
        """Shortest interval whose frame interval stays inside the schedule's tolerance."""
        headroom_s = self.critical_interval_s - self.config.continuity_margin_s - step_s
        if headroom_s <= 0:
            return self.config.longest_interval
        else:
            return min(
                self.config.longest_interval, math.ceil(prefill_s / headroom_s)
            )

    def first_audio_interval(self, step_s: float, prefill_s: float) -> int:
        """Longest interval whose admission wait still leaves first audio inside its SLO."""
        config = self.config
        budget_s = (
            config.first_audio_slo_s
            - prefill_s
            - config.first_window * step_s
            - config.vocoder_margin_s
        )
        if budget_s <= 0:
            return 0
        else:
            return min(config.longest_interval, int(budget_s / step_s))

    def settle(
        self, interval: int, step_s: float, arrival_rate: float
    ) -> tuple[float, float, float]:
        """What this interval settles on: the prefill, the prefills per second, and the burst.

        A prefill happens only when a request is waiting, so prefills per second are the smaller of
        the arrival rate and one per cycle. Below that crossover a longer interval batches nothing,
        and the run of frames a stream gets between two prefills is set by the gap between arrivals
        rather than by the interval, which is why a light load never stalls whatever the interval.
        """
        cohort, prefills = 1.0, max(arrival_rate, 1e-6)
        for _ in range(3):
            cycle_s = interval * step_s + self.prefill_estimate_s(cohort)
            prefills = min(max(arrival_rate, 1e-6), 1 / cycle_s)
            cohort = max(1.0, arrival_rate / prefills)
        prefill_s = self.prefill_estimate_s(cohort)
        burst_frames = max(1.0, (1 / prefills - prefill_s) / step_s)
        return prefill_s, prefills, burst_frames

    def modelled_frame_interval_s(
        self, interval: int, step_s: float, arrival_rate: float
    ) -> float:
        """The step stretched by the share of the time prefills take at this interval."""
        prefill_s, prefills, _ = self.settle(interval, step_s, arrival_rate)
        return step_s / (1 - min(0.9, prefills * prefill_s))

    def window_ends(self) -> list[int]:
        """Frame each window ends on, the last one cut to the frames a request actually produces."""
        config = self.config
        frames = max(config.followup_window, int(self.frames_per_request()) or 120)
        ends: list[int] = []
        for window in (
            config.first_window,
            *config.window_ramp,
            *[config.followup_window] * frames,
        ):
            if ends and ends[-1] >= frames:
                break
            else:
                ends.append(min(frames, (ends[-1] if ends else 0) + window))
        return ends

    def worst_burst_stall_s(
        self, interval: int, step_s: float, arrival_rate: float
    ) -> float:
        """Worst stall of a stream whose frames come in runs, against the window schedule.

        Frame k is ready at k * step + the prefills that fell before it. A window is decoded once
        its last frame is in, and the stream plays what it already has meanwhile, so the stalls that
        matter are in the opening ramp: window 2 needs ten more frames where window 1 bought four
        frames of audio, and on the default schedule the window ending at frame 58 binds hardest.
        What decides that stall is the prefill time accumulated before that frame, floor(57 / run)
        prefills, not the mean frame interval over the whole request: a longer run puts fewer
        prefills before it even though each one is longer.
        """
        prefill_s, _, burst_frames = self.settle(interval, step_s, arrival_rate)
        ready = lambda frame: frame * step_s + int((frame - 1) / burst_frames) * prefill_s  # noqa: E731
        ends = self.window_ends()
        playback_end_s = ready(ends[0]) + ends[0] * AUDIO_FRAME_SECONDS
        worst_s = 0.0
        for previous, end in zip(ends, ends[1:]):
            ready_s = ready(end)
            worst_s = max(worst_s, ready_s - playback_end_s)
            playback_end_s = (
                max(playback_end_s, ready_s) + (end - previous) * AUDIO_FRAME_SECONDS
            )
        return worst_s

    def wall_time_s(
        self, interval: int, step_s: float, arrival_rate: float, frames: float
    ) -> float:
        """Modelled wall time of one request: the admission wait it pays plus its own frames."""
        return interval * step_s / 2 + frames * self.modelled_frame_interval_s(
            interval, step_s, arrival_rate
        )

    def smooth_intervals(self, step_s: float, arrival_rate: float) -> list[int]:
        """Every interval whose burst keeps the worst stall inside the budget.

        A list, not a floor: the stall is **not** monotone in the interval. Whether a prefill lands
        inside a window or between two of them depends on how the run of frames lines up with the
        window boundaries, so the passing intervals come in bands. On this machine at C96 with a
        100 ms budget they are {12}, [20, 24] and [29, ...], because floor(57 / run) prefills fall
        before the window ending at frame 58 and that count steps down: 4 for a run of 14, 2 for a
        run of 20, 1 for a run of 29. A rule that only bounds the search from below can therefore
        pick an interval that does not pass its own test, and a run of 14 is a local worst case.
        """
        limit_s = self.config.max_stall_s * (1 - self.config.continuity_margin)
        return [
            level
            for level in range(1, self.config.longest_interval + 1)
            if self.worst_burst_stall_s(level, step_s, arrival_rate) <= limit_s
        ]

    def latency_interval(self, now_s: float, step_s: float, prefill_s: float) -> int:
        """The interval with the shortest modelled wall time, inside what the two SLOs allow."""
        arrival_rate = self.request_rate(now_s)
        frames = self.frames_per_request()
        longest = self.first_audio_interval(step_s, prefill_s)
        if arrival_rate <= 0 or frames <= 0:
            self.chosen_targets["unmeasured"] += 1
            return longest
        else:
            pass
        smooth = self.smooth_intervals(step_s, arrival_rate)
        allowed = [level for level in smooth if level <= longest]
        if allowed:
            self.chosen_targets["latency"] += 1
            candidates = allowed
        elif smooth:
            # note (0xtoward): no interval meets both SLOs. Keep continuity: a stall breaks every
            # window of a stream where late first audio is one event, and the longer interval also
            # shrinks the running batch, which brings the step back.
            self.conflicts += 1
            self.chosen_targets["conflict"] += 1
            candidates = [min(smooth)]
        else:
            self.conflicts += 1
            self.chosen_targets["neither"] += 1
            candidates = [max(1, longest)]
        return min(
            candidates,
            key=lambda level: self.wall_time_s(level, step_s, arrival_rate, frames),
        )

    def choose_interval(self, now_s: float = 0.0) -> int:
        step_s, prefill_s = self.step_s, self.prefill_s
        if self.config.is_longest_only:
            interval = self.config.longest_interval
        elif step_s is None or prefill_s is None:
            interval = 0
        elif self.config.is_latency_driven:
            interval = self.latency_interval(now_s, step_s, prefill_s)
        else:
            shortest = self.continuity_interval(step_s, prefill_s)
            longest = self.first_audio_interval(step_s, prefill_s)
            if shortest <= longest:
                interval = longest
            else:
                # note (0xtoward): no interval meets both SLOs at this load. Keep continuity: a
                # stall breaks every window of a stream while late first audio is one event, and
                # the longer interval also shrinks the running batch, which brings the step back.
                self.conflicts += 1
                interval = shortest
        self.chosen_intervals[interval] += 1
        return interval

    def first_audio_s(self, step_s: float, prefill_s: float) -> float:
        config = self.config
        return prefill_s + config.first_window * step_s + config.vocoder_margin_s

    def wait_cap_s(self, waiting_count: int = 1) -> float:
        """How long a waiting request may wait before the interval is cut short for it.

        The prefill it will pay for is the one admitting everything now waiting, which is known
        here, so the cap uses T_P(waiting_count) and not a mean of prefills already gone by.
        """
        step_s = self.step_s
        if step_s is None or self.prefill_s is None:
            return self.config.first_audio_slo_s
        else:
            # note (0xtoward): release one step early, the step already running delays the prefill.
            return max(
                0.0,
                self.config.first_audio_slo_s
                - self.first_audio_s(step_s, self.prefill_estimate_s(waiting_count))
                - step_s,
            )

    def should_release(
        self, waiting_count: int, oldest_wait_s: float, free_slots: int
    ) -> bool:
        if free_slots < min(waiting_count, self.config.release_min_admit):
            self.releases["held_full"] += 1
            return False
        elif oldest_wait_s >= self.wait_cap_s(waiting_count):
            self.releases["wait_cap"] += 1
            return True
        else:
            return False

    def report(self, now_s: float = 0.0) -> str:
        step_ms = 1000 * self.step_s if self.step_s is not None else 0.0
        prefill_ms = 1000 * self.prefill_s if self.prefill_s is not None else 0.0
        text = (
            f"Drama cadence: step {step_ms:.1f} ms, prefill {prefill_ms:.1f} ms, critical interval "
            f"{1000 * self.critical_interval_s:.1f} ms, wait cap {1000 * self.wait_cap_s():.0f} ms, "
            f"frames/request {self.frames_per_request():.0f}, rate {self.request_rate(now_s):.1f}/s "
            f"(arrivals {self.arrival_rate(now_s):.1f}/s), prefill fit "
            f"{1000 * self.prefill_estimate_s(0):.1f}+{1000 * (self.prefill_estimate_s(1) - self.prefill_estimate_s(0)):.2f}n ms, "
            f"targets {dict(self.chosen_targets)}, "
            f"intervals {dict(sorted(self.chosen_intervals.items()))}, slo conflicts "
            f"{self.conflicts}, releases {dict(sorted(self.releases.items()))}, decode ms by batch "
            f"{self.means(self.step_by_batch)}, prefill ms by cohort {self.means(self.prefill_by_count)}"
        )
        self.step_by_batch.clear()
        self.prefill_by_count.clear()
        self.chosen_intervals.clear()
        self.releases.clear()
        self.chosen_targets.clear()
        self.conflicts = 0
        return text

    def means(self, table: dict[int, list[float]]) -> dict[int, tuple[int, float]]:
        return {
            key: (count, round(1000 * total / count, 1))
            for key, (count, total) in sorted(table.items())
        }
