# SPDX-License-Identifier: Apache-2.0
"""How many decode steps Drama 3 runs between prefills, from the serving SLOs.

A prefill stops every running stream for one forward, and most of its cost does not grow with the
requests it admits: every active expert is read and the tower runs over the prompt rows. Run
interval decode steps between prefills and a stream's frames come out every

    frame interval = decode step + prefill / interval

so a longer interval buys back decode duty cycle, and what it buys is the fixed part of a prefill,
not the part that grows with the cohort.

Two SLOs bound the interval from opposite sides. A stream plays the audio of the window it has
while the next window is made, so the schedule tolerates frame intervals up to a critical interval
before a stall exceeds max_stall: that is the shortest interval worth running. A request that
arrives mid-run waits for the rest of the run before its prefill, so the first-audio SLO sets the
longest. Between the two the longest wins, because a longer interval leaves more of the time to
decode, which both raises throughput at a fixed number of streams and shortens the frame interval
at a fixed arrival rate. Only first audio pays for it.

Both bounds come from the measured decode step and prefill time, with no arrival rate in them, so
the rule reads the same whether the load is a fixed arrival rate or a fixed number of streams.
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field

AUDIO_FRAME_SECONDS = 2048 / 44100
BATCH_BUCKET = 8


@dataclass(frozen=True, kw_only=True)
class PrefillCadenceConfig:
    """The SLOs the interval follows, and the window schedule it reads them against."""

    longest_interval: int = 32
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
            window_ends.append((window_ends[-1] if window_ends else 0) + window)
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

    def __post_init__(self) -> None:
        config = self.config
        self.critical_interval_s = critical_frame_interval_s(
            config.max_stall_s,
            (config.first_window, *config.window_ramp),
            config.followup_window,
        )

    def observe_decode_step(self, step_s: float, batch_size: int) -> None:
        self.tally(self.step_by_batch, batch_size // BATCH_BUCKET * BATCH_BUCKET, step_s)
        self.step_s = self.smoothed(self.step_s, step_s)

    def observe_prefill(self, prefill_s: float, request_count: int) -> None:
        self.tally(self.prefill_by_count, request_count, prefill_s)
        self.prefill_s = self.smoothed(self.prefill_s, prefill_s)

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

    def choose_interval(self) -> int:
        step_s, prefill_s = self.step_s, self.prefill_s
        if self.config.is_longest_only:
            interval = self.config.longest_interval
        elif step_s is None or prefill_s is None:
            interval = 0
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

    def wait_cap_s(self) -> float:
        """How long a waiting request may wait before the interval is cut short for it."""
        step_s, prefill_s = self.step_s, self.prefill_s
        if step_s is None or prefill_s is None:
            return self.config.first_audio_slo_s
        else:
            # note (0xtoward): release one step early, the step already running delays the prefill.
            return max(
                0.0,
                self.config.first_audio_slo_s
                - self.first_audio_s(step_s, prefill_s)
                - step_s,
            )

    def should_release(
        self, waiting_count: int, oldest_wait_s: float, free_slots: int
    ) -> bool:
        if free_slots < min(waiting_count, self.config.release_min_admit):
            self.releases["held_full"] += 1
            return False
        elif oldest_wait_s >= self.wait_cap_s():
            self.releases["wait_cap"] += 1
            return True
        else:
            return False

    def report(self) -> str:
        step_ms = 1000 * self.step_s if self.step_s is not None else 0.0
        prefill_ms = 1000 * self.prefill_s if self.prefill_s is not None else 0.0
        text = (
            f"Drama cadence: step {step_ms:.1f} ms, prefill {prefill_ms:.1f} ms, critical interval "
            f"{1000 * self.critical_interval_s:.1f} ms, wait cap {1000 * self.wait_cap_s():.0f} ms, "
            f"intervals {dict(sorted(self.chosen_intervals.items()))}, slo conflicts "
            f"{self.conflicts}, releases {dict(sorted(self.releases.items()))}, decode ms by batch "
            f"{self.means(self.step_by_batch)}, prefill ms by cohort {self.means(self.prefill_by_count)}"
        )
        self.step_by_batch.clear()
        self.prefill_by_count.clear()
        self.chosen_intervals.clear()
        self.releases.clear()
        self.conflicts = 0
        return text

    def means(self, table: dict[int, list[float]]) -> dict[int, tuple[int, float]]:
        return {
            key: (count, round(1000 * total / count, 1))
            for key, (count, total) in sorted(table.items())
        }
