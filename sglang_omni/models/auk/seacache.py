# SPDX-License-Identifier: Apache-2.0
"""Trajectory-local SeaCache for AuK audio sampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, kw_only=True)
class SeaCacheConfig:
    threshold: float = 0.20
    max_skip_steps: int = 1
    force_compute_steps: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.threshold < float("inf"):
            raise ValueError("SeaCache threshold must be finite and nonnegative")
        if self.max_skip_steps < 1 or self.force_compute_steps < 1:
            raise ValueError("SeaCache skip and forced-compute steps must be positive")


def filter_audio(
    hidden: torch.Tensor, time: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Apply a mean-normalized flow Wiener filter along valid audio frames."""
    result = torch.zeros_like(hidden, dtype=torch.float32)
    a = time.float().clamp(1e-6, 1 - 1e-6)
    b = 1 - a
    for row, length in enumerate(lengths.tolist()):
        freq = torch.fft.rfftfreq(length, device=hidden.device)
        spectrum = 1 / (freq.square() + 1e-16)
        gain = (a * spectrum) / (a.square() * spectrum + b.square() + 1e-16)
        weights = torch.full_like(gain, 2.0)
        weights[0] = 1.0
        if length % 2 == 0:
            weights[-1] = 1.0
        gain = gain / (gain * weights).sum() * length
        signal = hidden[row, :length].float()
        result[row, :length] = torch.fft.irfft(
            torch.fft.rfft(signal, dim=0) * gain[:, None], n=length, dim=0
        )
    return result


def relative_l1(
    current: torch.Tensor, previous: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    """Return one relative distance per row, excluding padded frames."""
    valid = (
        torch.arange(current.shape[1], device=current.device)[None, :]
        < lengths[:, None]
    )
    valid = valid.unsqueeze(-1)
    numerator = ((current - previous).abs() * valid).sum(dim=(1, 2))
    denominator = (previous.abs() * valid).sum(dim=(1, 2)).clamp_min(1e-16)
    return numerator / denominator


@dataclass(kw_only=True)
class SeaCacheState:
    config: SeaCacheConfig
    total_steps: int
    step: int = 0
    previous_filtered_input: torch.Tensor | None = None
    previous_residual: torch.Tensor | None = None
    accumulated_distance: torch.Tensor | None = None
    consecutive_skips: int = 0
    computed_steps: int = 0
    cached_steps: int = 0
    reasons: dict[str, int] = field(default_factory=dict)
    timings: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list
    )

    def decide(
        self, hidden: torch.Tensor, time: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[bool, str]:
        """Choose one decision for the entire CFG-expanded batch."""
        if self.step >= self.total_steps:
            raise RuntimeError("SeaCache trajectory has already completed")
        if self.config.threshold == 0:
            reason = "disabled_threshold"
            should_compute = True
        elif self.step < self.config.force_compute_steps or self.step >= (
            self.total_steps - self.config.force_compute_steps
        ):
            reason = "boundary"
            should_compute = True
        elif self.previous_residual is None:
            reason = "initial"
            should_compute = True
        elif self.consecutive_skips >= self.config.max_skip_steps:
            reason = "max_skip"
            should_compute = True
        else:
            reason = "threshold"
            should_compute = False

        if self.config.threshold > 0:
            filtered = filter_audio(hidden, time, lengths)
            if self.previous_filtered_input is not None:
                distance = relative_l1(filtered, self.previous_filtered_input, lengths)
                if self.accumulated_distance is None:
                    self.accumulated_distance = distance
                else:
                    self.accumulated_distance += distance
                if not should_compute and bool(
                    (self.accumulated_distance >= self.config.threshold).any()
                ):
                    should_compute, reason = True, "threshold"
            self.previous_filtered_input = filtered

        if should_compute:
            self.accumulated_distance = None
            self.consecutive_skips = 0
            self.computed_steps += 1
        else:
            reason = "cache_hit"
            self.consecutive_skips += 1
            self.cached_steps += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.step += 1
        return should_compute, reason
