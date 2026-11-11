# SPDX-License-Identifier: Apache-2.0
"""Numerical qualification for optimized Flow outputs."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

# note (Codex): these are numerical screening budgets, not audio-quality guarantees.
DEFAULT_FLOW_ATOL = 1e-3
DEFAULT_FLOW_RTOL = 1e-2
NRMSE_REFERENCE_FLOOR = 1e-12


def finite_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    else:
        return bool(torch.isfinite(actual).all().item()) and torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )


@dataclass(frozen=True, kw_only=True)
class FlowPrecisionResult:
    accepted: bool
    byte_exact: bool
    schema_valid: bool
    finite: bool
    max_abs: float | None = None
    rmse: float | None = None
    worst_row_nrmse: float | None = None
    out_of_bounds: int | None = None


@dataclass(frozen=True, kw_only=True)
class FlowPrecisionGate:
    """Use exact checks by default; load the convolution budget explicitly from env."""

    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value >= 0 for value in (self.atol, self.rtol)
        ):
            raise ValueError("Flow tolerances must be finite non-negative numbers")
        else:
            pass

    @classmethod
    def from_environment(cls) -> FlowPrecisionGate:
        try:
            return cls(
                atol=float(
                    os.environ.get("FIRERED_TTS3_FLOW_ATOL", str(DEFAULT_FLOW_ATOL))
                ),
                rtol=float(
                    os.environ.get("FIRERED_TTS3_FLOW_RTOL", str(DEFAULT_FLOW_RTOL))
                ),
            )
        except ValueError as error:
            raise ValueError(
                "FIRERED_TTS3_FLOW_ATOL and FIRERED_TTS3_FLOW_RTOL "
                "must be finite non-negative numbers"
            ) from error

    def compare(
        self, actual: torch.Tensor, expected: torch.Tensor
    ) -> FlowPrecisionResult:
        schema_valid = (
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and actual.device == expected.device
            and actual.is_floating_point()
            and actual.ndim >= 1
            and actual.numel() > 0
        )
        if not schema_valid:
            return FlowPrecisionResult(
                accepted=False, byte_exact=False, schema_valid=False, finite=False
            )
        elif not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
            return FlowPrecisionResult(
                accepted=False, byte_exact=False, schema_valid=True, finite=False
            )
        else:
            byte_exact = finite_byte_equal(actual, expected)
            reference = expected.double()
            delta = actual.double() - reference
            error = delta.abs()
            out_of_bounds = int(
                (error > self.atol + self.rtol * reference.abs()).sum().item()
            )
            row_error = delta.reshape(delta.shape[0], -1).square().mean(dim=1).sqrt()
            row_reference = (
                reference.reshape(reference.shape[0], -1).square().mean(dim=1).sqrt()
            )
            # note (Codex): zero reference rows need a finite diagnostic denominator.
            row_nrmse = row_error / row_reference.clamp_min(NRMSE_REFERENCE_FLOOR)
            return FlowPrecisionResult(
                accepted=(
                    byte_exact if self.atol == self.rtol == 0 else out_of_bounds == 0
                ),
                byte_exact=byte_exact,
                schema_valid=True,
                finite=True,
                max_abs=float(error.max().item()),
                rmse=float(delta.square().mean().sqrt().item()),
                worst_row_nrmse=float(row_nrmse.max().item()),
                out_of_bounds=out_of_bounds,
            )
