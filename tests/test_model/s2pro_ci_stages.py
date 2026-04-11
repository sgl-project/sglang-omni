# SPDX-License-Identifier: Apache-2.0
"""Canonical stage identifiers for the S2-Pro benchmark CI."""

from __future__ import annotations

S2PRO_STAGE_NONSTREAM = "s2pro-stage-1-nonstream"
S2PRO_STAGE_STREAM = "s2pro-stage-2-stream"
S2PRO_STAGE_CONSISTENCY = "s2pro-stage-3-consistency"
S2PRO_STAGE_ALL = "all"

S2PRO_CI_STAGES = (
    S2PRO_STAGE_NONSTREAM,
    S2PRO_STAGE_STREAM,
    S2PRO_STAGE_CONSISTENCY,
)

