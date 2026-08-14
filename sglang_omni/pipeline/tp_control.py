# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible TP names for generic parallel control helpers."""

from sglang_omni.pipeline.parallel_control import (
    ParallelFollowerControlPlane,
    ParallelLeaderFanout,
    ParallelWorkMessage,
)

TPFollowerControlPlane = ParallelFollowerControlPlane
TPLeaderFanout = ParallelLeaderFanout
TPWorkMessage = ParallelWorkMessage

__all__ = ["TPFollowerControlPlane", "TPLeaderFanout", "TPWorkMessage"]
