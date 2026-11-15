# SPDX-License-Identifier: Apache-2.0
"""GPU regressions for shared plain and speculative decoding."""

from __future__ import annotations

import subprocess
import sys
from typing import Literal

import pytest


@pytest.mark.accelerator
@pytest.mark.parametrize("model_family", ["qwen3_asr", "moss_td"])
def test_plain_fallback_uses_one_target_worker(
    model_family: Literal["qwen3_asr", "moss_td"],
) -> None:
    """Each engine owns its process and keeps one target through both paths."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests.unit_test.fixtures.plain_fallback_lifecycle import probe_plain_fallback; probe_plain_fallback({model_family!r}, enable_cuda_graph=True, enable_radix_cache=True)",
        ],
        check=True,
        timeout=300,
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("model_family", ["qwen3_asr", "moss_td"])
@pytest.mark.parametrize("scenario", ["kv_pressure", "blocked_path"])
def test_plain_fallback_progress_with_shared_resource_limits(
    model_family: Literal["qwen3_asr", "moss_td"],
    scenario: Literal["kv_pressure", "blocked_path"],
) -> None:
    """Mixed paths must requeue under pressure and avoid idling runnable work."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests.unit_test.fixtures.plain_fallback import probe_plain_fallback_progress; probe_plain_fallback_progress({model_family!r}, {scenario!r}, enable_cuda_graph=True)",
        ],
        check=True,
        timeout=300,
    )
