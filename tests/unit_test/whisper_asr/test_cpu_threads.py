# SPDX-License-Identifier: Apache-2.0
"""A serving thread cap respects available CPUs and is opt-in."""

from types import SimpleNamespace

import pytest

from sglang_omni.models.whisper_asr import engine_builder
from sglang_omni.utils import cpu


@pytest.mark.parametrize(
    "requested_threads,available_cpu_count,quota_cpu_count,expected_threads",
    [(3, 2, None, [2]), (3, 8, None, [3]), (8, 8, 2, [2]), (None, 8, 2, [])],
)
def test_thread_limit_respects_affinity_and_cgroup_quota(
    monkeypatch: pytest.MonkeyPatch,
    requested_threads: int | None,
    available_cpu_count: int,
    quota_cpu_count: int | None,
    expected_threads: list[int],
) -> None:
    """An explicit thread count must not exceed a smaller cgroup quota."""
    monkeypatch.setattr(cpu, "cgroup_cpu_quota_count", lambda: quota_cpu_count)
    applied_threads: list[int] = []
    monkeypatch.setattr(engine_builder.torch, "set_num_threads", applied_threads.append)
    monkeypatch.setattr(
        cpu.os,
        "sched_getaffinity",
        lambda process_id: set(range(available_cpu_count)),
        raising=False,
    )
    builder = engine_builder.WhisperASREngineBuilder(
        encoder_attention_backend="auto",
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_pre_lm_encoder=False,
        cpu_intra_op_threads=requested_threads,
    )
    builder.setup_runtime_resources(None, SimpleNamespace())
    assert applied_threads == expected_threads
