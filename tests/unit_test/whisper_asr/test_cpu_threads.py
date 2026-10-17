# SPDX-License-Identifier: Apache-2.0
"""A serving thread cap respects available CPUs and is opt-in."""

from types import SimpleNamespace

import pytest

from sglang_omni.models.whisper_asr import engine_builder


@pytest.mark.parametrize(
    "requested_threads,available_cpu_count,expected_threads",
    [(3, 2, [2]), (3, 8, [3]), (None, 8, [])],
)
def test_thread_limit_is_applied_before_encoder_setup_and_respects_affinity(
    monkeypatch: pytest.MonkeyPatch,
    requested_threads: int | None,
    available_cpu_count: int,
    expected_threads: list[int],
) -> None:
    applied_threads: list[int] = []
    monkeypatch.setattr(engine_builder.torch, "set_num_threads", applied_threads.append)
    monkeypatch.setattr(
        engine_builder.os,
        "sched_getaffinity",
        lambda process_id: set(range(available_cpu_count)),
    )
    builder = engine_builder.WhisperASREngineBuilder(
        max_running_requests=4,
        max_new_tokens=32,
        mem_fraction_static=0.2,
        enable_pre_lm_encoder=False,
        cpu_intra_op_threads=requested_threads,
    )
    builder.setup_runtime_resources(None, SimpleNamespace())
    assert applied_threads == expected_threads
