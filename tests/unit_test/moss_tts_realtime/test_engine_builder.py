# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from sglang_omni.models.moss_tts_realtime.engine_builder import (
    MossTTSRealtimeEngineBuilder,
)


def _builder(**overrides: Any) -> MossTTSRealtimeEngineBuilder:
    values: dict[str, Any] = {
        "max_seq_len": 40960,
        "total_gpu_memory_fraction": 0.90,
        "max_sessions": 7,
        "max_held_sessions": 5,
        "max_active_turns": 3,
        "max_pending_text_tokens": 64,
        "max_pending_text_bytes": 2048,
        "max_input_updates": 32,
        "terminal_tombstone_limit": 77,
        "input_idle_timeout_s": 1.5,
        "turn_timeout_s": 2.5,
        "session_idle_ttl_s": 3.5,
    }
    values.update(overrides)
    return MossTTSRealtimeEngineBuilder(**values)


def test_falls_back_when_process_accounting_is_unavailable(monkeypatch) -> None:
    from sglang_omni.utils import gpu_memory

    builder = _builder()
    builder.gpu_id = 1
    builder.minimum_codec_mem_reserve = 0.10
    overrides = builder.generation_defaults(dtype="bfloat16")
    monkeypatch.setattr(gpu_memory, "get_process_gpu_memory_bytes", lambda _: None)

    builder.adjust_overrides(overrides)

    assert overrides["mem_fraction_static"] == pytest.approx(0.80)
    assert builder.profile_total_gpu_memory_fraction is None


def test_derives_colocated_codec_reserve_from_hbm_and_session_slots(
    monkeypatch,
) -> None:
    from sglang_omni.models.moss_tts_realtime import stages
    from sglang_omni.utils import gpu_memory

    gib = 1024**3
    builder = _builder(
        codec_model_path="codec",
        max_sessions=16,
        max_held_sessions=5,
        max_active_turns=16,
    )
    builder.gpu_id = 2
    calls: dict[str, Any] = {}

    def fake_estimate(model_path: str, *, stream_slots: int) -> tuple[int, int]:
        calls["estimate"] = (model_path, stream_slots)
        return 3 * gib, 2 * gib

    monkeypatch.setattr(
        stages,
        "estimate_moss_tts_realtime_codec_memory",
        fake_estimate,
    )
    monkeypatch.setattr(
        gpu_memory,
        "get_gpu_device_info",
        lambda gpu_id: (
            calls.__setitem__("gpu_id", gpu_id),
            gpu_memory.GpuDeviceInfo(gpu_id, gpu_id, "fake", 80 * gib),
        )[-1],
    )

    builder._derive_colocated_codec_memory_budget()

    # Session-scoped streaming slots: held sessions plus active turns.
    assert calls == {"gpu_id": 2, "estimate": ("codec", 21)}
    assert builder.gpu_memory_bytes == 80 * gib
    assert builder.codec_decoder_bytes == 3 * gib
    assert builder.codec_streaming_state_bytes == 2 * gib
    assert builder.codec_runtime_margin_bytes == 2 * gib
    assert builder.minimum_codec_mem_reserve == pytest.approx(0.088)


def test_rejects_explicit_ar_fraction_that_consumes_codec_reserve() -> None:
    builder = _builder()
    builder.gpu_id = 0
    builder.minimum_codec_mem_reserve = 0.10
    overrides = builder.generation_defaults(dtype="bfloat16")
    overrides["mem_fraction_static"] = 0.81

    with pytest.raises(ValueError, match="required decoder"):
        builder.adjust_overrides(overrides)


@pytest.mark.parametrize(
    ("max_sessions", "max_held_sessions", "max_active_turns", "expected"),
    [
        (10, 2, 3, 5),
        (7, 5, 3, 7),
    ],
)
def test_reserves_request_slots_for_held_sessions(
    max_sessions: int,
    max_held_sessions: int,
    max_active_turns: int,
    expected: int,
) -> None:
    builder = _builder(
        max_sessions=max_sessions,
        max_held_sessions=max_held_sessions,
        max_active_turns=max_active_turns,
    )

    assert builder.request_pool_capacity == expected
    assert (
        builder.generation_defaults(dtype="bfloat16")["max_running_requests"]
        == expected
    )
