# SPDX-License-Identifier: Apache-2.0
"""Single-terminal results must surface the finish reason like merged ones do."""

from __future__ import annotations

from sglang_omni.client.client import Client


def test_single_terminal_result_surfaces_finish_reason() -> None:
    """Any pipeline that terminates through one stage, not only Qwen3-TTS."""
    chunk = Client._default_result_builder(
        "req-1",
        {
            "modality": "audio",
            "sample_rate": 24000,
            "finish_reason": "length",
            "usage": {"prompt_tokens": 2, "completion_tokens": 2048},
        },
    )

    assert chunk.finish_reason == "length"


def test_single_terminal_result_without_finish_reason_stays_default() -> None:
    """A pipeline that reports no reason must not be relabeled."""
    chunk = Client._default_result_builder(
        "req-2",
        {"modality": "audio", "sample_rate": 24000},
    )

    assert chunk.finish_reason == "stop"
