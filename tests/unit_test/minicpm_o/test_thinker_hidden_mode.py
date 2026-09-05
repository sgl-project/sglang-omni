# SPDX-License-Identifier: Apache-2.0
"""The thinker only asks sglang for hidden states when a talker consumes them.

Decode CUDA graphs replay only on an exact hidden-mode match, so the text-only
pipeline (graphs captured without hidden states) must request NULL.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sglang.srt.managers.scheduler")

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode  # noqa: E402

from sglang_omni.models.minicpm_o.thinker_model_runner import (  # noqa: E402
    MiniCPMOThinkerModelRunner,
)


def _runner(speech_enabled: bool) -> MiniCPMOThinkerModelRunner:
    runner = MiniCPMOThinkerModelRunner.__new__(MiniCPMOThinkerModelRunner)
    runner._speech_enabled = speech_enabled
    return runner


def test_text_only_pipeline_requests_no_hidden_states():
    runner = _runner(speech_enabled=False)
    assert (
        runner.requested_capture_hidden_mode_prefill(None, []) is CaptureHiddenMode.NULL
    )
    assert (
        runner.requested_capture_hidden_mode_decode(None, []) is CaptureHiddenMode.NULL
    )


def test_speech_pipeline_requests_full_hidden_states():
    runner = _runner(speech_enabled=True)
    assert (
        runner.requested_capture_hidden_mode_prefill(None, []) is CaptureHiddenMode.FULL
    )
    assert (
        runner.requested_capture_hidden_mode_decode(None, []) is CaptureHiddenMode.FULL
    )
