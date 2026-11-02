# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MiniCPM-o transcription adapter."""

from __future__ import annotations

import pytest

from sglang_omni.serve.transcription_adapters import resolve_adapter
from sglang_omni.serve.transcription_adapters.minicpm_o import (
    MiniCPMOTranscriptionAdapter,
)


def test_resolve_adapter_matches_minicpm_o_architecture() -> None:
    assert isinstance(resolve_adapter(["MiniCPMO"]), MiniCPMOTranscriptionAdapter)


@pytest.mark.parametrize(
    ("raw_text", "transcript"),
    [
        (
            'What was that?\n|||{"wer": 0.0, "ins": 0, "del": 0, "sub": 0}',
            "What was that?",
        ),
        ('Thank you, Klaus.\n||{"wer": 0.0, "ins": 0', "Thank you, Klaus."),
        (
            'His three sons, capless and terrified,\n|| #answer|||{"wer": 0.0}',
            "His three sons, capless and terrified,",
        ),
        ("今天天气很好。", "今天天气很好。"),
        ("Either way || works.", "Either way || works."),
    ],
)
def test_postprocess_drops_trailing_evaluation_record(
    raw_text: str, transcript: str
) -> None:
    adapter = MiniCPMOTranscriptionAdapter()

    assert adapter.postprocess_text(raw_text) == transcript
