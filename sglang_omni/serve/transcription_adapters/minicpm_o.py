# SPDX-License-Identifier: Apache-2.0
"""Transcription output adapter for MiniCPM-o."""

from __future__ import annotations

import re

from sglang_omni.serve.transcription_adapters.base import (
    DefaultTranscriptionAdapter,
    register_transcription_adapter,
)

# note (Tianyao Wu): the checkpoint sometimes ends a transcript with an
# evaluation record on a new line, such as ||{"wer": 0.0, ...}.
_EVALUATION_RECORD_RE = re.compile(r"\n\|\|.*", re.DOTALL)


@register_transcription_adapter("MiniCPMO")
class MiniCPMOTranscriptionAdapter(DefaultTranscriptionAdapter):
    def postprocess_text(self, text: str) -> str:
        return _EVALUATION_RECORD_RE.sub("", text).strip()
