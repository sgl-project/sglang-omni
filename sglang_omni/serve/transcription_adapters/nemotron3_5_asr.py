# SPDX-License-Identifier: Apache-2.0
"""Output cleanup and locale detection for Nemotron 3.5 ASR."""

from __future__ import annotations

from sglang_omni.models.nemotron3_5_asr.text import (
    clean_nemotron_text,
    resolve_nemotron_locale,
)
from sglang_omni.serve.transcription_adapters.base import (
    DefaultTranscriptionAdapter,
    register_transcription_adapter,
)


@register_transcription_adapter("Nemotron3_5Asr")
class Nemotron3_5ASRTranscriptionAdapter(DefaultTranscriptionAdapter):
    """Keep locale tags long enough to populate ``verbose_json.language``."""

    def resolve_language(
        self,
        raw_text: str,
        requested_language: str | None,
    ) -> str | None:
        return resolve_nemotron_locale(raw_text, requested_language)

    def postprocess_text(self, text: str) -> str:
        return clean_nemotron_text(text)

    def postprocess_plain_text(self, text: str) -> str:
        return self.postprocess_text(text)


__all__ = ["Nemotron3_5ASRTranscriptionAdapter"]
