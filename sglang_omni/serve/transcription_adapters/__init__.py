# SPDX-License-Identifier: Apache-2.0
"""Transcription output adapters (verbose_json segment building).

Importing this package registers all built-in adapters.
"""

from __future__ import annotations

# Imported for its registration side effect (@register_transcription_adapter).
from sglang_omni.serve.transcription_adapters import (  # noqa: F401
    moss_transcribe_diarize,
)
from sglang_omni.serve.transcription_adapters.base import (
    TranscriptionAdapter,
    register_transcription_adapter,
    resolve_adapter,
)

__all__ = [
    "TranscriptionAdapter",
    "register_transcription_adapter",
    "resolve_adapter",
]
