# SPDX-License-Identifier: Apache-2.0
"""Request helpers for the ZONOS2 pipeline skeleton."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.zonos2.payload_types import ZONOS2State
from sglang_omni.proto import StagePayload


def build_zonos2_state_from_request(payload: StagePayload) -> ZONOS2State:
    """Build placeholder ZONOS2 state from an OpenAI-compatible TTS request."""
    request = payload.request
    extra_body: dict[str, Any] = getattr(request, "extra_body", None) or {}
    references = (
        getattr(request, "references", None) or extra_body.get("references") or []
    )
    ref_audio = getattr(request, "ref_audio", None) or extra_body.get("ref_audio")
    ref_text = getattr(request, "ref_text", None) or extra_body.get("ref_text")
    if references and isinstance(references, list):
        first_reference = references[0]
        if isinstance(first_reference, dict):
            ref_audio = ref_audio or first_reference.get("audio_path")
            ref_text = ref_text or first_reference.get("text")

    generation_kwargs = extra_body.get("generation_kwargs")
    return ZONOS2State(
        text=str(getattr(request, "input", "") or ""),
        language=str(extra_body.get("language") or "auto"),
        voice=getattr(request, "voice", None) or extra_body.get("voice"),
        ref_audio=ref_audio,
        ref_text=ref_text,
        generation_kwargs=(
            dict(generation_kwargs) if isinstance(generation_kwargs, dict) else {}
        ),
    )
