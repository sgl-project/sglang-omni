# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic audio frontend utilities."""

from __future__ import annotations

from typing import Any


def ensure_audio_list(audios: Any) -> list[Any]:
    """Normalize audio inputs into a list."""
    if audios is None:
        return []
    if isinstance(audios, list):
        return audios
    return [audios]


def build_audio_mm_inputs(hf_inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract standard audio tensors from HF processor outputs."""
    return {
        "input_features": hf_inputs.get("input_features"),
        "feature_attention_mask": hf_inputs.get("feature_attention_mask"),
        "audio_feature_lengths": hf_inputs.get("audio_feature_lengths"),
    }
