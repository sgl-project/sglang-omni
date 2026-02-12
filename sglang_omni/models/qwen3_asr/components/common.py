# SPDX-License-Identifier: Apache-2.0
"""Common utilities for Qwen3-ASR components."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.qwen3_asr.modeling.configuration_qwen3_asr import (
    Qwen3ASRThinkerConfig,
)


def load_thinker_config(model_id: str) -> Any:
    """Load the thinker configuration for the given model ID."""
    return Qwen3ASRThinkerConfig.from_pretrained(model_id)
