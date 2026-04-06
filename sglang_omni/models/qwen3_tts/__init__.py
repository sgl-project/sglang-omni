# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model package for sglang-omni."""

import logging

from sglang_omni.models.qwen3_tts.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from sglang_omni.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Register with HuggingFace AutoConfig / AutoModel
# ---------------------------------------------------------------------------
try:
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoConfig.register("qwen3_tts_talker", Qwen3TTSTalkerConfig)
    AutoConfig.register(
        "qwen3_tts_talker_code_predictor", Qwen3TTSTalkerCodePredictorConfig
    )
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
except ImportError:
    # transformers is optional; skip registration if unavailable.
    pass
except Exception as exc:
    logger.warning("Failed to register Qwen3-TTS with transformers: %s", exc)
