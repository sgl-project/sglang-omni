"""
FishQwen3 models with HuggingFace-style interface.

This module provides a clean separation between model architecture and loading logic,
following HuggingFace conventions while maintaining compatibility with existing code.
"""

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
    FishQwen3AudioDecoderConfig,
    FishQwen3AudioEncoderConfig,
    FishQwen3Config,
    FishQwen3ForSequenceClassificationConfig,
    FishQwen3OmniConfig,
    FishQwen3OmniForSequenceClassificationConfig,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
    FishQwen3AudioDecoder,
    FishQwen3AudioEncoder,
    FishQwen3ForCausalLM,
    FishQwen3ForSequenceClassification,
    FishQwen3Model,
    FishQwen3OmniForCausalLM,
    FishQwen3OmniForSequenceClassification,
    FishQwen3OmniOutput,
    FishQwen3OmniSequenceClassificationOutput,
    FishQwen3PreTrainedModel,
)

__all__ = [
    # Configurations
    "FishQwen3Config",
    "FishQwen3AudioDecoderConfig",
    "FishQwen3ForSequenceClassificationConfig",
    "FishQwen3AudioEncoderConfig",
    "FishQwen3OmniConfig",
    "FishQwen3OmniForSequenceClassificationConfig",
    # Models
    "FishQwen3Model",
    "FishQwen3ForCausalLM",
    "FishQwen3ForSequenceClassification",
    "FishQwen3PreTrainedModel",
    "FishQwen3OmniForCausalLM",
    "FishQwen3OmniForSequenceClassification",
    "FishQwen3AudioEncoder",
    "FishQwen3AudioDecoder",
    # Outputs
    "FishQwen3OmniOutput",
    "FishQwen3OmniSequenceClassificationOutput",
]
