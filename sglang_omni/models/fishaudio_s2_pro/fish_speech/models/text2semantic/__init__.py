"""
FishQwen3 models with HuggingFace-style interface.
"""

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
    FishQwen3AudioDecoderConfig,
    FishQwen3Config,
    FishQwen3OmniConfig,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
    FishQwen3AudioDecoder,
    FishQwen3ForCausalLM,
    FishQwen3Model,
    FishQwen3OmniForCausalLM,
    FishQwen3OmniOutput,
    FishQwen3PreTrainedModel,
)

__all__ = [
    # Configurations
    "FishQwen3Config",
    "FishQwen3AudioDecoderConfig",
    "FishQwen3OmniConfig",
    # Models
    "FishQwen3Model",
    "FishQwen3ForCausalLM",
    "FishQwen3PreTrainedModel",
    "FishQwen3OmniForCausalLM",
    "FishQwen3AudioDecoder",
    # Outputs
    "FishQwen3OmniOutput",
]
