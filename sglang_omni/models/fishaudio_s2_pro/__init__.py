# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro (FishQwen3OmniForCausalLM) model support for sglang-omni.

S2-Pro uses a fundamentally different architecture from S1-Mini:
- FishQwen3OmniForCausalLM instead of DualARTransformer
- HuggingFace PreTrainedTokenizerFast instead of FishTokenizer (tiktoken)
- 1D token input with VQ masks instead of multi-row [num_codebooks+1, seq_len]
- Post-norm hidden states, sqrt(num_codebooks+1) scaling
- RAS (Repetition Aware Sampling) + constrained decoding + top-k=30
"""

from .factory import create_s2pro_engine, create_s2pro_sglang_engine
from .runtime.s2pro_ar import S2ProRequestData, S2ProStepOutput
from .runtime.s2pro_sglang_ar import S2ProSGLangRequestData
from .tokenizer import Reference, S2ProTokenizerAdapter

__all__ = [
    "create_s2pro_engine",
    "create_s2pro_sglang_engine",
    "create_tts_pipeline_config",
    "S2ProRequestData",
    "S2ProSGLangRequestData",
    "S2ProStepOutput",
    "S2ProTokenizerAdapter",
    "Reference",
]


def __getattr__(name: str):
    if name == "create_tts_pipeline_config":
        from .pipeline.config import create_tts_pipeline_config

        return create_tts_pipeline_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
