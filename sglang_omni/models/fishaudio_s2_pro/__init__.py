# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro (FishQwen3OmniForCausalLM) model support for sglang-omni.

S2-Pro uses a fundamentally different architecture from S1-Mini:
- FishQwen3OmniForCausalLM instead of DualARTransformer
- HuggingFace PreTrainedTokenizerFast instead of FishTokenizer (tiktoken)
- 1D token input with VQ masks instead of multi-row [num_codebooks+1, seq_len]
- Post-norm hidden states, sqrt(num_codebooks+1) scaling
- RAS (Repetition Aware Sampling) + constrained decoding + top-k=30
"""

from . import config
from .factory import create_s2pro_sglang_engine
from .runtime.s2pro_ar import S2ProStepOutput
from .runtime.s2pro_sglang_ar import S2ProSGLangRequestData
from .tokenizer import Reference, S2ProTokenizerAdapter

__all__ = [
    "config",
    "create_s2pro_sglang_engine",
    "S2ProSGLangRequestData",
    "S2ProStepOutput",
    "S2ProTokenizerAdapter",
    "Reference",
]
