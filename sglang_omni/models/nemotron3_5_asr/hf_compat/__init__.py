# SPDX-License-Identifier: Apache-2.0
"""Narrow Transformers 5.13 compatibility backport for Nemotron 3.5 ASR.

The source modules in this package are adapted from Hugging Face Transformers
v5.13.0. Imports, registration, and the documented compatibility delta are
adjusted to run with the repository-pinned Transformers 5.12.1 stack. Remove
this package once the SGLang dependency set moves to Transformers 5.13 or newer.
"""

from .configuration_nemotron3_5_asr import Nemotron3_5AsrConfig
from .feature_extraction_nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor,
)
from .modeling_nemotron3_5_asr import Nemotron3_5AsrForRNNT
from .processing_nemotron3_5_asr import Nemotron3_5AsrProcessor

__all__ = [
    "Nemotron3_5AsrConfig",
    "Nemotron3_5AsrForRNNT",
    "Nemotron3_5AsrProcessor",
    "NemotronAsrStreamingFeatureExtractor",
]
