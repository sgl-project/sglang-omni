# SPDX-License-Identifier: Apache-2.0
"""Runtime module - model-type-specific batching and I/O logic."""

from .ar import (
    ARBatchData,
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARRequestData,
    ARResourceManager,
)
from .common import (
    EosIterationController,
    SimpleResourceManager,
    SinglePassIterationController,
)
from .encoder import (
    EncoderBatchData,
    EncoderBatchPlanner,
    EncoderInputPreparer,
    EncoderOutputProcessor,
    EncoderRequestData,
)
from .interfaces import (
    BatchPlanner,
    CacheManager,
    InputPreparer,
    IterationController,
    OutputProcessor,
    ResourceManager,
)
from .sglang_ar import (
    SGLangARRequestData,
    SGLangBatchPlanner,
    SGLangIterationController,
    SGLangModelRunner,
    SGLangOutputProcessor,
    SGLangResourceManager,
)

__all__ = [
    # Protocols
    "BatchPlanner",
    "CacheManager",
    "ResourceManager",
    "IterationController",
    "InputPreparer",
    "OutputProcessor",
    "SimpleResourceManager",
    "SinglePassIterationController",
    "EosIterationController",
    # Encoder
    "EncoderRequestData",
    "EncoderBatchData",
    "EncoderBatchPlanner",
    "EncoderInputPreparer",
    "EncoderOutputProcessor",
    # AR (Simple)
    "ARRequestData",
    "ARBatchData",
    "ARBatchPlanner",
    "ARResourceManager",
    "ARInputPreparer",
    "AROutputProcessor",
    # AR (SGLang)
    "SGLangARRequestData",
    "SGLangBatchPlanner",
    "SGLangResourceManager",
    "SGLangOutputProcessor",
    "SGLangIterationController",
    "SGLangModelRunner",
]
