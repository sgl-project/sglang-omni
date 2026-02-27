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
from .logits_processor import (
    FrequencyPenaltyProcessor,
    LogitsProcessor,
    LogitsProcessorPipeline,
    RepetitionPenaltyProcessor,
    SamplingContext,
    TemperatureProcessor,
    TopKProcessor,
    TopPProcessor,
    default_logits_pipeline,
)
from .sampler import (
    ArgmaxSampler,
    MultinomialNoSyncSampler,
    MultinomialSampler,
    Sampler,
    SamplerOutput,
)
from .tokenizer import (
    HFTokenizerAdapter,
    PromptBuilder,
    TokenizerAdapter,
    wrap_tokenizer,
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
    # Tokenizer
    "TokenizerAdapter",
    "PromptBuilder",
    "HFTokenizerAdapter",
    "wrap_tokenizer",
    # Logits Processor
    "LogitsProcessor",
    "LogitsProcessorPipeline",
    "SamplingContext",
    "TemperatureProcessor",
    "TopPProcessor",
    "TopKProcessor",
    "RepetitionPenaltyProcessor",
    "FrequencyPenaltyProcessor",
    "default_logits_pipeline",
    # Sampler
    "Sampler",
    "SamplerOutput",
    "ArgmaxSampler",
    "MultinomialSampler",
    "MultinomialNoSyncSampler",
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
]
