# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating OmniEngine instances."""

from __future__ import annotations

from typing import Any

import torch

from .engine import OmniEngine
from .model_runner import ModelRunner
from .runtime.ar import (
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARResourceManager,
)
from .runtime.cache import SimpleCacheManager
from .runtime.common import (
    EosIterationController,
    SimpleResourceManager,
    SinglePassIterationController,
)
from .runtime.dual_ar import (
    DualARBatchPlanner,
    DualARInputPreparer,
    DualARIterationController,
    DualAROutputProcessor,
    DualARResourceManager,
)
from .runtime.encoder import (
    EncoderBatchPlanner,
    EncoderInputPreparer,
    EncoderOutputProcessor,
)
from .runtime.logits_processor import LogitsProcessor, LogitsProcessorPipeline, default_logits_pipeline
from .runtime.radix_cache import DualARRadixCache
from .runtime.sampler import MultinomialNoSyncSampler, Sampler
from .runtime.tokenizer import wrap_tokenizer
from .scheduler import Scheduler


def create_encoder_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_batch_size: int = 32,
    pooling: str = "last",
    device: str = "cuda",
    use_cache: bool = False,
    cache_size: int | None = None,
) -> OmniEngine:
    """Create an encoder engine.

    Args:
        model: The encoder model (e.g., BERT, RoBERTa)
        tokenizer: Optional tokenizer (used to get pad_token_id)
        max_batch_size: Maximum batch size for scheduling
        pooling: Pooling strategy - "last", "mean", or "cls"
        device: Device to run on
        use_cache: Enable encoder output cache
        cache_size: Max cache entries (None for unbounded)

    Returns:
        OmniEngine configured for encoder models

    Example:
        from transformers import BertModel, BertTokenizer

        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        engine = create_encoder_engine(model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Hello world", return_tensors="pt")
        data = EncoderRequestData(input_ids=input_ids[0])

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # Returns embeddings tensor
    """
    # Get pad_token_id from tokenizer if available
    pad_token_id = 0
    if tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    scheduler = Scheduler(
        batch_planner=EncoderBatchPlanner(max_batch_size=max_batch_size),
        resource_manager=SimpleResourceManager(max_count=max_batch_size),
        iteration_controller=SinglePassIterationController(),
    )

    # Create model runner (stateless)
    model_runner = ModelRunner(
        model=model,
        input_preparer=EncoderInputPreparer(pad_token_id=pad_token_id),
        output_processor=EncoderOutputProcessor(pooling=pooling),
        device=device,
    )

    # Create cache manager (if needed)
    cache_manager = None
    if use_cache:
        cache_manager = SimpleCacheManager(max_size=cache_size)

    return OmniEngine(
        scheduler=scheduler, model_runner=model_runner, cache_manager=cache_manager
    )


def create_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> OmniEngine:
    """Create an AR engine (single request, HF KV cache).

    Args:
        model: The causal LM model (e.g., LLaMA, GPT-2)
        tokenizer: Tokenizer (used to get eos_token_id)
        max_seq_len: Maximum sequence length
        device: Device to run on

    Returns:
        OmniEngine configured for AR models

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        engine = create_ar_engine(model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
        data = ARRequestData(
            input_ids=input_ids[0],
            max_new_tokens=256,
            temperature=0.7,
        )

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # ARRequestData with output_ids

        generated_text = tokenizer.decode(result.output_ids)
    """
    # Get eos_token_id from tokenizer
    eos_token_id = 2
    if tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None) or 2

    def _stream_adapter(request, output):
        token = output.data
        if isinstance(token, tuple):
            token = token[0]
        if token is None:
            return None
        return int(token)

    scheduler = Scheduler(
        batch_planner=ARBatchPlanner(),
        resource_manager=ARResourceManager(max_count=1),
        iteration_controller=EosIterationController(
            eos_token_id=eos_token_id,
            max_length=max_seq_len,
        ),
        stream_adapter=_stream_adapter,
    )

    # Create model runner
    model_runner = ModelRunner(
        model=model,
        input_preparer=ARInputPreparer(),
        output_processor=AROutputProcessor(),
        device=device,
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)


def create_dual_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    *,
    num_codebooks: int = 4,
    codebook_size: int = 1024,
    max_new_tokens: int = 2048,
    device: str = "cuda",
    logits_processors: list[LogitsProcessor] | None = None,
    sampler: Sampler | None = None,
    use_radix_cache: bool = False,
    radix_cache_size: int = 256,
    use_cuda_graph: bool = False,
) -> OmniEngine:
    """Create an engine for DualARTransformer (FishAudio-S1) models.

    Args:
        model: A ``DualARTransformer`` instance (already loaded).
        tokenizer: ``FishTokenizer`` or any object with ``get_token_id()``.
        num_codebooks: Number of VQ codebooks (default 4).
        codebook_size: Size of each codebook (default 1024).
        max_new_tokens: Maximum decode steps.
        device: Device to run on.
        logits_processors: Optional extra logits processors (appended to
            the default pipeline).
        sampler: Override the default ``MultinomialNoSyncSampler``.
        use_radix_cache: Enable radix-tree prefix cache for voice
            cloning reference reuse.
        radix_cache_size: Maximum radix cache entries.
        use_cuda_graph: Capture CUDA Graphs for the decode step.

    Returns:
        ``OmniEngine`` configured for DualAR TTS.

    Example::

        from fish_speech.models.text2semantic.llama import DualARTransformer
        from fish_speech.tokenizer import FishTokenizer

        model = DualARTransformer.from_pretrained("checkpoints/openaudio-s1-mini", load_weights=True)
        tokenizer = FishTokenizer.from_pretrained("checkpoints/openaudio-s1-mini")

        engine = create_dual_ar_engine(model, tokenizer, device="cuda")
        await engine.start()

        adapter = FishTokenizerAdapter(tokenizer)
        values, audio_masks, audio_parts = adapter.build_prompt("Hello world")
        data = DualARRequestData(
            input_values=values,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            max_new_tokens=1024,
        )
        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # DualARRequestData with output_codes
    """
    adapter = wrap_tokenizer(tokenizer)
    im_end_id = adapter.eos_token_ids[0]

    semantic_begin_id = 0
    if hasattr(adapter, "semantic_begin_id"):
        semantic_begin_id = adapter.semantic_begin_id

    # Build logits pipeline
    slow_pipeline = default_logits_pipeline()
    fast_pipeline = default_logits_pipeline()
    if logits_processors:
        for proc in logits_processors:
            slow_pipeline.add(proc)
            fast_pipeline.add(proc)

    sampler = sampler or MultinomialNoSyncSampler()

    # Optionally wrap model with CUDA Graph runner
    actual_model = model
    if use_cuda_graph and torch.cuda.is_available():
        from .runtime.cuda_graph import DualARCudaGraphRunner

        actual_model = DualARCudaGraphRunner(
            model, num_codebooks=num_codebooks,
            codebook_size=codebook_size, device=device,
        )

    def _stream_adapter(request, output):
        step_out = output.data
        if step_out is None or not hasattr(step_out, "codes"):
            return None
        return step_out.codes

    scheduler = Scheduler(
        batch_planner=DualARBatchPlanner(),
        resource_manager=DualARResourceManager(max_count=1),
        iteration_controller=DualARIterationController(
            im_end_token_id=im_end_id,
            max_new_tokens=max_new_tokens,
        ),
        stream_adapter=_stream_adapter,
    )

    model_runner = ModelRunner(
        model=model,
        input_preparer=DualARInputPreparer(),
        output_processor=DualAROutputProcessor(
            model=model,
            slow_pipeline=slow_pipeline,
            fast_pipeline=fast_pipeline,
            sampler=sampler,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            semantic_begin_id=semantic_begin_id,
        ),
        device=device,
    )

    cache_manager = None
    if use_radix_cache:
        cache_manager = DualARRadixCache(max_entries=radix_cache_size)

    return OmniEngine(
        scheduler=scheduler,
        model_runner=model_runner,
        cache_manager=cache_manager,
    )
