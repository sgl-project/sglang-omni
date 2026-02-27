# SPDX-License-Identifier: Apache-2.0
"""Factory function for creating DualAR (FishAudio-S1) engines."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.engines.omni.engine import OmniEngine
from sglang_omni.engines.omni.model_runner import ModelRunner
from sglang_omni.engines.omni.runtime.logits_processor import (
    LogitsProcessor,
    default_logits_pipeline,
)
from sglang_omni.engines.omni.runtime.sampler import MultinomialNoSyncSampler, Sampler
from sglang_omni.engines.omni.scheduler import Scheduler

from .runtime.dual_ar import (
    DualARBatchPlanner,
    DualARInputPreparer,
    DualARIterationController,
    DualAROutputProcessor,
    DualARResourceManager,
)
from .runtime.radix_cache import DualARRadixCache
from .tokenizer import FishTokenizerAdapter


class _InferenceWrapper(torch.nn.Module):
    """Thin wrapper that routes ``forward()`` to ``forward_generate()``.

    ``ModelRunner`` calls ``model(**inputs)`` which dispatches to ``forward()``.
    During inference the KV-cache-aware ``forward_generate`` must be used
    instead.  This wrapper also calls ``setup_caches`` once at construction
    so the KV buffers are allocated before the first forward pass.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        max_batch_size: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self._model = model
        model.setup_caches(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=torch.bfloat16,
        )

    def forward(
        self, x: torch.Tensor, input_pos=None, audio_masks=None, audio_parts=None
    ):
        return self._model.forward_generate(
            x, input_pos=input_pos, audio_masks=audio_masks, audio_parts=audio_parts
        )

    def forward_generate_fast(self, x: torch.Tensor, input_pos=None):
        return self._model.forward_generate_fast(x, input_pos)

    def __getattr__(self, name: str):
        # Delegate attribute access to the wrapped model so that
        # OutputProcessor can reach e.g. fast_layers, fast_embeddings,
        # forward_generate_fast, layers, etc.
        if name == "_model":
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)


def create_dual_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    *,
    num_codebooks: int = 4,
    codebook_size: int = 1024,
    max_new_tokens: int = 2048,
    max_seq_len: int = 4096,
    device: str = "cuda",
    logits_processors: list[LogitsProcessor] | None = None,
    sampler: Sampler | None = None,
    use_radix_cache: bool = False,
    radix_cache_size: int = 256,
    use_compile: bool = False,
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
        use_compile: Use torch.compile for decode step optimization.

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
    adapter = FishTokenizerAdapter(tokenizer)
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

    # Wrap model for inference (forward → forward_generate, setup KV caches).
    actual_model = _InferenceWrapper(
        model,
        max_batch_size=1,
        max_seq_len=max_seq_len,
    )

    # Create radix cache (integrated directly into DualAR components,
    # NOT via the engine-level CacheManager protocol)
    radix_cache = None
    if use_radix_cache:
        radix_cache = DualARRadixCache(max_tokens=radix_cache_size)

    def _stream_adapter(request, output):
        step_out = output.data
        if step_out is None or not hasattr(step_out, "codes"):
            return None
        return step_out.codes

    scheduler = Scheduler(
        batch_planner=DualARBatchPlanner(radix_cache=radix_cache),
        resource_manager=DualARResourceManager(max_count=1, radix_cache=radix_cache),
        iteration_controller=DualARIterationController(
            im_end_token_id=im_end_id,
            max_new_tokens=max_new_tokens,
        ),
        stream_adapter=_stream_adapter,
    )

    output_processor = DualAROutputProcessor(
        model=actual_model,
        slow_pipeline=slow_pipeline,
        fast_pipeline=fast_pipeline,
        sampler=sampler,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        semantic_begin_id=semantic_begin_id,
        radix_cache=radix_cache,
        use_compile=use_compile,
    )

    model_runner = ModelRunner(
        model=actual_model,
        input_preparer=DualARInputPreparer(model=actual_model),
        output_processor=output_processor,
        device=device,
    )

    engine = OmniEngine(
        scheduler=scheduler,
        model_runner=model_runner,
    )
    # Attach radix cache so callers can inspect stats via engine.radix_cache
    engine.radix_cache = radix_cache
    return engine
