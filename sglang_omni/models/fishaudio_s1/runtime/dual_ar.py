# SPDX-License-Identifier: Apache-2.0
"""DualAR (FishAudio-S1) runtime support for sglang-omni.

DualARTransformer differs from standard LLMs in several ways:

1. Input tokens are multi-row: ``[num_codebooks+1, seq_len]``
   (row 0 = text/semantic tokens, rows 1..N = codebook values).
2. Each decode step produces ``num_codebooks+1`` tokens via a two-stage
   process: slow transformer → sample semantic token → fast transformer
   → autoregressively sample N codebook tokens.
3. The fast transformer has its own KV cache that is reset every step.
4. Stop condition is ``<|im_end|>`` on row 0, not a standard EOS token.

This module provides the full set of runtime components needed to plug
DualAR into the existing ``OmniEngine`` / ``ModelRunner`` architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from sglang_omni.engines.omni.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from sglang_omni.engines.omni.runtime.common import SimpleResourceManager
from sglang_omni.engines.omni.runtime.interfaces import ResourceManager
from sglang_omni.engines.omni.runtime.logits_processor import (
    LogitsProcessorPipeline,
    SamplingContext,
)
from sglang_omni.engines.omni.runtime.sampler import MultinomialNoSyncSampler, Sampler

if TYPE_CHECKING:
    from .radix_cache import DualARRadixCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DualARRequestData:
    """DualAR-specific request data (stored in ``SchedulerRequest.data``).

    ``input_values`` has shape ``[num_codebooks+1, seq_len]``:
    - Row 0:   text token IDs (with semantic tokens mapped via
               ``semantic_id_to_token_id``)
    - Rows 1-N: codebook indices (non-zero only at VQ positions)

    ``audio_masks`` / ``audio_parts`` come from
    ``ContentSequence.encode_for_inference()`` and are used during prefill
    for reference audio embedding injection.
    """

    input_values: torch.Tensor  # [num_codebooks+1, seq_len]
    audio_masks: torch.Tensor | None = None
    audio_parts: torch.Tensor | None = None

    num_codebooks: int = 4
    num_computed_tokens: int = 0
    output_codes: list[torch.Tensor] = field(default_factory=list)
    max_new_tokens: int | None = None

    temperature: float = 0.8
    top_p: float = 0.8
    repetition_penalty: float = 1.1

    # Managed by the runtime — callers should not set these directly.
    _slow_kv_cache: Any = None  # Opaque handle for slow transformer KV cache
    _previous_tokens: torch.Tensor | None = None  # [num_codebooks+1, window]

    # Radix cache bookkeeping
    _cached_prefix_len: int = 0
    _last_cache_node: Any = None
    _original_row0: torch.Tensor | None = None
    _cached_kv_data: list[tuple[Tensor, Tensor]] | None = None


@dataclass
class DualARBatchData:
    """DualAR-specific batch data.

    Currently single-request only (batch_size=1).
    """

    input_values: torch.Tensor  # [1, num_codebooks+1, seq_len]
    input_pos: torch.Tensor  # [seq_len]
    is_prefill: bool
    audio_masks: torch.Tensor | None = None
    audio_parts: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# KV cache snapshot / restore helpers
# ---------------------------------------------------------------------------


def snapshot_slow_kv(model: Any, length: int) -> list[tuple[Tensor, Tensor]]:
    """Clone slow KV cache slices ``[:, :, :length, :]`` from all layers.

    Works with fish-speech's ``KVCache`` class which has ``k_cache``/``v_cache``
    as registered buffers of shape ``[B, H, max_seq_len, D]``.
    """
    result = []
    for layer in model.layers:
        kv = layer.attention.kv_cache
        result.append(
            (
                kv.k_cache[:, :, :length, :].clone(),
                kv.v_cache[:, :, :length, :].clone(),
            )
        )
    return result


def restore_slow_kv(model: Any, kv_data: list[tuple[Tensor, Tensor]]) -> None:
    """Write saved KV slices back into model layer caches."""
    for (k, v), layer in zip(kv_data, model.layers):
        length = k.shape[2]
        layer.attention.kv_cache.k_cache[:, :, :length, :].copy_(k)
        layer.attention.kv_cache.v_cache[:, :, :length, :].copy_(v)


# ---------------------------------------------------------------------------
# Compilable decode step (for torch.compile)
# ---------------------------------------------------------------------------


def _multinomial_no_sync(probs: Tensor) -> Tensor:
    """Gumbel-max trick: sample without CUDA sync (compilable)."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _sample(
    logits: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    repetition_penalty: Tensor,
    previous_tokens: Tensor | None = None,
) -> Tensor:
    """Sample one token from logits (compilable, no Python objects)."""
    # Repetition penalty
    if previous_tokens is not None:
        prev = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=prev)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits = logits.clone()
        logits.scatter_(dim=-1, index=prev, src=score.to(logits.dtype))

    # Top-p filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(
        torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
    )
    sorted_mask = cum_probs > top_p
    sorted_mask[..., 0] = False  # keep at least one
    indices_to_remove = sorted_mask.scatter(
        dim=-1, index=sorted_indices, src=sorted_mask
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return _multinomial_no_sync(probs)


def decode_one_token(
    model: Any,
    x: Tensor,
    input_pos: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    repetition_penalty: Tensor,
    previous_tokens: Tensor | None,
    semantic_begin_id: int,
    num_codebooks: int,
    codebook_size: int,
) -> Tensor:
    """Full decode step: slow forward + sample + fast loop (compilable).

    This function mirrors fish-speech's ``decode_one_token_ar`` and is
    designed to be wrapped with ``torch.compile(mode='reduce-overhead',
    fullgraph=True)``.

    Returns:
        ``[num_codebooks+1, 1]`` tensor of sampled token IDs.
    """
    # 1. Slow forward
    out = model.forward_generate(x, input_pos)
    slow_logits = out.logits[:, -1, :]  # [1, vocab]
    hidden_states = out.hidden_states  # [1, 1, dim]

    # 2. Sample semantic token
    prev_row0 = previous_tokens[:, 0] if previous_tokens is not None else None
    semantic = _sample(
        slow_logits, temperature, top_p, repetition_penalty, prev_row0
    )  # [1, 1]
    codebooks = [semantic.squeeze(-1)]  # [1]

    # 3. Clear fast KV cache
    for layer in model.fast_layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
            layer.attention.kv_cache.k_cache.fill_(0)
            layer.attention.kv_cache.v_cache.fill_(0)

    # 4. Initial fast forward (seed with hidden states)
    fast_pos = torch.tensor([0], device=x.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states.squeeze(1), fast_pos)

    # 5. Embed semantic token for codebook loop
    cb_input = semantic.squeeze(-1) - semantic_begin_id
    cb_input = cb_input.clamp(min=0)
    cb_hidden = model.fast_embeddings(cb_input)
    codebooks.append(cb_input)

    # 6. Fast loop for remaining codebooks
    for cb_idx in range(1, num_codebooks):
        fast_pos = torch.tensor([cb_idx], device=x.device, dtype=torch.long)
        cb_logits = model.forward_generate_fast(cb_hidden, fast_pos)
        cb_logits = cb_logits[:, :, :codebook_size].squeeze(1)  # [1, codebook_size]

        prev_row = (
            previous_tokens[:, cb_idx + 1] if previous_tokens is not None else None
        )
        cb_token = _sample(
            cb_logits, temperature, top_p, repetition_penalty, prev_row
        )  # [1, 1]

        cb_hidden = model.fast_embeddings(cb_token.squeeze(-1))
        codebooks.append(cb_token.squeeze(-1))

    return torch.stack(codebooks, dim=1).T  # [num_codebooks+1, 1]


# ---------------------------------------------------------------------------
# BatchPlanner
# ---------------------------------------------------------------------------


class DualARBatchPlanner:
    """Batch planner for single-request DualAR execution."""

    def __init__(self, radix_cache: DualARRadixCache | None = None) -> None:
        self._radix_cache = radix_cache

    def select_requests(
        self,
        waiting: list[SchedulerRequest],
        running: list[SchedulerRequest],
        resource_manager: ResourceManager,
    ) -> list[SchedulerRequest]:
        if running:
            return [running[0]]
        if not waiting:
            return []
        request = waiting[0]
        if not resource_manager.can_allocate(request):
            return []
        resource_manager.allocate(request)
        return [request]

    def build_batch(self, requests: list[SchedulerRequest]) -> DualARBatchData:
        request = requests[0]
        data: DualARRequestData = request.data
        is_prefill = data.num_computed_tokens == 0

        if is_prefill:
            input_values = data.input_values
            input_pos = torch.arange(0, input_values.shape[1], dtype=torch.long)

            if self._radix_cache is not None:
                row0 = input_values[0].tolist()
                matched_len, kv_data, last_node = (
                    self._radix_cache.match_prefix(row0)
                )

                if matched_len > 0 and kv_data is not None:
                    data._original_row0 = input_values[0].clone()
                    data._cached_prefix_len = matched_len
                    data._last_cache_node = last_node
                    data._cached_kv_data = kv_data
                    self._radix_cache.inc_lock_ref(last_node)

                    input_values = input_values[:, matched_len:]
                    input_pos = torch.arange(
                        matched_len,
                        matched_len + input_values.shape[1],
                        dtype=torch.long,
                    )

                    if data.audio_masks is not None:
                        if matched_len >= data.audio_masks.shape[-1]:
                            data.audio_masks = None
                            data.audio_parts = None
                        else:
                            data.audio_masks = data.audio_masks[..., matched_len:]
                            if data.audio_parts is not None:
                                data.audio_parts = data.audio_parts[..., matched_len:]

                    logger.info(
                        "Radix cache hit: %d/%d tokens cached",
                        matched_len,
                        data._original_row0.shape[0],
                    )
        else:
            last_codes = data.output_codes[-1]  # [num_codebooks+1, 1]
            input_values = last_codes
            pos = data.num_computed_tokens + len(data.output_codes) - 1
            input_pos = torch.tensor([pos], dtype=torch.long)

        return DualARBatchData(
            input_values=(
                input_values.unsqueeze(0) if input_values.dim() == 2 else input_values
            ),
            input_pos=input_pos,
            is_prefill=is_prefill,
            audio_masks=data.audio_masks if is_prefill else None,
            audio_parts=data.audio_parts if is_prefill else None,
        )


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


class DualARResourceManager(SimpleResourceManager):
    """Clears DualAR KV caches on free and unlocks radix cache nodes."""

    def __init__(
        self,
        max_count: int = 32,
        radix_cache: DualARRadixCache | None = None,
    ) -> None:
        super().__init__(max_count=max_count)
        self._radix_cache = radix_cache

    def free(self, request: SchedulerRequest) -> None:
        super().free(request)
        data: DualARRequestData = request.data

        if data._last_cache_node is not None and self._radix_cache is not None:
            self._radix_cache.dec_lock_ref(data._last_cache_node)
            data._last_cache_node = None

        data._slow_kv_cache = None
        data._previous_tokens = None
        data._cached_kv_data = None
        data._original_row0 = None
        data._cached_prefix_len = 0


# ---------------------------------------------------------------------------
# InputPreparer
# ---------------------------------------------------------------------------


class DualARInputPreparer:
    """Convert ``DualARBatchData`` to model input kwargs.

    Also restores slow-transformer KV cache from radix cache hits
    so that the subsequent forward pass can attend to the cached prefix.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        batch_data: DualARBatchData = scheduler_output.batch_data

        # Restore cached KV into model before forward pass
        request = scheduler_output.requests[0]
        data: DualARRequestData = request.data
        if data._cached_kv_data is not None:
            restore_slow_kv(self._model, data._cached_kv_data)
            data._cached_kv_data = None

        result: dict[str, Any] = {
            "x": batch_data.input_values.to(device),
            "input_pos": batch_data.input_pos.to(device),
        }

        if batch_data.audio_masks is not None:
            result["audio_masks"] = batch_data.audio_masks.to(device)
        if batch_data.audio_parts is not None:
            result["audio_parts"] = batch_data.audio_parts.to(device)

        return result


# ---------------------------------------------------------------------------
# OutputProcessor
# ---------------------------------------------------------------------------


@dataclass
class DualARStepOutput:
    """Per-step output containing multi-codebook tokens."""

    codes: torch.Tensor  # [num_codebooks+1, 1]


class DualAROutputProcessor:
    """Two-stage sampling for DualARTransformer.

    1. Apply slow logits pipeline → sample semantic token from ``token_logits``.
    2. Feed hidden_states into fast transformer, autoregressively sample
       ``num_codebooks`` codebook tokens (each through fast logits pipeline).
    3. Stack into ``[num_codebooks+1, 1]`` and return.

    The fast transformer loop is deliberately kept inside the output
    processor (not the model runner) so the ``ModelRunner`` only sees a
    single ``model(**inputs)`` call for the slow transformer.
    """

    def __init__(
        self,
        model: Any,
        *,
        slow_pipeline: LogitsProcessorPipeline | None = None,
        fast_pipeline: LogitsProcessorPipeline | None = None,
        sampler: Sampler | None = None,
        num_codebooks: int = 4,
        codebook_size: int = 1024,
        semantic_begin_id: int = 0,
        radix_cache: DualARRadixCache | None = None,
        use_compile: bool = False,
    ) -> None:
        self._model = model
        self._slow_pipeline = slow_pipeline or LogitsProcessorPipeline()
        self._fast_pipeline = fast_pipeline or LogitsProcessorPipeline()
        self._sampler = sampler or MultinomialNoSyncSampler()
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._semantic_begin_id = semantic_begin_id
        self._radix_cache = radix_cache

        # torch.compile path
        self._compiled_decode = None
        self._compile_step = 0
        if use_compile:
            logger.info("Compiling decode_one_token with torch.compile ...")
            self._compiled_decode = torch.compile(
                decode_one_token,
                backend="inductor",
                mode="reduce-overhead",
                fullgraph=True,
            )

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        request = scheduler_output.requests[0]
        data: DualARRequestData = request.data
        batch_data: DualARBatchData = scheduler_output.batch_data

        # Use compiled path for decode steps (not prefill)
        is_decode = not batch_data.is_prefill
        if is_decode and self._compiled_decode is not None:
            codes = self._process_compiled(data, batch_data)
        else:
            codes = self._process_eager(model_output, data)

        # 3. Save KV to radix cache after prefill
        if batch_data.is_prefill and self._radix_cache is not None:
            total_len = data._cached_prefix_len + batch_data.input_values.shape[-1]
            kv_data = snapshot_slow_kv(self._model, total_len)
            row0_key = (
                data._original_row0.tolist()
                if data._original_row0 is not None
                else data.input_values[0].tolist()
            )
            self._radix_cache.insert(row0_key, kv_data)

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=DualARStepOutput(codes=codes),
                finished=False,
            )
        }

    def _process_compiled(
        self, data: DualARRequestData, batch_data: DualARBatchData
    ) -> Tensor:
        """Decode via compiled decode_one_token (slow + fast in one graph)."""
        device = batch_data.input_values.device
        x = batch_data.input_values.to(device)
        input_pos = batch_data.input_pos.to(device)

        temperature = torch.tensor([data.temperature], device=device)
        top_p = torch.tensor([data.top_p], device=device)
        rep_penalty = torch.tensor([data.repetition_penalty], device=device)

        # Build previous_tokens window: [num_codebooks+1, window_size]
        previous_tokens = self._get_compiled_window(data, device)

        from torch.nn.attention import SDPBackend, sdpa_kernel

        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            codes = self._compiled_decode(
                self._model._model if hasattr(self._model, "_model") else self._model,
                x,
                input_pos,
                temperature,
                top_p,
                rep_penalty,
                previous_tokens,
                self._semantic_begin_id,
                self._num_codebooks,
                self._codebook_size,
            )
        self._compile_step += 1
        if self._compile_step <= 3:
            logger.info("Compiled decode step %d (warmup)", self._compile_step)
        return codes.clone()

    def _process_eager(self, model_output: Any, data: DualARRequestData) -> Tensor:
        """Decode via eager execution (original path)."""
        ctx = SamplingContext(
            request_id="",
            temperature=data.temperature,
            top_p=data.top_p,
            repetition_penalty=data.repetition_penalty,
            previous_tokens=self._get_window(data, row=0),
            step=len(data.output_codes),
        )

        # 1. Slow transformer → semantic token
        token_logits = model_output.logits
        hidden_states = model_output.hidden_states

        slow_logits = token_logits[:, -1:, :].squeeze(1)
        slow_logits = self._slow_pipeline(slow_logits, ctx)
        semantic_out = self._sampler.sample(slow_logits, ctx)
        semantic_token = semantic_out.token_ids

        codebooks = [semantic_token]

        # 2. Fast transformer → codebook tokens
        self._clear_fast_kv_cache()

        fast_input_pos = torch.tensor(
            [0], device=hidden_states.device, dtype=torch.long
        )
        self._model.forward_generate_fast(hidden_states.squeeze(1), fast_input_pos)

        cb_input = semantic_token - self._semantic_begin_id
        cb_input = cb_input.clamp(min=0)
        cb_hidden = self._model.fast_embeddings(cb_input)
        codebooks.append(cb_input)

        for cb_idx in range(1, self._num_codebooks):
            fast_input_pos = torch.tensor(
                [cb_idx], device=cb_hidden.device, dtype=torch.long
            )
            cb_logits = self._model.forward_generate_fast(cb_hidden, fast_input_pos)
            cb_logits = cb_logits[:, :, : self._codebook_size]
            cb_logits = cb_logits.squeeze(1)

            fast_ctx = SamplingContext(
                request_id="",
                temperature=data.temperature,
                top_p=data.top_p,
                repetition_penalty=data.repetition_penalty,
                previous_tokens=self._get_window(data, row=cb_idx + 1),
                step=len(data.output_codes),
                metadata={"codebook_idx": cb_idx},
            )

            cb_logits = self._fast_pipeline(cb_logits, fast_ctx)
            cb_out = self._sampler.sample(cb_logits, fast_ctx)
            cb_hidden = self._model.fast_embeddings(cb_out.token_ids)
            codebooks.append(cb_out.token_ids)

        return torch.stack(codebooks, dim=1).T  # [num_codebooks+1, 1]

    _COMPILED_WINDOW: int = 16

    def _get_compiled_window(
        self, data: DualARRequestData, device: torch.device
    ) -> Tensor:
        """Build previous_tokens tensor for compiled path.

        Always returns a fixed-size ``[1, num_codebooks+1, WINDOW]`` tensor
        (zero-padded on the left) to avoid shape-triggered recompilations.
        """
        W = self._COMPILED_WINDOW
        rows = self._num_codebooks + 1
        if not data.output_codes:
            return torch.zeros((1, rows, W), dtype=torch.int, device=device)
        recent = data.output_codes[-W:]
        # Each code is [num_codebooks+1, 1], cat → [num_codebooks+1, actual_W]
        tokens = torch.cat(recent, dim=-1).to(device)
        actual = tokens.shape[-1]
        if actual < W:
            pad = torch.zeros((rows, W - actual), dtype=tokens.dtype, device=device)
            tokens = torch.cat([pad, tokens], dim=-1)
        return tokens.unsqueeze(0)  # [1, num_codebooks+1, W]

    def _clear_fast_kv_cache(self) -> None:
        for layer in self._model.fast_layers:
            if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
                layer.attention.kv_cache.k_cache.fill_(0)
                layer.attention.kv_cache.v_cache.fill_(0)

    def _get_window(
        self, data: DualARRequestData, row: int, window: int = 16
    ) -> torch.Tensor | None:
        if not data.output_codes:
            return None
        recent = data.output_codes[-window:]
        tokens = torch.cat([c[row : row + 1, :] for c in recent], dim=-1)
        return tokens.squeeze(0)


# ---------------------------------------------------------------------------
# IterationController
# ---------------------------------------------------------------------------


class DualARIterationController:
    """Stop when ``<|im_end|>`` appears in the semantic token (row 0)."""

    def __init__(self, im_end_token_id: int, max_new_tokens: int = 2048) -> None:
        self._im_end_id = im_end_token_id
        self._max_new_tokens = max_new_tokens

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: DualARRequestData = request.data
        step_out: DualARStepOutput = output.data
        data.output_codes.append(step_out.codes.clone())

        if data.num_computed_tokens == 0:
            data.num_computed_tokens = data.input_values.shape[1]

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        data: DualARRequestData = request.data
        step_out: DualARStepOutput = output.data

        semantic_token = step_out.codes[0, -1].item()
        if semantic_token == self._im_end_id:
            return True

        max_tok = data.max_new_tokens or self._max_new_tokens
        if len(data.output_codes) >= max_tok:
            return True

        return False
