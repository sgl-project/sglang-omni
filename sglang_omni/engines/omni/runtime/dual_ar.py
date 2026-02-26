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
from typing import Any

import torch

from ..types import RequestOutput, SchedulerOutput, SchedulerRequest
from .common import SimpleResourceManager
from .interfaces import ResourceManager
from .logits_processor import LogitsProcessorPipeline, SamplingContext
from .sampler import MultinomialNoSyncSampler, Sampler, SamplerOutput

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
# BatchPlanner
# ---------------------------------------------------------------------------


class DualARBatchPlanner:
    """Batch planner for single-request DualAR execution."""

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
            seq_len = data.input_values.shape[1]
            input_values = data.input_values
            input_pos = torch.arange(0, seq_len, dtype=torch.long)
        else:
            last_codes = data.output_codes[-1]  # [num_codebooks+1, 1]
            input_values = last_codes
            pos = data.num_computed_tokens + len(data.output_codes) - 1
            input_pos = torch.tensor([pos], dtype=torch.long)

        return DualARBatchData(
            input_values=input_values.unsqueeze(0) if input_values.dim() == 2 else input_values,
            input_pos=input_pos,
            is_prefill=is_prefill,
            audio_masks=data.audio_masks if is_prefill else None,
            audio_parts=data.audio_parts if is_prefill else None,
        )


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


class DualARResourceManager(SimpleResourceManager):
    """Clears DualAR KV caches on free."""

    def free(self, request: SchedulerRequest) -> None:
        super().free(request)
        data: DualARRequestData = request.data
        data._slow_kv_cache = None
        data._previous_tokens = None


# ---------------------------------------------------------------------------
# InputPreparer
# ---------------------------------------------------------------------------


class DualARInputPreparer:
    """Convert ``DualARBatchData`` to model input kwargs."""

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        batch_data: DualARBatchData = scheduler_output.batch_data

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
    ) -> None:
        self._model = model
        self._slow_pipeline = slow_pipeline or LogitsProcessorPipeline()
        self._fast_pipeline = fast_pipeline or LogitsProcessorPipeline()
        self._sampler = sampler or MultinomialNoSyncSampler()
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._semantic_begin_id = semantic_begin_id

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        request = scheduler_output.requests[0]
        data: DualARRequestData = request.data

        ctx = SamplingContext(
            request_id=request.request_id,
            temperature=data.temperature,
            top_p=data.top_p,
            repetition_penalty=data.repetition_penalty,
            previous_tokens=self._get_window(data, row=0),
            step=len(data.output_codes),
        )

        # --- Stage 1: slow transformer → semantic token ---
        token_logits = model_output.logits  # [batch, 1, vocab]
        hidden_states = model_output.hidden_states  # [batch, 1, dim]

        slow_logits = token_logits[:, -1:, :]  # [1, 1, vocab]
        slow_logits = slow_logits.squeeze(1)  # [1, vocab]
        slow_logits = self._slow_pipeline(slow_logits, ctx)
        semantic_out = self._sampler.sample(slow_logits, ctx)
        semantic_token = semantic_out.token_ids  # [1]

        codebooks = [semantic_token]

        # --- Stage 2: fast transformer → codebook tokens ---
        self._clear_fast_kv_cache()

        fast_input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
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
            cb_logits = cb_logits.squeeze(1)  # [1, codebook_size]

            fast_ctx = SamplingContext(
                request_id=request.request_id,
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

        codes = torch.stack(codebooks, dim=1).T  # [num_codebooks+1, 1]

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=DualARStepOutput(codes=codes),
                finished=False,
            )
        }

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

    def update_request(
        self, request: SchedulerRequest, output: RequestOutput
    ) -> None:
        data: DualARRequestData = request.data
        step_out: DualARStepOutput = output.data
        data.output_codes.append(step_out.codes.clone())

        if data.num_computed_tokens == 0:
            data.num_computed_tokens = data.input_values.shape[1]

    def is_finished(
        self, request: SchedulerRequest, output: RequestOutput
    ) -> bool:
        data: DualARRequestData = request.data
        step_out: DualARStepOutput = output.data

        semantic_token = step_out.codes[0, -1].item()
        if semantic_token == self._im_end_id:
            return True

        max_tok = data.max_new_tokens or self._max_new_tokens
        if len(data.output_codes) >= max_tok:
            return True

        return False
