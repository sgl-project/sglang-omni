# SPDX-License-Identifier: Apache-2.0
"""S2-Pro SGLang runtime with unified slow+fast head decode.

The text model's forward() runs the entire decode step:
  1. VQ embedding combination (from previous step's codebook values)
  2. Transformer forward (slow head, paged KV via RadixAttention)
  3. Constrained semantic sampling (argmax)
  4. Batched codebook generation (fast head, static KV cache)

The whole step is captured by CUDA graph for decode. Prefill still
injects VQ embeddings via input_embeds (not graphed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.mem_cache.common import release_kv_cache

from sglang_omni.engines.omni.runtime.sglang_ar import (
    SGLangARRequestData,
    SGLangBatchPlanner,
    SGLangResourceManager,
)
from sglang_omni.engines.omni.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

from .s2pro_ar import S2ProStepOutput

if TYPE_CHECKING:
    from sglang_omni.engines.ar.sglang_backend.model_worker import ModelWorker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class S2ProSGLangRequestData(SGLangARRequestData):
    # VQ embedding data for prefill
    vq_mask_tokens: torch.Tensor | None = None
    vq_parts: list[torch.Tensor] | None = None

    num_codebooks: int = 10
    codebook_size: int = 4096
    output_codes: list[torch.Tensor] = field(default_factory=list)
    max_new_tokens: int | None = None

    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 1.1

    ras_window: int = 16
    ras_temperature: float = 1.5
    ras_top_p: float = 0.95

    _previous_semantic_tokens: list[int] = field(default_factory=list)
    _last_codebook_values: torch.Tensor | None = None


# ---------------------------------------------------------------------------
# Codebook loop (standalone for torch.compile compatibility)
# ---------------------------------------------------------------------------


def _codebook_loop_impl(
    audio_decoder: Any,
    hidden_states: Tensor,
    semantic_token: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    top_k: int,
    num_codebooks: int,
    codebook_size: int,
    semantic_begin_id: int,
) -> Tensor:
    """Generate codebook tokens from audio decoder. Compilable with fullgraph=True."""
    audio_decoder.reset_caches()

    fast_input = audio_decoder.project_in(hidden_states.squeeze(1))
    fast_input = fast_input.unsqueeze(1)
    audio_decoder.forward_kvcached(fast_input, codebook_idx=0)

    sem_id = semantic_token.squeeze(-1) - semantic_begin_id
    sem_id = sem_id.clamp(min=0)
    cb_hidden = audio_decoder.embeddings(sem_id).unsqueeze(1)
    codebooks = [semantic_token.squeeze(-1), sem_id]

    for cb_idx in range(1, num_codebooks):
        cb_logits = audio_decoder.forward_kvcached(cb_hidden, codebook_idx=cb_idx)
        cb_logits = cb_logits[:, 0, :codebook_size]

        cb_token = _sample_with_topk(cb_logits, temperature, top_p, top_k=top_k)
        cb_hidden = audio_decoder.embeddings(cb_token.squeeze(-1)).unsqueeze(1)
        codebooks.append(cb_token.squeeze(-1))

    return torch.stack(codebooks, dim=1).T  # [num_codebooks+1, 1]


# ---------------------------------------------------------------------------
# OutputProcessor: two-stage sampling (semantic + codebooks)
# ---------------------------------------------------------------------------


class S2ProSGLangOutputProcessor:
    """Two-stage output processor for S2-Pro on SGLang backend.

    After the text model produces logits + hidden states via ForwardBatch,
    applies constrained decoding + RAS + codebook generation using the
    audio decoder's static KVCache.
    """

    def __init__(
        self,
        audio_decoder: Any,
        *,
        num_codebooks: int = 10,
        codebook_size: int = 4096,
        semantic_begin_id: int = 0,
        semantic_end_id: int = 0,
        im_end_id: int = 0,
        top_k: int = 30,
        ras_window: int = 16,
        ras_temperature: float = 1.5,
        ras_top_p: float = 0.95,
        use_torch_compile: bool = False,
        align_logits_to_bf16: bool = True,
    ) -> None:
        self._audio_decoder = audio_decoder
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._semantic_begin_id = semantic_begin_id
        self._semantic_end_id = semantic_end_id
        self._im_end_id = im_end_id
        self._top_k = top_k
        self._ras_window = ras_window
        self._ras_temperature = ras_temperature
        self._ras_top_p = ras_top_p
        self._align_logits_to_bf16 = align_logits_to_bf16
        self._semantic_bias: Tensor | None = None

        import functools

        _cb_fn = functools.partial(
            _codebook_loop_impl,
            audio_decoder,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            semantic_begin_id=semantic_begin_id,
        )
        if use_torch_compile:
            # Avoid CUDA-graph capture inside inductor for sampling stability.
            compile_mode = "max-autotune-no-cudagraphs"
            try:
                self._codebook_fn = torch.compile(
                    _cb_fn, mode=compile_mode, fullgraph=True
                )
            except Exception:
                logger.warning(
                    "torch.compile mode '%s' unavailable; fallback to max-autotune.",
                    compile_mode,
                )
                self._codebook_fn = torch.compile(
                    _cb_fn, mode="max-autotune", fullgraph=True
                )
        else:
            self._codebook_fn = _cb_fn

    @staticmethod
    def _truncate_to_bf16(logits: Tensor) -> Tensor:
        # Match fish-speech training/inference numerics:
        # keep fp32 sampling math but quantize logits to bf16 grid first.
        return logits.to(torch.bfloat16).to(torch.float32)

    def _get_semantic_bias(self, logits: Tensor) -> Tensor:
        if self._semantic_bias is None:
            bias = torch.full(
                (logits.shape[-1],),
                -float("inf"),
                device=logits.device,
                dtype=logits.dtype,
            )
            bias[self._semantic_begin_id : self._semantic_end_id + 1] = 0.0
            bias[self._im_end_id] = 0.0
            self._semantic_bias = bias
        return self._semantic_bias.to(device=logits.device, dtype=logits.dtype)

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Process text model output into S2-Pro step outputs with codebooks."""
        outputs = {}

        # model_output is GenerationBatchResult from SGLang
        # We need logits and hidden states from the text model
        logits_output = model_output.logits_output
        # logits_output.next_token_logits: (batch, vocab)
        # logits_output.hidden_states: (batch, dim) — last-token hidden states
        next_token_logits = logits_output.next_token_logits
        hidden_states = logits_output.hidden_states

        for i, sched_req in enumerate(scheduler_output.requests):
            data: S2ProSGLangRequestData = sched_req.data

            # Skip chunked prefill steps
            if data.req.is_chunked > 0:
                outputs[sched_req.request_id] = RequestOutput(
                    request_id=sched_req.request_id,
                    data=None,
                    finished=False,
                )
                continue

            token_logits = next_token_logits[i : i + 1]  # [1, vocab]
            if self._align_logits_to_bf16 and token_logits.is_floating_point():
                token_logits = self._truncate_to_bf16(token_logits)
            token_hidden = hidden_states[i : i + 1].unsqueeze(1)  # [1, 1, dim]

            codes = self._two_stage_decode(token_logits, token_hidden, data)

            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=S2ProStepOutput(codes=codes),
                finished=False,
            )

        return outputs

    @torch.no_grad()
    def _two_stage_decode(
        self,
        token_logits: Tensor,
        hidden_states: Tensor,
        data: S2ProSGLangRequestData,
    ) -> Tensor:
        """Semantic token sampling + codebook generation."""
        device = token_logits.device

        # Constrained decoding: mask non-semantic tokens
        semantic_bias = self._get_semantic_bias(token_logits)
        token_logits = token_logits + semantic_bias

        # RAS check
        use_ras = False
        if len(data._previous_semantic_tokens) > 0:
            recent = data._previous_semantic_tokens[-self._ras_window :]
            if len(recent) >= 2 and len(set(recent[-4:])) < len(recent[-4:]):
                use_ras = True

        if use_ras:
            temperature = torch.tensor([self._ras_temperature], device=device)
            top_p = torch.tensor([self._ras_top_p], device=device)
        else:
            temperature = torch.tensor([data.temperature], device=device)
            top_p = torch.tensor([data.top_p], device=device)

        rep_penalty = torch.tensor([data.repetition_penalty], device=device)
        prev_tokens = None
        if data._previous_semantic_tokens:
            prev_tokens = torch.tensor(
                data._previous_semantic_tokens[-16:],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)

        # Sample semantic token
        semantic_token = _sample_with_topk(
            token_logits,
            temperature,
            top_p,
            top_k=data.top_k,
            repetition_penalty=rep_penalty,
            previous_tokens=prev_tokens,
        )  # [1, 1]

        # Codebook generation (compiled or eager)
        return self._codebook_fn(
            hidden_states, semantic_token, temperature, top_p, data.top_k
        )


# ---------------------------------------------------------------------------
# IterationController
# ---------------------------------------------------------------------------


class S2ProSGLangIterationController:
    def __init__(
        self,
        tree_cache: Any,
        im_end_token_id: int,
        max_new_tokens: int = 2048,
    ) -> None:
        self.tree_cache = tree_cache
        self._im_end_id = im_end_token_id
        self._max_new_tokens = max_new_tokens

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: S2ProSGLangRequestData = request.data
        req = data.req

        if req.is_chunked > 0:
            output.data = None
            req.is_chunked -= 1
            return

        codes = output.data.codes.clone()
        data.output_codes.append(codes)

        semantic_token = codes[0, -1].item()
        data._previous_semantic_tokens.append(semantic_token)

        # Codebook values for next step's VQ embedding (clone is redundant
        # since `codes` is already cloned above, but guards against future edits)
        data._last_codebook_values = codes[1:, 0].clone()

        req.output_ids.append(semantic_token)

        if not req.finished() and req.decode_batch_idx == 0:
            self.tree_cache.cache_unfinished_req(req)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        data: S2ProSGLangRequestData = request.data

        if data.req.is_chunked > 0:
            return False

        semantic_token = output.data.codes[0, -1].item()
        if semantic_token == self._im_end_id:
            return True

        max_tok = data.max_new_tokens or self._max_new_tokens
        if len(data.output_codes) >= max_tok:
            return True

        return False


# ---------------------------------------------------------------------------
# ModelRunner: unified slow+fast head
# ---------------------------------------------------------------------------


class S2ProSGLangModelRunner:
    """Unified model runner: text model + codebook decode in one forward pass.

    Prefill: injects VQ embeddings via input_embeds.
    Decode: updates model's persistent VQ buffers, then model.forward()
    runs the full step (transformer + sampling + codebook loop).
    """

    def __init__(
        self,
        model_worker: "ModelWorker",
        batch_planner: SGLangBatchPlanner,
        semantic_begin_id: int,
        semantic_end_id: int,
    ):
        self.model_worker = model_worker
        self.batch_planner = batch_planner
        self._semantic_begin_id = semantic_begin_id
        self._semantic_end_id = semantic_end_id

    def _inject_vq_embeds_prefill(
        self,
        model_worker_batch: Any,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Compute input_embeds with VQ injection for prefill."""
        device = model_worker_batch.input_ids.device
        text_model = self.model_worker.model_runner.model
        audio_decoder = text_model._audio_decoder
        embed_tokens = text_model.get_embed_tokens()

        input_ids = model_worker_batch.input_ids
        text_embeds = embed_tokens(input_ids)

        offset = 0
        for sched_req in scheduler_output.requests:
            data: S2ProSGLangRequestData = sched_req.data
            req_len = data.req.extend_input_len

            if (
                data.vq_mask_tokens is not None
                and data.vq_parts is not None
                and len(data.vq_parts) > 0
            ):
                vq_mask = data.vq_mask_tokens.to(device)
                if vq_mask.dim() == 2:
                    vq_mask = vq_mask.squeeze(0)

                prefix_len = len(data.req.prefix_indices)
                mask_slice = vq_mask[prefix_len : prefix_len + req_len]

                parts = [p.to(device).T for p in data.vq_parts if p.dim() == 2]
                vq_parts_flat = torch.cat(parts, dim=0) if parts else None

                if vq_parts_flat is not None and mask_slice.any():
                    vq_before = (
                        vq_mask[:prefix_len].sum().item() if prefix_len > 0 else 0
                    )
                    num_vq_in_slice = mask_slice.sum().item()
                    vq_slice = vq_parts_flat[vq_before : vq_before + num_vq_in_slice]
                    req_embeds = text_embeds[offset : offset + req_len]
                    vq_embeds = audio_decoder.embed_text_dim(
                        req_embeds.unsqueeze(0),
                        vq_slice,
                        mask_slice.unsqueeze(0),
                    )
                    mask_indices = mask_slice.nonzero(as_tuple=True)[0] + offset
                    text_embeds[mask_indices] = vq_embeds.to(text_embeds.dtype)

            offset += req_len

        model_worker_batch.input_embeds = text_embeds

    def _update_vq_buffers(
        self,
        model_worker_batch: Any,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Update the text model's persistent VQ buffers for decode."""
        text_model = self.model_worker.model_runner.model
        input_ids = model_worker_batch.input_ids
        bs = input_ids.shape[0]

        is_semantic = (input_ids >= self._semantic_begin_id) & (
            input_ids <= self._semantic_end_id
        )
        text_model._vq_mask[:bs].copy_(is_semantic)

        for i, sched_req in enumerate(scheduler_output.requests):
            data: S2ProSGLangRequestData = sched_req.data
            if data._last_codebook_values is not None and is_semantic[i]:
                text_model._vq_codes[i].copy_(data._last_codebook_values)

    def _build_outputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        """Read codes from model's persistent output buffers."""
        text_model = self.model_worker.model_runner.model
        outputs = {}
        for i, sched_req in enumerate(scheduler_output.requests):
            data: S2ProSGLangRequestData = sched_req.data
            if data.req.is_chunked > 0:
                outputs[sched_req.request_id] = RequestOutput(
                    request_id=sched_req.request_id,
                    data=None,
                    finished=False,
                )
                continue

            # [num_codebooks+1] → [num_codebooks+1, 1] for existing format
            codes = text_model._output_codes[i].unsqueeze(-1)
            outputs[sched_req.request_id] = RequestOutput(
                request_id=sched_req.request_id,
                data=S2ProStepOutput(codes=codes),
                finished=False,
            )
        return outputs

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        schedule_batch = scheduler_output.batch_data
        model_worker_batch = schedule_batch.get_model_worker_batch()
        is_prefill = schedule_batch.forward_mode.is_extend()

        if is_prefill:
            self._inject_vq_embeds_prefill(model_worker_batch, scheduler_output)
        else:
            self._update_vq_buffers(model_worker_batch, scheduler_output)

        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.model_worker.model_runner
        )
        batch_result = self.model_worker.forward_batch_generation(forward_batch)

        if schedule_batch.is_prefill_only:
            batch_result.next_token_ids = torch.zeros(
                len(model_worker_batch.seq_lens),
                dtype=torch.long,
                device=model_worker_batch.input_ids.device,
            )

        self.batch_planner.record_last_batch(schedule_batch)

        # Codes produced by model.forward() → read from output buffers
        outputs = self._build_outputs(scheduler_output)

        text_model = self.model_worker.model_runner.model
        bs = len(scheduler_output.requests)
        schedule_batch.output_ids = text_model._output_semantic_ids[:bs].clone()

        req_ids = [req.request_id for req in scheduler_output.requests]
        req_id_to_index = {req_id: idx for idx, req_id in enumerate(req_ids)}

        return ModelRunnerOutput(
            outputs=outputs,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


class S2ProSGLangResourceManager(SGLangResourceManager):
    def free(self, request: SchedulerRequest) -> None:
        data: S2ProSGLangRequestData = request.data
        release_kv_cache(data.req, self.tree_cache)
        data._previous_semantic_tokens.clear()
        data._last_codebook_values = None
