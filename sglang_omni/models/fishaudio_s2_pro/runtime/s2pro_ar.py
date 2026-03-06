# SPDX-License-Identifier: Apache-2.0
"""S2-Pro (FishQwen3OmniForCausalLM) runtime for sglang-omni.

Key differences from S1's DualAR runtime:
1. Input is 1D token IDs with VQ masks (not multi-row [num_codebooks+1, seq_len])
2. Forward call: model.forward_kvcached(input_ids, input_pos, input_embeds)
3. Hidden states: post-norm (output.token_hidden_states)
4. Codebook generation: model.audio_decoder.project_in() + forward_kvcached()
5. Codebook scaling: sqrt(num_codebooks+1) always
6. Sampling: top-p + top-k=30 + RAS (Repetition Aware Sampling) + constrained decoding
7. KV cache shape: [B, max_seq_len, H, D] (flash_attn style)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from sglang_omni.engines.omni.runtime.common import SimpleResourceManager
from sglang_omni.engines.omni.runtime.interfaces import ResourceManager
from sglang_omni.engines.omni.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class S2ProRequestData:
    """S2-Pro-specific request data (stored in ``SchedulerRequest.data``).

    Unlike DualAR, input is 1D token IDs with separate VQ mask/parts:
    - ``input_ids``: [seq_len] token IDs
    - ``vq_mask_tokens``: [seq_len] boolean mask for VQ positions
    - ``vq_parts``: list of [num_codebooks, T_i] VQ code tensors
    """

    input_ids: torch.Tensor  # [seq_len]
    vq_mask_tokens: torch.Tensor | None = None  # [seq_len] bool
    vq_parts: list[torch.Tensor] | None = None  # list of [num_codebooks, T_i]

    num_codebooks: int = 10
    codebook_size: int = 4096
    num_computed_tokens: int = 0
    output_codes: list[torch.Tensor] = field(default_factory=list)
    max_new_tokens: int | None = None

    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    repetition_penalty: float = 1.1

    # RAS (Repetition Aware Sampling) parameters
    ras_window: int = 16
    ras_temperature: float = 1.5
    ras_top_p: float = 0.95

    # Managed by the runtime
    _previous_semantic_tokens: list[int] = field(default_factory=list)
    _last_codebook_values: torch.Tensor | None = (
        None  # [num_codebooks] for embed_one_token
    )
    _cache_node: Any = None  # TreeNode from radix cache (for lock_ref lifecycle)


@dataclass
class S2ProBatchData:
    """S2-Pro-specific batch data. Single-request only (batch_size=1)."""

    input_ids: torch.Tensor  # [1, seq_len]
    input_pos: torch.Tensor  # [seq_len]
    input_embeds: torch.Tensor | None = None  # [1, seq_len, dim] pre-computed
    is_prefill: bool = True


# ---------------------------------------------------------------------------
# KV cache snapshot / restore helpers (flash_attn shape)
# ---------------------------------------------------------------------------


def snapshot_slow_kv(model: Any, length: int) -> list[tuple[Tensor, Tensor]]:
    """Clone slow KV cache slices ``[:, :length, :, :]`` from all layers.

    S2-Pro KV cache shape: [B, max_seq_len, H, D] (flash_attn style).
    """
    result = []
    for layer in model.text_model.model.layers:
        kv = layer.attention.kv_cache
        result.append(
            (
                kv.k_cache[:, :length, :, :].clone(),
                kv.v_cache[:, :length, :, :].clone(),
            )
        )
    return result


def restore_slow_kv(model: Any, kv_data: list[tuple[Tensor, Tensor]]) -> None:
    """Write saved KV slices back into model layer caches."""
    for (k, v), layer in zip(kv_data, model.text_model.model.layers):
        length = k.shape[1]
        layer.attention.kv_cache.k_cache[:, :length, :, :].copy_(k)
        layer.attention.kv_cache.v_cache[:, :length, :, :].copy_(v)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _multinomial_no_sync(probs: Tensor) -> Tensor:
    """Gumbel-max trick: sample without CUDA sync (compilable)."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def _sample_with_topk(
    logits: Tensor,
    temperature: Tensor,
    top_p: Tensor,
    top_k: int = 30,
    repetition_penalty: Tensor | None = None,
    previous_tokens: Tensor | None = None,
) -> Tensor:
    """Sample one token with top-k + top-p filtering."""
    # Repetition penalty
    if previous_tokens is not None and repetition_penalty is not None:
        prev = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=prev)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits = logits.clone()
        logits.scatter_(dim=-1, index=prev, src=score.to(logits.dtype))

    # Top-k filtering
    if top_k > 0:
        top_k_logits, top_k_indices = torch.topk(
            logits, min(top_k, logits.size(-1)), dim=-1
        )
        logits = torch.full_like(logits, -float("Inf"))
        logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # Top-p filtering
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cum_probs > top_p
    sorted_mask[..., 0] = False  # keep at least one
    indices_to_remove = sorted_mask.scatter(
        dim=-1, index=sorted_indices, src=sorted_mask
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))

    # Temperature
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return _multinomial_no_sync(probs)


# ---------------------------------------------------------------------------
# BatchPlanner
# ---------------------------------------------------------------------------


class S2ProBatchPlanner:
    """Batch planner for single-request S2-Pro execution."""

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

    def build_batch(self, requests: list[SchedulerRequest]) -> S2ProBatchData:
        request = requests[0]
        data: S2ProRequestData = request.data
        is_prefill = data.num_computed_tokens == 0

        if is_prefill:
            input_ids = data.input_ids
            input_pos = torch.arange(0, input_ids.shape[0], dtype=torch.long)

            return S2ProBatchData(
                input_ids=input_ids.unsqueeze(0),  # [1, seq_len]
                input_pos=input_pos,
                input_embeds=None,  # Will be computed by InputPreparer
                is_prefill=True,
            )
        else:
            # Decode step: feed the last generated token
            last_output = data.output_codes[-1]  # [num_codebooks+1, 1]
            semantic_token = last_output[0, 0]  # scalar token ID
            pos = data.num_computed_tokens + len(data.output_codes) - 1
            input_pos = torch.tensor([pos], dtype=torch.long)

            return S2ProBatchData(
                input_ids=semantic_token.unsqueeze(0).unsqueeze(0),  # [1, 1]
                input_pos=input_pos,
                input_embeds=None,  # Will be computed by InputPreparer
                is_prefill=False,
            )


# ---------------------------------------------------------------------------
# ResourceManager
# ---------------------------------------------------------------------------


class S2ProResourceManager(SimpleResourceManager):
    """Clears S2-Pro KV caches on free and releases radix cache locks."""

    def __init__(self, max_count: int = 1, radix_cache: Any | None = None) -> None:
        super().__init__(max_count=max_count)
        self._radix_cache = radix_cache

    def free(self, request: SchedulerRequest) -> None:
        super().free(request)
        data: S2ProRequestData = request.data

        # Release radix cache lock
        if self._radix_cache is not None and data._cache_node is not None:
            self._radix_cache.dec_lock_ref(data._cache_node)
            data._cache_node = None

        data._previous_semantic_tokens.clear()
        data._last_codebook_values = None


# ---------------------------------------------------------------------------
# InputPreparer
# ---------------------------------------------------------------------------


class S2ProInputPreparer:
    """Convert ``S2ProBatchData`` to model input kwargs.

    For prefill: uses model.embed() to handle VQ embeddings.
    For decode: uses model.embed_one_token() to combine text + VQ codes.

    When a ``radix_cache`` is provided, prefill checks for cached KV states.
    On a cache hit the matched prefix KV is restored and only the suffix
    tokens are forwarded through the model.
    """

    def __init__(self, model: Any, radix_cache: Any | None = None) -> None:
        self._model = model
        self._radix_cache = radix_cache

    def prepare(
        self,
        scheduler_output: SchedulerOutput,
        device: torch.device,
    ) -> dict[str, Any]:
        batch_data: S2ProBatchData = scheduler_output.batch_data
        request = scheduler_output.requests[0]
        data: S2ProRequestData = request.data

        input_ids = batch_data.input_ids.to(device)
        input_pos = batch_data.input_pos.to(device)

        if batch_data.is_prefill:
            # --- Radix cache: check for prefix match ---
            matched_len = 0
            if self._radix_cache is not None:
                from .radix_cache import restore_kv_to_model

                token_list = data.input_ids.tolist()
                matched_len, cached_kv, cache_node = self._radix_cache.match_prefix(
                    token_list
                )

                if matched_len > 0 and cached_kv is not None:
                    # Restore cached KV into model layers
                    actual_model = getattr(self._model, "_model", self._model)
                    restore_kv_to_model(actual_model, cached_kv, matched_len)

                    # Lock the cache node for this request's lifetime
                    self._radix_cache.inc_lock_ref(cache_node)
                    data._cache_node = cache_node

                    # Advance num_computed_tokens so decode doesn't re-prefill
                    data.num_computed_tokens = matched_len

                    logger.info(
                        "Radix cache: restored %d/%d tokens, forwarding suffix of %d",
                        matched_len,
                        len(token_list),
                        len(token_list) - matched_len,
                    )

            # Prefill: use model.embed() with VQ masks and parts
            # If cache hit, embed the FULL sequence but only forward the suffix
            # (model.embed handles VQ replacement on full sequence)
            vq_mask_tokens = None
            vq_parts_flat = None

            if data.vq_mask_tokens is not None:
                vq_mask_tokens = data.vq_mask_tokens.to(device).unsqueeze(
                    0
                )  # [1, seq_len]

            if data.vq_parts is not None and len(data.vq_parts) > 0:
                parts = []
                for p in data.vq_parts:
                    p = p.to(device)
                    if p.dim() == 2:
                        parts.append(p.T)  # [T_i, num_codebooks]
                parts_flat = torch.cat(parts, dim=0) if parts else None
                vq_parts_flat = parts_flat

            input_embeds = self._model.embed(
                input_ids=input_ids,
                vq_parts=vq_parts_flat,
                vq_mask_tokens=vq_mask_tokens,
            )

            if matched_len > 0:
                # Slice to suffix only — KV for prefix is already in cache
                input_embeds = input_embeds[:, matched_len:, :]
                input_ids = input_ids[:, matched_len:]
                input_pos = input_pos[matched_len:]

            return {
                "input_ids": input_ids,
                "input_pos": input_pos,
                "input_embeds": input_embeds,
            }
        else:
            # Decode step: embed one token with VQ codebook values
            vq_parts = data._last_codebook_values
            if vq_parts is not None:
                vq_parts = vq_parts.to(device).unsqueeze(0)  # [1, num_codebooks]

            input_embeds = self._model.embed_one_token(
                token_ids=input_ids,
                vq_parts=vq_parts,
            )

            return {
                "input_ids": input_ids,
                "input_pos": input_pos,
                "input_embeds": input_embeds,
            }


# ---------------------------------------------------------------------------
# OutputProcessor
# ---------------------------------------------------------------------------


@dataclass
class S2ProStepOutput:
    """Per-step output containing multi-codebook tokens."""

    codes: torch.Tensor  # [num_codebooks+1, 1]


class S2ProOutputProcessor:
    """Two-stage sampling for FishQwen3OmniForCausalLM.

    1. forward_kvcached → token_logits + token_hidden_states (post-norm)
    2. Apply constrained decoding (mask non-semantic tokens) + RAS
    3. Sample semantic token with top-k + top-p
    4. audio_decoder: project_in(hidden_states) → forward_kvcached for each codebook
    5. Stack into [num_codebooks+1, 1] and return
    """

    def __init__(
        self,
        model: Any,
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
    ) -> None:
        self._model = model
        self._num_codebooks = num_codebooks
        self._codebook_size = codebook_size
        self._semantic_begin_id = semantic_begin_id
        self._semantic_end_id = semantic_end_id
        self._im_end_id = im_end_id
        self._top_k = top_k
        self._ras_window = ras_window
        self._ras_temperature = ras_temperature
        self._ras_top_p = ras_top_p

        # Pre-compute semantic logit bias mask
        # We allow tokens in [semantic_begin_id, semantic_end_id] and im_end_id
        self._semantic_bias = None

    def _get_semantic_bias(self, logits: Tensor) -> Tensor:
        """Create constrained decoding mask: only allow semantic tokens and im_end."""
        if (
            self._semantic_bias is None
            or self._semantic_bias.shape[-1] != logits.shape[-1]
        ):
            bias = torch.full(
                (logits.shape[-1],),
                -float("Inf"),
                device=logits.device,
                dtype=logits.dtype,
            )
            bias[self._semantic_begin_id : self._semantic_end_id + 1] = 0.0
            bias[self._im_end_id] = 0.0
            self._semantic_bias = bias
        return self._semantic_bias.to(device=logits.device, dtype=logits.dtype)

    def cleanup(self) -> None:
        self._model = None
        self._semantic_bias = None

    def process(
        self,
        model_output: Any,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, RequestOutput]:
        request = scheduler_output.requests[0]
        data: S2ProRequestData = request.data
        batch_data: S2ProBatchData = scheduler_output.batch_data

        codes = self._process_eager(model_output, data)

        return {
            request.request_id: RequestOutput(
                request_id=request.request_id,
                data=S2ProStepOutput(codes=codes),
                finished=False,
            )
        }

    @torch.no_grad()
    def _process_eager(self, model_output: Any, data: S2ProRequestData) -> Tensor:
        """Decode via eager execution."""
        device = model_output.token_logits.device

        # 1. Get logits and hidden states from slow transformer
        token_logits = model_output.token_logits[:, -1:, :].squeeze(1)  # [1, vocab]
        hidden_states = model_output.token_hidden_states[:, -1:, :]  # [1, 1, dim]

        # 2. Apply constrained decoding: mask non-semantic tokens
        semantic_bias = self._get_semantic_bias(token_logits)
        token_logits = token_logits + semantic_bias

        # 3. RAS: check if semantic token is repeated in recent window
        use_ras = False
        if len(data._previous_semantic_tokens) > 0:
            recent = data._previous_semantic_tokens[-self._ras_window :]
            # Check for repetitions in window
            if len(recent) >= 2 and len(set(recent[-4:])) < len(recent[-4:]):
                use_ras = True

        # 4. Set sampling params (RAS overrides if triggered)
        if use_ras:
            temperature = torch.tensor([self._ras_temperature], device=device)
            top_p = torch.tensor([self._ras_top_p], device=device)
        else:
            temperature = torch.tensor([data.temperature], device=device)
            top_p = torch.tensor([data.top_p], device=device)

        rep_penalty = torch.tensor([data.repetition_penalty], device=device)

        # Build previous tokens for repetition penalty
        prev_tokens = None
        if data._previous_semantic_tokens:
            prev_tokens = torch.tensor(
                data._previous_semantic_tokens[-16:],
                device=device,
                dtype=torch.long,
            ).unsqueeze(
                0
            )  # [1, window]

        # 5. Sample semantic token
        semantic_token = _sample_with_topk(
            token_logits,
            temperature,
            top_p,
            top_k=data.top_k,
            repetition_penalty=rep_penalty,
            previous_tokens=prev_tokens,
        )  # [1, 1]

        codebooks = [semantic_token.squeeze(-1)]  # [1]

        # 6. Fast decoder (audio_decoder): generate codebook tokens
        audio_decoder = self._model.audio_decoder
        audio_decoder.reset_caches()

        # Project hidden states to fast decoder dimension
        fast_input = audio_decoder.project_in(hidden_states.squeeze(1))  # [1, dim_fast]
        fast_input = fast_input.unsqueeze(1)  # [1, 1, dim_fast]

        # Seed fast decoder with projected hidden states
        audio_decoder.forward_kvcached(fast_input, codebook_idx=0)

        # Embed semantic token for first codebook step
        sem_id = semantic_token.squeeze(-1) - self._semantic_begin_id
        sem_id = sem_id.clamp(min=0)
        cb_hidden = audio_decoder.embeddings(sem_id).unsqueeze(1)  # [1, 1, dim_fast]
        codebooks.append(sem_id)

        # Autoregressively generate remaining codebooks
        for cb_idx in range(1, self._num_codebooks):
            cb_logits = audio_decoder.forward_kvcached(
                cb_hidden, codebook_idx=cb_idx
            )  # [1, 1, codebook_vocab]
            cb_logits = cb_logits[:, 0, : self._codebook_size]  # [1, codebook_size]

            cb_token = _sample_with_topk(
                cb_logits,
                temperature,
                top_p,
                top_k=data.top_k,
            )  # [1, 1]

            cb_hidden = audio_decoder.embeddings(cb_token.squeeze(-1)).unsqueeze(1)
            codebooks.append(cb_token.squeeze(-1))

        return torch.stack(codebooks, dim=1).T  # [num_codebooks+1, 1]


# ---------------------------------------------------------------------------
# IterationController
# ---------------------------------------------------------------------------


class S2ProIterationController:
    """Stop when ``<|im_end|>`` appears in the semantic token (row 0).

    When a ``radix_cache`` is provided, the KV cache is inserted after
    the first decode step (i.e. right after prefill completes).
    """

    def __init__(
        self,
        im_end_token_id: int,
        max_new_tokens: int = 2048,
        radix_cache: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self._im_end_id = im_end_token_id
        self._max_new_tokens = max_new_tokens
        self._radix_cache = radix_cache
        self._model = model

    def update_request(self, request: SchedulerRequest, output: RequestOutput) -> None:
        data: S2ProRequestData = request.data
        step_out: S2ProStepOutput = output.data
        codes = step_out.codes.clone()
        data.output_codes.append(codes)

        # Track semantic tokens for RAS
        semantic_token = codes[0, -1].item()
        data._previous_semantic_tokens.append(semantic_token)

        # Store codebook values for next step's embed_one_token
        # codes shape: [num_codebooks+1, 1] — rows 1..N are codebook indices
        data._last_codebook_values = codes[1:, 0]  # [num_codebooks]

        if data.num_computed_tokens == 0:
            data.num_computed_tokens = data.input_ids.shape[0]

        # --- Radix cache: insert KV after first decode (prefill just finished) ---
        # len == 1 because we just appended the first output code above
        if (
            len(data.output_codes) == 1
            and self._radix_cache is not None
            and self._model is not None
        ):
            self._cache_prefill_kv(data)

    def _cache_prefill_kv(self, data: S2ProRequestData) -> None:
        """Extract KV from model and insert into radix cache after prefill."""
        from .radix_cache import extract_kv_from_model

        try:
            actual_model = getattr(self._model, "_model", self._model)
            seq_len = data.num_computed_tokens
            kv_data = extract_kv_from_model(actual_model, seq_len)
            token_list = data.input_ids.tolist()
            already = self._radix_cache.insert(token_list, kv_data)
            logger.info(
                "Radix cache: inserted %d tokens (already cached: %d)",
                len(token_list),
                already,
            )
        except Exception:
            logger.warning("Failed to insert KV into radix cache", exc_info=True)

    def is_finished(self, request: SchedulerRequest, output: RequestOutput) -> bool:
        data: S2ProRequestData = request.data
        step_out: S2ProStepOutput = output.data

        semantic_token = step_out.codes[0, -1].item()
        if semantic_token == self._im_end_id:
            return True

        max_tok = data.max_new_tokens or self._max_new_tokens
        if len(data.output_codes) >= max_tok:
            return True

        return False
