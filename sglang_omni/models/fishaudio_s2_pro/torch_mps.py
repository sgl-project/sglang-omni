# SPDX-License-Identifier: Apache-2.0
"""Eager Fish Slow-AR and Fast-AR execution for the Torch/MPS backend.

Keep Fish's interleaved BF16 RoPE, reference embeddings, and codebook sampler;
only replace the CUDA attention/cache execution contract. One scheduler request
owns the native Slow-AR cache at a time.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.audio_decoder import (
    TransformerBlock,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.utils import (
    apply_rotary_emb,
    precompute_freqs_cis,
)
from sglang_omni.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni.models.fishaudio_s2_pro.sglang_model import S2ProSGLangTextModel


def sdpa_attention(q, k, v, *, causal: bool = False):
    """Fish NHD attention, with explicit GQA expansion supported by MPS."""
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    repeats = q.shape[1] // k.shape[1]
    if repeats != 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2)


def fast_attention(q, k_cache, v_cache, k, v, cache_position: int):
    """Append one Fast-AR position and attend only to its initialized prefix."""
    if k is None or v is None:
        raise ValueError("Fish MPS Fast-AR expects one new key/value position")
    if q.shape[1] != 1 or k.shape[1] != 1 or v.shape[1] != 1:
        raise ValueError("Fish MPS Fast-AR expects one new key/value position")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("Fish MPS Fast-AR query/key/value batch sizes must match")
    if not 0 <= cache_position < k_cache.shape[1]:
        raise ValueError("Fish MPS Fast-AR cache position is out of bounds")
    k_cache[:, cache_position : cache_position + 1].copy_(k)
    v_cache[:, cache_position : cache_position + 1].copy_(v)
    return sdpa_attention(
        q, k_cache[:, : cache_position + 1], v_cache[:, : cache_position + 1]
    )


def seeded_choice(probs, seeds, positions):
    """CPU equivalent of SGLang's MurmurHash3/Gumbel sampler for a small top-k.

    MurmurHash3 x86_32 hashes four little-endian words: seed low/high, step,
    column. int64 arithmetic plus a uint32 mask avoids both Triton and MPS's
    unsupported uint64/float64 operations. No process-global RNG is modified.
    """
    mask = 0xFFFFFFFF
    seed = seeds.cpu().long().reshape(-1, 1)
    step = positions.cpu().long().reshape(-1, 1)
    columns = torch.arange(probs.shape[1], dtype=torch.long)[None]
    h = torch.zeros((probs.shape[0], probs.shape[1]), dtype=torch.long)
    for word in (seed & mask, (seed >> 32) & mask, step & mask, columns):
        k = (word * 0xCC9E2D51) & mask
        k = ((k << 15) | (k >> 17)) & mask
        k = (k * 0x1B873593) & mask
        h = h ^ k
        h = ((h << 13) | (h >> 19)) & mask
        h = (h * 5 + 0xE6546B64) & mask
    h = h ^ 16
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & mask
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & mask
    h = h ^ (h >> 16)
    uniform = h.double() / mask
    noise = -(
        -uniform.log().clamp(min=torch.finfo(torch.float64).min, max=-(2.0**-32))
    ).log()
    return (
        (noise + probs.log().cpu().double())
        .argmax(dim=1, keepdim=True)
        .to(probs.device)
    )


class S2ProTorchMpsTextModel(S2ProSGLangTextModel):
    """Native Torch Slow-AR layers with the existing Fish decode-buffer API."""

    def __init__(self, config: Any, quant_config: Any = None, **kwargs):
        nn.Module.__init__(self)
        if quant_config is not None:
            raise ValueError("Fish Torch/MPS currently requires unquantized weights")
        tc = config.text_config
        if tc.use_moe:
            raise ValueError("Fish Torch/MPS requires the dense S2-Pro checkpoint")
        self.vocab_size = tc.vocab_size
        self.hidden_size = tc.dim
        self.num_layers = tc.n_layer
        self.start_layer, self.end_layer = 0, tc.n_layer
        self.tie_word_embeddings = tc.tie_word_embeddings
        self._vq_ready = False
        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.dim)
        self.layers = nn.ModuleList([TransformerBlock(tc) for _ in range(tc.n_layer)])
        self.norm = nn.RMSNorm(tc.dim, eps=tc.norm_eps)
        if not tc.tie_word_embeddings:
            self.lm_head = nn.Linear(tc.dim, tc.vocab_size, bias=False)
        # torch.polar used by the canonical helper is constructed on CPU.
        with torch.device("cpu"):
            phases = precompute_freqs_cis(tc.max_seq_len, tc.head_dim, tc.rope_base)
        self.register_buffer("freqs_cis", phases, persistent=False)
        self._request_caches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        parameters = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            if name.startswith("text_model.model."):
                target = name.removeprefix("text_model.model.")
                if target == "embeddings.weight":
                    target = "embed_tokens.weight"
            elif name == "text_model.lm_head.weight" and not self.tie_word_embeddings:
                target = "lm_head.weight"
            else:
                continue
            if target not in parameters:
                raise ValueError(f"Unexpected Fish Torch/MPS text weight: {name}")
            parameter = parameters[target]
            if parameter.shape != value.shape:
                raise ValueError(f"Fish Torch/MPS weight shape mismatch: {name}")
            parameter.data.copy_(value)
            loaded.add(target)
        missing = parameters.keys() - loaded
        if missing:
            raise ValueError(f"Missing Fish Torch/MPS weights: {sorted(missing)}")
        return loaded

    def clear_request(self, request_id: str) -> None:
        self._request_caches.pop(request_id, None)

    def _sample_semantic_choice(self, probs, seeds, positions):
        # The upstream seeded sampler uses uint64 and float64, unsupported by
        # MPS. Only its small top-k distribution crosses to CPU; preserve its
        # exact hash/Gumbel algorithm without invoking torch.compile.
        if int(seeds[0]) < 0:
            return torch.multinomial(probs, num_samples=1)
        return seeded_choice(probs, seeds, positions)

    @torch.inference_mode()
    def forward_native(
        self, input_ids, *, request_id, input_embeds=None, prefill=False
    ):
        if prefill:
            self._request_caches.clear()
            previous = [None] * len(self.layers)
            offset = 0
        else:
            if request_id not in self._request_caches:
                raise RuntimeError(f"Fish Torch/MPS has no cache for {request_id}")
            previous = self._request_caches[request_id]
            offset = previous[0][0].shape[1]
        if input_ids.ndim != 1 or (not prefill and input_ids.numel() != 1):
            raise ValueError("Fish Torch/MPS requires one unchunked request")
        end = offset + input_ids.numel()
        if end > self.freqs_cis.shape[0]:
            raise ValueError("Fish Torch/MPS request exceeds the RoPE context")
        x = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        if input_embeds is None and not prefill and self._vq_ready:
            codes = self._vq_codes[:1] + self._vq_codebook_offsets[None]
            combined = (x + self._vq_codebook_embeddings(codes).sum(1)) * self._vq_scale
            x = torch.where(self._vq_mask[:1, None], combined, x)
        x = x.unsqueeze(0)
        freqs = self.freqs_cis[offset:end].to(x.device)
        caches = []
        for layer, cache in zip(self.layers, previous):
            attn = layer.attention
            h = layer.attention_norm(x)
            q_size, kv_size = (
                attn.n_head * attn.head_dim,
                attn.n_local_heads * attn.head_dim,
            )
            q, k, v = attn.wqkv(h).split([q_size, kv_size, kv_size], dim=-1)
            q = q.view(1, h.shape[1], attn.n_head, attn.head_dim)
            k = k.view(1, h.shape[1], attn.n_local_heads, attn.head_dim)
            v = v.view(1, h.shape[1], attn.n_local_heads, attn.head_dim)
            if attn.attention_qk_norm:
                q, k = attn.q_norm(q), attn.k_norm(k)
            q, k = apply_rotary_emb(q, freqs), apply_rotary_emb(k, freqs)
            if cache is not None:
                k, v = torch.cat((cache[0], k), dim=1), torch.cat((cache[1], v), dim=1)
            caches.append((k, v))
            h = sdpa_attention(q, k, v, causal=prefill).reshape(1, h.shape[1], q_size)
            x = x + attn.wo(h)
            x = x + layer.feed_forward(layer.ffn_norm(x))
        hidden = self.norm(x[:, -1])
        logits = (
            F.linear(hidden, self.embed_tokens.weight)
            if self.tie_word_embeddings
            else self.lm_head(hidden)
        )
        if self._vq_ready:
            self._decode_codebooks(logits, hidden)
        self._request_caches[request_id] = caches
        return logits, hidden


class FishS2ProTorchMpsRunner(FishS2ProModelRunner):
    """Preserve Fish request/stream adapters and scheduler-owned completion."""

    def lookahead_eligible(self, batch):
        return False

    def _forward_native(self, forward_batch, schedule_batch, requests, *, prefill):
        from sglang.srt.managers.scheduler import GenerationBatchResult

        if len(requests) != 1:
            raise RuntimeError("Fish Torch/MPS requires max_running_requests=1")
        self.model.forward_native(
            schedule_batch.input_ids,
            request_id=requests[0].request_id,
            input_embeds=forward_batch.input_embeds if prefill else None,
            prefill=prefill,
        )
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=self.model._output_semantic_ids[:1].clone(),
            can_run_cuda_graph=False,
        )

    def custom_prefill_forward(self, forward_batch, schedule_batch, requests):
        return self._forward_native(
            forward_batch, schedule_batch, requests, prefill=True
        )

    def custom_decode_forward(self, forward_batch, schedule_batch, requests):
        return self._forward_native(
            forward_batch, schedule_batch, requests, prefill=False
        )

    def on_request_finished(self, request_id, req_data):
        self.model.clear_request(request_id)

    def abort_request(self, request_id):
        self.model.clear_request(request_id)
