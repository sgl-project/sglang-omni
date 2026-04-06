import logging
import math
import os
import re as stdlib_re
import time
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin

import numpy as np
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from apex.normalization import FusedRMSNorm
    rms_norm = FusedRMSNorm
except ImportError:
    from torch.nn import RMSNorm as RMSNorm
    rms_norm = RMSNorm

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "ar": "Arabic",
    "hi": "Hindi",
    "it": "Italian",
    "pt": "Portuguese",
}


# ---- Audio special tokens ----

class AudioSpecialTokens(str, Enum):
    empty_audio = "[EMPTY_AUDIO]"
    end_audio = "[END_AUDIO]"

    @staticmethod
    def all_special_tokens() -> list["AudioSpecialTokens"]:
        return [token for token in AudioSpecialTokens]

    @staticmethod
    def id(token: "AudioSpecialTokens") -> int:
        return AudioSpecialTokens.all_special_tokens().index(token)


# ---- Dataclasses (copied from vLLM) ----

@dataclass
class AcousticTransformerArgs:
    input_dim: int
    dim: int = 768
    n_layers: int = 3
    head_dim: int = 128
    hidden_dim: int = 2048
    n_heads: int = 6
    n_kv_heads: int = 2
    use_biases: bool = False
    norm_eps: float = 1e-5
    sigma: float = 1e-5


@dataclass
class MultimodalAudioModelArgs:
    semantic_codebook_size: int
    acoustic_codebook_size: int
    n_acoustic_codebook: int
    acoustic_transformer_args: AcousticTransformerArgs

    @property
    def codebook_sizes(self) -> list[int]:
        return [
            self.semantic_codebook_size,
            *[self.acoustic_codebook_size for _ in range(self.n_acoustic_codebook)],
        ]

    def get_codebook_sizes(
        self, pad_to_multiple: int | None = 128, include_special_tokens: bool = True
    ) -> list[int]:
        def _round_up_to_multiple_of_number(n: int, multiple: int) -> int:
            return multiple * ((n + multiple - 1) // multiple)

        result_codebook_sizes = []
        for i, cb_size in enumerate(self.codebook_sizes):
            if include_special_tokens:
                cb_size += len(AudioSpecialTokens.all_special_tokens())
            if pad_to_multiple is not None:
                cb_size = _round_up_to_multiple_of_number(cb_size, pad_to_multiple)
            result_codebook_sizes.append(cb_size)
        return result_codebook_sizes


def from_nested_dict(cls, d):
    if not is_dataclass(cls):
        return d
    kwargs = {}
    for f in fields(cls):
        value = d.get(f.name, getattr(cls, f.name, None))
        field_type = f.type
        origin = get_origin(field_type)
        if origin is Union:
            args = get_args(field_type)
            non_none_types = [a for a in args if a is not type(None)]
            if len(non_none_types) == 1:
                field_type = non_none_types[0]
        if is_dataclass(field_type) and isinstance(value, dict):
            value = from_nested_dict(field_type, value)
        kwargs[f.name] = value
    return cls(**kwargs)


# ---- Acoustic Transformer components (copied from vLLM) ----

def _repeat_interleave(t: torch.Tensor, repeats: int) -> torch.Tensor:
    return t.unsqueeze(3).expand([-1, -1, -1, repeats, -1]).flatten(2, 3)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int) -> tuple[torch.Tensor, torch.Tensor]:
    if repeats > 1:
        keys = _repeat_interleave(keys, repeats=repeats)
        values = _repeat_interleave(values, repeats=repeats)
    return keys, values


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, use_biases: bool) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=use_biases)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BidirectionalAttention(nn.Module):
    def __init__(self, args: AcousticTransformerArgs, layer_id: int) -> None:
        super().__init__()
        self.args = args
        self.n_local_heads: int = args.n_heads
        self.n_local_kv_heads: int = args.n_kv_heads
        self.repeats = self.n_local_heads
        self.layer_id = layer_id
        self.head_dim = args.head_dim
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=args.use_biases)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=args.use_biases)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=args.use_biases)
        self.softmax_scale: float = self.args.head_dim**-0.5
        self.repeats = self.n_local_heads // self.n_local_kv_heads

    def _native_attention(self, query, key, value):
        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        attn = attn.softmax(-1)
        attn = attn @ value
        return attn.transpose(1, 2).contiguous()

    def _forward_attention(self, query, key, value):
        key, value = repeat_kv(key, value, repeats=self.repeats)
        bsz, seqlen, _, _ = query.shape
        output = self._native_attention(query, key, value)
        return output.view(bsz, seqlen, -1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if x.dim() == 2:
            bsz, (seqlen, _) = 1, x.shape
        else:
            bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        output = self._forward_attention(query=xq, key=xk, value=xv, **kwargs)
        output = output.view(bsz, seqlen, self.n_local_heads * self.head_dim)
        return self.wo(output).squeeze(0)


class AcousticTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: AcousticTransformerArgs) -> None:
        super().__init__()
        self._layer_id = layer_id
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = BidirectionalAttention(args, layer_id=layer_id)
        self.feed_forward = FeedForward(args.dim, args.hidden_dim, args.use_biases)
        self.attention_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.ffn_norm = rms_norm(args.dim, eps=args.norm_eps)
        self.args = args

    @property
    def layer_id(self) -> int:
        return self._layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x))
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = torch.exp(-math.log(theta) * torch.arange(dim // 2).float() / (dim // 2))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = torch.einsum("bi, j -> bj", t, self.inv_freq)
        return torch.cat((emb.cos(), emb.sin()), dim=-1)


class FlowMatchingAudioTransformer(nn.Module):
    def __init__(self, audio_model_args: dict) -> None:
        super().__init__()
        if "codebook_sizes" in audio_model_args:
            codebook_sizes = [int(c) for c in audio_model_args.pop("codebook_sizes").split(",")]
            audio_model_args.update({
                "semantic_codebook_size": codebook_sizes[0],
                "acoustic_codebook_size": codebook_sizes[1],
                "n_acoustic_codebook": len(codebook_sizes) - 1,
            })
        self.model_args: MultimodalAudioModelArgs = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        assert isinstance(self.model_args, MultimodalAudioModelArgs)
        args = self.model_args.acoustic_transformer_args
        self.acoustic_transformer_args = args
        assert isinstance(self.acoustic_transformer_args, AcousticTransformerArgs)

        self.num_non_acoustic_embeddings = 1
        self.num_acoustic_codebooks = len(self.model_args.get_codebook_sizes()) - self.num_non_acoustic_embeddings

        self.sigma = args.sigma

        acoustic_codebook_sizes = self.model_args.get_codebook_sizes(
            pad_to_multiple=None, include_special_tokens=False
        )[1:]
        assert len(set(acoustic_codebook_sizes)) == 1
        self.acoustic_embeddings_levels = acoustic_codebook_sizes[0]
        self.acoustic_embeddings_dim = len(acoustic_codebook_sizes)

        self._init_audio_embeddings_layer()
        self._init_output_layer()
        self._init_layers()

        self._end_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
        self._empty_audio_token_id = AudioSpecialTokens.id(AudioSpecialTokens.empty_audio)

        self._acoustic_decode_iters = 16
        self._cfg_alpha = 1.2
        self._noise_scale = 1.0
        self.register_buffer("_timesteps", torch.linspace(0, 1, self._acoustic_decode_iters), persistent=False)

    def load_weight(self, weight: tuple[str, torch.Tensor]) -> str:
        params_dict = dict(self.named_parameters())
        name, loaded_weight = weight
        if name not in params_dict:
            logger.warning(f"{name} not found in FlowMatchingAudioTransformer (UNUSED)")
            return name
        param = params_dict[name]
        param.data.copy_(loaded_weight)
        return name

    def _init_audio_embeddings_layer(self) -> None:
        self.time_embedding = TimeEmbedding(self.acoustic_transformer_args.dim)
        input_dim = self.acoustic_embeddings_dim
        self.input_projection = nn.Linear(input_dim, self.acoustic_transformer_args.dim, bias=False)
        self.time_projection = nn.Linear(self.acoustic_transformer_args.dim, self.acoustic_transformer_args.dim, bias=False)
        self.llm_projection = nn.Linear(self.acoustic_transformer_args.input_dim, self.acoustic_transformer_args.dim, bias=False)

    def _init_output_layer(self) -> None:
        padded_codebook_sizes = self.model_args.get_codebook_sizes(pad_to_multiple=128)
        self.semantic_codebook_output = nn.Linear(
            self.acoustic_transformer_args.dim, padded_codebook_sizes[0], self.acoustic_transformer_args.use_biases,
        )
        self.acoustic_codebook_output = nn.Linear(
            in_features=self.acoustic_transformer_args.dim,
            out_features=self.model_args.n_acoustic_codebook,
            bias=False,
        )

    def _init_layers(self) -> None:
        self.layers_ids: list[int] = list(range(self.acoustic_transformer_args.n_layers))
        self.layers = nn.ModuleDict()
        for layer_id in self.layers_ids:
            block = AcousticTransformerBlock(layer_id=layer_id, args=self.acoustic_transformer_args)
            self.layers[str(layer_id)] = block
        self.norm = rms_norm(self.acoustic_transformer_args.dim, self.acoustic_transformer_args.norm_eps)

    def forward_attention_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_id in self.layers_ids:
            layer = self.layers[str(layer_id)]
            h = layer(h)
        return h

    def decode_one_frame(self, semantic_code: torch.Tensor, llm_hidden: torch.Tensor) -> torch.Tensor:
        B = semantic_code.shape[0]
        should_decode = semantic_code != self._end_audio_token_id
        x_0 = torch.randn(B, self.model_args.n_acoustic_codebook).to(dtype=llm_hidden.dtype, device=llm_hidden.device)
        x_0 = self._noise_scale * x_0
        timesteps = self._timesteps.to(dtype=llm_hidden.dtype)
        llm_hidden_zero = torch.zeros_like(llm_hidden)

        sampled = x_0
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            t_emb = self.time_embedding(t.view(-1, 1).repeat(B, 1)).to(llm_hidden.dtype)
            x_batched = torch.cat([sampled, sampled], dim=0)
            llm_batched = torch.cat([llm_hidden, llm_hidden_zero], dim=0)
            t_emb_batched = torch.cat([t_emb, t_emb], dim=0)
            v_all = self._predict_velocity(x_t=x_batched, llm_output=llm_batched, t_emb=t_emb_batched)
            v_t, uncond_v_t = v_all[:B], v_all[B:]
            v_t = self._cfg_alpha * v_t + (1 - self._cfg_alpha) * uncond_v_t
            sampled = sampled + v_t * dt

        sampled = torch.clamp(sampled, -1, 1)
        scaled_x = ((sampled + 1) / 2) * (self.acoustic_embeddings_levels - 1)
        output_codes = scaled_x.round().long()
        output_codes[~should_decode] = self._empty_audio_token_id
        return output_codes + len(AudioSpecialTokens)

    def _predict_velocity(self, x_t, llm_output, t_emb):
        x_t = x_t.to(llm_output.dtype)
        t_emb = self.time_projection(t_emb)
        llm_output = self.llm_projection(llm_output)
        acoustic_and_semantic_embeddings = [
            self.input_projection(x_t.unsqueeze(1)),
            t_emb.unsqueeze(1),
            llm_output.unsqueeze(1),
        ]
        acoustic_transformer_inputs = torch.concatenate(acoustic_and_semantic_embeddings, dim=1)
        attn_output = self.forward_attention_layers(acoustic_transformer_inputs)
        final_hidden = self.norm(attn_output)
        final_hidden = final_hidden.view(-1, acoustic_transformer_inputs.shape[1], final_hidden.shape[-1])
        v_t = self.acoustic_codebook_output(final_hidden[:, 0, :])
        return v_t

    def forward(self, llm_hidden: torch.Tensor) -> torch.Tensor:
        semantic_logit = self.semantic_codebook_output(llm_hidden).float()
        semantic_logit[:, self._empty_audio_token_id] = -float("inf")
        semantic_logit[:, (len(AudioSpecialTokens) + self.model_args.semantic_codebook_size):] = -float("inf")
        semantic_code = semantic_logit.argmax(dim=-1, keepdim=True)
        acoustic_codes = self.decode_one_frame(semantic_code.squeeze(1), llm_hidden)
        audio_codes = torch.concatenate([semantic_code, acoustic_codes], dim=1)
        return audio_codes


# ---- MultiVocabEmbeddings (copied from vLLM audio_tokenizer) ----

class MultiVocabEmbeddings(nn.Module):
    def __init__(self, audio_model_args: dict, embedding_dim: int) -> None:
        super().__init__()
        self.model_args = from_nested_dict(MultimodalAudioModelArgs, audio_model_args)
        self.codebook_sizes = list(self.model_args.get_codebook_sizes(pad_to_multiple=None))
        self.offsets = torch.from_numpy(np.cumsum([0] + self.codebook_sizes[:-1]))
        self.total_vocab_size = sum(self.codebook_sizes)
        padded_size = 128 * ((self.total_vocab_size + 127) // 128)
        self.embeddings = nn.Embedding(padded_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.offsets = self.offsets.to(input_ids.device)
        input_ids = input_ids + self.offsets[torch.newaxis, :, torch.newaxis]
        return self.embeddings(input_ids)


# ---- Standalone LLM: ported from vLLM's LlamaModel / LlamaDecoderLayer ----
# Uses flash_attn_func for attention (same kernel as vLLM).
# RMSNorm with fused residual pattern matches vLLM exactly.
# RoPE uses the same cos_sin_cache approach as vLLM.


class _RMSNorm(nn.Module):
    """RMSNorm matching vLLM's implementation, with optional fused residual add."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is not None:
            x = x + residual
        residual = x
        x_float = x.float()
        norm = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        out = (x_float * norm).to(residual.dtype) * self.weight
        return out, residual


class _RotaryEmbedding(nn.Module):
    """Neox-style rotary embeddings matching vLLM's RotaryEmbedding exactly."""

    def __init__(self, head_dim: int, max_position_embeddings: int, base: float, dtype: torch.dtype):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self._cache_dtype = dtype
        self._build_cache()

    def _build_cache(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(self._cache_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.unsqueeze(-2).to(q.dtype)
        sin = sin.unsqueeze(-2).to(q.dtype)

        q = q.view(num_tokens, -1, self.head_dim)
        q1, q2 = q.chunk(2, dim=-1)
        q = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1).flatten(1)

        k = k.view(num_tokens, -1, self.head_dim)
        k1, k2 = k.chunk(2, dim=-1)
        k = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).flatten(1)

        return q, k


class _LlamaAttention(nn.Module):
    """Attention layer matching vLLM's LlamaAttention, using PyTorch SDPA."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 max_position_embeddings, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False)

        self.rotary_emb = _RotaryEmbedding(head_dim, max_position_embeddings, rope_theta,
                                            dtype=torch.bfloat16)

    def forward(self, positions, hidden_states, kv_cache=None):
        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k = self.rotary_emb(positions, q, k)

        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=0)
            v = torch.cat([v_cache, v], dim=0)
        new_kv = (k, v)

        # Transpose to (batch=1, heads, seq, head_dim) for SDPA
        q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, num_tokens, head_dim]
        k = k.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, kv_len, head_dim]
        v = v.unsqueeze(0).transpose(1, 2)  # [1, num_kv_heads, kv_len, head_dim]

        # Expand KV heads for GQA: repeat each KV head to match query head groups
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        is_causal = kv_cache is None and num_tokens > 1
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).squeeze(0).reshape(num_tokens, -1)
        return self.o_proj(attn_out), new_kv


class _LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _LlamaDecoderLayer(nn.Module):
    """Decoder layer matching vLLM's LlamaDecoderLayer exactly:
    fused residual + RMSNorm pattern, flash_attn attention."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, max_position_embeddings, rope_theta, rms_norm_eps):
        super().__init__()
        self.self_attn = _LlamaAttention(
            hidden_size, num_heads, num_kv_heads, head_dim,
            max_position_embeddings, rope_theta,
        )
        self.mlp = _LlamaMLP(hidden_size, intermediate_size)
        self.input_layernorm = _RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(hidden_size, rms_norm_eps)

    def forward(self, positions, hidden_states, residual, kv_cache=None):
        if residual is None:
            residual = hidden_states
            hidden_states, _ = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, new_kv = self.self_attn(positions, hidden_states, kv_cache)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual, new_kv


class _LlamaModel(nn.Module):
    """Standalone LlamaModel matching vLLM's, with flash_attn."""

    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
                 head_dim, intermediate_size, max_position_embeddings, rope_theta, rms_norm_eps):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            _LlamaDecoderLayer(
                hidden_size, num_heads, num_kv_heads, head_dim,
                intermediate_size, max_position_embeddings, rope_theta, rms_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = _RMSNorm(hidden_size, rms_norm_eps)

    def forward(self, inputs_embeds, positions, past_key_values=None):
        hidden_states = inputs_embeds
        residual = None
        new_kvs = []
        for i, layer in enumerate(self.layers):
            kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, residual, new_kv = layer(positions, hidden_states, residual, kv)
            new_kvs.append(new_kv)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, new_kvs


# ---- Main model ----

class VoxtralTTSAudioGeneration(nn.Module):
    """Voxtral TTS generation model.

    LLM backbone is a standalone LlamaModel ported from vLLM, using flash_attn.
    """

    def __init__(self, text_config, audio_model_args: dict, embedding_dim: int):
        """Args:
            text_config: VoxtralTextConfig dataclass with dim, n_layers, etc.
            audio_model_args: dict for FlowMatchingAudioTransformer & MultiVocabEmbeddings.
            embedding_dim: typically text_config.dim.
        """
        super().__init__()
        self.n_heads = text_config.n_heads
        self.n_kv_heads = text_config.n_kv_heads
        self.head_dim = text_config.head_dim
        self.hidden_size = text_config.dim

        self.language_model = _LlamaModel(
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.dim,
            num_layers=text_config.n_layers,
            num_heads=text_config.n_heads,
            num_kv_heads=text_config.n_kv_heads,
            head_dim=text_config.head_dim,
            intermediate_size=text_config.hidden_dim,
            max_position_embeddings=text_config.max_seq_len,
            rope_theta=text_config.rope_theta,
            rms_norm_eps=text_config.norm_eps,
        )
        self.acoustic_transformer = FlowMatchingAudioTransformer(audio_model_args)
        self.audio_token_embedding = MultiVocabEmbeddings(
            audio_model_args=audio_model_args, embedding_dim=embedding_dim,
        )

    # ---- Forward ----

    def forward_llm(self, inputs_embeds, position_ids, past_key_values=None,
                    use_cache=True, do_layer_debug=False):
        """Run LLM forward. Returns (hidden_states, past_key_values).
        inputs_embeds: [B, seq_len, dim] — squeezed to [seq_len, dim] for flash_attn."""
        embeds = inputs_embeds.squeeze(0) if inputs_embeds.dim() == 3 else inputs_embeds
        positions = position_ids.flatten()

        if do_layer_debug and past_key_values is None:
            hidden, new_kvs = self._forward_with_layer_debug(embeds, positions)
            return hidden.unsqueeze(0), new_kvs if use_cache else None

        hidden, new_kvs = self.language_model(embeds, positions, past_key_values)
        return hidden.unsqueeze(0), new_kvs if use_cache else None

    # ---- Per-layer debug (mirrors vLLM's _forward_with_layer_debug) ----

    @torch.no_grad()
    def _forward_with_layer_debug(self, inputs_embeds, positions):
        """Manually iterate layers with per-layer logging, matching vLLM's debug.
        Also returns KV cache so decode steps can continue properly."""
        model = self.language_model
        hidden_states = inputs_embeds
        residual = None
        new_kvs = []

        last = hidden_states[-1]
        logger.info(
            "[SGL-DEBUG] embed: shape=%s, mean=%.6f, std=%.4f | last_tok: mean=%.6f, std=%.4f",
            list(hidden_states.shape),
            hidden_states.float().mean().item(), hidden_states.float().std().item(),
            last.float().mean().item(), last.float().std().item(),
        )

        for i, layer in enumerate(model.layers):
            hidden_states, residual, new_kv = layer(positions, hidden_states, residual)
            new_kvs.append(new_kv)
            state = hidden_states + residual
            last = state[-1]
            logger.info(
                "[SGL-DEBUG] layer %2d: shape=%s, mean=%.6f, std=%.4f | last_tok: mean=%.6f, std=%.4f",
                i, list(state.shape),
                state.float().mean().item(), state.float().std().item(),
                last.float().mean().item(), last.float().std().item(),
            )

        hidden_states, _ = model.norm(hidden_states, residual)
        last = hidden_states[-1]
        logger.info(
            "[SGL-DEBUG] final_norm: shape=%s, mean=%.6f, std=%.4f | last_tok: mean=%.6f, std=%.4f, norm=%.4f",
            list(hidden_states.shape),
            hidden_states.float().mean().item(), hidden_states.float().std().item(),
            last.float().mean().item(), last.float().std().item(), last.float().norm().item(),
        )
        return hidden_states, new_kvs

    # ---- Weight loading (mirrors vLLM's load_weights) ----

    _MISTRAL_TO_HF_RULES = [
        (r"^layers\.(\d+)\.attention\.wq\.weight$", r"layers.\1.self_attn.q_proj.weight"),
        (r"^layers\.(\d+)\.attention\.wk\.weight$", r"layers.\1.self_attn.k_proj.weight"),
        (r"^layers\.(\d+)\.attention\.wv\.weight$", r"layers.\1.self_attn.v_proj.weight"),
        (r"^layers\.(\d+)\.attention\.wo\.weight$", r"layers.\1.self_attn.o_proj.weight"),
        (r"^layers\.(\d+)\.attention_norm\.weight$", r"layers.\1.input_layernorm.weight"),
        (r"^layers\.(\d+)\.feed_forward\.w1\.weight$", r"layers.\1.mlp.gate_proj.weight"),
        (r"^layers\.(\d+)\.feed_forward\.w2\.weight$", r"layers.\1.mlp.down_proj.weight"),
        (r"^layers\.(\d+)\.feed_forward\.w3\.weight$", r"layers.\1.mlp.up_proj.weight"),
        (r"^layers\.(\d+)\.ffn_norm\.weight$", r"layers.\1.post_attention_layernorm.weight"),
    ]

    @staticmethod
    def _permute_qk_weight(w: torch.Tensor, n_heads: int, head_dim: int, hidden_size: int) -> torch.Tensor:
        attn_in = head_dim * n_heads
        return (
            w.view(n_heads, attn_in // n_heads // 2, 2, hidden_size)
            .transpose(1, 2)
            .reshape(attn_in, hidden_size)
        )

    def load_weights(self, checkpoint_dir: str, device: str = "cpu"):
        """Load weights from Mistral-format safetensors checkpoint."""
        import glob
        from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

        safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not safetensors_files:
            raise RuntimeError(f"No .safetensors files found in {checkpoint_dir}")

        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        hidden_size = self.hidden_size

        llm_state = {}
        llm_count = 0
        at_count = 0
        emb_loaded = False

        # Remapping rules for acoustic transformer and audio embeddings (same as vLLM)
        vllm_remapping = [
            (r"^acoustic_transformer\.(.*)$", r"\1"),
            (r"^audio_tokenizer\.(.*)$", r"\1"),
            (r"^mm_audio_embeddings\.audio_codebook_embeddings\.embeddings\.(weight|bias)", r"audio_token_embedding.embeddings.\1"),
            (r"^mm_audio_embeddings\.tok_embeddings\.weight", r"tok_embeddings.weight"),
        ]

        for name, tensor in safetensors_weights_iterator(safetensors_files):
            # LLM weights
            hf_name = self._remap_mistral_to_hf(name)
            if hf_name is not None:
                if ".attention.wq." in name:
                    tensor = self._permute_qk_weight(tensor, n_heads, head_dim, hidden_size)
                elif ".attention.wk." in name:
                    tensor = self._permute_qk_weight(tensor, n_kv_heads, head_dim, hidden_size)
                llm_state[hf_name] = tensor
                llm_count += 1
                continue

            # Acoustic transformer weights
            if name.startswith("acoustic_transformer."):
                short = name[len("acoustic_transformer."):]
                self.acoustic_transformer.load_weight((short, tensor))
                at_count += 1
                continue

            # Audio token embedding weights
            if name == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight":
                self.audio_token_embedding.embeddings.weight.data.copy_(tensor)
                emb_loaded = True
                continue

        missing, unexpected = self.language_model.load_state_dict(llm_state, strict=False)
        logger.info("LLM weights: %d loaded, %d missing, %d unexpected", llm_count, len(missing), len(unexpected))
        if missing:
            logger.warning("Missing LLM keys (first 5): %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected LLM keys (first 5): %s", unexpected[:5])
        logger.info("Acoustic transformer weights: %d loaded", at_count)
        logger.info("Audio token embedding loaded: %s", emb_loaded)

    def _remap_mistral_to_hf(self, name: str) -> str | None:
        if name == "norm.weight":
            return "norm.weight"
        if name == "mm_audio_embeddings.tok_embeddings.weight":
            return "embed_tokens.weight"
        for pattern, repl in self._MISTRAL_TO_HF_RULES:
            if stdlib_re.match(pattern, name):
                return stdlib_re.sub(pattern, repl, name)
        return None

    # ---- Class method to build from checkpoint (replaces stages.py logic) ----

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str, device: str = "cuda:0"):
        """Build the full model from a Mistral-format checkpoint directory."""
        from dataclasses import asdict
        from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

        config = VoxtralModelConfig.from_model_path(checkpoint_dir)
        audio_model_args_dict = asdict(config.audio_model_args)

        logger.info("Starting to load model %s ...", checkpoint_dir)
        t0 = time.perf_counter()
        mem_before = torch.cuda.memory_allocated(device) if device.startswith("cuda") else 0

        logger.info("Building VoxtralTTSAudioGeneration with meta device (fast init) ...")
        with torch.device("meta"):
            model = cls(
                text_config=config.text_config,
                audio_model_args=audio_model_args_dict,
                embedding_dim=config.text_config.dim,
            )
        model = model.to_empty(device="cpu")

        for layer in model.language_model.layers:
            layer.self_attn.rotary_emb._build_cache()

        # Rebuild acoustic transformer buffers lost during meta-device init
        at = model.acoustic_transformer
        at._timesteps = torch.linspace(0, 1, at._acoustic_decode_iters)
        dim = at.acoustic_transformer_args.dim
        inv_freq = torch.exp(-math.log(10000.0) * torch.arange(dim // 2).float() / (dim // 2))
        at.time_embedding.inv_freq = inv_freq

        model.load_weights(checkpoint_dir)

        load_time = time.perf_counter() - t0
        logger.info("Loading weights took %.2f seconds", load_time)

        model = model.to(dtype=torch.bfloat16, device=device).eval()

        mem_after = torch.cuda.memory_allocated(device) if device.startswith("cuda") else 0
        mem_used_gib = (mem_after - mem_before) / (1024**3)
        total_time = time.perf_counter() - t0
        logger.info("Model loading took %.2f GiB and %.2f seconds", mem_used_gib, total_time)

        # Load voice embeddings
        voice_embeddings = {}
        voice_dir = os.path.join(checkpoint_dir, "voice_embedding")
        if os.path.isdir(voice_dir):
            for fname in os.listdir(voice_dir):
                if fname.endswith(".pt"):
                    voice_name = fname[:-3]
                    emb = torch.load(os.path.join(voice_dir, fname), map_location=device)
                    voice_embeddings[voice_name] = emb.to(dtype=torch.bfloat16)
            logger.info("Loaded %d voice embeddings: %s", len(voice_embeddings), list(voice_embeddings.keys()))

        return model, voice_embeddings, config
