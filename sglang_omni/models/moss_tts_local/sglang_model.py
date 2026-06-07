# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-TTS Local model.

Time-synchronous variant of MOSS-TTS: the Qwen3 backbone emits one global latent
per frame (under the sglang forward path), then a small depth transformer
autoregressively predicts the text channel + N RVQ codebooks within that frame
(RQ-Transformer style). The depth transformer runs eager in the model runner,
outside the captured graph.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3 import Qwen3Model
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_tts.payload_types import moss_tts_special_token_defaults

logger = logging.getLogger(__name__)


def _as_qwen3_config(config: Any) -> Any:
    from transformers import Qwen3Config

    if isinstance(config, Qwen3Config):
        return config
    if isinstance(config, dict):
        return Qwen3Config(**config)
    if hasattr(config, "to_dict"):
        return Qwen3Config(**config.to_dict())
    return config


class _RMSNormUpcast(nn.Module):
    """Qwen3RMSNorm: variance in float32, weight applied in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(in_dtype)


class _MossRMSNorm(nn.Module):
    """Upstream MossTTSRMSNorm: computed entirely in the input dtype (no upcast)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class _SwiGLUMLP(nn.Module):
    def __init__(self, input_size: int, ffn_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(input_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _DepthAttention(nn.Module):
    """Qwen3 GQA attention with q/k norm and no positional embedding (causal)."""

    def __init__(self, hidden: int, n_heads: int, n_kv: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, hidden, bias=False)
        self.q_norm = _RMSNormUpcast(head_dim)
        self.k_norm = _RMSNormUpcast(head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(b, t, self.n_kv, self.head_dim))
        v = self.v_proj(x).view(b, t, self.n_kv, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scaling
        )
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class _DepthLayer(nn.Module):
    def __init__(self, hidden, ffn, n_heads, n_kv, head_dim, eps):
        super().__init__()
        self.input_layernorm = _RMSNormUpcast(hidden, eps)
        self.self_attn = _DepthAttention(hidden, n_heads, n_kv, head_dim)
        self.post_attention_layernorm = _RMSNormUpcast(hidden, eps)
        self.mlp = _SwiGLUMLP(hidden, ffn, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class _DepthTransformer(nn.Module):
    def __init__(self, n_layers, hidden, ffn, n_heads, n_kv, head_dim, eps):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DepthLayer(hidden, ffn, n_heads, n_kv, head_dim, eps)
                for _ in range(n_layers)
            ]
        )
        self.norm = _RMSNormUpcast(hidden, eps)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MossTTSLocalSGLangModel(torch.nn.Module):
    """MOSS-TTS Local: Qwen3 backbone + depth transformer over RVQ codebooks."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = self._normalize_config(config)
        self.quant_config = quant_config
        self.hidden_size = int(self.config.hidden_size)
        self.channels = int(self.config.channels)
        self.audio_pad_code = int(self.config.audio_pad_code)

        self.embedding_list = nn.ModuleList(
            [
                nn.Embedding(int(self.config.vocab_size_list[idx]), self.hidden_size)
                for idx in range(self.channels)
            ]
        )
        self.model = Qwen3Model(
            config=self.config.language_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        lc = self.config.language_config
        n_heads = int(lc.num_attention_heads)
        n_kv = int(getattr(lc, "num_key_value_heads", n_heads))
        head_dim = int(getattr(lc, "head_dim", None) or self.hidden_size // n_heads)
        eps = float(getattr(lc, "rms_norm_eps", 1e-6))
        local_hidden = int(self.config.local_hidden_size)
        add_ffn = int(self.config.additional_mlp_ffn_hidden_size)
        self.local_transformer = _DepthTransformer(
            int(self.config.local_num_layers),
            local_hidden,
            int(self.config.local_ffn_hidden_size),
            n_heads,
            n_kv,
            head_dim,
            eps,
        )
        self.speech_embedding_to_local_mlp = _SwiGLUMLP(
            self.hidden_size, add_ffn, local_hidden
        )
        self.local_to_speech_embedding_mlps = nn.ModuleList(
            [
                _SwiGLUMLP(local_hidden, add_ffn, self.hidden_size)
                for _ in range(self.channels)
            ]
        )
        self.layer_norm_before_lm_heads = nn.ModuleList(
            [_MossRMSNorm(self.hidden_size, eps) for _ in range(self.channels)]
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(
                    self.hidden_size, int(self.config.vocab_size_list[idx]), bias=False
                )
                for idx in range(self.channels)
            ]
        )
        self._pad_token_per_channel = self._compute_pad_token_per_channel()

        try:
            from sglang.srt.server_args import get_global_server_args

            max_batch_size = int(get_global_server_args().max_running_requests)
        except Exception:
            max_batch_size = 1
        max_batch_size = max(1, max_batch_size)
        weight = self.embedding_list[0].weight
        self._decode_input_embedding = nn.Embedding(
            max_batch_size, self.hidden_size, device=weight.device, dtype=weight.dtype
        )
        self._decode_input_embedding.weight.requires_grad_(False)

        # Depth-axis KV cache (one slot per channel) — keeps the depth loop O(channels).
        self._local_n_heads = n_heads
        self._local_n_kv = n_kv
        self._local_head_dim = head_dim
        self._local_n_layers = int(self.config.local_num_layers)
        self._local_scaling = float(head_dim) ** -0.5
        self._pred_k = torch.zeros(
            self._local_n_layers,
            max_batch_size,
            n_kv,
            self.channels,
            head_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        self._pred_v = torch.zeros_like(self._pred_k)

    @staticmethod
    def _normalize_config(config: Any) -> Any:
        language_config = _as_qwen3_config(getattr(config, "language_config", None))
        config.language_config = language_config
        config.hidden_size = int(
            getattr(config, "hidden_size", getattr(language_config, "hidden_size"))
        )
        config.vocab_size = int(
            getattr(config, "vocab_size", getattr(language_config, "vocab_size"))
        )
        config.n_vq = int(getattr(config, "n_vq", 32))
        config.channels = int(getattr(config, "channels", config.n_vq + 1))
        audio_vocab_size = int(getattr(config, "audio_vocab_size", 1024))
        if not getattr(config, "vocab_size_list", None):
            config.vocab_size_list = [config.vocab_size] + [audio_vocab_size + 1] * (
                config.channels - 1
            )
        if not getattr(config, "pad_token", None):
            text_pad = int(getattr(config, "pad_token_id", 0) or 0)
            audio_pad = int(getattr(config, "audio_pad_code", audio_vocab_size))
            config.pad_token = [text_pad] + [audio_pad] * (config.channels - 1)
        for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
            if getattr(config, attr, None) is None:
                setattr(config, attr, default)
        config.local_hidden_size = int(getattr(config, "local_hidden_size", 1536))
        config.local_num_layers = int(getattr(config, "local_num_layers", 4))
        config.local_ffn_hidden_size = int(
            getattr(config, "local_ffn_hidden_size", 8960)
        )
        config.additional_mlp_ffn_hidden_size = int(
            getattr(config, "additional_mlp_ffn_hidden_size", 2048)
        )
        return config

    def _compute_pad_token_per_channel(self) -> list[int]:
        pad = getattr(self.config, "pad_token", None)
        if isinstance(pad, (list, tuple)) and pad:
            pad_ids = [int(v) if v is not None else 0 for v in pad]
            if len(pad_ids) < self.channels:
                pad_ids.extend([pad_ids[-1]] * (self.channels - len(pad_ids)))
            return pad_ids[: self.channels]
        return [int(getattr(self.config, "pad_token_id", 0) or 0)] + [
            self.audio_pad_code
        ] * (self.channels - 1)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    @property
    def device(self) -> torch.device:
        return self.embedding_list[0].weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embedding_list[0].weight.dtype

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._prepare_multi_modal_inputs(input_ids)

    def _prepare_multi_modal_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            total = int(input_ids.shape[0])
            if total % self.channels == 0:
                rows = input_ids.view(total // self.channels, self.channels)
            else:
                rows = torch.empty(
                    (total, self.channels),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                for idx, pad_id in enumerate(self._pad_token_per_channel):
                    rows[:, idx].fill_(int(pad_id))
                rows[:, 0] = input_ids
        elif input_ids.dim() == 2:
            rows = input_ids
        else:
            raise ValueError(
                f"MOSS-TTS Local input_ids bad shape {tuple(input_ids.shape)}"
            )
        if int(rows.shape[-1]) != self.channels:
            raise ValueError(
                f"MOSS-TTS Local expected {self.channels} channels, got {rows.shape[-1]}"
            )
        embeds = torch.zeros(
            rows.shape[0], self.hidden_size, device=rows.device, dtype=self.dtype
        )
        for idx, layer in enumerate(self.embedding_list):
            embeds = embeds + layer(rows[:, idx])
        return embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_embeds_are_projected: bool = False,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected
        if input_embeds is None:
            forward_mode = getattr(forward_batch, "forward_mode", None)
            is_decode = forward_mode is not None and bool(forward_mode.is_decode())
            if is_decode:
                input_embeds = self._decode_input_embedding(input_ids)
            elif self.pp_group.is_first_rank:
                input_embeds = self._prepare_multi_modal_inputs(input_ids)

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states

        # Sampling/logits run in the runner (depth loop); return sampled hidden states.
        sample_hidden_states = self._select_sample_hidden_states(
            hidden_states, forward_batch
        )
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits, hidden_states=sample_hidden_states
        )

    @staticmethod
    def _select_sample_hidden_states(
        hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        if forward_mode is None or not forward_mode.is_extend():
            return hidden_states
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            return hidden_states[-1:].contiguous()
        last_index = (
            torch.cumsum(
                extend_seq_lens.to(device=hidden_states.device, dtype=torch.long), dim=0
            )
            - 1
        )
        return hidden_states[last_index]

    def _depth_layer_cached(
        self, layer_idx: int, x: torch.Tensor, cache_len: int, b: int
    ) -> torch.Tensor:
        """One depth layer for a single new token, attending to cache 0..cache_len."""
        layer = self.local_transformer.layers[layer_idx]
        attn = layer.self_attn
        nh, nkv, hd = self._local_n_heads, self._local_n_kv, self._local_head_dim

        normed = layer.input_layernorm(x)
        q = attn.q_norm(attn.q_proj(normed).view(b, nh, hd))
        k = attn.k_norm(attn.k_proj(normed).view(b, nkv, hd))
        v = attn.v_proj(normed).view(b, nkv, hd)
        self._pred_k[layer_idx, :b, :, cache_len, :] = k
        self._pred_v[layer_idx, :b, :, cache_len, :] = v
        keys = self._pred_k[layer_idx, :b, :, : cache_len + 1, :]
        vals = self._pred_v[layer_idx, :b, :, : cache_len + 1, :]
        if nh != nkv:
            rep = nh // nkv
            keys = keys.repeat_interleave(rep, dim=1)
            vals = vals.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(2), keys, vals, scale=self._local_scaling
        )
        x = x + attn.o_proj(out.transpose(1, 2).reshape(b, nh * hd))
        return x + layer.mlp(layer.post_attention_layernorm(x))

    def _predictor_one_token(
        self, x: torch.Tensor, cache_len: int, b: int
    ) -> torch.Tensor:
        for layer_idx in range(self._local_n_layers):
            x = self._depth_layer_cached(layer_idx, x, cache_len, b)
        return self.local_transformer.norm(x)

    @torch.no_grad()
    def decode_frames(self, backbone_hidden: torch.Tensor, sampler) -> torch.Tensor:
        """Depth-decode one frame: returns (B, channels) tokens. ``sampler(channel,
        logits) -> tokens`` carries per-request sampling params (channel 0 = text)."""
        backbone_hidden = backbone_hidden.to(self.dtype)
        b = backbone_hidden.shape[0]
        if b > self._pred_k.shape[1]:
            raise RuntimeError(
                f"MOSS-TTS Local depth batch {b} exceeds the pre-allocated KV "
                f"cache ({self._pred_k.shape[1]}); raise max_running_requests"
            )
        cur = self.speech_embedding_to_local_mlp(backbone_hidden)
        rows = torch.empty(
            (b, self.channels), dtype=torch.long, device=backbone_hidden.device
        )
        for k in range(self.channels):
            last = self._predictor_one_token(cur, k, b)
            logits = self.lm_heads[k](
                self.layer_norm_before_lm_heads[k](
                    self.local_to_speech_embedding_mlps[k](last)
                )
            )
            if k != 0:
                logits[:, self.audio_pad_code] = float("-inf")
            tok = sampler(k, logits)
            rows[:, k] = tok
            cur = self.speech_embedding_to_local_mlp(self.embedding_list[k](tok))
        return rows

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for original_name, loaded_weight in weights:
            name = original_name
            # Backbone weights (model.language_model.*) use sglang's fused qkv/gate_up.
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model.") :]
                layer_id = get_layer_id(name)
                if (
                    layer_id is not None
                    and hasattr(self.model, "start_layer")
                    and not (self.model.start_layer <= layer_id < self.model.end_layer)
                ):
                    continue
                if "rotary_emb.inv_freq" in name:
                    continue
                done = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if mapped.endswith(".bias") and mapped not in params_dict:
                        done = True
                        break
                    param = params_dict.get(mapped)
                    if param is None:
                        break
                    param.weight_loader(param, loaded_weight, shard_id)
                    done = True
                    break
                if done:
                    continue
                param = params_dict.get(name)
                if param is not None:
                    self._load_param(param, loaded_weight)
                else:
                    logger.warning("MOSS-TTS Local backbone param %s not found", name)
                continue

            # Depth/head/embedding weights map 1:1 (drop the leading model. prefix).
            if name.startswith("model.embedding_list."):
                name = name[len("model.") :]
            param = params_dict.get(name)
            if param is not None:
                self._load_param(param, loaded_weight)
            else:
                logger.warning("MOSS-TTS Local parameter %s not found", original_name)

    @staticmethod
    def _load_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = MossTTSLocalSGLangModel
