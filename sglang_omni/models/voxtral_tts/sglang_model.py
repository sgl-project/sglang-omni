# SPDX-License-Identifier: Apache-2.0
"""SGLang-backed Voxtral TTS autoregressive text backbone."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.voxtral_tts.acoustic_transformer import (
    FlowMatchingAudioTransformer,
)
from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig
from sglang_omni.models.voxtral_tts.voxtral_tts_audio_generation import (
    MultiVocabEmbeddings,
    _interleave_qk_weight,
)
from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RadixAttention,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
)


class VoxtralSGLangAttention(nn.Module):
    def __init__(self, cfg: Any, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = cfg.n_heads
        self.num_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = QKVParallelLinear(
            cfg.dim,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            cfg.dim,
            bias=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=cfg.max_seq_len,
            base=cfg.rope_theta,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class VoxtralSGLangDecoderLayer(nn.Module):
    def __init__(self, cfg: Any, layer_id: int, prefix: str = "") -> None:
        super().__init__()
        self.self_attn = VoxtralSGLangAttention(
            cfg,
            layer_id=layer_id,
            prefix=add_prefix("attention", prefix),
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            cfg.dim,
            [cfg.hidden_dim, cfg.hidden_dim],
            bias=False,
            prefix=add_prefix("feed_forward.gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            cfg.hidden_dim,
            cfg.dim,
            bias=False,
            prefix=add_prefix("feed_forward.w2", prefix),
        )
        self.attention_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states, residual


class VoxtralSGLangTextModel(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(
            [
                VoxtralSGLangDecoderLayer(
                    cfg,
                    layer_id=idx,
                    prefix=f"layers.{idx}",
                )
                for idx in range(cfg.n_layers)
            ]
        )
        self.norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.start_layer = 0
        self.end_layer = cfg.n_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        for idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[idx](
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class VoxtralSGLangTTSModel(nn.Module):
    """Voxtral TTS model with SGLang-managed text KV cache."""

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        del config, quant_config, prefix
        super().__init__()
        server_args = __import__(
            "sglang.srt.server_args", fromlist=["get_global_server_args"]
        ).get_global_server_args()
        self.model_path = server_args.model_path
        self.voxtral_config = VoxtralModelConfig.from_model_path(self.model_path)
        text_cfg = self.voxtral_config.text_config
        self.language_model = VoxtralSGLangTextModel(text_cfg)
        audio_args = asdict(self.voxtral_config.audio_model_args)
        self.acoustic_transformer = FlowMatchingAudioTransformer(audio_args)
        self.audio_token_embedding = MultiVocabEmbeddings(
            audio_model_args=audio_args,
            embedding_dim=text_cfg.dim,
        )
        self.audio_token_id = self.voxtral_config.audio_model_args.audio_token_id
        self.hidden_size = text_cfg.dim

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend():
            last_index = self._extend_last_index(forward_batch, hidden_states.device)
            hidden_states = hidden_states[last_index]
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    @staticmethod
    def _extend_last_index(
        forward_batch: ForwardBatch,
        device: torch.device,
    ) -> torch.Tensor:
        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            return torch.tensor([forward_batch.input_ids.shape[0] - 1], device=device)
        return torch.cumsum(extend_seq_lens.to(device=device), dim=0) - 1

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if self._load_text_weight(name, loaded_weight, params):
                continue
            if name.startswith("acoustic_transformer."):
                self.acoustic_transformer.load_weight(
                    (name[len("acoustic_transformer.") :], loaded_weight)
                )
                continue
            if (
                name
                == "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
            ):
                self.audio_token_embedding.embeddings.weight.data.copy_(loaded_weight)

    def _load_text_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
    ) -> bool:
        if name == "norm.weight":
            return self._copy_weight(
                "language_model.norm.weight", loaded_weight, params
            )
        if name == "mm_audio_embeddings.tok_embeddings.weight":
            return self._copy_weight(
                "language_model.embed_tokens.weight", loaded_weight, params
            )

        import re

        match = re.match(r"^layers\.(\d+)\.(.+)$", name)
        if match is None:
            return False
        layer_idx, suffix = match.group(1), match.group(2)
        prefix = f"language_model.layers.{layer_idx}"
        if suffix == "attention.wq.weight":
            return self._load_qkv(prefix, "q", loaded_weight, params)
        if suffix == "attention.wk.weight":
            return self._load_qkv(prefix, "k", loaded_weight, params)
        if suffix == "attention.wv.weight":
            return self._load_qkv(prefix, "v", loaded_weight, params)
        mapping = {
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention_norm.weight": "attention_norm.weight",
            "ffn_norm.weight": "ffn_norm.weight",
            "feed_forward.w1.weight": ("gate_up_proj.weight", 0),
            "feed_forward.w3.weight": ("gate_up_proj.weight", 1),
            "feed_forward.w2.weight": "down_proj.weight",
        }
        target = mapping.get(suffix)
        if target is None:
            return False
        if isinstance(target, tuple):
            target_name, shard_id = target
            param = params[f"{prefix}.{target_name}"]
            param.weight_loader(param, loaded_weight, shard_id)
            return True
        return self._copy_weight(f"{prefix}.{target}", loaded_weight, params)

    def _load_qkv(
        self,
        prefix: str,
        shard_id: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
    ) -> bool:
        param = params[f"{prefix}.self_attn.qkv_proj.weight"]
        layer = self.language_model.layers[int(prefix.split(".")[-1])]
        if shard_id == "q":
            loaded_weight = _interleave_qk_weight(
                loaded_weight,
                layer.self_attn.num_heads,
                layer.self_attn.head_dim,
            )
        elif shard_id == "k":
            loaded_weight = _interleave_qk_weight(
                loaded_weight,
                layer.self_attn.num_kv_heads,
                layer.self_attn.head_dim,
            )
        param.weight_loader(param, loaded_weight, shard_id)
        return True

    @staticmethod
    def _copy_weight(
        target: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
    ) -> bool:
        param = params.get(target)
        if param is None:
            return False
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        return True


EntryClass = VoxtralSGLangTTSModel
