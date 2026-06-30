# SPDX-License-Identifier: Apache-2.0
"""SGLang-native CosyVoice3 AR speech-token LM (tts_engine stage).

CosyVoice3LM is the autoregressive language model that predicts discrete speech
tokens. Its backbone is a Qwen2-0.5B transformer (hidden 896, 24 layers, 14 heads,
2 KV heads, head_dim 64, intermediate 4864, rms_norm_eps 1e-6, rope_theta 1e6).
It mirrors fishaudio's ``S2ProSGLangTextModel`` skeleton with Qwen2 deltas:

  * attention uses qkv BIAS and NO qk-norm,
  * the INPUT embedding is ``speech_embedding`` (speech tokens, 6761x896),
  * the OUTPUT head is ``llm_decoder`` (Linear 896->6761, no bias, untied),
  * ``text_embedding`` (151936x896) exists only so the Qwen2 token embedding from
    ``llm.pt`` loads cleanly; it is consumed by preprocessing, never in forward.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from torch import Tensor, nn

from sglang_omni.vendor.sglang.core import ForwardBatch
from sglang_omni.vendor.sglang.layers import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RadixAttention,
    ReplicatedLinear,
    RMSNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
    get_rope,
)
from sglang_omni.vendor.sglang.utils import make_layers

logger = logging.getLogger(__name__)


class CosyVoice3Attention(nn.Module):
    """Qwen2 self-attention: fused qkv WITH bias, RoPE (neox), no qk-norm."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim**-0.5

        # Qwen2 attention projections carry a bias term.
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            bias=True,
        )
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope_base,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            num_heads,
            head_dim,
            self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
    ) -> Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class CosyVoice3DecoderLayer(nn.Module):
    """Qwen2 decoder layer: attention + gated SiLU MLP + two RMSNorms."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        layer_id: int,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn = CosyVoice3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layer_id=layer_id,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        positions: Tensor,
        hidden_states: Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden_states = torch.nn.functional.silu(gate) * up
        del gate, up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states, residual


class CosyVoice3LM(nn.Module):
    """AR speech-token LM for CosyVoice3 (Qwen2-0.5B backbone)."""

    def __init__(
        self,
        config: Any = None,
        quant_config: Any = None,
        *,
        speech_vocab_size: int = 6761,
        text_vocab_size: int = 151936,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_layers: int = 24,
        num_heads: int = 14,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        rope_base: float = 1000000.0,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        del quant_config

        # The AR engine bootstraps the Qwen2 backbone from the CosyVoice-BlankEN
        # subdir (a real HF Qwen2 config.json); read shapes from it when present.
        if config is not None:
            hidden_size = getattr(config, "hidden_size", hidden_size)
            intermediate_size = getattr(config, "intermediate_size", intermediate_size)
            num_layers = getattr(config, "num_hidden_layers", num_layers)
            num_heads = getattr(config, "num_attention_heads", num_heads)
            num_kv_heads = getattr(config, "num_key_value_heads", num_kv_heads)
            cfg_head_dim = getattr(config, "head_dim", None)
            head_dim = cfg_head_dim if cfg_head_dim else hidden_size // num_heads
            rope_base = getattr(config, "rope_theta", rope_base)
            max_position_embeddings = getattr(
                config, "max_position_embeddings", max_position_embeddings
            )
            rms_norm_eps = getattr(config, "rms_norm_eps", rms_norm_eps)
            text_vocab_size = getattr(config, "vocab_size", text_vocab_size)

        self.speech_vocab_size = speech_vocab_size
        self.text_vocab_size = text_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # INPUT embedding: speech tokens (used at decode + by preprocessing prompt).
        self.speech_embedding = VocabParallelEmbedding(speech_vocab_size, hidden_size)
        # Qwen2 token embedding: loaded from llm.pt, consumed ONLY by preprocessing.
        self.text_embedding = nn.Embedding(text_vocab_size, hidden_size)

        self.start_layer = 0
        self.end_layer = num_layers
        self.layers = make_layers(
            num_layers,
            lambda idx, prefix: CosyVoice3DecoderLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                layer_id=idx,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
            ),
            prefix="layers",
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # OUTPUT head: untied speech-token logits (no bias).
        self.llm_decoder = ReplicatedLinear(
            hidden_size,
            speech_vocab_size,
            bias=False,
        )

    def get_input_embeddings(self):
        return self.speech_embedding

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[Tensor] = None,
        input_embeds_are_projected: bool = False,
    ) -> LogitsProcessorOutput:
        # For CosyVoice3 the prompt embeds built in preprocessing are already the
        # final hidden states, so the projection flag carries no extra work here.
        del input_embeds_are_projected

        if input_embeds is None and forward_batch.input_embeds is not None:
            input_embeds = forward_batch.input_embeds

        if input_embeds is not None:
            # Prefill: ModelRunner supplies prompt embeds via forward_batch.
            hidden_states = input_embeds
        else:
            # Decode: embed the single fed-back speech token.
            hidden_states = self.speech_embedding(input_ids)

        residual = None
        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states, residual = self.layers[layer_idx](
                positions, hidden_states, forward_batch, residual
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Extend: prune to the last-token position of each sequence.
        if forward_batch.forward_mode.is_extend():
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_index]

        logits, _ = self.llm_decoder(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=logits,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> set[str]:
        """Overlay CosyVoice3 ``llm.pt`` onto the Qwen2 backbone.

        Checkpoint keys are wrapped as ``llm.model.model.*`` (transformer body),
        ``llm.model.lm_head.weight`` (unused tied head -> SKIP), plus the bare
        ``speech_embedding.weight`` / ``llm_decoder.weight``. q/k/v (weight AND
        bias) fuse into ``qkv_proj``; gate/up fuse into ``gate_up_proj``; the HF
        ``mlp.`` wrapper level is stripped to match our flat decoder layout.
        """
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        loaded_names: set[str] = set()
        # Track which shards of each fused (qkv/gate_up) param actually loaded, so a missing
        # k/v/up shard is caught — marking the fused param "loaded" after any single shard
        # would hide it from the coverage check. Expected shard set per fused param name:
        expected_shards: dict[str, set] = {}
        for _pn, _wn, _sid in stacked_params_mapping:
            expected_shards.setdefault(_pn, set()).add(_sid)
        loaded_shards: dict[str, set] = {}
        shard_param_name: dict[str, str] = {}
        for name, loaded_weight in weights:
            # Strip the Qwen2 wrapper prefixes (order matters: the longer prefix
            # must be tried first). Bare keys keep their name.
            if name.startswith("llm.model.model."):
                name = name[len("llm.model.model.") :]
            elif name.startswith("llm.model."):
                name = name[len("llm.model.") :]

            # Untied lm_head from the Qwen2 backbone is unused (head = llm_decoder).
            if name.endswith("lm_head.weight"):
                continue

            # Qwen2 token embedding feeds preprocessing only, not forward().
            if name == "embed_tokens.weight":
                param = params_dict["text_embedding.weight"]
                _default_weight_loader(param, loaded_weight)
                loaded_names.add("text_embedding.weight")
                continue

            # Drop the HF mlp wrapper level: our layers hold gate_up_proj/down_proj
            # directly (no ``mlp`` submodule).
            name = name.replace(".mlp.", ".")

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                param = params_dict.get(mapped)
                if param is None:
                    continue
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_names.add(mapped)
                loaded_shards.setdefault(mapped, set()).add(shard_id)
                shard_param_name[mapped] = param_name
                handled = True
                break
            if handled:
                continue

            param = params_dict.get(name)
            if param is None:
                logger.debug("Skipping weight: %s", name)
                continue
            weight_loader = getattr(param, "weight_loader", _default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_names.add(name)

        # Every fused param that loaded must have received ALL its shards (both the BlankEN
        # and llm.pt checkpoints provide complete q/k/v and gate/up), else a shard is missing.
        for mapped, shards in loaded_shards.items():
            want = expected_shards[shard_param_name[mapped]]
            if shards != want:
                raise RuntimeError(
                    f"CosyVoice3 fused weight {mapped!r} is missing shards "
                    f"{sorted(str(s) for s in want - shards)}"
                )

        # Return the set of populated parameter names. load_weights runs twice: once for the
        # BlankEN backbone during engine build -- a no-op through this mapping, because BlankEN's
        # `model.*` / `lm_head.*` keys don't match the llm.pt-oriented prefix/fusion handling
        # (which yields flat `layers.*` / bare head names), so they all fall through and are
        # skipped -- and once for the llm.pt overlay, which supplies the complete weight set. The
        # two-checkpoint coverage check therefore belongs at the llm.pt call site (stages.py).
        return loaded_names


def _default_weight_loader(param: nn.Parameter, loaded_weight: Tensor) -> None:
    param.data.copy_(loaded_weight)


EntryClass = CosyVoice3LM
