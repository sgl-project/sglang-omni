# SPDX-License-Identifier: Apache-2.0
"""Native shared-trunk, independent-text-tail, and independent-audio-tail model."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader

from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig
from sglang_omni.models.moss_speech.model_layers import (
    MossSpeechDecoderLayer,
    MossSpeechRMSNorm,
    request_linear,
    request_norm,
)

logger = logging.getLogger(__name__)


@dataclass
class MossSpeechModelOutput(LogitsProcessorOutput):
    text_logits: torch.Tensor | None = None
    audio_logits: torch.Tensor | None = None


class MossSpeechSGLangModel(torch.nn.Module):
    """Return both channel logits through the standard graph-compatible output."""

    def __init__(
        self,
        config: MossSpeechConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        init_device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError("MOSS-Speech inference does not support quantization")
        else:
            pass
        assert config.audio_vocab_size is not None
        self.config: MossSpeechConfig = config
        self.dtype: torch.dtype = torch.bfloat16
        self.hidden_size: int = config.hidden_size
        self.text_vocab_size: int = config.vocab_size
        self.audio_vocab_size: int = config.audio_vocab_size
        self.num_shared_layers: int = config.num_shared_layers
        self.num_modality_layers: int = config.num_modality_layers
        self.total_attention_layers: int = (
            config.num_shared_layers + 2 * config.num_modality_layers
        )
        self.head_dim: int = config.head_dim
        self.embed_tokens: VocabParallelEmbedding = VocabParallelEmbedding(
            self.text_vocab_size, self.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        self.audio_embed: VocabParallelEmbedding = VocabParallelEmbedding(
            self.audio_vocab_size, self.hidden_size, prefix=f"{prefix}.audio_embed"
        )
        self.layers: torch.nn.ModuleList = torch.nn.ModuleList(
            MossSpeechDecoderLayer(config, i, prefix=f"{prefix}.layers.{i}")
            for i in range(self.num_shared_layers)
        )
        self.text_block: torch.nn.ModuleList = torch.nn.ModuleList(
            MossSpeechDecoderLayer(
                config, self.num_shared_layers + i, prefix=f"{prefix}.text_block.{i}"
            )
            for i in range(self.num_modality_layers)
        )
        self.audio_block: torch.nn.ModuleList = torch.nn.ModuleList(
            MossSpeechDecoderLayer(
                config,
                self.num_shared_layers + self.num_modality_layers + i,
                prefix=f"{prefix}.audio_block.{i}",
            )
            for i in range(self.num_modality_layers)
        )
        self.text_norm: MossSpeechRMSNorm = MossSpeechRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.audio_norm: MossSpeechRMSNorm = MossSpeechRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.text_lm_head: ParallelLMHead = ParallelLMHead(
            self.text_vocab_size, self.hidden_size
        )
        self.audio_lm_head: ParallelLMHead = ParallelLMHead(
            self.audio_vocab_size, self.hidden_size
        )
        self.decode_input_embedding: torch.nn.Embedding = torch.nn.Embedding(
            config.moss_decode_buffer_bs,
            self.hidden_size,
            device=init_device or torch.device("cpu"),
            dtype=self.dtype,
        )
        self.decode_input_embedding.weight.requires_grad_(False)
        assert self.attention_layer_ids() == list(range(self.total_attention_layers))

    def attention_layer_ids(self) -> list[int]:
        return [
            layer.self_attn.attn.layer_id
            for layer in list(self.layers)
            + list(self.text_block)
            + list(self.audio_block)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        # Required by the shared prefill forwarding interface.
        omni_prefill_rids: list[str] | None = None,
    ) -> MossSpeechModelOutput:
        is_decode = forward_batch.forward_mode.is_decode()
        if input_embeds is not None:
            hidden_states = input_embeds
        elif is_decode:
            hidden_states = self.decode_input_embedding(input_ids)
        else:
            raise ValueError("MOSS-Speech prefill requires dual-channel embeddings")
        if is_decode:
            token_lengths = [1] * hidden_states.shape[0]
        else:
            assert forward_batch.extend_seq_lens_cpu is not None
            token_lengths = list(forward_batch.extend_seq_lens_cpu)
        assert sum(token_lengths) == hidden_states.shape[0]
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, token_lengths, residual
            )
        text_hidden_states, text_residual = hidden_states, residual
        audio_hidden_states = hidden_states.clone()
        audio_residual = residual.clone() if residual is not None else None
        for layer in self.text_block:
            text_hidden_states, text_residual = layer(
                positions,
                text_hidden_states,
                forward_batch,
                token_lengths,
                text_residual,
            )
        for layer in self.audio_block:
            audio_hidden_states, audio_residual = layer(
                positions,
                audio_hidden_states,
                forward_batch,
                token_lengths,
                audio_residual,
            )
        if text_residual is None or audio_residual is None:
            raise RuntimeError("MOSS-Speech requires nonempty modality tails")
        else:
            pass
        normalized_text_hidden_states, _ = request_norm(
            self.text_norm, text_hidden_states, token_lengths, text_residual
        )
        normalized_audio_hidden_states, _ = request_norm(
            self.audio_norm, audio_hidden_states, token_lengths, audio_residual
        )
        if is_decode:
            last_token_indices = slice(None)
        else:
            last_token_indices = (
                torch.tensor(
                    token_lengths, device=normalized_text_hidden_states.device
                ).cumsum(0)
                - 1
            )
        text_logits = request_linear(
            normalized_text_hidden_states, self.text_lm_head.weight, token_lengths
        )[last_token_indices, : self.text_vocab_size]
        audio_logits = request_linear(
            normalized_audio_hidden_states, self.audio_lm_head.weight, token_lengths
        )[last_token_indices, : self.audio_vocab_size]
        return MossSpeechModelOutput(
            next_token_logits=text_logits,
            text_logits=text_logits,
            audio_logits=audio_logits,
        )

    @staticmethod
    def map_checkpoint_name(name: str) -> tuple[str, str | None] | None:
        renames = {
            "model.embed_tokens.weight": "embed_tokens.weight",
            "model.audio_embed.weight": "audio_embed.weight",
            "model.text_norm.weight": "text_norm.weight",
            "model.audio_norm.weight": "audio_norm.weight",
            "text_lm_head.weight": "text_lm_head.weight",
            "audio_lm_head.weight": "audio_lm_head.weight",
        }
        if name in renames:
            return renames[name], None
        else:
            pass
        for checkpoint_prefix, native_prefix in (
            ("model.shared_block.layers.", "layers."),
            ("model.text_block.layers.", "text_block."),
            ("model.audio_block.layers.", "audio_block."),
        ):
            if name.startswith(checkpoint_prefix):
                suffix = name[len(checkpoint_prefix) :]
                break
            else:
                pass
        else:
            return None
        fused = {
            "self_attn.q_proj.weight": ("self_attn.qkv_proj.weight", "q"),
            "self_attn.k_proj.weight": ("self_attn.qkv_proj.weight", "k"),
            "self_attn.v_proj.weight": ("self_attn.qkv_proj.weight", "v"),
            "mlp.gate_proj.weight": ("mlp.gate_up_proj.weight", "gate"),
            "mlp.up_proj.weight": ("mlp.gate_up_proj.weight", "up"),
        }
        layer_index, separator, leaf = suffix.partition(".")
        assert separator, f"Malformed checkpoint layer key {name}"
        if leaf in fused:
            target, shard = fused[leaf]
            return f"{native_prefix}{layer_index}.{target}", shard
        else:
            return f"{native_prefix}{suffix}", None

    def shard_row_span(self, shard: str) -> tuple[int, int]:
        query_rows = self.config.num_attention_heads * self.head_dim
        key_value_rows = self.config.num_key_value_heads * self.head_dim
        intermediate_rows = self.config.intermediate_size
        return {
            "q": (0, query_rows),
            "k": (query_rows, query_rows + key_value_rows),
            "v": (query_rows + key_value_rows, query_rows + 2 * key_value_rows),
            "gate": (0, intermediate_rows),
            "up": (intermediate_rows, 2 * intermediate_rows),
        }[shard]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        parameters = dict(self.named_parameters())
        shards_seen: dict[str, set[str]] = {}
        direct_seen: set[str] = set()
        source_names: set[str] = set()
        for name, tensor in weights:
            if name in source_names:
                raise RuntimeError(f"Duplicate checkpoint source {name}")
            else:
                source_names.add(name)
            mapped = self.map_checkpoint_name(name)
            if mapped is None:
                raise RuntimeError(f"Unexpected checkpoint key {name}")
            else:
                target, shard = mapped
            parameter = parameters[target]
            if shard is None:
                expected_shape = parameter.shape
            else:
                start, end = self.shard_row_span(shard)
                expected_shape = parameter[start:end].shape
            if tensor.shape != expected_shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(expected_shape)}"
                )
            elif shard is None:
                default_weight_loader(parameter, tensor)
                direct_seen.add(target)
            else:
                with torch.no_grad():
                    parameter[start:end].copy_(tensor.to(parameter.dtype))
                shards_seen.setdefault(target, set()).add(shard)
        layer_prefixes = [f"layers.{i}." for i in range(self.num_shared_layers)] + [
            f"{block}.{i}."
            for block in ("text_block", "audio_block")
            for i in range(self.num_modality_layers)
        ]
        expected_direct = {
            "embed_tokens.weight",
            "audio_embed.weight",
            "text_norm.weight",
            "audio_norm.weight",
            "text_lm_head.weight",
            "audio_lm_head.weight",
        }
        expected_shards = {}
        for layer_prefix in layer_prefixes:
            expected_direct.update(
                layer_prefix + leaf
                for leaf in (
                    "self_attn.o_proj.weight",
                    "self_attn.q_norm.weight",
                    "self_attn.k_norm.weight",
                    "mlp.down_proj.weight",
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                )
            )
            expected_shards[layer_prefix + "self_attn.qkv_proj.weight"] = {
                "q",
                "k",
                "v",
            }
            expected_shards[layer_prefix + "mlp.gate_up_proj.weight"] = {"gate", "up"}
        expected_sources = 6 + self.total_attention_layers * 11
        if (
            direct_seen != expected_direct
            or shards_seen != expected_shards
            or len(source_names) != expected_sources
        ):
            raise RuntimeError(
                "MOSS-Speech checkpoint does not cover every independent model tensor"
            )
        else:
            logger.info(f"MOSS-Speech loaded {expected_sources} source tensors")


EntryClass = MossSpeechSGLangModel
ARCH_KEY = "MossSpeechForCausalLM"
