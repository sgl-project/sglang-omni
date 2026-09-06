# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-TTS-Nano global-local model."""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.activation import NewGELU
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

from sglang_omni.models.moss_tts_local.local_transformer import MossTTSLocalTransformer
from sglang_omni.models.moss_tts_local.sglang_model import MossTTSLocalSGLangModel
from sglang_omni.models.moss_tts_local.state_pool import MossTTSLocalDecodeStatePool

logger = logging.getLogger(__name__)


def _as_gpt2_config(config: Any) -> Any:
    from transformers import GPT2Config

    if isinstance(config, GPT2Config):
        return config
    if isinstance(config, dict):
        return GPT2Config(**config)
    if hasattr(config, "to_dict"):
        return GPT2Config(**config.to_dict())
    return config


class MossTTSNanoGPT2Attention(torch.nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        total_num_heads = int(config.num_attention_heads)
        tp_size = int(get_parallel().tp_size)
        if total_num_heads % tp_size != 0:
            raise ValueError(
                f"MOSS-TTS-Nano heads={total_num_heads} must divide tp={tp_size}"
            )
        self.num_heads = total_num_heads // tp_size
        self.head_dim = hidden_size // total_num_heads
        self.c_attn = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            total_num_heads,
            total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("c_attn", prefix),
        )
        self.c_proj = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("c_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(config.max_position_embeddings),
            base=float(getattr(config, "rope_base", 10000.0)),
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        hidden_states = self.attn(q, k, v, forward_batch)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states


class MossTTSNanoGPT2MLP(torch.nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        inner_size = int(config.n_inner or 4 * hidden_size)
        self.fc_in = ColumnParallelLinear(
            hidden_size,
            inner_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc_in", prefix),
        )
        self.fc_out = RowParallelLinear(
            inner_size,
            hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc_out", prefix),
        )
        self.act = NewGELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class MossTTSNanoGPT2Block(torch.nn.Module):
    def __init__(
        self,
        layer_id: int,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        eps = float(config.layer_norm_epsilon)
        self.ln_1 = torch.nn.LayerNorm(hidden_size, eps=eps)
        self.attn = MossTTSNanoGPT2Attention(
            layer_id,
            config,
            quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.ln_2 = torch.nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = MossTTSNanoGPT2MLP(
            config,
            quant_config,
            prefix=add_prefix("mlp", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attn(
            positions,
            self.ln_1(hidden_states),
            forward_batch,
        )
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.ln_2(hidden_states))


class MossTTSNanoGPT2Model(torch.nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.start_layer = 0
        self.end_layer = int(config.num_hidden_layers)
        self.h = torch.nn.ModuleList(
            [
                MossTTSNanoGPT2Block(
                    layer_id,
                    config,
                    quant_config,
                    prefix=add_prefix(f"h.{layer_id}", prefix),
                )
                for layer_id in range(self.start_layer, self.end_layer)
            ]
        )
        self.ln_f = torch.nn.LayerNorm(
            int(config.hidden_size),
            eps=float(config.layer_norm_epsilon),
        )

    def forward(
        self,
        *,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.h:
            hidden_states = layer(positions, hidden_states, forward_batch)
        return self.ln_f(hidden_states)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        del quantization_param_path


class MossTTSNanoSGLangModel(MossTTSLocalSGLangModel):
    """Nano GPT-2+RoPE backbone with the shared frame-local decode contract."""

    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        torch.nn.Module.__init__(self)
        self.pp_group = get_pp_group()
        self.config = self._normalize_nano_config(config)
        self.quant_config = quant_config
        self.hidden_size = int(self.config.hidden_size)
        self.n_vq = int(self.config.n_vq)

        self.embedding_list = torch.nn.ModuleList()
        if self.pp_group.is_first_rank:
            for index in range(int(self.config.channels)):
                self.embedding_list.append(
                    VocabParallelEmbedding(
                        int(self.config.vocab_size_list[index]),
                        self.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"embedding_list.{index}", prefix),
                    )
                )
        else:
            for _ in range(int(self.config.channels)):
                self.embedding_list.append(PPMissingLayer())

        gpt2_config = self.config.gpt2_config
        self.model = MossTTSNanoGPT2Model(
            gpt2_config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.local_transformer = MossTTSLocalTransformer(
            hidden_size=self.hidden_size,
            num_heads=int(gpt2_config.n_head),
            inner_size=int(gpt2_config.n_inner or 4 * self.hidden_size),
            num_layers=int(self.config.local_transformer_layers),
            max_positions=self.n_vq + 1,
            rope_base=float(getattr(gpt2_config, "rope_base", 10000.0)),
            layer_norm_eps=float(gpt2_config.layer_norm_epsilon),
            activation="gelu_new",
        )
        # The official generator samples only these two rows of the tied text
        # head, in this order. Keeping the two-row projection preserves the
        # Local runner's 0=continue / 1=stop contract without a 16K matmul.
        self.local_text_lm_head = torch.nn.Linear(
            self.hidden_size,
            2,
            bias=False,
        )

        max_batch_size = None
        try:
            from sglang.srt.server_args import get_global_server_args

            max_batch_size = get_global_server_args().max_running_requests
        except Exception:
            max_batch_size = None
        weight = self._first_embedding_weight()
        self._decode_input_embedding = torch.nn.Embedding(
            int(max_batch_size or 1),
            self.hidden_size,
            device=weight.device,
            dtype=weight.dtype,
        )
        self._decode_input_embedding.weight.requires_grad_(False)
        self._state_pool = MossTTSLocalDecodeStatePool(self)
        self._compiled_frame_sampler = None
        self._large_vocab_frame_sampler = None
        self._frame_compile_configured = False

    @staticmethod
    def _normalize_nano_config(config: Any) -> Any:
        gpt2_config = _as_gpt2_config(getattr(config, "gpt2_config", None))
        if gpt2_config is None:
            raise ValueError("MOSS-TTS-Nano config is missing gpt2_config")
        config.gpt2_config = gpt2_config
        config.language_config = gpt2_config
        config.hidden_size = int(gpt2_config.hidden_size)
        config.vocab_size = int(gpt2_config.vocab_size)
        config.n_vq = int(getattr(config, "n_vq", 16))
        config.channels = config.n_vq + 1
        sizes = list(getattr(config, "audio_codebook_sizes", []) or [])
        if len(sizes) != config.n_vq:
            sizes = [int(getattr(config, "audio_vocab_size", 1024))] * config.n_vq
        if len(set(int(size) for size in sizes)) != 1:
            raise ValueError("MOSS-TTS-Nano requires equal audio codebook sizes")
        config.audio_vocab_size = int(sizes[0])
        config.audio_pad_code = int(config.audio_pad_token_id)
        config.vocab_size_list = [config.vocab_size] + [
            config.audio_vocab_size + 1
        ] * config.n_vq
        config.pad_token = [int(config.pad_token_id)] + [
            int(config.audio_pad_token_id)
        ] * config.n_vq
        return config

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
        del input_embeds_are_projected, pp_proxy_tensors
        if input_embeds is None:
            forward_mode = getattr(forward_batch, "forward_mode", None)
            is_decode = (
                forward_mode is not None
                and hasattr(forward_mode, "is_decode")
                and bool(forward_mode.is_decode())
            )
            if is_decode:
                input_embeds = self._decode_input_embedding(input_ids)
            else:
                input_embeds = self._prepare_multi_modal_inputs(input_ids)

        hidden_states = self.model(
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        sample_hidden_states = self._select_sample_hidden_states(
            hidden_states,
            forward_batch,
        )
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sample_hidden_states,
        )

    @torch.no_grad()
    def _decode_frame_graphable(
        self,
        hidden_states: torch.Tensor,
        text_temperature: torch.Tensor,
        text_top_p: torch.Tensor,
        text_top_k: torch.Tensor,
        audio_temperature: torch.Tensor,
        audio_top_p: torch.Tensor,
        audio_top_k: torch.Tensor,
        seeds: torch.Tensor,
        base_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_hidden = self.local_transformer.step(
            hidden_states.to(dtype=self.dtype), 0
        )
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight).float()
        stop_choice = self._sample_seeded_branchless(
            text_logits,
            temperature=text_temperature,
            top_p=text_top_p,
            top_k=text_top_k,
            seeds=seeds,
            positions=base_positions,
        )

        slot_ids = torch.full_like(
            seeds, int(self.config.audio_assistant_slot_token_id)
        )
        end_ids = torch.full_like(seeds, int(self.config.audio_end_token_id))
        text_ids = torch.where(stop_choice == 0, slot_ids, end_ids)
        current = self.local_transformer.step(self.embedding_list[0](text_ids), 1)
        feedback = self.embedding_list[0](slot_ids)

        codes = []
        for channel in range(self.n_vq):
            head_weight = self._audio_embedding_weight(channel)
            logits = F.linear(current, head_weight).float()
            code = self._sample_seeded_branchless(
                logits,
                temperature=audio_temperature,
                top_p=audio_top_p,
                top_k=audio_top_k,
                seeds=seeds,
                positions=base_positions + channel + 1,
            )
            codes.append(code)
            code_embed = F.embedding(code, head_weight)
            feedback = feedback + code_embed
            if channel + 1 < self.n_vq:
                current = self.local_transformer.step(
                    code_embed.to(dtype=self.dtype), channel + 2
                )
        return stop_choice, torch.stack(codes, dim=-1), feedback

    @torch.no_grad()
    def decode_frame(
        self,
        hidden_states: torch.Tensor,
        *,
        sample_text: Callable[[torch.Tensor], torch.Tensor],
        sample_audio: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_hidden = self.local_transformer.step(
            hidden_states.to(dtype=self.dtype), 0
        )
        text_logits = F.linear(local_hidden, self.local_text_lm_head.weight)
        stop_choice = sample_text(text_logits.float())

        slot_ids = torch.full_like(
            stop_choice, int(self.config.audio_assistant_slot_token_id)
        )
        end_ids = torch.full_like(stop_choice, int(self.config.audio_end_token_id))
        text_ids = torch.where(stop_choice == 0, slot_ids, end_ids)
        current = self.local_transformer.step(self.embedding_list[0](text_ids), 1)

        codes = []
        for channel in range(self.n_vq):
            head_weight = self._audio_embedding_weight(channel)
            logits = F.linear(current, head_weight)
            code = sample_audio(logits.float(), channel)
            codes.append(code)
            if channel + 1 < self.n_vq:
                current = self.local_transformer.step(
                    F.embedding(code, head_weight).to(dtype=self.dtype),
                    channel + 2,
                )
        return stop_choice, torch.stack(codes, dim=-1)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        text_candidate_ids = torch.tensor(
            [
                int(self.config.audio_assistant_slot_token_id),
                int(self.config.audio_end_token_id),
            ],
            dtype=torch.long,
        )

        for original_name, loaded_weight in weights:
            if original_name in {
                "text_lm_head.weight",
                "transformer.wte.weight",
            }:
                if original_name == "transformer.wte.weight":
                    self._load_param(
                        params_dict["embedding_list.0.weight"],
                        loaded_weight,
                    )
                candidate_weight = loaded_weight.index_select(
                    0,
                    text_candidate_ids.to(device=loaded_weight.device),
                )
                self._load_param(
                    params_dict["local_text_lm_head.weight"],
                    candidate_weight,
                )
                continue
            if original_name.startswith("audio_lm_heads."):
                continue
            if original_name.startswith("audio_embeddings.") and original_name.endswith(
                ".weight"
            ):
                mapped = self._map_audio_embedding_name(original_name)
                if mapped is not None and mapped in params_dict:
                    param = params_dict[mapped]
                    rows = int(loaded_weight.shape[0])
                    with torch.no_grad():
                        param.data[:rows].copy_(
                            loaded_weight.to(device=param.device, dtype=param.dtype)
                        )
                continue
            if original_name.startswith(
                "local_transformer.wte."
            ) or original_name.startswith("local_transformer.wpe."):
                continue

            name = original_name
            if name.startswith("transformer."):
                name = "model." + name[len("transformer.") :]
            layer_id = get_layer_id(name)
            if layer_id is not None and not (
                self.model.start_layer <= layer_id < self.model.end_layer
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            param = params_dict.get(name)
            if param is None:
                if name.endswith(".bias"):
                    continue
                logger.warning(
                    "MOSS-TTS-Nano parameter %s not found",
                    original_name,
                )
                continue
            self._load_param(param, loaded_weight)

        self._zero_audio_pad_rows()

    def get_embed_and_head(self) -> tuple[list[Any], list[Any]]:
        embed_weights = [
            getattr(layer, "weight", None) for layer in self.embedding_list
        ]
        return embed_weights, [self.local_text_lm_head.weight]


EntryClass = MossTTSNanoSGLangModel
