# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MOSS-TTS Delay model wrapper."""

from __future__ import annotations

import logging
from copy import copy
from typing import Any, Iterable, Optional, Tuple

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
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


class MossTTSDelaySGLangModel(torch.nn.Module):
    """MOSS-TTS Delay AR backbone with one text channel and N RVQ channels."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]

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

        self.embedding_list = torch.nn.ModuleList()
        if self.pp_group.is_first_rank or (
            bool(getattr(self.config, "tie_word_embeddings", False))
            and self.pp_group.is_last_rank
        ):
            for idx in range(self.config.channels):
                self.embedding_list.append(
                    VocabParallelEmbedding(
                        int(self.config.vocab_size_list[idx]),
                        self.hidden_size,
                        quant_config=quant_config,
                        prefix=add_prefix(f"embedding_list.{idx}", prefix),
                    )
                )
        else:
            for _ in range(self.config.channels):
                self.embedding_list.append(PPMissingLayer())

        self.model = Qwen3Model(
            config=self.config.language_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        self.lm_heads = torch.nn.ModuleList()
        if self.pp_group.is_last_rank:
            for idx in range(self.config.channels):
                self.lm_heads.append(
                    ParallelLMHead(
                        num_embeddings=int(self.config.vocab_size_list[idx]),
                        embedding_dim=self.hidden_size,
                        prefix=add_prefix(f"lm_heads.{idx}", prefix),
                    )
                )
        else:
            for _ in range(self.config.channels):
                self.lm_heads.append(PPMissingLayer())

        self.logits_processors = torch.nn.ModuleList(
            [
                self._make_logits_processor(self.config, idx)
                for idx in range(self.config.channels)
            ]
        )
        self._pad_token_per_channel = self._compute_pad_token_per_channel()

        max_batch_size = getattr(getattr(self, "config", None), "max_batch_size", None)
        try:
            from sglang.srt.server_args import get_global_server_args

            max_batch_size = get_global_server_args().max_running_requests
        except Exception:
            max_batch_size = max_batch_size or 1
        weight = self._first_embedding_weight()
        self._decode_input_embedding = torch.nn.Embedding(
            int(max_batch_size or 1),
            self.hidden_size,
            device=weight.device,
            dtype=weight.dtype,
        )
        self._decode_input_embedding.weight.requires_grad_(False)

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
        config.language_config.channels = config.channels
        config.language_config.vocab_size_list = list(config.vocab_size_list)
        config.language_config.pad_token = list(config.pad_token)
        return config

    def _first_embedding_weight(self) -> torch.Tensor:
        for layer in self.embedding_list:
            weight = getattr(layer, "weight", None)
            if isinstance(weight, torch.Tensor):
                return weight
        return torch.empty((), dtype=torch.float32)

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    @property
    def device(self) -> torch.device:
        return self._first_embedding_weight().device

    @property
    def dtype(self) -> torch.dtype:
        return self._first_embedding_weight().dtype

    def _compute_pad_token_per_channel(self) -> list[int]:
        pad = getattr(self.config, "pad_token", None)
        if isinstance(pad, (list, tuple)) and pad:
            pad_ids = [int(value) if value is not None else 0 for value in pad]
            if len(pad_ids) < self.config.channels:
                pad_ids.extend([pad_ids[-1]] * (self.config.channels - len(pad_ids)))
            return pad_ids[: self.config.channels]
        return [int(getattr(self.config, "pad_token_id", 0) or 0)] + [
            int(getattr(self.config, "audio_pad_code", 0) or 0)
        ] * (self.config.channels - 1)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._prepare_multi_modal_inputs(input_ids)

    def _prepare_multi_modal_inputs(self, input_ids: torch.LongTensor) -> torch.Tensor:
        if input_ids.dim() == 1:
            channels = int(self.config.channels)
            total_tokens = int(input_ids.shape[0])
            if total_tokens % channels == 0:
                input_ids_2d = input_ids.view(total_tokens // channels, channels)
            else:
                input_ids_2d = torch.empty(
                    (total_tokens, channels),
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                for idx, pad_id in enumerate(self._pad_token_per_channel):
                    input_ids_2d[:, idx].fill_(int(pad_id))
                input_ids_2d[:, 0] = input_ids
        elif input_ids.dim() == 2:
            input_ids_2d = input_ids
        else:
            raise ValueError(
                "MOSS-TTS input_ids must be rank-1 flattened rows or rank-2 "
                f"multi-channel rows, got shape {tuple(input_ids.shape)}"
            )

        if int(input_ids_2d.shape[-1]) != int(self.config.channels):
            raise ValueError(
                f"MOSS-TTS expected {self.config.channels} channels, "
                f"got {input_ids_2d.shape[-1]}"
            )

        weight = self._first_embedding_weight()
        embeds = torch.zeros(
            input_ids_2d.shape[0],
            self.hidden_size,
            device=input_ids_2d.device,
            dtype=weight.dtype,
        )
        for idx, embed_layer in enumerate(self.embedding_list):
            embeds = embeds + embed_layer(input_ids_2d[:, idx])
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
            is_decode = (
                forward_mode is not None
                and hasattr(forward_mode, "is_decode")
                and bool(forward_mode.is_decode())
            )
            if is_decode:
                input_embeds = self._decode_input_embedding(input_ids)
            elif self.pp_group.is_first_rank:
                input_embeds = self._prepare_multi_modal_inputs(input_ids)
            else:
                input_embeds = None

        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if not self.pp_group.is_last_rank:
            return hidden_states

        sample_hidden_states = self._select_sample_hidden_states(
            hidden_states,
            forward_batch,
        )
        # The MOSS runner samples 33 channels from hidden states after the
        # graph-captured backbone returns. Keeping logits outside model.forward
        # avoids SGLang graph replay dropping customized_info and avoids reusing
        # the text-vocab graph logits buffer for small audio heads.
        dummy_logits = sample_hidden_states.new_empty(
            (sample_hidden_states.shape[0], 1)
        )
        return LogitsProcessorOutput(
            next_token_logits=dummy_logits,
            hidden_states=sample_hidden_states,
        )

    @staticmethod
    def _make_logits_processor(config: Any, channel: int) -> LogitsProcessor:
        """Per-channel LogitsProcessor sized to that channel's own vocab.

        sglang's ``_get_logits`` slices the head output to ``config.vocab_size``
        to strip the ParallelLMHead vocab padding (1025 -> 1088 for the audio
        heads). Sharing the text config (vocab 152k) for the audio heads leaves
        the padding columns 1025..1087 in the audio logits; with the padding
        weights ~0 they get sampled at the audio temperature as out-of-range
        codes, which corrupt frames and drive the high-WER failure tail.

        Per-channel sizing referenced from
        https://github.com/sgl-project/sglang-omni/pull/608.
        """
        channel_config = copy(config)
        channel_config.vocab_size = int(config.vocab_size_list[channel])
        return LogitsProcessor(channel_config)

    def compute_channel_outputs(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> list[LogitsProcessorOutput]:
        logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
        logits_metadata.next_token_logits_buffer = None
        logits_metadata.forward_mode = ForwardMode.DECODE
        return [
            processor(
                None,
                hidden_states=hidden_states,
                lm_head=self.lm_heads[idx],
                logits_metadata=logits_metadata,
            )
            for idx, processor in enumerate(self.logits_processors)
        ]

    def compute_channel_logits(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> list[torch.Tensor]:
        return [
            output.next_token_logits
            for output in self.compute_channel_outputs(hidden_states, forward_batch)
        ]

    @staticmethod
    def _select_sample_hidden_states(
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_extend = (
            forward_mode is not None
            and hasattr(forward_mode, "is_extend")
            and bool(forward_mode.is_extend())
        )
        if not is_extend:
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
            if name.startswith("language_model."):
                name = "model." + name[len("language_model.") :]

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if (
                "rotary_emb.inv_freq" in name
                or "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
                or "projector" in name
            ):
                continue

            if name.startswith("emb_ext.") and name.endswith(".weight"):
                mapped = self._map_audio_embedding_name(name)
                if mapped is not None and mapped in params_dict:
                    self._load_param(params_dict[mapped], loaded_weight)
                continue

            if name == "model.embed_tokens.weight":
                mapped = "embedding_list.0.weight"
                if mapped in params_dict:
                    self._load_param(params_dict[mapped], loaded_weight)

            if name.startswith("lm_heads.") and name.endswith(".weight"):
                if name in params_dict:
                    self._load_param(params_dict[name], loaded_weight)
                continue

            mapped_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    mapped_stacked = True
                    break
                param = params_dict.get(mapped_name)
                if param is None:
                    break
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                mapped_stacked = True
                break
            if mapped_stacked:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict.get(name)
            if param is not None:
                self._load_param(param, loaded_weight)
            else:
                logger.warning(f"MOSS-TTS parameter {original_name} not found")

    @staticmethod
    def _map_audio_embedding_name(name: str) -> str | None:
        try:
            idx = int(name.split(".")[1]) + 1
        except (IndexError, ValueError):
            return None
        return f"embedding_list.{idx}.weight"

    @staticmethod
    def _load_param(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)

    def get_embed_and_head(self) -> tuple[list[Any], list[Any]]:
        embed_weights = [
            getattr(layer, "weight", None) for layer in self.embedding_list
        ]
        head_weights = [getattr(head, "weight", None) for head in self.lm_heads]
        return embed_weights, head_weights

    def set_embed_and_head(self, embed_list: list[Any], head_list: list[Any]) -> None:
        if embed_list is not None:
            for idx, embed in enumerate(embed_list[: len(self.embedding_list)]):
                if embed is not None and hasattr(self.embedding_list[idx], "weight"):
                    self.embedding_list[idx].weight = embed
        if head_list is not None:
            for idx, head in enumerate(head_list[: len(self.lm_heads)]):
                if head is not None and hasattr(self.lm_heads[idx], "weight"):
                    self.lm_heads[idx].weight = head
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)


EntryClass = MossTTSDelaySGLangModel
