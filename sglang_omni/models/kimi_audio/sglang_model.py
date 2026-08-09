# SPDX-License-Identifier: Apache-2.0
"""SGLang-native Kimi-Audio model for text output."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2DecoderLayer
from sglang.srt.utils import add_prefix
from torch import nn


class KimiVQAdaptor(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.hidden_size, config.hidden_size, bias=True),
            nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class KimiAudioTextModel(nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.text_blank_id = int(getattr(config, "kimia_text_blank", 151666))
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    config=config,
                    layer_id=index,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{index}", prefix),
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vq_adaptor = KimiVQAdaptor(config)

    def _prefill_streams(
        self, forward_batch: ForwardBatch, input_ids: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if not forward_batch.forward_mode.is_extend():
            return None, None, None, None

        mm_inputs = forward_batch.mm_inputs
        extend_lens = forward_batch.extend_seq_lens_cpu
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        if (
            mm_inputs is None
            or extend_lens is None
            or prefix_lens is None
            or not (len(mm_inputs) == len(extend_lens) == len(prefix_lens))
        ):
            raise ValueError("Kimi-Audio extend batch is missing stream metadata")

        audio_ids: list[torch.Tensor] = []
        text_ids: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        features: list[torch.Tensor] = []
        cursor = 0
        for mm_input, prefix_len, extend_len in zip(
            mm_inputs, prefix_lens, extend_lens, strict=True
        ):
            prefix_len = int(prefix_len)
            extend_len = int(extend_len)
            segment = input_ids[cursor : cursor + extend_len]
            cursor += extend_len

            if mm_input is None or not mm_input.mm_items:
                audio_ids.append(torch.full_like(segment, self.text_blank_id))
                text_ids.append(segment)
                masks.append(torch.zeros_like(segment, dtype=torch.bool))
                continue

            item = mm_input.mm_items[0]
            prompt_text = item.text_input_ids
            prompt_mask = item.continuous_mask
            prompt_len = int(prompt_text.numel())
            prompt_end = min(prefix_len + extend_len, prompt_len)
            prompt_count = max(prompt_end - prefix_len, 0)

            if prompt_count:
                audio_ids.append(segment[:prompt_count])
                text_ids.append(
                    prompt_text[prefix_len:prompt_end].to(device=input_ids.device)
                )
                mask_slice = prompt_mask[prefix_len:prompt_end]
                masks.append(mask_slice)
                if bool(mask_slice.any()):
                    if item.feature is None:
                        raise ValueError(
                            "Kimi-Audio continuous features are unavailable for "
                            "prompt recomputation"
                        )
                    feature_start = int(prompt_mask[:prefix_len].sum().item())
                    feature_end = feature_start + int(mask_slice.sum().item())
                    features.append(item.feature[feature_start:feature_end])

            if prompt_count < extend_len:
                generated = segment[prompt_count:]
                audio_ids.append(torch.full_like(generated, self.text_blank_id))
                text_ids.append(generated)
                masks.append(torch.zeros_like(generated, dtype=torch.bool))

        if cursor != input_ids.numel():
            raise ValueError(
                "Kimi-Audio extend lengths do not match the flattened token batch"
            )
        flat_audio = torch.cat(audio_ids)
        flat_text = torch.cat(text_ids)
        flat_mask = torch.cat([mask.to(device=input_ids.device) for mask in masks])
        if not (
            flat_audio.numel()
            == flat_text.numel()
            == flat_mask.numel()
            == input_ids.numel()
        ):
            raise ValueError(
                "Kimi-Audio prefill metadata does not match the flattened token batch"
            )
        flat_features = torch.cat(features) if features else None
        return flat_audio, flat_text, flat_mask, flat_features

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        audio_ids, text_ids, continuous_mask, continuous_features = (
            self._prefill_streams(forward_batch, input_ids)
        )
        if text_ids is None:
            # Decode input_ids are the previously sampled text tokens. Audio is
            # fixed to text_blank for Kimi's text-only generation mode.
            text_ids = input_ids
            audio_ids = torch.full_like(input_ids, self.text_blank_id)
        else:
            text_ids = text_ids.to(device=input_ids.device)
            audio_ids = audio_ids.to(device=input_ids.device)

        audio_embeds = self.embed_tokens(audio_ids)
        if continuous_mask is not None and bool(continuous_mask.any()):
            mask = continuous_mask.to(device=audio_embeds.device)
            if continuous_features is None or continuous_features.shape[0] != int(
                mask.sum().item()
            ):
                raise ValueError("Kimi-Audio continuous feature rows do not match mask")
            adapted = self.vq_adaptor(
                continuous_features.to(
                    device=audio_embeds.device, dtype=audio_embeds.dtype
                )
            )
            audio_embeds[mask] = (audio_embeds[mask] + adapted) * math.sqrt(2.0)
        hidden_states = audio_embeds + self.embed_tokens(text_ids)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class KimiAudioForTextGeneration(nn.Module):
    """Text-output subset of ``MoonshotKimiaForCausalLM``."""

    default_bitsandbytes_target_modules: ClassVar[list[str]] = [
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
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if get_pp_group().world_size != 1:
            raise ValueError("Kimi-Audio text generation currently supports PP=1 only")
        self.config = config
        self.model = KimiAudioTextModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            # The checkpoint stores a full union-vocabulary matrix. The
            # LogitsProcessor below crops its output to the valid text range.
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        logits_config = type("KimiTextLogitsConfig", (), {})()
        logits_config.vocab_size = int(config.kimia_text_output_vocab)
        self.logits_processor = LogitsProcessor(logits_config)
        self.start_layer = 0
        self.end_layer = int(config.num_hidden_layers)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **_: Any,
    ) -> Any:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params = dict(self.named_parameters())
        loaded_params: set[str] = set()
        loaded_stacked_shards: dict[str, set[str | int]] = {}
        for original_name, loaded_weight in weights:
            name = original_name
            if "rotary_emb." in name:
                continue
            loaded = False
            for param_name, weight_name, shard_id in stacked:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in params:
                    params[name].weight_loader(params[name], loaded_weight, shard_id)
                    loaded_params.add(name)
                    loaded_stacked_shards.setdefault(name, set()).add(shard_id)
                loaded = True
                break
            if loaded:
                continue
            param = params.get(name)
            if param is None:
                # The MIMO audio branch and audio head are intentionally absent
                # from this text-output model.
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
            loaded_params.add(name)

        problems = sorted(set(params) - loaded_params)
        expected_stacked_shards = {
            "qkv_proj": {"q", "k", "v"},
            "gate_up_proj": {0, 1},
        }
        for name in sorted(loaded_params):
            for marker, expected in expected_stacked_shards.items():
                if marker not in name:
                    continue
                actual = loaded_stacked_shards.get(name, set())
                if actual != expected:
                    missing_shards = sorted(expected - actual, key=str)
                    problems.append(f"{name} (missing shards {missing_shards})")
                break
        if problems:
            raise RuntimeError(
                "Kimi-Audio checkpoint is missing required text-generation weights: "
                + ", ".join(problems)
            )


EntryClass = KimiAudioForTextGeneration

__all__ = ["KimiAudioForTextGeneration", "KimiAudioTextModel"]
