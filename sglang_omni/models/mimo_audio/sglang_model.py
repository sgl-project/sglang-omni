# SPDX-License-Identifier: Apache-2.0
"""SGLang-native MiMo Qwen2 model for audio-input-to-text inference."""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Sequence
from typing import Any

import torch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import add_prefix
from torch import nn
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model as HFQwen2Model

from .prompt import AUDIO_CHANNELS, GROUP_SIZE, SPEECH_EMPTY_IDS

logger = logging.getLogger(__name__)

OUTPUT_ONLY_WEIGHT_PREFIXES = (
    "local_transformer.",
    "local_transformer_lm_heads.",
    "hidden_states_downcast.",
    "speech_embeddings_to_local.",
)


def parse_channel_values(
    value: str | int | Sequence[int],
    *,
    channels: int = AUDIO_CHANNELS,
) -> tuple[int, ...]:
    if isinstance(value, str):
        values = tuple(int(part) for part in value.split("-"))
    elif isinstance(value, int):
        values = (value,) * channels
    else:
        values = tuple(int(part) for part in value)
    if len(values) != channels:
        raise ValueError(f"expected {channels} channel values, got {len(values)}")
    return values


def is_output_only_weight(name: str) -> bool:
    return name.startswith(OUTPUT_ONLY_WEIGHT_PREFIXES)


def make_input_local_config(config: Qwen2Config) -> Qwen2Config:
    local = copy.deepcopy(config)
    local.hidden_size = int(getattr(config, "input_local_dim", 1024))
    local.num_hidden_layers = int(getattr(config, "input_local_layers", 6))
    local.num_attention_heads = int(getattr(config, "local_attn_heads", 64))
    local.num_key_value_heads = local.num_attention_heads
    local.head_dim = local.hidden_size // local.num_attention_heads
    local.intermediate_size = local.hidden_size * 4
    local.attention_dropout = float(getattr(config, "local_attn_dropout", 0.1))
    local.vocab_size = 1
    local.tie_word_embeddings = False
    local.use_cache = False
    local.layer_types = ["full_attention"] * local.num_hidden_layers
    # Passing an explicit all-visible mask below is only unambiguously
    # non-causal with eager attention across supported Transformers versions.
    local._attn_implementation = "eager"
    return local


class MiMoInputPatchEncoder(nn.Module):
    """Turn `[frames, 8]` RVQ codes into one global embedding per 4 frames."""

    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.group_size = int(getattr(config, "group_size", GROUP_SIZE))
        self.audio_channels = int(getattr(config, "audio_channels", AUDIO_CHANNELS))
        if self.group_size != GROUP_SIZE or self.audio_channels != AUDIO_CHANNELS:
            raise ValueError(
                "Milestone 1 supports MiMo group_size=4 and audio_channels=8, "
                f"got {self.group_size} and {self.audio_channels}"
            )

        self.speech_vocab_sizes = parse_channel_values(
            getattr(config, "speech_vocab_size", "1025-1025-129-129-129-129-129-129")
        )
        self.speech_empty_ids = parse_channel_values(
            getattr(config, "speech_zeroemb_idx", SPEECH_EMPTY_IDS)
        )
        self.input_local_config = make_input_local_config(config)
        self.input_local_transformer = HFQwen2Model(self.input_local_config)
        # Official MiMo supplies `inputs_embeds`; this table has no checkpoint
        # tensor and would waste memory.
        self.input_local_transformer.embed_tokens = None
        self.speech_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, self.input_local_config.hidden_size)
                for vocab_size in self.speech_vocab_sizes
            ]
        )
        self.speech_group_downcast = nn.Linear(
            self.input_local_config.hidden_size * self.group_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        codes = torch.as_tensor(audio_codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] != self.audio_channels:
            raise ValueError(
                "MiMo audio codes must have shape "
                f"[frames, {self.audio_channels}], got {tuple(codes.shape)}"
            )
        if codes.shape[0] == 0 or codes.shape[0] % self.group_size:
            raise ValueError(
                "MiMo audio-code frames must be non-empty and divisible by "
                f"group_size={self.group_size}, got {codes.shape[0]}"
            )
        for channel, vocab_size in enumerate(self.speech_vocab_sizes):
            if torch.any(codes[:, channel] < 0) or torch.any(
                codes[:, channel] >= vocab_size
            ):
                raise ValueError(
                    f"MiMo channel {channel} code is outside [0, {vocab_size})"
                )

        device = self.speech_embeddings[0].weight.device
        codes = codes.to(device)
        groups = codes.shape[0] // self.group_size
        grouped = codes.reshape(groups, self.group_size, self.audio_channels)
        speech_embeds = torch.zeros(
            (groups, self.group_size, self.input_local_config.hidden_size),
            device=device,
            dtype=self.speech_embeddings[0].weight.dtype,
        )
        for channel, embedding in enumerate(self.speech_embeddings):
            speech_embeds.add_(embedding(grouped[:, :, channel]))

        # Each four-frame patch is a separate batch item. An explicit
        # all-visible eager-attention mask reproduces official
        # `input_full_attention=True`.
        output = self.input_local_transformer(
            inputs_embeds=speech_embeds,
            attention_mask={"full_attention": None},
            use_cache=False,
            return_dict=True,
        )
        return self.speech_group_downcast(output.last_hidden_state.reshape(groups, -1))


class MiMoAudioModel(nn.Module):
    """MiMo global Qwen2 plus input-side patch encoder, with text-only output."""

    default_bitsandbytes_target_modules = (
        Qwen2ForCausalLM.default_bitsandbytes_target_modules
    )
    bitsandbytes_stacked_params_mapping = (
        Qwen2ForCausalLM.bitsandbytes_stacked_params_mapping
    )

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = Qwen2ForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.input_patch_encoder = MiMoInputPatchEncoder(config)
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        self.loaded_weight_names: set[str] = set()
        self.skipped_output_weight_names: set[str] = set()
        self.unexpected_weight_names: set[str] = set()

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return torch.cat(
            [self.input_patch_encoder(item.feature) for item in items],
            dim=0,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
            positions=positions,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        global_weights: list[tuple[str, torch.Tensor]] = []
        custom_params = dict(
            self.input_patch_encoder.named_parameters(remove_duplicate=False)
        )
        loaded_custom_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.startswith(("model.", "lm_head.")):
                global_weights.append((name, loaded_weight))
                self.loaded_weight_names.add(name)
                continue
            if is_output_only_weight(name):
                self.skipped_output_weight_names.add(name)
                continue

            custom_name = name
            if custom_name.startswith("input_local_transformer."):
                custom_name = "input_local_transformer." + custom_name.removeprefix(
                    "input_local_transformer."
                )
            if custom_name not in custom_params:
                self.unexpected_weight_names.add(name)
                continue
            param = custom_params[custom_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_custom_params.add(custom_name)
            self.loaded_weight_names.add(name)

        self.language_model.load_weights(global_weights)
        missing_custom_params = set(custom_params) - loaded_custom_params
        if missing_custom_params:
            missing = ", ".join(sorted(missing_custom_params)[:10])
            raise ValueError(f"Missing required MiMo input weights: {missing}")
        if self.unexpected_weight_names:
            unexpected = ", ".join(sorted(self.unexpected_weight_names)[:10])
            raise ValueError(f"Unexpected MiMo checkpoint weights: {unexpected}")
        logger.info(
            "MiMo Milestone 1 loaded %d tensors and intentionally skipped "
            "%d output-only tensors",
            len(self.loaded_weight_names),
            len(self.skipped_output_weight_names),
        )
        return self.loaded_weight_names


EntryClass = MiMoAudioModel

__all__ = [
    "OUTPUT_ONLY_WEIGHT_PREFIXES",
    "EntryClass",
    "MiMoAudioModel",
    "MiMoInputPatchEncoder",
    "is_output_only_weight",
    "make_input_local_config",
    "parse_channel_values",
]
