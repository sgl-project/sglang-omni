# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR-3B: Whisper(RoPE) audio tower + MLP adapter + dense Qwen2 LM.

Audio embeddings are scattered into ``<|audio|>`` placeholder positions via
``general_mm_embed_routine``, matching the qwen3_asr / higgs_audio_asr pattern.
"""

import logging
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
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

from .audio_tower import ArkAudioMLPAdapter
from .configuration_arkasr import ArkasrConfig

logger = logging.getLogger(__name__)


class ArkasrForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: ArkasrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_token_id = int(getattr(config, "audio_token_id", 151663))

        # audio_encoder = whisper tower + MLP frame-merge adapter (checkpoint name)
        self.audio_encoder = ArkAudioMLPAdapter(config)
        self.language_model = Qwen2ForCausalLM(
            config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encode each request's mel features to LLM-space audio embeddings.

        Each item.feature is (num_mel_bins, T) or (1, num_mel_bins, T). We run
        the tower per item (variable T) and concatenate along the token axis so
        the flat sequence lines up with the scattered <|audio|> positions.
        """
        device = next(self.audio_encoder.parameters()).device
        dtype = self.audio_encoder.dtype
        outs = []
        for item in items:
            feat = item.feature.to(device=device, dtype=dtype)
            if feat.dim() == 2:
                feat = feat.unsqueeze(0)  # (1, mel, T)
            emb = self.audio_encoder(feat)  # (1, Sa, H)
            outs.append(emb.reshape(-1, emb.size(-1)))
        return torch.cat(outs, dim=0)

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        llm_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        tie = bool(getattr(self.config, "tie_word_embeddings", False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if tie and "lm_head.weight" in name:
                continue

            # checkpoint layout:
            #   audio_encoder.whisper.*   audio_encoder.layer_norm.*  audio_encoder.adapting.*
            #   model.*  (Qwen2 decoder)  lm_head.*
            is_audio = name.startswith("audio_encoder.")
            if not is_audio:
                if name.startswith("model."):
                    name = "language_model." + name
                elif name.startswith("lm_head."):
                    name = "language_model." + name

            if is_audio:
                # audio tower params load directly (no qkv stacking: q/k/v are separate
                # Linear layers in WhisperRoPESdpaAttention, matching the checkpoint)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.debug("arkasr: skip unmatched audio weight %s", name)
                    continue
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(
                    param, loaded_weight
                )
                continue

            for param_name, weight_name, shard_id in llm_stacked_params:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(".bias") and mapped not in params_dict:
                    continue
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.debug("arkasr: skip unmatched llm weight %s", name)
                    continue
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(
                    param, loaded_weight
                )


EntryClass = ArkasrForConditionalGeneration
