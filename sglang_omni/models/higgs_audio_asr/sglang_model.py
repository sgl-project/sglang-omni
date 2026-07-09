# SPDX-License-Identifier: Apache-2.0
"""Higgs-Audio-v3-STT model compatible with HuggingFace weights.

Whisper-style audio tower + conv/MLP projector feeding a dense
Qwen3-1.7B causal LM. Audio embeddings (12.5/s, one contiguous span per
request covering all 4 s chunks) are scattered into ``<|AUDIO|>``
placeholder positions via ``general_mm_embed_routine``.
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
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.utils import add_prefix

from .audio_tower import HiggsAudioFeatureProjector, HiggsAudioTower
from .configuration_higgs_audio_asr import HiggsAudio3Config, higgs_audio_token_lengths

logger = logging.getLogger(__name__)

# Checkpoint components that only exist for audio *generation* (the TTS
# side of the higgs-audio family) — not loaded for STT.
_SKIP_PREFIXES = (
    "audio_codebook_embeddings.",
    "audio_decoder_proj.audio_lm_head.",
)


class HiggsAudioASRForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: HiggsAudio3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.audio_tower = HiggsAudioTower(config)
        self.audio_encoder_proj = HiggsAudioFeatureProjector(config)
        self.language_model = Qwen3ForCausalLM(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Batch-encode padded 4 s chunks, project, slice each chunk to its
        valid (downsampled) length, concatenate in order."""
        device = next(self.audio_tower.parameters()).device
        dtype = self.audio_tower.dtype

        embeds: list[torch.Tensor] = []
        for item in items:
            features = item.feature.to(device=device, dtype=dtype)
            if features.dim() == 2:
                features = features.unsqueeze(0)  # (1, mels, T)

            mask = getattr(item, "feature_attention_mask", None)
            if mask is not None:
                mel_lens = mask.to(device).long().sum(dim=-1)
            else:
                mel_lens = torch.full(
                    (features.shape[0],), features.shape[-1],
                    dtype=torch.long, device=device,
                )
            valid_lens = higgs_audio_token_lengths(mel_lens.cpu())

            encoded = self.audio_tower(features)          # (chunks, T_out, 1280)
            projected = self.audio_encoder_proj(encoded)  # (chunks, T', hidden)
            for i, valid in enumerate(valid_lens.tolist()):
                embeds.append(projected[i, : int(valid)])

        return torch.cat(embeds, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.AUDIO: self.get_audio_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map the checkpoint's flat layout onto this module tree.

        Checkpoint -> module:
          audio_tower.*                           -> audio_tower.*
          audio_encoder_proj.*                    -> audio_encoder_proj.*
          embed_tokens.weight                     -> language_model.model.embed_tokens.weight
          layers.N.*                              -> language_model.model.layers.N.*
          norm.weight                             -> language_model.model.norm.weight
          audio_decoder_proj.text_lm_head.weight  -> language_model.lm_head.weight
        """
        llm_stacked_params = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb" in name:
                continue
            if name.startswith(_SKIP_PREFIXES):
                continue

            if name == "audio_decoder_proj.text_lm_head.weight":
                name = "language_model.lm_head.weight"
            elif name.startswith("embed_tokens."):
                name = f"language_model.model.{name}"
            elif name.startswith("layers."):
                name = f"language_model.model.{name}"
            elif name.startswith("norm."):
                name = f"language_model.model.{name}"
            # audio_tower.* / audio_encoder_proj.* load by name as-is.

            is_llm = name.startswith("language_model.")
            stacked_params = llm_stacked_params if is_llm else []

            for param_name, weight_name, shard_id in stacked_params:
                if weight_name not in name:
                    continue
                name_tmp = name.replace(weight_name, param_name)
                if name_tmp.endswith(".bias") and name_tmp not in params_dict:
                    continue
                if name_tmp not in params_dict:
                    continue
                param = params_dict[name_tmp]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.debug("higgs-audio-asr: skipping unmapped key %s", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = HiggsAudioASRForConditionalGeneration
