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

from .audio_lengths import arkasr_num_audio_tokens
from .audio_tower import ArkAudioMLPAdapter
from .configuration_arkasr import ArkasrConfig

logger = logging.getLogger(__name__)


class ArkasrForConditionalGeneration(nn.Module):
    DEFAULT_ENCODER_MAX_BATCH_SIZE = 8

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
        self.encoder_max_batch_size = self.DEFAULT_ENCODER_MAX_BATCH_SIZE
        self.encoder_cuda_graph_runner = None

    def set_encoder_max_batch_size(self, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError(
                f"encoder_max_batch_size must be >= 1, got {max_batch_size}"
            )
        self.encoder_max_batch_size = max_batch_size

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Encode cache-miss audio items in bounded sequential microbatches.

        SGLang calls this once for every audio embedding-cache miss in a
        forward batch.  The flattened result must retain item order and exactly
        match every item's precomputed audio-token span.  Request concurrency
        is intentionally independent from encoder batch size: large miss
        batches are split into sequential encoder forwards to bound activation
        memory.
        """
        if not items:
            raise ValueError(
                "ARK-ASR get_audio_feature requires at least one audio item"
            )
        outputs = []
        for start in range(0, len(items), self.encoder_max_batch_size):
            outputs.append(
                self._encode_audio_batch(
                    items[start : start + self.encoder_max_batch_size]
                )
            )
        return torch.cat(outputs, dim=0)

    def _encode_audio_batch(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Pad and encode one bounded batch of audio items."""
        device = next(self.audio_encoder.parameters()).device
        dtype = self.audio_encoder.dtype

        features: list[torch.Tensor] = []
        mel_lengths: list[int] = []
        mel_bins: int | None = None
        for item in items:
            if item.feature is None:
                raise ValueError("ARK-ASR audio item is missing mel features")
            feature = item.feature
            if feature.ndim == 2:
                feature = feature.unsqueeze(0)
            if feature.ndim != 3 or feature.shape[0] != 1:
                raise ValueError(
                    "ARK-ASR expects item.feature shaped [mel_bins, T] or "
                    f"[1, mel_bins, T], got {tuple(feature.shape)}"
                )
            if mel_bins is None:
                mel_bins = int(feature.shape[1])
            elif feature.shape[1] != mel_bins:
                raise ValueError(
                    "ARK-ASR audio features in one batch must share mel_bins; "
                    f"expected {mel_bins}, got {feature.shape[1]}"
                )

            frames = int(feature.shape[-1])
            mask = getattr(item, "feature_attention_mask", None)
            if mask is None:
                valid_frames = frames
            else:
                if mask.ndim == 1:
                    mask = mask.unsqueeze(0)
                if mask.shape != (1, frames):
                    raise ValueError(
                        "ARK-ASR feature_attention_mask shape "
                        f"{tuple(mask.shape)} does not match feature frames "
                        f"{(1, frames)}"
                    )
                mask = mask.to(dtype=torch.bool)
                valid_frames = int(mask.sum().item())
                is_right_padded = bool(mask[0, :valid_frames].all()) and not bool(
                    mask[0, valid_frames:].any()
                )
                if valid_frames <= 0 or not is_right_padded:
                    raise ValueError(
                        "ARK-ASR feature_attention_mask must mark a non-empty "
                        "right-padded valid prefix"
                    )

            features.append(feature[:, :, :valid_frames])
            mel_lengths.append(valid_frames)

        assert mel_bins is not None
        batch_size = len(features)
        max_frames = max(mel_lengths)
        batched = torch.zeros(
            (batch_size, mel_bins, max_frames), device=device, dtype=dtype
        )
        for index, (feature, length) in enumerate(zip(features, mel_lengths)):
            batched[index, :, :length] = feature[0, :, :length].to(
                device=device, dtype=dtype, non_blocking=True
            )

        encoded = None
        if self.encoder_cuda_graph_runner is not None:
            encoded = self.encoder_cuda_graph_runner.run(batched, mel_lengths)

        if encoded is None:
            # note (guozhihao-224): B=1 is unpadded; leave mask=None to match
            # the unmasked encoder forward.
            if batch_size == 1:
                encoder_mask = None
            else:
                frame_index = torch.arange(max_frames, device=device).unsqueeze(0)
                encoder_mask = frame_index < torch.tensor(
                    mel_lengths, device=device, dtype=torch.long
                ).unsqueeze(1)
            encoded = self.audio_encoder(batched, attention_mask=encoder_mask)
        outputs = []
        for batch_index, mel_length in enumerate(mel_lengths):
            num_tokens = arkasr_num_audio_tokens(
                mel_length, self.audio_encoder.merge_factor
            )
            outputs.append(encoded[batch_index, :num_tokens, :])
        return torch.cat(outputs, dim=0)

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
