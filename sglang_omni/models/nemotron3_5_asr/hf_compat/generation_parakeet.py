# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Modified by the SGLang-Omni project for Transformers 5.12.1 compatibility.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transformers 5.13 Parakeet generation delta on the pinned 5.12 base."""

import torch
from transformers.generation import GenerationMixin
from transformers.models.parakeet.generation_parakeet import (
    ParakeetRNNTDecoderCache,
    ParakeetRNNTGenerateOutput,
)
from transformers.models.parakeet.generation_parakeet import (
    ParakeetRNNTGenerationMixin as _ParakeetRNNTGenerationMixin,
)


class ParakeetRNNTGenerationMixin(_ParakeetRNNTGenerationMixin):
    """Forward Nemotron streaming kwargs added by Transformers 5.13."""

    def _prepare_model_inputs(self, *args, **kwargs):
        # Call GenerationMixin directly to avoid running the pinned 5.12
        # Parakeet override before we can forward cache-aware encoder kwargs.
        inputs, input_name, model_kwargs = GenerationMixin._prepare_model_inputs(
            self, *args, **kwargs
        )
        explicit = {"input_features", "attention_mask", "output_attention_mask"}
        irrelevant_prefix = (
            "decoder_",
            "cross_attn",
            "use_cache",
            "past_key_values",
            "cache_params",
        )
        encoder_kwargs = {
            key: value
            for key, value in model_kwargs.items()
            if key not in explicit and not key.startswith(irrelevant_prefix)
        }

        encoder_outputs = self.get_audio_features(
            input_features=inputs,
            attention_mask=model_kwargs.get("attention_mask"),
            output_attention_mask=True,
            **encoder_kwargs,
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        if encoder_outputs.attention_mask is not None:
            encoder_valid_lengths = encoder_outputs.attention_mask.sum(-1)
        else:
            batch_size = encoder_outputs.last_hidden_state.shape[0]
            encoder_valid_lengths = torch.full(
                (batch_size,),
                encoder_outputs.last_hidden_state.shape[1],
                dtype=torch.long,
                device=encoder_outputs.last_hidden_state.device,
            )
        model_kwargs["encoder_valid_lengths"] = encoder_valid_lengths
        model_kwargs["encoder_frame_idxs"] = torch.zeros(
            inputs.shape[0], device=inputs.device, dtype=torch.long
        )
        return inputs, input_name, model_kwargs


__all__ = [
    "ParakeetRNNTDecoderCache",
    "ParakeetRNNTGenerateOutput",
    "ParakeetRNNTGenerationMixin",
]
