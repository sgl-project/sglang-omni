# SPDX-License-Identifier: Apache-2.0
"""SGLang text-backbone wrapper for MiniCPM-o.

The MiniCPM-o checkpoint stores the text backbone under the ``llm.`` prefix
next to standalone multimodal towers (``vpm.`` / ``resampler.`` / ``apm.`` /
``audio_projection_layer.`` / ``tts.``). Our pipeline owns those towers as
separate stages, so this wrapper keeps only the text model and LM head and
routes checkpoint names onto a stock SGLang Qwen3 dense model.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen3 import Qwen3ForCausalLM

from sglang_omni.models.minicpm_o.hf_config import derive_text_config

# Checkpoint prefixes owned by other pipeline stages; never loaded here.
_NON_TEXT_PREFIXES = (
    "vpm.",
    "resampler.",
    "apm.",
    "audio_projection_layer.",
    "tts.",
)


class MiniCPMOThinkerForCausalLM(nn.Module):
    """MiniCPM-o text backbone without the multimodal towers."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.root_config = config
        self.config = derive_text_config(config)
        self.language_model = Qwen3ForCausalLM(
            self.config,
            quant_config,
            prefix=prefix,
        )

    @property
    def thinker(self) -> "MiniCPMOThinkerForCausalLM":
        # ThinkerModelRunner expects model.thinker.model.embed_tokens.
        return self

    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        return self.language_model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        def _text_weights():
            for name, loaded_weight in weights:
                if name.startswith(_NON_TEXT_PREFIXES):
                    continue
                if name.startswith("llm."):
                    yield name[len("llm.") :], loaded_weight

        self.language_model.load_weights(_text_weights())


EntryClass = MiniCPMOThinkerForCausalLM
