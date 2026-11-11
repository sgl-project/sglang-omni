# SPDX-License-Identifier: Apache-2.0
"""ASR MTP draft with fine-tuned prediction weights and shared verifier weights."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sglang_omni.modules.ming_audio.thinker import Ming3AudioMTP

MTP_MODEL_ID = "wirybeaver/Ming-3.0-asr-tiny-mtp"
MTP_MODEL_REVISION = "5d6dabd0ba4c739c011951f8982d9d75077ab0e4"


class Ming3ASRMTP(Ming3AudioMTP):
    """Use the fine-tuned ASR prediction block with the original shared head."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        checkpoint = hf_hub_download(
            MTP_MODEL_ID,
            "model.safetensors",
            revision=MTP_MODEL_REVISION,
        )
        fine_tuned = load_file(checkpoint, device="cpu")
        layer_prefix = f"model.layers.{self.config.num_hidden_layers}."

        super().load_weights(
            (name, fine_tuned[name] if name.startswith(layer_prefix) else weight)
            for name, weight in weights
        )
