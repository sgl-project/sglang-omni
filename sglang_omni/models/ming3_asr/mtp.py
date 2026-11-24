# SPDX-License-Identifier: Apache-2.0
"""ASR MTP draft with fine-tuned prediction weights and shared verifier weights."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from safetensors.torch import load_file
from transformers.utils.hub import cached_file

from sglang_omni.modules.ming_audio.thinker import Ming3AudioMTP


class Ming3ASRMTP(Ming3AudioMTP):
    """Use the fine-tuned ASR prediction block with the original shared head."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        checkpoint = cached_file(
            self.config.mtp_weights_path,
            "model.safetensors",
            revision=self.config.mtp_weights_revision,
        )
        fine_tuned = load_file(checkpoint, device="cpu")
        layer_prefix = f"model.layers.{self.config.num_hidden_layers}."

        super().load_weights(
            (name, fine_tuned.pop(name) if name.startswith(layer_prefix) else weight)
            for name, weight in weights
        )
        if fine_tuned:
            raise ValueError(
                f"Unused ASR MTP weights in {checkpoint}: {sorted(fine_tuned)}"
            )
        else:
            pass
