# SPDX-License-Identifier: Apache-2.0
"""YuE2 autoregressive backbone on SGLang's Qwen3 implementation."""

from collections.abc import Iterable

import torch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from transformers import AutoConfig, Qwen3Config

from sglang_omni.models.yue2.shared_weights import is_acoustic_only


class Yue2ARConfig(Qwen3Config):
    model_type = "yue2"


class Yue2ARForCausalLM(Qwen3ForCausalLM):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        # note (ruiling): Qwen3 fuses any name containing q_proj or gate_proj, so the
        # acoustic layers' nar_* copies must never reach its loader.
        super().load_weights(
            (name, tensor) for name, tensor in weights if not is_acoustic_only(name)
        )


def register_yue2_config() -> None:
    AutoConfig.register(Yue2ARConfig.model_type, Yue2ARConfig, exist_ok=True)


EntryClass = Yue2ARForCausalLM
