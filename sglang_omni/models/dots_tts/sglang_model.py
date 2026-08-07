# SPDX-License-Identifier: Apache-2.0
"""SGLang Qwen2 backbone with the dots.tts continuous-latent head."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from torch import nn

from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead
from sglang_omni.models.weight_loader import default_weight_loader


class DotsTTSSGLangModel(nn.Module):
    """Keep AR execution in SGLang; add only dots-specific model operators."""

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        llm_config = getattr(config, "llm_config", None)
        dots_config = getattr(config, "dots_tts_config", None)
        checkpoint = str(getattr(config, "name_or_path", ""))
        if llm_config is None or not isinstance(dots_config, dict) or not checkpoint:
            raise ValueError(
                "dots.tts requires its top-level config and checkpoint path"
            )
        self.qwen2 = Qwen2ForCausalLM(
            llm_config,
            quant_config=quant_config,
            prefix=f"{prefix}.qwen2" if prefix else "qwen2",
        )
        self.flow = DotsTTSFlowHead(
            dots_config,
            llm_hidden_size=int(llm_config.hidden_size),
            latent_stats_path=str(Path(checkpoint) / "latent_stats.pt"),
            optimize=False,
        )

    def get_input_embeddings(self):
        return self.qwen2.get_input_embeddings()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        qwen_weights: list[tuple[str, torch.Tensor]] = []
        flow_params = dict(self.flow.named_parameters())
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith("llm."):
                qwen_weights.append((name.removeprefix("llm."), tensor))
                continue
            if name.startswith(
                (
                    "audio_encoder.",
                    "dec_mi_layer.",
                    "decoder.",
                    "enc_mi_layer.",
                    "model.",
                    "post_proj.",
                    "pre_proj.",
                    "resample.",
                )
            ):
                continue
            parameter = flow_params.get(name)
            assert parameter is not None, (
                f"Unexpected dots.tts checkpoint weight {name!r}; expected an "
                "llm weight, a separately loaded codec weight, or a flow parameter"
            )
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(f"flow.{name}")
        self.qwen2.load_weights(qwen_weights)
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        del kwargs
        if input_embeds is None:
            input_embeds = forward_batch.input_embeds
        hidden_states = self.qwen2.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        # note (db-ol): dots consumes hidden states only and the runner
        # overwrites next_token_ids, dummy logits just satisfy the contract.
        if forward_batch.forward_mode.is_extend():
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            request_count = (
                int(extend_seq_lens.numel()) if extend_seq_lens is not None else 1
            )
        else:
            request_count = int(hidden_states.shape[0])
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((request_count, 1)),
            hidden_states=hidden_states,
        )


EntryClass = DotsTTSSGLangModel

__all__ = ["DotsTTSSGLangModel", "EntryClass"]
