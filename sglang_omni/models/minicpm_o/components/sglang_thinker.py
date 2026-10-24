# SPDX-License-Identifier: Apache-2.0
"""Run the MiniCPM-o text backbone on SGLang's Qwen3 model."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from transformers import PretrainedConfig

from sglang_omni.models.minicpm_o.hf_config import derive_text_config

# note (0xtoward): verify rows kept for the talker, room for every running request's block.
SPECULATIVE_HIDDEN_ROWS = 8192

NON_TEXT_PREFIXES = (
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
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
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
        self.speculative_hidden_rows: torch.Tensor | None = None
        self.prefill_last_hidden: torch.Tensor | None = None

    @property
    def thinker(self) -> "MiniCPMOThinkerForCausalLM":
        # note (MayDomine): the shared thinker runner expects this backbone view.
        return self

    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def get_embed_and_head(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.language_model.get_embed_and_head()

    def keep_speculative_hidden_rows(
        self,
        module: nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        output: torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]],
    ) -> None:
        """Keep every verify row, and the last row of each prefilled request."""
        forward_batch = args[2] if len(args) > 2 else kwargs["forward_batch"]
        hidden = output[0] if isinstance(output, tuple) else output
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_target_verify():
            self.speculative_hidden_rows[: hidden.shape[0]].copy_(hidden)
        elif forward_mode.is_extend():
            last_rows = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            )
            self.prefill_last_hidden = hidden[last_rows.long()].clone()
        else:
            pass

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        self.language_model.set_eagle3_layers_to_capture(layer_ids)
        # note (0xtoward): the captured verify graph rewrites this fixed buffer on replay.
        head = self.language_model.lm_head.weight
        self.speculative_hidden_rows = torch.empty(
            SPECULATIVE_HIDDEN_ROWS,
            self.config.hidden_size,
            dtype=head.dtype,
            device=head.device,
        )
        self.language_model.model.register_forward_hook(
            self.keep_speculative_hidden_rows, with_kwargs=True
        )

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        self.language_model.set_dflash_layers_to_capture(layer_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        omni_prefill_rids: list[str] | tuple[str, ...] | None = None,
        **kwargs: object,
    ) -> LogitsProcessorOutput:
        # note (Junnan Li): Request ids are part of the prefill interface; text-only prefill ignores them.
        return self.language_model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def _text_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                if name.startswith(NON_TEXT_PREFIXES):
                    continue
                else:
                    pass
                if name.startswith("llm."):
                    yield name[len("llm.") :], loaded_weight
                else:
                    pass

        self.language_model.load_weights(_text_weights())


EntryClass = MiniCPMOThinkerForCausalLM
