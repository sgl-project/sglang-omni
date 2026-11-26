# SPDX-License-Identifier: Apache-2.0
"""Haru speech components on the SGLang Qwen3.5 backbone."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from haru.config import HaruConfig
from haru.models.haru import AudioProjector, HaruModel
from haru.models.meanflow_head import MeanFlowHead
from haru.models.vibevoice_codec import VibeVoiceCodec
from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.utils import add_prefix
from torch import nn


class HaruSGLangModel(HaruModel):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        # note (BeiningWu): The Haru initializer allocates a Transformers backbone whose cache would be separate from SGLang's.
        nn.Module.__init__(self)
        haru_config = HaruConfig.from_dict(config.haru_config)
        self.config: HaruConfig = haru_config
        self.lm: Qwen3_5ForCausalLM = Qwen3_5ForCausalLM(
            config,
            quant_config=quant_config,
            prefix=add_prefix("lm", prefix),
        )
        self.model_audio_proj: AudioProjector = AudioProjector(
            haru_config.latent_dim, haru_config.hidden_size
        )
        self.user_audio_proj: AudioProjector = AudioProjector(
            haru_config.latent_dim, haru_config.hidden_size
        )
        self.head: MeanFlowHead = MeanFlowHead(haru_config.head)
        self.codec: VibeVoiceCodec = VibeVoiceCodec(haru_config.codec)
        self.register_buffer("latent_scale", torch.ones((), dtype=torch.float32))
        self.register_buffer("latent_bias", torch.zeros((), dtype=torch.float32))
        self.logits_processor: LogitsProcessor = LogitsProcessor(config)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.lm.get_input_embeddings()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        omni_prefill_rids: list[str] | tuple[str, ...] | None = None,
    ) -> LogitsProcessorOutput:
        # note (BeiningWu): Omni forwards request IDs with fused prefill inputs. This model carries no per-request state.
        hidden_states = self.lm(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        assert isinstance(hidden_states, torch.Tensor)
        return self.logits_processor(
            input_ids, hidden_states, self.get_input_embeddings(), forward_batch
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        speech_weights: dict[str, torch.Tensor] = {}

        def backbone_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith("lm."):
                    yield name.removeprefix("lm."), weight
                else:
                    speech_weights[name] = weight

        loaded_backbone = self.lm.load_weights(backbone_weights())
        unmatched = self.load_state_dict(speech_weights, strict=False)
        # note (BeiningWu): Gated DeltaNet exposes its parameter objects through both the layer and its attention module.
        expected_backbone = {name for name, _ in self.lm.named_parameters()}
        unloaded = {
            *(name for name in unmatched.missing_keys if not name.startswith("lm.")),
            *(f"lm.{name}" for name in expected_backbone - loaded_backbone),
        }
        if unloaded or unmatched.unexpected_keys:
            raise ValueError(
                f"Unexpected Haru checkpoint weights {unmatched.unexpected_keys}; "
                f"missing weights {sorted(unloaded)}"
            )
        else:
            pass
        return {
            *speech_weights,
            *(f"lm.{name}" for name in loaded_backbone),
        }
