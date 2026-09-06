# SPDX-License-Identifier: Apache-2.0
"""SGLang text-only model wrapper for Cosmos3-Nano."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.models.qwen3_vl import Qwen3LLMModel
from sglang.srt.runtime_context import get_server_args
from sglang.srt.utils import add_prefix

from sglang_omni.models.cosmos3.hf_config import Cosmos3OmniConfig

_TEXT_ROOTS = ("embed_tokens.", "layers.", "norm.", "lm_head.")
_GEN_WEIGHT_MARKERS = (
    "_moe_gen",
    ".add_k_proj.",
    ".add_q_proj.",
    ".add_v_proj.",
    ".norm_added_k.",
    ".norm_added_q.",
    ".to_add_out.",
)
_ATTENTION_NAME_MAP = (
    (".self_attn.to_q.", ".self_attn.q_proj."),
    (".self_attn.to_k.", ".self_attn.k_proj."),
    (".self_attn.to_v.", ".self_attn.v_proj."),
    (".self_attn.to_out.", ".self_attn.o_proj."),
    (".self_attn.norm_q.", ".self_attn.q_norm."),
    (".self_attn.norm_k.", ".self_attn.k_norm."),
)


def resolve_cosmos3_text_positions(
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Return Cosmos3's explicit three-axis text mRoPE positions.

    Cosmos3 assigns the same monotonically increasing position to the
    temporal, height, and width axes for every text token. SGLang computes
    that ``[3, num_tokens]`` tensor on ``ForwardBatch`` for mRoPE models. The
    scheduler populates that tensor before invoking the model.
    """

    return forward_batch.mrope_positions


def map_cosmos3_text_weight_name(name: str) -> str | None:
    """Map one flat Cosmos3 text tensor name to the SGLang Qwen3 tree.

    The checkpoint interleaves UND/text and GEN tensors under ``layers.*``.
    Returning ``None`` is intentional: those GEN tensors belong to a later
    stage and must not reach Qwen3's packed-weight loader.
    """

    bare_name = name.removeprefix("model.")
    if not bare_name.startswith(_TEXT_ROOTS):
        return None
    if any(marker in bare_name for marker in _GEN_WEIGHT_MARKERS):
        return None

    for source, target in _ATTENTION_NAME_MAP:
        bare_name = bare_name.replace(source, target)
    return bare_name


class Cosmos3TextForCausalLM(Qwen3ForCausalLM):
    """Cosmos3-Nano text model backed by SGLang's dense Qwen3 kernels."""

    def __init__(
        self,
        config: Cosmos3OmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        torch.nn.Module.__init__(self)
        text_config = config.text_config
        text_config.tie_word_embeddings = bool(config.tie_word_embeddings)
        text_config.vision_config = config.vision_config

        self.pp_group = get_pp_group()
        self.config = text_config
        self.quant_config = quant_config
        self.model = Qwen3LLMModel(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and text_config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    text_config.vocab_size,
                    text_config.hidden_size,
                    quant_config=quant_config,
                    use_attn_tp_group=get_server_args().enable_dp_lm_head,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False
        self.root_config = config

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Any = None,
    ) -> torch.Tensor:
        # Qwen3ForCausalLM normally consumes the scheduler's flat positions.
        # Cosmos3 follows Qwen3-VL's explicit T/H/W text-position convention,
        # so route SGLang's computed mRoPE positions into every decoder layer.
        positions = resolve_cosmos3_text_positions(forward_batch)
        if input_deepstack_embeds is None:
            return super().forward(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            input_deepstack_embeds=input_deepstack_embeds,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if not self.pp_group.is_last_rank:
            return hidden_states
        if get_embedding:
            return self.pooler(hidden_states, forward_batch)
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Any:
        mapped_weights = (
            (mapped_name, loaded_weight)
            for name, loaded_weight in weights
            if (mapped_name := map_cosmos3_text_weight_name(name)) is not None
        )
        return super().load_weights(mapped_weights)


EntryClass = Cosmos3TextForCausalLM

__all__ = [
    "Cosmos3TextForCausalLM",
    "EntryClass",
    "map_cosmos3_text_weight_name",
    "resolve_cosmos3_text_positions",
]
