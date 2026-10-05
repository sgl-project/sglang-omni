# SPDX-License-Identifier: Apache-2.0
"""SGLang text-only thinker wrapper for Qwen3-Omni.

The upstream SGLang Qwen3-Omni class builds ``thinker.audio_tower`` and
``thinker.visual`` inside the thinker process. Our pipeline already owns those
encoders as standalone stages and injects their embeddings before thinker
prefill, so this wrapper keeps only the text model and LM head.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel, load_fused_expert_weights
from sglang.srt.utils import add_prefix, logger
from transformers import PretrainedConfig

from sglang_omni.models.qwen3_omni.components.thinker_fused_rope import (
    install_thinker_fused_rope,
)
from sglang_omni.quantization import get_weight_preprocessor


def config_uses_mrope(config: PretrainedConfig) -> bool:
    """Return whether the exact Qwen text config declares M-RoPE."""
    for field in ("rope_parameters", "rope_scaling"):
        value = getattr(config, field, None)
        if isinstance(value, Mapping) and value.get("mrope_section") is not None:
            return True
        else:
            pass
    return False


@dataclass(kw_only=True)
class DecodeLiveRows:
    """The real rows of the current decode forward, shared by every MoE top-k."""

    is_live_row: torch.Tensor | None = None


class PaddedRowsTopK(nn.Module):
    """Top-k that gives a decode graph's padded rows the experts of row 0.

    A decode graph replays at its bucket size, and the padded rows keep stale
    token ids, so each would route to its own experts and the fused MoE would
    load their weights for outputs it drops. Real rows route as before.
    """

    def __init__(self, topk: TopK, live_rows: DecodeLiveRows) -> None:
        super().__init__()
        self.topk = topk
        self.live_rows = live_rows

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: torch.Tensor | None = None,
        expert_location_dispatch_info: ExpertLocationDispatchInfo | None = None,
    ) -> TopKOutput:
        topk_output = self.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
        is_live_row = self.live_rows.is_live_row
        # note (ratish): with a padding count (expert parallel) SGLang masks padded
        # rows itself, and runners that route inside the expert kernel return no ids.
        if (
            is_live_row is None
            or num_token_non_padded is not None
            or not isinstance(topk_output, StandardTopKOutput)
        ):
            return topk_output
        else:
            pass
        return StandardTopKOutput(
            topk_weights=topk_output.topk_weights,
            topk_ids=torch.where(
                is_live_row.unsqueeze(1), topk_output.topk_ids, topk_output.topk_ids[:1]
            ),
            router_logits=topk_output.router_logits,
        )

    def empty_topk_output(
        self, device: torch.device, *, layer_id: int | None = None
    ) -> TopKOutput:
        return self.topk.empty_topk_output(device, layer_id=layer_id)


class Qwen3OmniThinkerForCausalLM(nn.Module):
    """Qwen3-Omni thinker text model without duplicated audio/vision towers."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.root_config = config
        self.thinker_config = getattr(config, "thinker_config", config)
        self.config = getattr(self.thinker_config, "text_config", self.thinker_config)
        self.is_mrope_enabled = config_uses_mrope(self.config)

        self.model = Qwen3MoeLLMModel(
            config=self.config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        # note (ratish): the default deepstack addition order lowers image and video
        # accuracy; drop this once the default passes the visual checks.
        self.model.use_hf_deepstack_order = True
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
        self.logits_processor = LogitsProcessor(self.config)
        self.fused_rope_gate = install_thinker_fused_rope(self.model)
        self.decode_live_rows = DecodeLiveRows()
        for layer in self.model.layers:
            if isinstance(layer, Qwen3MoeDecoderLayer) and isinstance(
                layer.mlp, Qwen3MoeSparseMoeBlock
            ):
                layer.mlp.topk = PaddedRowsTopK(layer.mlp.topk, self.decode_live_rows)
            else:
                pass

    @property
    def thinker(self) -> "Qwen3OmniThinkerForCausalLM":
        # Existing Qwen thinker runner/hook code expects model.thinker.model.
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: PPProxyTensors | None = None,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
        omni_prefill_rids: list[str] | tuple[str, ...] | None = None,
    ) -> LogitsProcessorOutput:
        del get_embedding, omni_prefill_rids
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        else:
            pass
        if self.fused_rope_gate is not None:
            self.fused_rope_gate.evaluate(positions, forward_batch)
        else:
            pass
        # note (ratish): padded decode-graph rows write KV slot 0, which the
        # allocator never hands out; a one-row decode has no padded rows.
        if (
            forward_batch.forward_mode.is_decode()
            and forward_batch.out_cache_loc.shape[0] > 1
        ):
            self.decode_live_rows.is_live_row = forward_batch.out_cache_loc != 0
        else:
            self.decode_live_rows.is_live_row = None

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            input_deepstack_embeds=input_deepstack_embeds,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        """Load only thinker text/LM-head weights from the Omni checkpoint."""
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            ("gate_up_proj", "up_proj", 1),
            ("gate_up_proj", "gate_proj", 0),
        ]
        base_expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        num_experts = self.config.num_experts

        preprocess_weight = get_weight_preprocessor(
            self.root_config, fp8_scale_inverted=True
        )

        for name, loaded_weight in weights:
            name = name.replace("model.language_model.", "model.")
            if name.startswith("thinker."):
                name = name[len("thinker.") :]
            elif name.startswith(("talker.", "code2wav.")):
                continue
            else:
                pass

            if name.startswith(("audio_tower.", "visual.")):
                continue
            else:
                pass

            is_fused_expert = False
            expert_params_mapping = base_expert_params_mapping

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping
                else:
                    pass

                if weight_name not in name:
                    continue
                else:
                    pass
                if "mlp.experts" in name:
                    continue
                else:
                    pass

                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(ignore_suffixes) and mapped not in params_dict:
                    continue
                else:
                    pass
                param = params_dict.get(mapped)
                if param is None:
                    continue
                else:
                    pass
                loaded_weight = preprocess_weight(mapped, loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    else:
                        pass
                    is_expert_weight = True
                    mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        loaded = loaded_weight.transpose(-1, -2)
                        if "experts.gate_up_proj" in name:
                            gate_weight, up_weight = loaded.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                mapped, params_dict, gate_weight, "w1", num_experts
                            )
                            load_fused_expert_weights(
                                mapped, params_dict, up_weight, "w3", num_experts
                            )
                        else:
                            load_fused_expert_weights(
                                mapped,
                                params_dict,
                                loaded,
                                shard_id,
                                num_experts,
                            )
                    else:
                        if (
                            mapped.endswith(ignore_suffixes)
                            and mapped not in params_dict
                        ):
                            continue
                        else:
                            pass
                        param = params_dict.get(mapped)
                        if param is None:
                            continue
                        else:
                            pass
                        loaded_weight = preprocess_weight(mapped, loaded_weight)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(
                            param,
                            loaded_weight,
                            mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    break
                else:
                    if is_expert_weight:
                        continue
                    else:
                        pass
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    else:
                        pass
                    param = params_dict.get(name)
                    if param is not None:
                        loaded_weight = preprocess_weight(name, loaded_weight)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    elif name.startswith(("model.", "lm_head.")):
                        logger.warning(
                            "Loaded thinker weight %s not found in text-only params",
                            name,
                        )
                    else:
                        pass


EntryClass = Qwen3OmniThinkerForCausalLM
