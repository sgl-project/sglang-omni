# SPDX-License-Identifier: Apache-2.0
"""SGLang text-only thinker wrapper for Qwen3-Omni.

The upstream SGLang Qwen3-Omni class builds ``thinker.audio_tower`` and
``thinker.visual`` inside the thinker process. Our pipeline already owns those
encoders as standalone stages and injects their embeddings before thinker
prefill, so this wrapper keeps only the text model and LM head.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel, load_fused_expert_weights
from sglang.srt.utils import add_prefix, logger


class Qwen3OmniThinkerForCausalLM(nn.Module):
    """Qwen3-Omni thinker text model without duplicated audio/vision towers."""

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.root_config = config
        self.thinker_config = getattr(config, "thinker_config", config)
        self.config = getattr(self.thinker_config, "text_config", self.thinker_config)

        self.model = Qwen3MoeLLMModel(
            config=self.config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
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

    @property
    def thinker(self) -> "Qwen3OmniThinkerForCausalLM":
        # Existing Qwen thinker runner/hook code expects model.thinker.model.
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        get_embedding: bool = False,
        pp_proxy_tensors: Any | None = None,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ):
        del get_embedding
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

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

        for name, loaded_weight in weights:
            name = name.replace("model.language_model.", "model.")
            if name.startswith("thinker."):
                name = name[len("thinker.") :]
            elif name.startswith(("talker.", "code2wav.")):
                continue

            if name.startswith(("audio_tower.", "visual.")):
                continue

            is_fused_expert = False
            expert_params_mapping = base_expert_params_mapping

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue

                mapped = name.replace(weight_name, param_name)
                if mapped.endswith(ignore_suffixes) and mapped not in params_dict:
                    continue
                param = params_dict.get(mapped)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
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
                        param = params_dict.get(mapped)
                        if param is None:
                            continue
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
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    param = params_dict.get(name)
                    if param is not None:
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    elif name.startswith(("model.", "lm_head.")):
                        logger.warning(
                            "Loaded thinker weight %s not found in text-only params",
                            name,
                        )


EntryClass = Qwen3OmniThinkerForCausalLM
