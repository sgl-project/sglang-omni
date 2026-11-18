"""SGLang pointwise kernels for the trained causal Realization and Intent modules."""

import logging

import torch
from peft import PeftModel
from sglang.kernels.ops.activation.activation import (
    silu_and_mul_with_activation_rounding,
)
from sglang.srt.layers.layernorm import RMSNorm
from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

logger = logging.getLogger(__name__)


class SGLangQwenMLP(Qwen3VLTextMLP):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        value = self.up_proj(hidden_states)
        activated = silu_and_mul_with_activation_rounding(torch.cat((gate, value), -1))
        return self.down_proj(activated)


def use_intent_kernels(model: PeftModel | Qwen3VLForConditionalGeneration) -> None:
    base = model.get_base_model() if isinstance(model, PeftModel) else model
    language = base.model.language_model
    for layer in language.layers:
        for owner, name in (
            (layer, "input_layernorm"),
            (layer, "post_attention_layernorm"),
            (layer.self_attn, "q_norm"),
            (layer.self_attn, "k_norm"),
        ):
            original = owner.get_submodule(name)
            norm = RMSNorm(
                original.weight.numel(),
                eps=original.variance_epsilon,
                cast_x_before_out_mul=True,
                weight_dtype=original.weight.dtype,
            )
            norm.weight = original.weight
            owner.set_submodule(name, norm)
        # Device dispatch may retain a bound forward on the loaded instance.
        with torch.device("meta"):
            mlp = SGLangQwenMLP(layer.mlp.config)
        mlp.gate_proj = layer.mlp.gate_proj
        mlp.up_proj = layer.mlp.up_proj
        mlp.down_proj = layer.mlp.down_proj
        layer.mlp = mlp
    original = language.norm
    language.norm = RMSNorm(
        original.weight.numel(),
        eps=original.variance_epsilon,
        cast_x_before_out_mul=True,
        weight_dtype=original.weight.dtype,
    )
    language.norm.weight = original.weight
    logger.info("Intent kernels: SGLang HF-rounding RMSNorm and SwiGLU")
