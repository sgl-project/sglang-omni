# SPDX-License-Identifier: Apache-2.0
"""A Llama-shaped ``config.json`` for the temporal transformer.

The checkpoint's own config carries no architecture. The Moshi backbone is a
Llama with two twists SGLang's Llama already knows how to honour from config:
interleaved (GPT-J style) rotary embeddings and an unusually small RMSNorm
epsilon. The Moshi-specific parts (embeddings, depformer) take their shapes
from ``architecture`` directly, not from this config.
"""

from __future__ import annotations

from sglang_omni.models.personaplex.architecture import TEMPORAL_TRANSFORMER, TEXT_CARD

PERSONAPLEX_ARCH = "PersonaPlexForCausalLM"
DEFAULT_CONTEXT_LENGTH = 8192


def build_backbone_config(*, context_length: int = DEFAULT_CONTEXT_LENGTH) -> dict:
    spec = TEMPORAL_TRANSFORMER
    return {
        "architectures": [PERSONAPLEX_ARCH],
        "model_type": "llama",
        "hidden_size": spec.dim,
        "intermediate_size": spec.ffn_hidden,
        "num_hidden_layers": spec.num_layers,
        "num_attention_heads": spec.num_heads,
        "num_key_value_heads": spec.num_heads,
        "head_dim": spec.head_dim,
        "vocab_size": TEXT_CARD,
        "rms_norm_eps": spec.rms_norm_eps,
        "rope_theta": spec.rope_max_period,
        "rope_is_neox_style": False,
        "max_position_embeddings": int(context_length),
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }


__all__ = ["DEFAULT_CONTEXT_LENGTH", "PERSONAPLEX_ARCH", "build_backbone_config"]
