# SPDX-License-Identifier: Apache-2.0
"""Convert and validate the checkpoint text configuration for the Qwen3.5 backbone."""

from __future__ import annotations

import math
from collections.abc import Mapping

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig

FISH_TEXT_MODEL_TYPE = "fish_omni_text"
# note (SunskyXH): fish_omni_text keys Qwen3_5MoeTextConfig has no field for;
# their values are checked against what the backbone hard-codes, then dropped.
FISH_ONLY_TEXT_KEYS = (
    "model_type",
    "layer_types",
    "mlp_layer_types",
    "rotary_interleaved",
    "use_qk_norm",
    "layernorm_zero_centered_gamma",
    "shared_expert_gate",
)
# Values the backbone hard-codes, with the fish_omni_text default when absent.
FISH_TEXT_INVARIANTS = (
    ("rotary_interleaved", False, False),
    ("use_qk_norm", True, True),
    ("attn_output_gate", True, True),
    ("layernorm_zero_centered_gamma", False, True),
)


class Drama3TextConfig(Qwen3_5MoeTextConfig):
    """Backbone config whose full-attention layer ids continue into the decoder tower.

    The tower's layers keep their key/value cache in the backbone's paged pool,
    under the ids that follow the backbone's own layers; the linear-attention
    layers and the backbone layer count are unchanged.
    """

    def __init__(self, *, tower_layer_count: int = 0, **kwargs: object) -> None:
        self.tower_layer_count: int = tower_layer_count
        super().__init__(**kwargs)

    @property
    def full_attention_layer_ids(self) -> list[int]:
        tower_layer_ids = range(
            self.num_hidden_layers, self.num_hidden_layers + self.tower_layer_count
        )
        return [*super().full_attention_layer_ids, *tower_layer_ids]


def checked_int(field_value: object, field_name: str, minimum: int) -> int:
    if (
        isinstance(field_value, bool)
        or not isinstance(field_value, int)
        or field_value < minimum
    ):
        raise ValueError(
            f"{field_name} must be an integer >= {minimum}; got {field_value!r}"
        )
    else:
        return field_value


def checked_number(field_value: object, field_name: str, upper_bound: float) -> float:
    """A finite number in (0, upper_bound]."""
    if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
        raise ValueError(f"{field_name} must be a number; got {field_value!r}")
    elif not (math.isfinite(field_value) and 0.0 < field_value <= upper_bound):
        raise ValueError(
            f"{field_name} must be finite and in (0, {upper_bound}]; got {field_value}"
        )
    else:
        return float(field_value)


def checked_rope_parameters(rope_parameters: object) -> dict[str, float]:
    if not isinstance(rope_parameters, Mapping):
        raise ValueError(
            f"text_config.rope_parameters must be a dict; got {rope_parameters!r}"
        )
    elif rope_parameters.get("rope_type", rope_parameters.get("type", "default")) != (
        "default"
    ):
        raise NotImplementedError(
            f"rope_type of {dict(rope_parameters)!r} is not supported; a scaled "
            f"rope must not load as the default rope"
        )
    elif "mrope_section" in rope_parameters:
        raise NotImplementedError("mrope_section is not supported on this backbone")
    else:
        return {
            "rope_theta": checked_number(
                rope_parameters.get("rope_theta"),
                "rope_parameters.rope_theta",
                math.inf,
            ),
            "partial_rotary_factor": checked_number(
                rope_parameters.get("partial_rotary_factor", 1.0),
                "rope_parameters.partial_rotary_factor",
                1.0,
            ),
        }


def fish_text_kwargs(text_config: Mapping[str, object]) -> dict[str, object]:
    """Validate a fish_omni_text dict and return Qwen3_5MoeTextConfig kwargs for it."""
    num_hidden_layers = checked_int(
        text_config.get("num_hidden_layers"), "text_config.num_hidden_layers", 1
    )
    layer_types = text_config.get("layer_types")
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != num_hidden_layers
        or not set(layer_types) <= {"linear_attention", "full_attention"}
        or "full_attention" not in layer_types
    ):
        raise ValueError(
            f"text_config.layer_types must list linear_attention / full_attention "
            f"for each of the {num_hidden_layers} layers, with at least one "
            f"full_attention; got {layer_types!r}"
        )
    else:
        pass
    full_attention_interval = layer_types.index("full_attention") + 1
    if any(
        (layer_type == "full_attention")
        != ((layer_index + 1) % full_attention_interval == 0)
        for layer_index, layer_type in enumerate(layer_types)
    ):
        raise NotImplementedError(
            "text_config.layer_types is not periodic; Qwen3_5MoeTextConfig only "
            "expresses every full_attention_interval-th layer as full attention"
        )
    elif text_config.get("mlp_layer_types") != ["moe"] * num_hidden_layers:
        raise NotImplementedError(
            "every text_config.mlp_layer_types entry must be moe for the "
            "Qwen3.5 MoE backbone"
        )
    else:
        pass
    for field_name, checkpoint_default, required_value in FISH_TEXT_INVARIANTS:
        if text_config.get(field_name, checkpoint_default) is not required_value:
            raise NotImplementedError(
                f"text_config.{field_name} must be {required_value} (the backbone hard-codes "
                f"it); got {text_config.get(field_name, checkpoint_default)!r}"
            )
        else:
            pass
    if text_config.get("shared_expert_gate", True) is not True and text_config.get(
        "shared_expert_intermediate_size"
    ):
        raise NotImplementedError(
            "text_config.shared_expert_gate=False is not supported: the backbone "
            "always gates its shared expert"
        )
    else:
        pass
    kwargs = {
        field_name: field_value
        for field_name, field_value in text_config.items()
        if field_name not in FISH_ONLY_TEXT_KEYS
    }
    kwargs["full_attention_interval"] = full_attention_interval
    return kwargs


def checked_text_config(text_kwargs: dict[str, object]) -> Drama3TextConfig:
    """Build the backbone config; its RoPE fields must all repeat rope_parameters."""
    rope_parameters = text_kwargs.get("rope_parameters")
    rope = checked_rope_parameters(rope_parameters)
    for field_name, field_value in (
        ("rope_theta", rope["rope_theta"]),
        ("partial_rotary_factor", rope["partial_rotary_factor"]),
        ("rope_scaling", rope_parameters),
    ):
        if text_kwargs.get(field_name) in (None, field_value):
            text_kwargs[field_name] = field_value
        else:
            raise ValueError(
                f"text_config.{field_name}={text_kwargs[field_name]!r} disagrees with "
                f"rope_parameters={rope_parameters!r}"
            )
    if text_kwargs.get("shared_expert_intermediate_size") is None:
        text_kwargs["shared_expert_intermediate_size"] = 0
    else:
        pass
    text_kwargs.setdefault("attn_output_gate", True)
    text_config = Drama3TextConfig(**text_kwargs)
    for field_name, field_value in (
        ("hidden_size", text_config.hidden_size),
        ("num_hidden_layers", text_config.num_hidden_layers),
        ("num_attention_heads", text_config.num_attention_heads),
        ("num_key_value_heads", text_config.num_key_value_heads),
        ("head_dim", text_config.head_dim),
        ("vocab_size", text_config.vocab_size),
        ("max_position_embeddings", text_config.max_position_embeddings),
        ("num_experts", text_config.num_experts),
        ("num_experts_per_tok", text_config.num_experts_per_tok),
        ("moe_intermediate_size", text_config.moe_intermediate_size),
        ("linear_conv_kernel_dim", text_config.linear_conv_kernel_dim),
        ("linear_key_head_dim", text_config.linear_key_head_dim),
        ("linear_value_head_dim", text_config.linear_value_head_dim),
        ("linear_num_key_heads", text_config.linear_num_key_heads),
        ("linear_num_value_heads", text_config.linear_num_value_heads),
        ("full_attention_interval", text_config.full_attention_interval),
    ):
        checked_int(field_value, f"text_config.{field_name}", 1)
    checked_int(
        text_config.shared_expert_intermediate_size,
        "text_config.shared_expert_intermediate_size",
        0,
    )
    checked_number(text_config.rms_norm_eps, "text_config.rms_norm_eps", math.inf)
    rotary_dim = int(text_config.head_dim * text_config.partial_rotary_factor)
    if text_config.num_attention_heads % text_config.num_key_value_heads != 0:
        raise ValueError(
            f"text_config.num_attention_heads={text_config.num_attention_heads} is "
            f"not a multiple of num_key_value_heads={text_config.num_key_value_heads}"
        )
    elif text_config.linear_num_value_heads % text_config.linear_num_key_heads != 0:
        raise ValueError(
            f"text_config.linear_num_value_heads={text_config.linear_num_value_heads} "
            f"is not a multiple of linear_num_key_heads="
            f"{text_config.linear_num_key_heads}"
        )
    elif text_config.num_experts_per_tok > text_config.num_experts:
        raise ValueError(
            f"text_config.num_experts_per_tok={text_config.num_experts_per_tok} "
            f"exceeds num_experts={text_config.num_experts}"
        )
    elif text_config.full_attention_interval > text_config.num_hidden_layers:
        raise ValueError(
            f"text_config.full_attention_interval leaves no full attention layer "
            f"in {text_config.num_hidden_layers} layers"
        )
    elif text_config.attn_output_gate is not True:
        raise NotImplementedError("text_config.attn_output_gate must be True")
    elif rotary_dim == 0 or rotary_dim % 2 != 0:
        raise ValueError(
            f"text_config.head_dim * partial_rotary_factor must give a positive "
            f"even rotary width; got {rotary_dim}"
        )
    else:
        return text_config
