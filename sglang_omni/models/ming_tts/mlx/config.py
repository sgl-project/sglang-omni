# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, fields

from pydantic import JsonValue


@dataclass
class TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    model_type: str = "bailing_moe"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 600000.0
    rope_scaling: dict[str, JsonValue] | None = None
    max_position_embeddings: int = 32768
    first_k_dense_replace: int = 0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    multi_gate: bool = False
    use_qkv_bias: bool = False
    use_bias: bool = False

    def __post_init__(self) -> None:
        if self.model_type != "bailing_moe" or self.num_experts <= 0:
            raise ValueError(
                "Ming MLX supports the A3B MoE branch, not the dense model"
            )
        else:
            pass
        if (
            min(
                self.hidden_size,
                self.head_dim,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.num_hidden_layers,
            )
            <= 0
        ):
            raise ValueError(
                "Backbone dimensions, heads and layer count must be positive"
            )
        else:
            pass
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        else:
            pass
        if self.head_dim % 2 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "RoPE requires even head_dim and GQA requires divisible heads"
            )
        else:
            pass
        if self.rope_scaling is not None:
            if (
                self.rope_scaling.get("type") != "3D"
                or self.rope_scaling.get("factor") is not None
            ):
                raise ValueError("Ming MLX supports only unscaled 3D RoPE")
            else:
                pass
            sections = self.mrope_section
            if (
                len(sections) != 3
                or min(sections) < 0
                or sum(sections) != self.head_dim // 2
            ):
                raise ValueError(
                    "mrope_section must partition half the head dimension into three axes"
                )
            else:
                pass
        else:
            pass

    @property
    def mrope_section(self) -> tuple[int, ...]:
        if self.rope_scaling is None:
            return (self.head_dim // 2, 0, 0)
        else:
            pass
        return tuple(self.rope_scaling.get("mrope_section", (16, 24, 24)))

    @classmethod
    def from_dict(cls, params: dict[str, JsonValue]) -> TextConfig:
        for name in (
            "use_qk_norm",
            "use_sliding_window",
            "moe_router_enable_expert_bias",
        ):
            if params.get(name, False):
                raise ValueError(f"Ming MLX does not support {name}")
            else:
                pass
        for name in ("n_group", "num_expert_group", "topk_group", "router_dtype"):
            if params.get(name) not in (None, 0):
                raise ValueError(f"Ming MLX does not support {name}")
            else:
                pass
        if params.get("hidden_act", "silu") != "silu" or params.get(
            "score_function"
        ) not in (None, "softmax"):
            raise ValueError("Ming MLX requires SiLU experts and softmax routing")
        else:
            pass
        shared = params.get("moe_shared_expert_intermediate_size")
        if shared is not None and shared != params["moe_intermediate_size"]:
            raise ValueError("Shared and routed expert intermediate sizes must match")
        else:
            pass
        return cls(**{f.name: params[f.name] for f in fields(cls) if f.name in params})


@dataclass
class AcousticConfig:
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    qk_norm: str | None = None
    pe_attn_head: int | None = None

    def __post_init__(self) -> None:
        if min(self.hidden_size, self.num_heads, self.depth) <= 0:
            raise ValueError("Acoustic dimensions, heads and depth must be positive")
        else:
            pass
        if (
            self.hidden_size % self.num_heads
            or (self.hidden_size // self.num_heads) % 2
        ):
            raise ValueError("Acoustic attention requires an even head dimension")
        else:
            pass
        if self.qk_norm is not None or self.pe_attn_head not in (None, self.num_heads):
            raise ValueError(
                "Ming acoustic path requires full-head RoPE without Q/K norm"
            )
        else:
            pass

    @classmethod
    def from_dict(cls, params: dict[str, JsonValue]) -> AcousticConfig:
        if params.get("spk_dim") is not None:
            raise ValueError("Ming MLX uses spk_head, not a DiT speaker token")
        else:
            pass
        return cls(**{f.name: params[f.name] for f in fields(cls) if f.name in params})


@dataclass
class ModelConfig:
    llm_config: TextConfig | dict[str, JsonValue]
    ditar_config: dict[str, JsonValue]
    aggregator_config: dict[str, JsonValue]
    audio_tokenizer_config: dict[str, JsonValue]
    model_type: str = "bailingmm"

    def __post_init__(self) -> None:
        if self.model_type != "bailingmm":
            raise ValueError("Expected the Ming bailingmm composite checkpoint")
        else:
            pass
        if isinstance(self.llm_config, dict):
            self.llm_config = TextConfig.from_dict(self.llm_config)
        else:
            pass
        AcousticConfig.from_dict(self.ditar_config)
        AcousticConfig.from_dict(self.aggregator_config)
        if min(self.patch_size, self.history_patch_size, self.latent_dim) <= 0:
            raise ValueError("Patch, history and latent dimensions must be positive")
        else:
            pass

    @property
    def patch_size(self) -> int:
        return int(self.ditar_config["patch_size"])

    @property
    def history_patch_size(self) -> int:
        return int(self.ditar_config["history_patch_size"])

    @property
    def latent_dim(self) -> int:
        return int(self.audio_tokenizer_config["enc_kwargs"]["latent_dim"])

    @classmethod
    def from_dict(cls, params: dict[str, JsonValue]) -> ModelConfig:
        return cls(**{f.name: params[f.name] for f in fields(cls) if f.name in params})
