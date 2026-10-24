"""Dense and mixture-of-experts configuration for SenseNova."""

import copy
import logging

from pydantic import JsonValue
from transformers import Qwen3Config, Qwen3MoeConfig
from transformers.configuration_utils import PretrainedConfig

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)

# Modified for SGLang; see this directory's README.md for upstream source.


logger = logging.getLogger(__name__)


def restore_legacy_rope_theta(
    config: Qwen3Config | Qwen3MoeConfig,
) -> None:
    """Expose the v4 rope attribute expected by the vendored model code."""
    if hasattr(config, "rope_theta"):
        return
    else:
        pass
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    config.rope_theta = float(rope_parameters.get("rope_theta", 10000.0))


class NEOLLMConfig(Qwen3Config):
    """Config for the dense Qwen3 backbone used by NEO-Unify."""

    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        **kwargs: JsonValue,
    ) -> None:
        super().__init__(**kwargs)
        restore_legacy_rope_theta(self)
        self.rope_theta_hw: float = rope_theta_hw
        self.max_position_embeddings_hw: int = max_position_embeddings_hw


class NEOMoELLMConfig(Qwen3MoeConfig):
    """Config for the Qwen3-MoE backbone used by NEO-Unify."""

    def __init__(
        self,
        rope_theta_hw: float = 10000.0,
        max_position_embeddings_hw: int = 10000,
        gen_num_experts: int | None = None,
        gen_num_experts_per_tok: int | None = None,
        gen_moe_intermediate_size: int | None = None,
        **kwargs: JsonValue,
    ) -> None:
        super().__init__(**kwargs)
        restore_legacy_rope_theta(self)
        self.rope_theta_hw: float = rope_theta_hw
        self.max_position_embeddings_hw: int = max_position_embeddings_hw

        self.gen_num_experts: int = (
            int(gen_num_experts)
            if gen_num_experts is not None
            else int(self.num_experts)
        )
        self.gen_num_experts_per_tok: int = (
            int(gen_num_experts_per_tok)
            if gen_num_experts_per_tok is not None
            else int(self.num_experts_per_tok)
        )
        self.gen_moe_intermediate_size: int = (
            int(gen_moe_intermediate_size)
            if gen_moe_intermediate_size is not None
            else int(self.moe_intermediate_size)
        )

        existing = getattr(self, "layer_types", None)
        if not existing or len(existing) != self.num_hidden_layers:
            use_swa = (
                bool(getattr(self, "use_sliding_window", False))
                and getattr(self, "sliding_window", None) is not None
            )
            max_window_layers = int(getattr(self, "max_window_layers", 0) or 0)
            self.layer_types: list[str] = [
                (
                    "sliding_attention"
                    if (use_swa and i >= max_window_layers)
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]
        else:
            pass


def is_moe_llm_config(
    llm_config: dict[str, JsonValue] | Qwen3Config | Qwen3MoeConfig
) -> bool:
    """Detect whether an llm_config (dict or object) targets a MoE backbone."""
    if isinstance(llm_config, dict):
        model_type = llm_config.get("model_type", "")
        archs = llm_config.get("architectures") or []
        has_num_experts = "num_experts" in llm_config
    else:
        model_type = getattr(llm_config, "model_type", "")
        archs = getattr(llm_config, "architectures", None) or []
        has_num_experts = hasattr(llm_config, "num_experts")

    if isinstance(model_type, str) and "moe" in model_type.lower():
        return True
    else:
        pass
    for arch in archs:
        arch_str = str(arch)
        if "Moe" in arch_str or "MoE" in arch_str:
            return True
        else:
            pass
    num_experts = (
        llm_config.get("num_experts", 0)
        if isinstance(llm_config, dict)
        else getattr(llm_config, "num_experts", 0)
    )
    return bool(has_num_experts and num_experts and int(num_experts) > 1)


def build_llm_config(
    llm_config: dict[str, JsonValue] | NEOLLMConfig | NEOMoELLMConfig
) -> NEOLLMConfig | NEOMoELLMConfig:
    """Instantiate the right LLM config object from a dict or pre-built config."""
    if isinstance(llm_config, dict):
        if is_moe_llm_config(llm_config):
            return NEOMoELLMConfig(**llm_config)
        else:
            pass
        return NEOLLMConfig(**llm_config)
    else:
        pass
    return llm_config


class NEOChatConfig(PretrainedConfig):
    model_type: str = "neo_chat"
    is_composition: bool = True

    def __init__(
        self,
        vision_config: dict[str, JsonValue] | NEOVisionConfig | None = None,
        llm_config: dict[str, JsonValue] | NEOLLMConfig | NEOMoELLMConfig | None = None,
        use_backbone_lora: int = 0,
        use_llm_lora: int = 0,
        downsample_ratio: float = 0.5,
        template: str | None = None,
        **kwargs: JsonValue,
    ) -> None:
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"architectures": ["NEOVisionModel"]}
            logger.info(
                "vision_config is None. Initializing the NEOVisionConfig with default values."
            )
        else:
            pass

        if llm_config is None:
            llm_config = {"architectures": ["Qwen3ForCausalLM"]}
            logger.info(
                "llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`)."
            )
        else:
            pass
        assert (
            "architectures" in llm_config
        ), "Should specify architecture in llm_config"

        if isinstance(vision_config, dict):
            self.vision_config: NEOVisionConfig = NEOVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.llm_config: NEOLLMConfig | NEOMoELLMConfig = build_llm_config(llm_config)

        self.use_backbone_lora: int = use_backbone_lora
        self.use_llm_lora: int = use_llm_lora
        self.downsample_ratio: float = downsample_ratio
        self.template: str | None = template
        self.tie_word_embeddings: bool = self.llm_config.tie_word_embeddings

    @property
    def is_moe_llm(self) -> bool:
        """Convenience flag so callers can switch between dense / MoE LLM."""
        return isinstance(self.llm_config, NEOMoELLMConfig)

    def to_dict(self) -> dict[str, JsonValue]:
        """Serializes this instance to a Python dictionary. Override the default [~PretrainedConfig.to_dict]."""
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template

        return output
