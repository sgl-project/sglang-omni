# Modified for SGLang; see this directory's README.md for upstream source.

import copy

from transformers import Qwen3Config, Qwen3MoeConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_neo_vit import NEOVisionConfig

logger = logging.get_logger(__name__)


def _restore_legacy_rope_theta(config) -> None:
    """Expose the v4 rope attribute expected by the vendored model code."""
    if hasattr(config, "rope_theta"):
        return
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    config.rope_theta = float(rope_parameters.get("rope_theta", 10000.0))


class NEOLLMConfig(Qwen3Config):
    """Dense Qwen3 config with extra RoPE settings for the spatial height and width axes."""

    def __init__(
        self, rope_theta_hw=10000.0, max_position_embeddings_hw=10000, **kwargs
    ):
        super().__init__(**kwargs)
        _restore_legacy_rope_theta(self)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw


class NEOMoELLMConfig(Qwen3MoeConfig):
    """Qwen3-MoE config with separate understanding and image generation expert branches.

    Each generation branch setting falls back to its understanding branch value when unset.
    """

    def __init__(
        self,
        rope_theta_hw=10000.0,
        max_position_embeddings_hw=10000,
        gen_num_experts=None,
        gen_num_experts_per_tok=None,
        gen_moe_intermediate_size=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        _restore_legacy_rope_theta(self)
        self.rope_theta_hw = rope_theta_hw
        self.max_position_embeddings_hw = max_position_embeddings_hw

        self.gen_num_experts = (
            int(gen_num_experts)
            if gen_num_experts is not None
            else int(self.num_experts)
        )
        self.gen_num_experts_per_tok = (
            int(gen_num_experts_per_tok)
            if gen_num_experts_per_tok is not None
            else int(self.num_experts_per_tok)
        )
        self.gen_moe_intermediate_size = (
            int(gen_moe_intermediate_size)
            if gen_moe_intermediate_size is not None
            else int(self.moe_intermediate_size)
        )

        # note (Richard Wang): Qwen3Attention reads config.layer_types, which older Qwen3MoeConfig leaves unset.
        existing = getattr(self, "layer_types", None)
        if not existing or len(existing) != self.num_hidden_layers:
            use_swa = (
                bool(getattr(self, "use_sliding_window", False))
                and getattr(self, "sliding_window", None) is not None
            )
            max_window_layers = int(getattr(self, "max_window_layers", 0) or 0)
            self.layer_types = [
                (
                    "sliding_attention"
                    if (use_swa and i >= max_window_layers)
                    else "full_attention"
                )
                for i in range(self.num_hidden_layers)
            ]


def _is_moe_llm_config(llm_config) -> bool:
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
    for arch in archs:
        arch_str = str(arch)
        if "Moe" in arch_str or "MoE" in arch_str:
            return True
    return (
        bool(has_num_experts)
        and getattr(llm_config, "num_experts", 0)
        and int(getattr(llm_config, "num_experts", 0)) > 1
    )


def _build_llm_config(llm_config):
    if isinstance(llm_config, dict):
        if _is_moe_llm_config(llm_config):
            return NEOMoELLMConfig(**llm_config)
        return NEOLLMConfig(**llm_config)
    return llm_config


class NEOChatConfig(PretrainedConfig):
    model_type = "neo_chat"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        downsample_ratio=0.5,
        template=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {"architectures": ["NEOVisionModel"]}
            logger.info(
                "vision_config is None. Initializing the NEOVisionConfig with default values."
            )

        if llm_config is None:
            llm_config = {"architectures": ["Qwen3ForCausalLM"]}
            logger.info(
                "llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`)."
            )
        assert (
            "architectures" in llm_config
        ), "Should specify architecture in llm_config"

        if isinstance(vision_config, dict):
            self.vision_config = NEOVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.llm_config = _build_llm_config(llm_config)

        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template

        return output
