"""SenseNova model registration and compatibility exports."""

from __future__ import annotations

from transformers import AutoConfig, AutoModel

from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    HAS_FLASH_ATTN as has_flash_attn,
)
from sglang_omni.models.sensenova_u1.neo_unify.attention_backend import (
    effective_attn_backend,
    get_attn_backend,
    set_attn_backend,
)
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOChatConfig,
    NEOLLMConfig,
    NEOMoELLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_vit import NEOVisionModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import Qwen3ForCausalLM
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)

# Modified for SGLang; see this directory's README.md for upstream source.


__all__ = [
    "NEOChatConfig",
    "NEOLLMConfig",
    "NEOMoELLMConfig",
    "NEOVisionConfig",
    "NEOChatModel",
    "NEOVisionModel",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "register",
    "set_attn_backend",
    "get_attn_backend",
    "effective_attn_backend",
    "has_flash_attn",
]


def register() -> None:
    """Register NEO-Unify types with transformers.Auto*."""
    AutoConfig.register("neo_vision", NEOVisionConfig, exist_ok=True)
    AutoConfig.register("neo_chat", NEOChatConfig, exist_ok=True)

    AutoModel.register(NEOVisionConfig, NEOVisionModel, exist_ok=True)
    AutoModel.register(NEOChatConfig, NEOChatModel, exist_ok=True)
