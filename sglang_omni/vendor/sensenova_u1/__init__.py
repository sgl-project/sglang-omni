# Modified for SGLang; see this directory's README.md for upstream source.

from __future__ import annotations

from .configuration_neo_chat import NEOChatConfig, NEOLLMConfig, NEOMoELLMConfig
from .configuration_neo_vit import NEOVisionConfig
from .modeling_neo_chat import NEOChatModel
from .modeling_neo_vit import NEOVisionModel
from .modeling_qwen3 import _HAS_FLASH_ATTN as has_flash_attn
from .modeling_qwen3 import (
    Qwen3ForCausalLM,
    effective_attn_backend,
    get_attn_backend,
    set_attn_backend,
)
from .modeling_qwen3_moe import Qwen3MoeForCausalLM

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


_REGISTERED = False


def register() -> None:
    """Register SenseNova configuration and model classes with Transformers."""
    global _REGISTERED
    if _REGISTERED:
        return

    from transformers import AutoConfig, AutoModel

    AutoConfig.register("neo_vision", NEOVisionConfig, exist_ok=True)
    AutoConfig.register("neo_chat", NEOChatConfig, exist_ok=True)

    AutoModel.register(NEOVisionConfig, NEOVisionModel, exist_ok=True)
    AutoModel.register(NEOChatConfig, NEOChatModel, exist_ok=True)

    _REGISTERED = True
