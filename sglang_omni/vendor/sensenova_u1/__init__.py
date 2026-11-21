# Modified for SGLang; see this directory's README.md for upstream source.
from __future__ import annotations

from .configuration_neo_chat import NEOChatConfig
from .configuration_neo_vit import NEOVisionConfig
from .modeling_neo_chat import NEOChatModel
from .modeling_neo_vit import NEOVisionModel

__all__ = ["register"]


def register() -> None:
    """Register SenseNova configuration and model classes with Transformers."""
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("neo_vision", NEOVisionConfig, exist_ok=True)
    AutoConfig.register("neo_chat", NEOChatConfig, exist_ok=True)

    AutoModel.register(NEOVisionConfig, NEOVisionModel, exist_ok=True)
    AutoModel.register(NEOChatConfig, NEOChatModel, exist_ok=True)
