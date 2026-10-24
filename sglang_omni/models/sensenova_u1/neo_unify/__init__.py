"""SenseNova model registration with transformers.Auto*."""

from __future__ import annotations

from transformers import AutoConfig, AutoModel

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOChatConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_vit import (
    NEOVisionConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_chat import NEOChatModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_vit import NEOVisionModel

# Modified for SGLang; see this directory's README.md for upstream source.


__all__ = ["register"]


def register() -> None:
    """Register NEO-Unify types with transformers.Auto*."""
    AutoConfig.register("neo_vision", NEOVisionConfig, exist_ok=True)
    AutoConfig.register("neo_chat", NEOChatConfig, exist_ok=True)

    AutoModel.register(NEOVisionConfig, NEOVisionModel, exist_ok=True)
    AutoModel.register(NEOChatConfig, NEOChatModel, exist_ok=True)
