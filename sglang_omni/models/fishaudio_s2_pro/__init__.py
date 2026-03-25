# SPDX-License-Identifier: Apache-2.0
"""FishAudio S2-Pro (FishQwen3OmniForCausalLM) model support for sglang-omni."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from . import config

if TYPE_CHECKING:
    from .factory import create_s2pro_sglang_engine
    from .runtime.s2pro_ar import S2ProStepOutput
    from .runtime.s2pro_sglang_ar import S2ProSGLangRequestData
    from .tokenizer import Reference, S2ProTokenizerAdapter


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "create_s2pro_sglang_engine": (".factory", "create_s2pro_sglang_engine"),
    "S2ProSGLangRequestData": (".runtime.s2pro_sglang_ar", "S2ProSGLangRequestData"),
    "S2ProStepOutput": (".runtime.s2pro_ar", "S2ProStepOutput"),
    "S2ProTokenizerAdapter": (".tokenizer", "S2ProTokenizerAdapter"),
    "Reference": (".tokenizer", "Reference"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "config",
    "create_s2pro_sglang_engine",
    "S2ProSGLangRequestData",
    "S2ProStepOutput",
    "S2ProTokenizerAdapter",
    "Reference",
]
