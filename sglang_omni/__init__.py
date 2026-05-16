# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni package exports.

Keep top-level imports lightweight so callers can import subpackages such as
``sglang_omni.config`` or ``sglang_omni.scheduling`` without immediately
loading the full pipeline runtime and heavyweight scheduler dependencies.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS: dict[str, tuple[str, str]] = {
    # client
    "AbortLevel": ("sglang_omni.client.types", "AbortLevel"),
    "AbortResult": ("sglang_omni.client.types", "AbortResult"),
    "Client": ("sglang_omni.client.client", "Client"),
    "GenerateChunk": ("sglang_omni.client.types", "GenerateChunk"),
    "GenerateRequest": ("sglang_omni.client.types", "GenerateRequest"),
    "Message": ("sglang_omni.client.types", "Message"),
    "SamplingParams": ("sglang_omni.client.types", "SamplingParams"),
    "UsageInfo": ("sglang_omni.client.types", "UsageInfo"),
    # pipeline
    "Coordinator": ("sglang_omni.pipeline.coordinator", "Coordinator"),
    "AggregatedInput": ("sglang_omni.pipeline.stage.input", "AggregatedInput"),
    "DirectInput": ("sglang_omni.pipeline.stage.input", "DirectInput"),
    "InputHandler": ("sglang_omni.pipeline.stage.input", "InputHandler"),
    "Stage": ("sglang_omni.pipeline.stage.runtime", "Stage"),
    # protocol
    "AbortMessage": ("sglang_omni.proto.messages", "AbortMessage"),
    "CompleteMessage": ("sglang_omni.proto.messages", "CompleteMessage"),
    "DataReadyMessage": ("sglang_omni.proto.messages", "DataReadyMessage"),
    "OmniRequest": ("sglang_omni.proto.request", "OmniRequest"),
    "RequestState": ("sglang_omni.proto.request", "RequestState"),
    "StageInfo": ("sglang_omni.proto.stage", "StageInfo"),
}

__all__ = ["__version__", *_EXPORTS.keys()]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
