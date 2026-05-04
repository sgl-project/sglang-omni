# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni package exports.

Keep top-level imports lightweight so callers can import subpackages such as
``sglang_omni_v1.config`` or ``sglang_omni_v1.scheduling`` without immediately
loading the full pipeline runtime and heavyweight scheduler dependencies.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

_EXPORTS: dict[str, tuple[str, str]] = {
    # client
    "AbortLevel": ("sglang_omni_v1.client.types", "AbortLevel"),
    "AbortResult": ("sglang_omni_v1.client.types", "AbortResult"),
    "Client": ("sglang_omni_v1.client.client", "Client"),
    "GenerateChunk": ("sglang_omni_v1.client.types", "GenerateChunk"),
    "GenerateRequest": ("sglang_omni_v1.client.types", "GenerateRequest"),
    "Message": ("sglang_omni_v1.client.types", "Message"),
    "SamplingParams": ("sglang_omni_v1.client.types", "SamplingParams"),
    "UsageInfo": ("sglang_omni_v1.client.types", "UsageInfo"),
    # pipeline
    "Coordinator": ("sglang_omni_v1.pipeline.coordinator", "Coordinator"),
    "AggregatedInput": ("sglang_omni_v1.pipeline.stage.input", "AggregatedInput"),
    "DirectInput": ("sglang_omni_v1.pipeline.stage.input", "DirectInput"),
    "InputHandler": ("sglang_omni_v1.pipeline.stage.input", "InputHandler"),
    "Stage": ("sglang_omni_v1.pipeline.stage.runtime", "Stage"),
    # protocol
    "AbortMessage": ("sglang_omni_v1.proto.messages", "AbortMessage"),
    "CompleteMessage": ("sglang_omni_v1.proto.messages", "CompleteMessage"),
    "DataReadyMessage": ("sglang_omni_v1.proto.messages", "DataReadyMessage"),
    "OmniRequest": ("sglang_omni_v1.proto.request", "OmniRequest"),
    "RequestState": ("sglang_omni_v1.proto.request", "RequestState"),
    "StageInfo": ("sglang_omni_v1.proto.stage", "StageInfo"),
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
