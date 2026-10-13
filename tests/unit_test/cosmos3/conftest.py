# SPDX-License-Identifier: Apache-2.0
"""Scoped native API doubles for adapter contract tests, without CUDA imports."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.models.cosmos3 import stages
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling import native_chat_scheduler
from sglang_omni.utils import device as device_utils


@pytest.fixture
def native(monkeypatch):
    """Small native API doubles; never import the CUDA runtime on this host."""
    calls = SimpleNamespace(generation=[], reasoner=[], shutdown=[], apps=[])
    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(index=index if index is not None else 0),
    )
    monkeypatch.setattr(current_platform, "set_device", lambda device: None)
    monkeypatch.setattr(native_chat_scheduler, "warm_up", lambda *args: None)
    monkeypatch.setattr(stages, "model_request_fields", lambda args: frozenset())

    def media_app(server_args, request_fields):
        calls.apps.append(server_args)
        return SimpleNamespace(server_args=server_args)

    monkeypatch.setattr(stages, "NativeMediaApp", media_app)

    def module(name, **attributes):
        # Patch every parent so the test also works with no SGLang installed.
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            key = ".".join(parts[:end])
            if key not in installed:
                value = ModuleType(key)
                value.__path__ = []
                monkeypatch.setitem(sys.modules, key, value)
                installed[key] = value
                if end > 1:
                    setattr(
                        installed[".".join(parts[: end - 1])], parts[end - 1], value
                    )
        for key, value in attributes.items():
            setattr(installed[name], key, value)

    installed = {}

    class Generator:
        local_scheduler_process = None
        supports_cancellation = True

        @classmethod
        def from_server_args(cls, args):
            calls.generation.append(args)
            generator = cls()
            generator.server_args = args
            return generator

        def shutdown(self):
            calls.shutdown.append("generation")

    class Engine:
        def __init__(self, **kwargs):
            calls.reasoner.append(kwargs)
            self.tokenizer_manager = object()
            self.template_manager = object()

        def shutdown(self):
            calls.shutdown.append("reasoner")

    module("sglang", Engine=Engine)
    module(
        "sglang.multimodal_gen.runtime.entrypoints.diffusion_generator",
        DiffGenerator=Generator,
    )
    module(
        "sglang.multimodal_gen.runtime.server_args",
        ServerArgs=SimpleNamespace(from_kwargs=lambda **kw: SimpleNamespace(**kw)),
        set_global_server_args=lambda args: None,
    )
    module(
        "sglang.multimodal_gen.runtime.platforms.plugins",
        apply_plugin_hooks=lambda: None,
    )
    module(
        "sglang.multimodal_gen.runtime.server_warmup",
        is_realtime_serving=lambda args: False,
        should_run_explicit_client_warmup=lambda args: False,
        should_run_synthetic_server_warmup=lambda args: False,
        supports_synthetic_warmup=lambda args: False,
    )
    module("sglang.srt.entrypoints.openai.protocol", ChatCompletionRequest=object)
    module(
        "sglang.srt.entrypoints.openai.serving_chat",
        OpenAIServingChat=lambda *args: object(),
    )
    calls.modules = installed
    return calls
