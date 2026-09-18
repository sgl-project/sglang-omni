# SPDX-License-Identifier: Apache-2.0
"""Scoped native API doubles for adapter contract tests, without CUDA imports."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI


@pytest.fixture
def native(monkeypatch):
    """Small native API doubles; never import the CUDA runtime on this host."""
    calls = SimpleNamespace(generation=[], reasoner=[], shutdown=[], apps=[])

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

        @classmethod
        def from_server_args(cls, args):
            calls.generation.append(args)
            return cls()

        def shutdown(self):
            calls.shutdown.append("generation")

    class Engine:
        def __init__(self, **kwargs):
            calls.reasoner.append(kwargs)
            self.tokenizer_manager = object()
            self.template_manager = object()

        def shutdown(self):
            calls.shutdown.append("reasoner")

    def create_app(args):
        calls.apps.append(args)
        return FastAPI()

    class AsyncSchedulerClient:
        def initialize(self, server_args, *, worker_failure=None):
            pass

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
        "sglang.multimodal_gen.runtime.entrypoints.http_server", create_app=create_app
    )
    module(
        "sglang.multimodal_gen.runtime.scheduler_client",
        AsyncSchedulerClient=AsyncSchedulerClient,
    )
    module("sglang.srt.entrypoints.openai.protocol", ChatCompletionRequest=object)
    module(
        "sglang.srt.entrypoints.openai.serving_chat",
        OpenAIServingChat=lambda *args: object(),
    )
    calls.modules = installed
    return calls
