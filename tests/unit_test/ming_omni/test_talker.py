# SPDX-License-Identifier: Apache-2.0
"""Speech/talker terminal tests."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import ModuleType

import numpy as np
import pytest


def test_ming_talker_add_request_propagates_generation_errors(
    monkeypatch,
) -> None:
    module_name = "sglang_omni.models.ming_omni.components.talker_executor"
    parent_name = "sglang_omni.models.ming_omni.components"
    sys.modules.pop(module_name, None)

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.Tensor = object

    def no_grad():
        def decorator(fn):
            return fn

        return decorator

    fake_torch.no_grad = no_grad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    try:
        module = importlib.import_module(module_name)
        from sglang_omni.proto import OmniRequest, StagePayload

        executor = module.MingTalkerExecutor(model_path="/fake/model/path")
        payload = StagePayload(
            request_id="req-1",
            request=OmniRequest(inputs="hello"),
            data={},
        )
        injected = RuntimeError("talker failed")

        monkeypatch.setattr(executor, "_extract_text", lambda _payload: "hello")

        def raise_error(_text):
            raise injected

        monkeypatch.setattr(executor, "_generate_speech", raise_error)

        with pytest.raises(RuntimeError, match="talker failed"):
            asyncio.run(executor.add_request(payload))
    finally:
        sys.modules.pop(module_name, None)
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, "talker_executor"):
            delattr(parent, "talker_executor")


def test_ming_talker_generation_failures_are_not_empty_successes(monkeypatch) -> None:
    module_name = "sglang_omni.models.ming_omni.components.talker_executor"
    parent_name = "sglang_omni.models.ming_omni.components"
    sys.modules.pop(module_name, None)

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.Tensor = object

    def no_grad():
        def decorator(fn):
            return fn

        return decorator

    fake_torch.no_grad = no_grad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    try:
        module = importlib.import_module(module_name)

        executor = module.MingTalkerExecutor(model_path="/fake/model/path")
        executor._talker = object()
        with pytest.raises(RuntimeError, match="no supported generation method"):
            executor._generate_speech("hello")

        class EmptyTalker:
            def omni_audio_generation(self, **_kwargs):
                yield None, None, None, None

        executor._talker = EmptyTalker()
        executor._vae = object()
        with pytest.raises(RuntimeError, match="produced no audio"):
            executor._generate_speech("hello")
    finally:
        sys.modules.pop(module_name, None)
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, "talker_executor"):
            delattr(parent, "talker_executor")


def test_ming_talker_skips_text_only_requests(monkeypatch) -> None:
    module_name = "sglang_omni.models.ming_omni.components.talker_executor"
    parent_name = "sglang_omni.models.ming_omni.components"
    sys.modules.pop(module_name, None)

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.Tensor = object

    def no_grad():
        def decorator(fn):
            return fn

        return decorator

    fake_torch.no_grad = no_grad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    try:
        module = importlib.import_module(module_name)
        from sglang_omni.proto import OmniRequest, StagePayload

        executor = module.MingTalkerExecutor(model_path="/fake/model/path")
        payload = StagePayload(
            request_id="req-text",
            request=OmniRequest(
                inputs="hello",
                metadata={"output_modalities": ["text"]},
            ),
            data={"thinker_out": {"output_ids": [1, 2, 3]}},
        )
        monkeypatch.setattr(
            executor,
            "_extract_text",
            lambda _payload: pytest.fail("text-only request should not decode text"),
        )
        monkeypatch.setattr(
            executor,
            "_generate_speech",
            lambda _text: pytest.fail("text-only request should not generate speech"),
        )

        async def run_request():
            await executor.add_request(payload)
            return await executor.get_result()

        result = asyncio.run(run_request())

        assert result.request_id == "req-text"
        assert result.data["audio_waveform"] is None
        assert result.data["skipped"] is True
    finally:
        sys.modules.pop(module_name, None)
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, "talker_executor"):
            delattr(parent, "talker_executor")


def _waveform_payload(values: np.ndarray) -> dict:
    return {
        "audio_waveform": values.tobytes(),
        "audio_waveform_dtype": str(values.dtype),
        "audio_waveform_shape": list(values.shape),
        "sample_rate": 44100,
    }


def test_default_result_builder_merges_decode_and_talker_audio() -> None:
    from sglang_omni.client.client import Client

    waveform = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    chunk = Client._default_result_builder(
        "req-1",
        {
            "decode": {
                "text": "hi",
                "finish_reason": "length",
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            "talker": _waveform_payload(waveform),
        },
    )

    assert chunk.text == "hi"
    assert chunk.finish_reason == "length"
    assert chunk.modality == "audio"
    assert chunk.sample_rate == 44100
    np.testing.assert_array_equal(chunk.audio_data, waveform)
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 2
    assert chunk.usage.completion_tokens == 1
    assert chunk.usage.total_tokens == 3


def test_default_result_builder_keeps_text_modality_for_skipped_talker() -> None:
    from sglang_omni.client.client import Client

    chunk = Client._default_result_builder(
        "req-text",
        {
            "decode": {"text": "hello"},
            "talker": {
                "audio_waveform": None,
                "sample_rate": 44100,
                "duration": 0.0,
                "skipped": True,
            },
        },
    )

    assert chunk.text == "hello"
    assert chunk.modality == "text"
    assert chunk.audio_data is None


def test_default_result_builder_still_merges_decode_and_code2wav_audio() -> None:
    from sglang_omni.client.client import Client

    waveform = np.array([1.0, 0.5], dtype=np.float32)

    chunk = Client._default_result_builder(
        "req-2",
        {
            "decode": {"text": "hello"},
            "code2wav": {
                **_waveform_payload(waveform),
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            },
        },
    )

    assert chunk.text == "hello"
    assert chunk.modality == "audio"
    np.testing.assert_array_equal(chunk.audio_data, waveform)
    assert chunk.usage is not None
    assert chunk.usage.prompt_tokens == 5
    assert chunk.usage.completion_tokens == 7
    assert chunk.usage.total_tokens == 12


def test_ming_talker_audio_result_includes_modality_and_text_usage(monkeypatch) -> None:
    module_name = "sglang_omni.models.ming_omni.components.talker_executor"
    parent_name = "sglang_omni.models.ming_omni.components"
    sys.modules.pop(module_name, None)

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.Tensor = object

    def no_grad():
        def decorator(fn):
            return fn

        return decorator

    fake_torch.no_grad = no_grad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    try:
        module = importlib.import_module(module_name)
        from sglang_omni.proto import OmniRequest, StagePayload

        class TensorLike:
            def numel(self) -> int:
                return 4

        class FakeWaveform:
            def cpu(self):
                return self

            def float(self):
                return self

            def numpy(self):
                return np.array([0.1, -0.1], dtype=np.float32)

        executor = module.MingTalkerExecutor(model_path="/fake/model/path")
        payload = StagePayload(
            request_id="req-audio",
            request=OmniRequest(inputs=[]),
            data={
                "prompt": {"input_ids": TensorLike()},
                "thinker_out": {"output_ids": [1, 2, 3]},
            },
        )
        monkeypatch.setattr(executor, "_extract_text", lambda _payload: "hello")
        monkeypatch.setattr(
            executor,
            "_generate_speech",
            lambda _text: (FakeWaveform(), 44100, 0.2),
        )

        async def run_request():
            await executor.add_request(payload)
            return await executor.get_result()

        result = asyncio.run(run_request())

        assert result.data["modality"] == "audio"
        assert result.data["usage"] == {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
        }
    finally:
        sys.modules.pop(module_name, None)
        parent = sys.modules.get(parent_name)
        if parent is not None and hasattr(parent, "talker_executor"):
            delattr(parent, "talker_executor")
