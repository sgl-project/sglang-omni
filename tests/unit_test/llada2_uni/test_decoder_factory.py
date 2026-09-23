# SPDX-License-Identifier: Apache-2.0
"""Image decoder stage configuration and runtime ownership."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from sglang_omni.models.llada2_uni.config import LLaDA2UniOmniPipelineConfig


def test_decoder_config_roundtrip():
    original = LLaDA2UniOmniPipelineConfig(model_path="unused")
    data = original.model_dump()
    decoder = next(stage for stage in data["stages"] if stage["name"] == "image_decode")
    assert decoder["process"] == "image_decode"
    decoder["factory"].update(backend="sglang", attention_backend="torch_sdpa")

    rebuilt = LLaDA2UniOmniPipelineConfig.model_validate(data)
    stage = next(stage for stage in rebuilt.stages if stage.name == "image_decode")
    assert stage.factory.model_extra == {
        "attention_backend": "torch_sdpa",
        "backend": "sglang",
    }


def test_sglang_decoder_factory_owns_runtime(monkeypatch):
    from sglang_omni.models.llada2_uni import merge, stages
    from sglang_omni.models.llada2_uni.components import decoder_runtime, image_decoder
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    events: list[str] = []
    settings: dict[str, object] = {}

    class Runtime:
        @contextmanager
        def compute_context(self):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        def close(self):
            events.append("close")

    runtime = Runtime()

    def initialize(path, **kwargs):
        assert path == "unused"
        settings.update(kwargs)
        return runtime

    class Decoder:
        def __init__(self, **kwargs):
            assert events[-1] == "enter"
            assert kwargs["backend"] == "sglang"
            assert kwargs["runtime"] is runtime

        def decode(self, tokens, h, w, **kwargs):
            assert events[-1] == "enter"
            assert kwargs == {"decode_mode": "decoder-turbo", "num_steps": 8}
            assert tokens == [3, 4] and (h, w) == (1, 2)
            return Image.new("RGB", (8, 8))

    monkeypatch.setattr(decoder_runtime, "initialize_decoder_runtime", initialize)
    monkeypatch.setattr(image_decoder, "LLaDA2ImageDecoder", Decoder)
    monkeypatch.setattr(
        merge,
        "extract_image_vq_tokens",
        lambda state: ([3, 4], 1, 2, {"decode_mode": "decoder-turbo"}),
    )

    scheduler = stages.create_image_decode_executor(
        "unused",
        device="cpu",
        backend="sglang",
        attention_backend="torch_sdpa",
    )
    payload = SimpleNamespace(data={})
    assert scheduler._fn(payload) is payload
    assert payload.data["format"] == "png" and payload.data["image"]
    assert settings == {
        "gpu_id": None,
        "dtype": settings["dtype"],
        "attention_backend": "torch_sdpa",
    }
    assert events == ["enter", "exit", "enter", "exit"]

    monkeypatch.setattr(SimpleScheduler, "start", lambda self: events.append("done"))
    scheduler.start()
    assert events[-2:] == ["done", "close"]


def test_image_request_without_vq_tokens_fails(monkeypatch):
    from sglang_omni.models.llada2_uni import merge, stages
    from sglang_omni.models.llada2_uni.components import image_decoder

    monkeypatch.setattr(
        image_decoder,
        "LLaDA2ImageDecoder",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(merge, "extract_image_vq_tokens", lambda _state: None)

    scheduler = stages.create_image_decode_executor("unused", device="cpu")
    with pytest.raises(ValueError, match="did not produce image VQ tokens"):
        scheduler._fn(SimpleNamespace(data={"task_kind": "t2i"}))
