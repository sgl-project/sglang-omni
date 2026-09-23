# SPDX-License-Identifier: Apache-2.0
"""Image decoder stage configuration and runtime ownership."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sglang_omni.models.llada2_uni.config import LLaDA2ImageDecoderStageConfig, Variants


@pytest.mark.parametrize("variant", [name for name in Variants if name != "text"])
def test_decoder_config_roundtrip(variant):
    config_cls = Variants[variant]
    original = config_cls(model_path="unused")
    assert (
        config_cls.stage_config_types["image_decode"] is LLaDA2ImageDecoderStageConfig
    )
    assert isinstance(
        original.stage_named("image_decode"), LLaDA2ImageDecoderStageConfig
    )
    data = original.model_dump()
    decoder = next(stage for stage in data["stages"] if stage["name"] == "image_decode")
    assert decoder["process"] == "image_decode"
    decoder["factory"].update(backend="sglang", attention_backend="torch_sdpa")

    rebuilt = config_cls.model_validate(data)
    stage = next(stage for stage in rebuilt.stages if stage.name == "image_decode")
    assert isinstance(stage, LLaDA2ImageDecoderStageConfig)
    assert stage.factory.attention_backend == "torch_sdpa"
    assert stage.factory.backend == "sglang"


@pytest.mark.parametrize("backend,degrees", [("sglang", (1, 1)), ("diffusers", (2, 1))])
def test_sp_decoder_rejects_incompatible_configuration(backend, degrees):
    with pytest.raises(ValueError):
        LLaDA2ImageDecoderStageConfig(
            name="image_decode",
            factory_path="pkg.create",
            gpu=[0, 1],
            sp_size=2,
            factory={
                "backend": backend,
                "ulysses_degree": degrees[0],
                "ring_degree": degrees[1],
            },
        )


@pytest.mark.parametrize("variant", [name for name in Variants if name != "text"])
def test_sp_decoder_configuration_roundtrip(variant):
    config_cls = Variants[variant]
    config = config_cls(model_path="unused").model_dump()
    decoder = next(
        stage for stage in config["stages"] if stage["name"] == "image_decode"
    )
    decoder.update(sp_size=2, gpu=[0, 1])
    decoder["factory"].update(backend="sglang", ulysses_degree=2)
    rebuilt = config_cls.model_validate(config)
    stage = next(stage for stage in rebuilt.stages if stage.name == "image_decode")
    assert stage.sp_size == 2 and stage.tp_size == 1
    assert stage.factory.ulysses_degree == 2 and stage.gpu == [0, 1]
    if variant == "interleaved":
        assert stage.factory.interleaved_nonterminal
        assert not stage.terminal and stage.next == "interleaved_collect"


def test_ring_rejects_sdpa_without_changing_backend():
    with pytest.raises(ValueError, match="ring parallelism requires"):
        LLaDA2ImageDecoderStageConfig(
            name="image_decode",
            factory_path="pkg.create",
            gpu=[0, 1],
            sp_size=2,
            factory={"backend": "sglang", "ring_degree": 2},
        )


@pytest.mark.parametrize("sp_rank,sp_size", [(0, 1), (0, 2), (1, 2)])
@pytest.mark.parametrize("interleaved", [False, True])
def test_sglang_decoder_factory_owns_runtime(
    monkeypatch, sp_rank, sp_size, interleaved
):
    from sglang_omni.models.llada2_uni import merge, stages
    from sglang_omni.models.llada2_uni.components import decoder_runtime, image_decoder
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
    from sglang_omni.utils import device

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
            return Image.new("RGB", (8, 8)) if sp_rank == 0 else None

    monkeypatch.setattr(decoder_runtime, "initialize_decoder_runtime", initialize)
    monkeypatch.setattr(image_decoder, "LLaDA2ImageDecoder", Decoder)
    monkeypatch.setattr(
        device, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
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
        sp_rank=sp_rank,
        sp_size=sp_size,
        stage_role=(
            "single" if sp_size == 1 else ("leader" if sp_rank == 0 else "follower")
        ),
        nccl_port=23456 if sp_size > 1 else None,
        ulysses_degree=sp_size,
        interleaved_nonterminal=interleaved,
    )
    assert scheduler.allow_multiple_inflight_per_request == interleaved
    payload = SimpleNamespace(
        request_id="r",
        data=(
            {
                "task_kind": "interleaved",
                "stream_state": {"interleaved": {"frame_index": 2}},
            }
            if interleaved
            else {}
        ),
    )
    original = payload.data.copy()
    result = scheduler._fn(payload)
    if sp_rank == 0:
        assert result is payload
        if interleaved:
            assert payload.data["kind"] == "interleaved_frame"
            assert payload.data["frame"]["index"] == 2
            assert payload.data["frame"]["image"]["data"]
        else:
            assert payload.data["format"] == "png" and payload.data["image"]
    else:
        assert result is None and payload.data == original
    assert settings == {
        "gpu_id": None,
        "dtype": settings["dtype"],
        "attention_backend": "torch_sdpa",
        "sp_rank": sp_rank,
        "sp_size": sp_size,
        "stage_role": (
            "single" if sp_size == 1 else ("leader" if sp_rank == 0 else "follower")
        ),
        "nccl_port": 23456 if sp_size > 1 else None,
        "ulysses_degree": sp_size,
        "ring_degree": 1,
    }
    assert events == ["enter", "exit", "enter", "exit"]

    monkeypatch.setattr(SimpleScheduler, "start", lambda self: events.append("done"))
    scheduler.start()
    assert events[-2:] == ["done", "close"]
