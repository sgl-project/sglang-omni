# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.cosmos3 import stages
from sglang_omni.models.cosmos3.payload_types import Cosmos3PipelineState
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache


def test_preprocessing_factory_returns_simple_scheduler(monkeypatch) -> None:
    sentinel = object()
    calls: list[tuple[str, int | None, bool]] = []

    def fake_preprocessor(
        model_path: str,
        max_seq_len: int | None,
        enable_vision: bool,
    ):
        calls.append((model_path, max_seq_len, enable_vision))
        return sentinel

    monkeypatch.setattr(stages, "Cosmos3TextPreprocessor", fake_preprocessor)

    scheduler = stages.create_preprocessing_executor(
        "nvidia/Cosmos3-Nano",
        thinker_max_seq_len=8192,
    )

    assert isinstance(scheduler, SimpleScheduler)
    assert scheduler._fn is sentinel
    assert calls == [("nvidia/Cosmos3-Nano", 8192, True)]


def test_text_factory_passes_through_tensor_parallelism(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    server_args = SimpleNamespace()
    scheduler = object()

    def fake_build(model_path: str, context_length: int, **kwargs):
        captured.update(
            model_path=model_path,
            context_length=context_length,
            kwargs=kwargs,
        )
        return server_args

    def fake_create(received_server_args, gpu_id: int, **kwargs):
        captured.update(
            server_args=received_server_args,
            gpu_id=gpu_id,
            scheduler_kwargs=kwargs,
        )
        return scheduler

    def fake_validate(**kwargs):
        captured["validation_kwargs"] = kwargs

    monkeypatch.setattr(stages, "build_sglang_server_args", fake_build)
    monkeypatch.setattr(stages, "create_thinker_scheduler", fake_create)
    monkeypatch.setattr(stages, "validate_generation_batch_policy", fake_validate)

    result = stages.create_sglang_text_executor_from_config(
        "nvidia/Cosmos3-Nano",
        thinker_max_seq_len=4096,
        gpu_id=2,
        tp_rank=1,
        tp_size=2,
        server_args_overrides={"max_running_requests": 8},
    )

    assert result is scheduler
    assert captured["model_path"] == "nvidia/Cosmos3-Nano"
    assert captured["context_length"] == 4096
    assert captured["kwargs"] == {
        "max_running_requests": 8,
        "cuda_graph_max_bs": 8,
        "torch_compile_max_bs": 8,
        "cuda_graph_bs": [1, 2, 4, 8],
        "disable_cuda_graph": False,
        "sampling_backend": "pytorch",
        "tp_size": 2,
    }
    assert captured["validation_kwargs"] == {
        "model_name": "Cosmos3 thinker",
        "server_args": server_args,
    }
    assert captured["server_args"] is server_args
    assert captured["gpu_id"] == 2
    assert captured["scheduler_kwargs"]["tp_rank"] == 1


class _FakeVisionModel:
    spatial_merge_size = 2
    out_hidden_size = 4
    deepstack_layers = 1
    visual_dtype_bytes = 4

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **inputs):
        self.calls += 1
        grid = inputs["image_grid_thw"]
        token_count = grid.prod(-1) // 4
        total = int(token_count.sum().item())
        embeds = torch.arange(total * 4, dtype=torch.float32).reshape(total, 4)
        return {
            "image_embeds": embeds,
            "image_grid_thw": grid,
            "image_token_counts": token_count,
            "deepstack_visual_embeds_image": [embeds + 1],
        }


def _vision_payload(request_id: str, cache_key: str) -> StagePayload:
    state = Cosmos3PipelineState(
        encoder_inputs={
            "vision_encoder": {
                "pixel_values": torch.ones((4, 6)),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
                "cache_key": cache_key,
            }
        }
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=None),
        data=state.to_dict(),
    )


def test_vision_encoder_batches_and_deduplicates_same_cache_key() -> None:
    model = _FakeVisionModel()
    cache = StageOutputCache(max_size=4, cache_device="cpu")
    outputs = stages._batch_vision_requests(
        [
            _vision_payload("first", "same-image"),
            _vision_payload("second", "same-image"),
        ],
        model=model,
        cache=cache,
    )

    assert model.calls == 1
    assert len(cache) == 1
    first = Cosmos3PipelineState.from_dict(outputs[0].data)
    second = Cosmos3PipelineState.from_dict(outputs[1].data)
    assert torch.equal(
        first.encoder_outs["vision_encoder"]["image_embeds"],
        second.encoder_outs["vision_encoder"]["image_embeds"],
    )
