# SPDX-License-Identifier: Apache-2.0
"""CPU/Gloo numerical checks; no checkpoint or accelerator is required."""

import asyncio
import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from sglang_omni.models.sensenova_u1.cancellation import (
    GenerationScope,
    SenseNovaCancelledError,
    generation_checkpoint,
)
from sglang_omni.models.sensenova_u1.tensor_parallel import (
    SenseNovaTensorParallelScheduler,
    apply_tensor_parallel,
    initialize_tp,
    synchronize_tp_noise,
    validate_tp_settings,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fixtures.pipeline_fakes import make_stage_payload
from tests.unit_test.pipeline.helpers import make_stage


def make_model():
    config = SimpleNamespace(
        model_type="qwen3",
        hidden_size=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        intermediate_size=32,
    )
    model = nn.Module()
    model.language_model = nn.Module()
    backbone = model.language_model.model = nn.Module()
    backbone.config = config
    backbone.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        attention = layer.self_attn = nn.Module()
        attention.num_key_value_groups = 2
        for suffix in ("", "_mot_gen"):
            for name, width in (("q_proj", 64), ("k_proj", 32), ("v_proj", 32)):
                setattr(attention, name + suffix, nn.Linear(16, width, bias=True))
            setattr(attention, "o_proj" + suffix, nn.Linear(64, 16, bias=True))
        for name in ("mlp", "mlp_mot_gen"):
            mlp = nn.Module()
            mlp.intermediate_size = 32
            mlp.gate_proj = nn.Linear(16, 32, bias=False)
            mlp.up_proj = nn.Linear(16, 32, bias=False)
            mlp.down_proj = nn.Linear(32, 16, bias=False)
            setattr(layer, name, mlp)
        backbone.layers.append(layer)
    model.pixel_head = nn.Linear(16, 3)
    return model.eval()


def forward(model, inputs, suffix, caches=None, update_cache=True):
    """Exercise GQA, prefix KV and repeated image denoising with local heads."""
    next_caches = []
    for index, layer in enumerate(model.language_model.model.layers):
        attention = layer.self_attn
        batch, sequence = inputs.shape[:2]

        def project(
            name, attention=attention, inputs=inputs, batch=batch, sequence=sequence
        ):
            return (
                getattr(attention, name + suffix)(inputs)
                .view(batch, sequence, -1, 4)
                .transpose(1, 2)
            )

        query, key, value = (project(name) for name in ("q_proj", "k_proj", "v_proj"))
        if caches is not None:
            key = torch.cat([caches[index][0], key], dim=2)
            value = torch.cat([caches[index][1], value], dim=2)
        else:
            pass
        next_caches.append((key, value) if update_cache else caches[index])
        output = F.scaled_dot_product_attention(query, key, value, enable_gqa=True)
        output = output.transpose(1, 2).reshape(batch, sequence, -1)
        inputs = inputs + getattr(attention, "o_proj" + suffix)(output)
        mlp = layer.mlp_mot_gen if suffix else layer.mlp
        inputs = inputs + mlp.down_proj(
            F.silu(mlp.gate_proj(inputs)) * mlp.up_proj(inputs)
        )
    return model.pixel_head(inputs), next_caches


def distributed_parity(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(17)
        reference = make_model()
        sharded = copy.deepcopy(reference)
        apply_tensor_parallel(sharded, rank, size, dist.group.WORLD)
        assert (
            sharded.language_model.model.layers[0].mlp.intermediate_size == 32 // size
        )
        torch.testing.assert_close(
            sharded.pixel_head.weight, reference.pixel_head.weight
        )
        with torch.inference_mode():
            for batch in (1, 2):
                prefix = torch.randn(batch, 5, 16)
                expected, full_cache = forward(reference, prefix, "")
                actual, local_cache = forward(sharded, prefix, "")
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                assert local_cache[0][0].shape[1] == 8 // size
                for _ in range(3):
                    image = torch.randn(batch, 3, 16)
                    expected, _ = forward(
                        reference, image, "_mot_gen", full_cache, False
                    )
                    actual, _ = forward(sharded, image, "_mot_gen", local_cache, False)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            noise = torch.full((2, 3, 4, 4), float(rank))
            synchronize_tp_noise(sharded, noise)
            torch.testing.assert_close(noise, torch.zeros_like(noise))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4, 8])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_distributed_attention_mlp_and_prefix_cache_parity(tmp_path, size):
    mp.spawn(
        distributed_parity,
        args=(size, (tmp_path / "rendezvous").as_uri()),
        nprocs=size,
        join=True,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "qwen3_moe"),
        ("num_key_value_heads", 1),
        ("intermediate_size", 31),
    ],
)
def test_invalid_backbone_is_rejected_before_mutation(field, value):
    model = make_model()
    original = model.language_model.model.layers[0].self_attn.q_proj
    setattr(model.language_model.model.config, field, value)
    with pytest.raises(ValueError):
        apply_tensor_parallel(model, 0, 2, None)
    assert model.language_model.model.layers[0].self_attn.q_proj is original


def test_tp_one_is_a_noop_and_double_sharding_is_rejected():
    model = make_model()
    original = model.language_model.model.layers[0].self_attn.q_proj
    apply_tensor_parallel(model, 0, 1, None)
    assert model.language_model.model.layers[0].self_attn.q_proj is original
    apply_tensor_parallel(model, 0, 2, None)
    with pytest.raises(ValueError, match="already been applied"):
        apply_tensor_parallel(model, 0, 2, None)


def test_sharding_copies_checkpoint_slices_without_random_init(monkeypatch):
    """Sharding randomly initialized every projection on the host before discarding it."""
    model = make_model()
    # Holding the originals keeps their storage addresses from being reused by shards.
    original_parameters = list(model.language_model.parameters())
    original_storages = {
        parameter.untyped_storage().data_ptr() for parameter in original_parameters
    }
    initialized_device_types: list[str] = []
    kaiming_uniform = nn.init.kaiming_uniform_

    def record_kaiming_uniform(tensor, *args, **kwargs):
        initialized_device_types.append(tensor.device.type)
        return kaiming_uniform(tensor, *args, **kwargs)

    monkeypatch.setattr(nn.init, "kaiming_uniform_", record_kaiming_uniform)
    apply_tensor_parallel(model, 1, 2, None)

    assert initialized_device_types
    assert set(initialized_device_types) == {"meta"}
    sharded_parameters = list(model.language_model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in sharded_parameters)
    assert all(parameter.is_contiguous() for parameter in sharded_parameters)
    assert original_storages.isdisjoint(
        parameter.untyped_storage().data_ptr() for parameter in sharded_parameters
    )


@pytest.mark.parametrize(
    "rank,size,batch", [(0, 3, 1), (-1, 2, 1), (2, 2, 1), (0, 2, 2)]
)
def test_invalid_launch_settings(rank, size, batch):
    with pytest.raises(ValueError):
        validate_tp_settings(rank, size, batch)


def test_tp_startup_checks_device_and_port():
    with pytest.raises(ValueError, match="CUDA or Ascend"):
        initialize_tp(torch.device("cpu"), 0, 2, 29500)
    with pytest.raises(ValueError, match="nccl_port"):
        initialize_tp(torch.device("cuda:0"), 0, 2, None)


def test_single_rank_noise_does_not_require_distributed():
    noise = torch.randn(1, 3, 4, 4)
    expected = noise.clone()
    synchronize_tp_noise(nn.Module(), noise)
    torch.testing.assert_close(noise, expected)


@pytest.mark.parametrize("device_type,backend", [("cuda", "nccl"), ("npu", "hccl")])
def test_group_initialization_uses_local_device_and_launcher_port(
    monkeypatch, device_type, backend
):
    calls = {}
    resolved_device = SimpleNamespace(type=device_type)
    monkeypatch.setattr(
        torch,
        device_type,
        SimpleNamespace(set_device=lambda value: calls.update(device=value)),
        raising=False,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        dist, "init_process_group", lambda **kwargs: calls.update(kwargs)
    )
    world_group = object()
    monkeypatch.setattr(dist, "group", SimpleNamespace(WORLD=world_group))
    assert initialize_tp(resolved_device, 1, 2, 29991) is world_group
    assert calls["device"] is resolved_device
    assert calls["backend"] == backend
    assert calls["init_method"] == "tcp://127.0.0.1:29991"
    assert calls["rank"] == 1
    assert calls["world_size"] == 2


def cancel_and_resume_generation(rank: int, rendezvous: str) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    execution_loop = asyncio.new_event_loop()
    try:

        def compute(payload: StagePayload) -> StagePayload:
            with GenerationScope(
                [payload],
                scheduler.is_aborted,
                torch.device("cpu"),
                torch.distributed.group.WORLD,
            ):
                generation_checkpoint()
                total = torch.ones((), dtype=torch.int32)
                torch.distributed.all_reduce(total)
                if rank == 0 and payload.request_id == "cancelled":
                    scheduler.abort(payload.request_id)
                else:
                    pass
                payload.data = int(total.item())
                generation_checkpoint()
                return payload

        scheduler = SenseNovaTensorParallelScheduler(
            compute, report_cancellation=rank == 0, shutdown_join_timeout_s=20.0
        )
        cancelled = StagePayload("cancelled", OmniRequest("prompt"), None)
        scheduler.run_single(
            IncomingMessage("cancelled", "new_request", cancelled), execution_loop
        )
        if rank == 0:
            failure = scheduler.outbox.get_nowait()
            assert failure.type == "error"
            assert isinstance(failure.data, SenseNovaCancelledError)
        else:
            assert scheduler.outbox.empty()
        assert scheduler.compute_error is None
        subsequent = StagePayload("next", OmniRequest("prompt"), None)
        scheduler.run_single(
            IncomingMessage("next", "new_request", subsequent), execution_loop
        )
        result = scheduler.outbox.get_nowait()
        assert result.type == "result"
        assert result.data.data == 2
    finally:
        execution_loop.close()
        torch.distributed.destroy_process_group()


def test_asymmetric_cancellation_leaves_tp_group_reusable(tmp_path: Path) -> None:
    rendezvous = (tmp_path / "cancellation-rendezvous").as_uri()
    torch.multiprocessing.spawn(
        cancel_and_resume_generation, args=(rendezvous,), nprocs=2, join=True
    )


def test_follower_stage_survives_cooperative_cancellation() -> None:
    """A cancelled TP request stopped the follower stage, so the next request hung the TP group."""

    async def run() -> None:
        completed_request_ids: list[str] = []

        def compute(payload: StagePayload) -> StagePayload:
            with GenerationScope(
                [payload], scheduler.is_aborted, torch.device("cpu"), None
            ):
                generation_checkpoint()
                completed_request_ids.append(payload.request_id)
                return payload

        scheduler = SenseNovaTensorParallelScheduler(
            compute, report_cancellation=False, shutdown_join_timeout_s=5.0
        )
        follower = make_stage(role="follower", scheduler=scheduler)
        follower.on_abort("cancelled")
        await follower.execute(make_stage_payload(request_id="cancelled"))
        await follower.execute(make_stage_payload(request_id="next"))
        scheduler.stop()
        await asyncio.to_thread(scheduler.start)
        follower.end_outbox_drain()
        await follower.drain_outbox()

        assert completed_request_ids == ["next"]
        assert scheduler.compute_error is None
        assert scheduler.outbox.empty()

    asyncio.run(run())
