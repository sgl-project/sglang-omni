# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

from sglang_omni.models.llada2_uni import stages


def test_thinker_factory_forwards_tensor_parallel_runtime(
    monkeypatch,
) -> None:
    captured = {}
    server_args = SimpleNamespace(
        dllm_algorithm="LowConfidence",
        mem_fraction_static=0.6,
    )

    def fake_build_server_args(model_path, **kwargs):
        captured["model_path"] = model_path
        captured["server_args_kwargs"] = kwargs
        return server_args

    def fake_create_scheduler(value, gpu_id, **kwargs):
        captured["server_args"] = value
        captured["gpu_id"] = gpu_id
        captured["scheduler_kwargs"] = kwargs
        return "scheduler"

    monkeypatch.setattr(
        "sglang_omni.scheduling.sglang_backend.build_sglang_server_args",
        fake_build_server_args,
    )
    monkeypatch.setattr(
        "sglang_omni.models.llada2_uni.bootstrap.create_dllm_thinker_scheduler",
        fake_create_scheduler,
    )

    result = stages.create_sglang_dllm_thinker_executor_from_config(
        "/models/llada2",
        gpu_id=0,
        tp_rank=1,
        tp_size=2,
        nccl_port=23456,
        server_args_overrides={"max_running_requests": 4},
    )

    assert result == "scheduler"
    assert captured["model_path"] == "/models/llada2"
    assert captured["server_args_kwargs"]["tp_size"] == 2
    assert captured["server_args_kwargs"]["max_running_requests"] == 4
    assert captured["server_args"] is server_args
    assert captured["gpu_id"] == 0
    assert captured["scheduler_kwargs"] == {
        "tp_rank": 1,
        "nccl_port": 23456,
    }


def test_dllm_scheduler_requires_tp_work_fanout() -> None:
    from sglang_omni.scheduling.dllm_scheduler import DllmScheduler

    scheduler = DllmScheduler(
        tp_worker=object(),
        tree_cache=object(),
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        server_args=SimpleNamespace(
            chunked_prefill_size=32,
            max_running_requests=2,
        ),
        model_config=object(),
        dllm_config=SimpleNamespace(
            block_size=32,
            max_running_requests=2,
        ),
        request_builder=lambda value: value,
        result_adapter=lambda value: value,
    )

    assert scheduler.requires_tp_work_fanout is True
