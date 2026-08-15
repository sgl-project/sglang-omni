# SPDX-License-Identifier: Apache-2.0
"""Stage factories for native SenseNova U1 serving paths."""

from __future__ import annotations

from typing import Any


def create_sensenova_u1_flow_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    vendor_root: str | None = None,
    attn_backend: str = "auto",
    max_concurrency: int = 1,
) -> Any:
    from sglang_omni.models.sensenova_u1.flow_matching import (
        SenseNovaU1FlowMatchingRunner,
    )
    from sglang_omni.models.sensenova_u1.sglang_model import (
        assert_no_hf_modeling_imported,
        block_hf_modeling_imports,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    assert_no_hf_modeling_imported(context="before native U1 flow stage factory")
    with block_hf_modeling_imports():
        runner = SenseNovaU1FlowMatchingRunner(
            model_path=model_path,
            vendor_root=vendor_root,
            device=device,
            dtype=dtype,
            attn_backend=attn_backend,
        )
    assert_no_hf_modeling_imported(context="after native U1 flow stage factory")
    return SimpleScheduler(
        runner.complete_payload,
        max_concurrency=max_concurrency,
    )


def create_sensenova_u1_interleave_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    vendor_root: str | None = None,
    attn_backend: str = "auto",
    max_concurrency: int = 1,
) -> Any:
    from sglang_omni.models.sensenova_u1.interleave import (
        SenseNovaU1InterleaveRunner,
    )
    from sglang_omni.models.sensenova_u1.sglang_model import (
        assert_no_hf_modeling_imported,
        block_hf_modeling_imports,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    assert_no_hf_modeling_imported(
        context="before native U1 interleave stage factory"
    )
    with block_hf_modeling_imports():
        runner = SenseNovaU1InterleaveRunner(
            model_path=model_path,
            vendor_root=vendor_root,
            device=device,
            dtype=dtype,
            attn_backend=attn_backend,
        )
    assert_no_hf_modeling_imported(
        context="after native U1 interleave stage factory"
    )
    return SimpleScheduler(
        runner.complete_payload,
        max_concurrency=max_concurrency,
    )


def create_sensenova_u1_native_executor(
    model_path: str,
    *,
    device: str = "cpu",
    dtype: str = "bfloat16",
    load_weights: bool = False,
) -> Any:
    from sglang_omni.models.sensenova_u1.sglang_model import (
        SenseNovaU1NativeLoadExecutor,
        assert_no_hf_modeling_imported,
        block_hf_modeling_imports,
    )

    assert_no_hf_modeling_imported(context="before native U1 stage factory")
    with block_hf_modeling_imports():
        executor = SenseNovaU1NativeLoadExecutor(
            model_path=model_path,
            device=device,
            dtype=dtype,
            load_weights=load_weights,
        )
    assert_no_hf_modeling_imported(context="after native U1 stage factory")
    return executor


def create_sensenova_u1_native_serving_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    attention_backend: str = "triton",
    mem_fraction_static: float = 0.65,
    max_total_tokens: int = 4096,
    max_running_requests: int = 2,
    max_concurrency: int = 4,
    max_batch_wait_ms: int = 10,
    enable_cuda_graph: bool = True,
    cuda_graph_bs: list[int] | None = None,
) -> Any:
    from sglang_omni.models.sensenova_u1.native_serving import (
        SenseNovaU1NativeServingExecutor,
    )
    from sglang_omni.models.sensenova_u1.sglang_model import (
        assert_no_hf_modeling_imported,
        block_hf_modeling_imports,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    assert_no_hf_modeling_imported(context="before native U1 serving stage factory")
    with block_hf_modeling_imports():
        executor = SenseNovaU1NativeServingExecutor(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
            mem_fraction_static=mem_fraction_static,
            max_total_tokens=max_total_tokens,
            max_running_requests=max(max_running_requests, max_concurrency),
            enable_radix_cache=True,
            disable_cuda_graph=not enable_cuda_graph,
            cuda_graph_bs=cuda_graph_bs or [1, 8, 16],
        )
    assert_no_hf_modeling_imported(context="after native U1 serving stage factory")
    return SimpleScheduler(
        executor.complete_payload,
        batch_compute_fn=executor.complete_payload_batch,
        max_batch_size=max_concurrency,
        max_batch_wait_ms=max_batch_wait_ms,
        max_concurrency=1,
        shutdown_callback=lambda: None,
    )


__all__ = [
    "create_sensenova_u1_flow_executor",
    "create_sensenova_u1_interleave_executor",
    "create_sensenova_u1_native_executor",
    "create_sensenova_u1_native_serving_executor",
]
