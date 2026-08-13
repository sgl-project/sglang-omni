# SPDX-License-Identifier: Apache-2.0
"""Stage factories for SenseNova U1 fallback paths."""

from __future__ import annotations

from typing import Any


def create_sensenova_u1_vqa_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    vendor_root: str | None = None,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    attn_backend: str = "auto",
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    max_concurrency: int = 1,
    load_with_info: bool = False,
) -> Any:
    from sglang_omni.models.sensenova_u1.hf_runner import (
        SenseNovaU1UnderstandingRunner,
    )
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    runner = SenseNovaU1UnderstandingRunner(
        model_path=model_path,
        vendor_root=vendor_root,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        load_with_info=load_with_info,
    )

    def _complete(payload):
        return runner.complete_payload(
            payload,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    return SimpleScheduler(_complete, max_concurrency=max_concurrency)


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
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    runner = SenseNovaU1FlowMatchingRunner(
        model_path=model_path,
        vendor_root=vendor_root,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
    )
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
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    runner = SenseNovaU1InterleaveRunner(
        model_path=model_path,
        vendor_root=vendor_root,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
    )
    return SimpleScheduler(
        runner.complete_payload,
        max_concurrency=max_concurrency,
    )


__all__ = [
    "create_sensenova_u1_flow_executor",
    "create_sensenova_u1_interleave_executor",
    "create_sensenova_u1_vqa_executor",
]
