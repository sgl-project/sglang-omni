# SPDX-License-Identifier: Apache-2.0
"""Nsight-Python targets for MOSS-TTS-Realtime local frame graphs."""

from __future__ import annotations

import torch

from benchmarks.eval.benchmark_moss_tts_realtime_local_attention import (
    _frame_callable,
    _new_module,
)


def _capture(
    backend: str, batch_size: int
) -> tuple[torch.cuda.CUDAGraph, object, object]:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(100 + batch_size)
    hidden_states = torch.randn(
        batch_size,
        2048,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    codes = torch.randint(
        0,
        1024,
        (batch_size, 15),
        device=device,
        generator=generator,
    )
    module = _new_module(
        backend,
        device=device,
        dtype=dtype,
        seed=7,
    )
    module.ensure_kv_cache(batch_size, device, dtype)
    run_frame = _frame_callable(module, hidden_states, codes)
    for _ in range(3):
        run_frame()
    module.freeze_kv_cache()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_frame()
    state = (module, hidden_states, codes)
    return graph, state, run_frame


_GRAPHS = {
    ("sdpa", 1): _capture("sdpa", 1),
    ("fa3", 1): _capture("fa3", 1),
    ("sdpa", 16): _capture("sdpa", 16),
    ("fa3", 16): _capture("fa3", 16),
}


def sdpa_batch1() -> None:
    _GRAPHS[("sdpa", 1)][0].replay()


def fa3_batch1() -> None:
    _GRAPHS[("fa3", 1)][0].replay()


def sdpa_batch16() -> None:
    _GRAPHS[("sdpa", 16)][0].replay()


def fa3_batch16() -> None:
    _GRAPHS[("fa3", 16)][0].replay()


def sdpa_batch1_eager() -> None:
    _GRAPHS[("sdpa", 1)][2]()


def fa3_batch1_eager() -> None:
    _GRAPHS[("fa3", 1)][2]()


def sdpa_batch16_eager() -> None:
    _GRAPHS[("sdpa", 16)][2]()


def fa3_batch16_eager() -> None:
    _GRAPHS[("fa3", 16)][2]()
