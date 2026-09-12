# SPDX-License-Identifier: Apache-2.0
"""The builder hands out a resolved ServerArgs.

A real config.json is needed because the resolution pipeline returns before
the cuda graph handler on a dummy model path. The device is pinned to cuda
the way upstream's own resolution tests pin it, so the record resolves the
same on an accelerator-less host.
"""

from __future__ import annotations

from pathlib import Path

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.model_executor.cuda_graph_config import Backend

from sglang_omni.scheduling.generation_batch_policy import (
    get_decode_cuda_graph_max_bs,
    get_prefill_cuda_graph_backend,
)
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)
from tests.unit_test.fixtures.mini_checkpoint import write_mini_llama_checkpoint


def test_builder_record_is_resolved_with_the_cuda_graph_config_declared(
    tmp_path: Path,
) -> None:
    server_args = build_sglang_server_args(
        write_mini_llama_checkpoint(tmp_path), context_length=2048, device="cuda"
    )

    assert server_args._resolution_finished is True
    assert server_args.cuda_graph_config is None
    cuda_graph_config = resolution_result(server_args, "cuda_graph_config")
    assert cuda_graph_config.prefill.backend == Backend.DISABLED


def test_accessors_read_the_declared_cuda_graph_config(tmp_path: Path) -> None:
    server_args = build_sglang_server_args(
        write_mini_llama_checkpoint(tmp_path),
        context_length=2048,
        device="cuda",
        cuda_graph_max_bs=8,
    )

    declared = resolution_result(server_args, "cuda_graph_config")

    assert get_prefill_cuda_graph_backend(server_args) == Backend.DISABLED
    assert declared.decode.max_bs is not None
    assert get_decode_cuda_graph_max_bs(server_args) == declared.decode.max_bs


def test_encoder_mem_reserve_reads_the_declared_fraction(tmp_path: Path) -> None:
    server_args = build_sglang_server_args(
        write_mini_llama_checkpoint(tmp_path), context_length=2048, device="cuda"
    )
    declared = resolution_result(server_args, "mem_fraction_static")

    assert server_args.mem_fraction_static is None
    assert declared is not None

    apply_encoder_mem_reserve(server_args, 0.1)

    assert resolution_result(server_args, "mem_fraction_static") == round(
        declared - 0.1, 3
    )
