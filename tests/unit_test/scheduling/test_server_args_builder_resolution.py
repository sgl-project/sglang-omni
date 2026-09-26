# SPDX-License-Identifier: Apache-2.0
"""The builder hands out a resolved ServerArgs.

A real config.json is needed because the resolution pipeline returns before
the cuda graph handler on a dummy model path. The device is pinned to cuda
the way upstream's own resolution tests pin it, so the record resolves the
same on an accelerator-less host.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
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
from sglang_omni.utils.gpu_compat import (
    apply_gpu_compat_env_defaults,
    apply_torch_compile_cache_env,
)
from tests.unit_test.fixtures.mini_checkpoint import write_mini_llama_checkpoint

DEFAULT_TORCHINDUCTOR_CACHE_DIRECTORY = str(
    Path.home() / ".cache" / "sglang-omni" / "torchinductor"
)


def test_builder_record_is_resolved_with_the_cuda_graph_config_declared(
    tmp_path: Path,
) -> None:
    server_args = build_sglang_server_args(
        write_mini_llama_checkpoint(tmp_path), context_length=2048, device="cuda"
    )

    assert (
        server_args._resolution_finished is True
    )  # noqa: leading-underscore  # upstream name
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


def test_builder_defaults_torch_compile_on(tmp_path: Path) -> None:
    server_args = build_sglang_server_args(
        write_mini_llama_checkpoint(tmp_path), context_length=2048, device="cuda"
    )

    assert resolution_result(server_args, "enable_torch_compile") is True


def test_builder_preserves_explicit_torch_compile_off(tmp_path: Path) -> None:
    server_args = build_sglang_server_args(
        write_mini_llama_checkpoint(tmp_path),
        context_length=2048,
        device="cuda",
        enable_torch_compile=False,
    )

    assert resolution_result(server_args, "enable_torch_compile") is False


def test_torchinductor_cache_directory_defaults_to_sglang_omni_home_cache() -> None:
    env: dict[str, str] = {}

    cache_directory = apply_torch_compile_cache_env(env)

    assert cache_directory == DEFAULT_TORCHINDUCTOR_CACHE_DIRECTORY
    assert env["TORCHINDUCTOR_CACHE_DIR"] == DEFAULT_TORCHINDUCTOR_CACHE_DIRECTORY


def test_torchinductor_cache_directory_keeps_existing_value() -> None:
    env = {"TORCHINDUCTOR_CACHE_DIR": "/var/cache/custom-torchinductor"}

    assert apply_torch_compile_cache_env(env) == "/var/cache/custom-torchinductor"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/var/cache/custom-torchinductor"


def test_server_args_builder_pins_torchinductor_cache_directory_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)

    build_sglang_server_args(
        write_mini_llama_checkpoint(tmp_path),
        context_length=2048,
        device="cuda",
    )

    assert (
        os.environ["TORCHINDUCTOR_CACHE_DIR"] == DEFAULT_TORCHINDUCTOR_CACHE_DIRECTORY
    )


def test_gpu_compat_env_defaults_pin_torchinductor_cache_directory_when_unset() -> None:
    env = {"FLASHINFER_USE_CUDA_NORM": "0"}

    apply_gpu_compat_env_defaults(env)

    assert env["TORCHINDUCTOR_CACHE_DIR"] == DEFAULT_TORCHINDUCTOR_CACHE_DIRECTORY
