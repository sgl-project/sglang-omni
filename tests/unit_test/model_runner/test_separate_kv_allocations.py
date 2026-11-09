# SPDX-License-Identifier: Apache-2.0
"""A compiled engine's MHA pool keeps K and V out of one allocation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

import sglang_omni.model_runner.sglang_model_runner as runner_mod
from sglang_omni.model_runner.sglang_model_runner import (
    OmniKVCacheConfigurator,
    SeparateKVAllocationsPool,
)

LAYER_NUM = 2
PAGE_SIZE = 16
HEAD_NUM = 2
HEAD_DIM = 128


def make_pool(pool_class: type[MHATokenToKVPool]) -> MHATokenToKVPool:
    return pool_class(
        size=PAGE_SIZE * 7,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        layer_num=LAYER_NUM,
        device="cpu",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def test_k_and_v_share_no_storage_and_keep_the_base_pool_shape() -> None:
    """K and V in one allocation made every compiled decode step clone all of it."""
    base_pool = make_pool(MHATokenToKVPool)
    pool = make_pool(SeparateKVAllocationsPool)
    for layer_id in range(LAYER_NUM):
        k_buffer, v_buffer = pool.get_kv_buffer(layer_id)
        base_k_buffer, base_v_buffer = base_pool.get_kv_buffer(layer_id)
        assert (
            k_buffer.untyped_storage().data_ptr()
            != v_buffer.untyped_storage().data_ptr()
        )
        assert k_buffer.shape == base_k_buffer.shape
        assert v_buffer.shape == base_v_buffer.shape


@pytest.mark.parametrize(
    ("enable_torch_compile", "kind", "default", "expected"),
    [
        (True, "mha", MHATokenToKVPool, SeparateKVAllocationsPool),
        (False, "mha", MHATokenToKVPool, MHATokenToKVPool),
        (True, "mla", MLATokenToKVPool, MLATokenToKVPool),
    ],
)
def test_the_pool_class_keeps_the_fused_layout_unless_a_plain_mha_pool_compiles(
    monkeypatch, enable_torch_compile, kind, default, expected
) -> None:
    monkeypatch.setattr(
        runner_mod,
        "get_exec",
        lambda: SimpleNamespace(
            graph=SimpleNamespace(enable_torch_compile=enable_torch_compile)
        ),
    )
    configurator = OmniKVCacheConfigurator.__new__(OmniKVCacheConfigurator)

    resolved = configurator._resolve_kv_pool_class(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        kind=kind, default=default
    )

    assert resolved is expected
