# SPDX-License-Identifier: Apache-2.0
"""Tests for quantization argument passthrough."""
from __future__ import annotations

from sglang_omni.engines.ar.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)


def test_quantization_passthrough():
    sa = build_sglang_server_args(
        "dummy/path",
        context_length=2048,
        quantization="awq",
    )
    assert sa.quantization == "awq"


def test_no_quantization_by_default():
    sa = build_sglang_server_args(
        "dummy/path",
        context_length=2048,
    )
    assert sa.quantization is None


def test_kv_cache_dtype_passthrough():
    sa = build_sglang_server_args(
        "dummy/path",
        context_length=2048,
        kv_cache_dtype="fp8_e5m2",
    )
    assert sa.kv_cache_dtype == "fp8_e5m2"


def test_quantization_via_overrides_still_works():
    """Verify the existing **overrides path also works."""
    sa = build_sglang_server_args(
        "dummy/path",
        context_length=2048,
        **{"quantization": "gptq"},
    )
    assert sa.quantization == "gptq"
