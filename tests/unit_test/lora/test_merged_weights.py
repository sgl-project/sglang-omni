# SPDX-License-Identifier: Apache-2.0
"""Unit tests for host-backed merged LoRA weight management."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from sglang_omni.lora.merged_weights import (
    DEFAULT_DEVICE_PARK_HEADROOM_BYTES,
    MergedLinearWeights,
    ParkingPolicy,
    ScratchBufferPool,
    copy_to_host_staged,
)


def test_parking_policy_selects_device_or_host() -> None:
    policy = ParkingPolicy(headroom_bytes=1024)
    assert policy.should_park_on_device(512, 2048) is True
    assert policy.should_park_on_device(512, 2047) is False
    assert policy.should_park_on_device(64, None) is False

    host_only = ParkingPolicy(force_host_parking=True)
    assert host_only.should_park_on_device(64, 10**12) is False
    assert DEFAULT_DEVICE_PARK_HEADROOM_BYTES == 4 * 1024**3

    with pytest.raises(ValueError, match="headroom_bytes"):
        ParkingPolicy(headroom_bytes=-1)
    with pytest.raises(ValueError, match="multiplier"):
        ParkingPolicy(multiplier=0.0)


def test_copy_to_host_staged_fallbacks() -> None:
    src = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    pinned_calls = 0

    def fake_pin_success(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal pinned_calls
        pinned_calls += 1
        return tensor

    out, mode = copy_to_host_staged(src, prefer_pinned=True, pin_fn=fake_pin_success)
    assert mode == "pinned_arena"
    assert torch.equal(out, src)
    assert pinned_calls == 1

    attempts = 0

    def fail_arena_then_succeed(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("arena pin OOM")
        return tensor

    out_chunked, chunk_mode = copy_to_host_staged(
        src,
        prefer_pinned=True,
        pin_fn=fail_arena_then_succeed,
        chunk_bytes=16,
    )
    assert chunk_mode == "chunked_pinned"
    assert torch.equal(out_chunked, src)
    assert attempts >= 2

    def always_fail_pin(tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("pinning unavailable")

    out_pageable, page_mode = copy_to_host_staged(
        src,
        prefer_pinned=True,
        pin_fn=always_fail_pin,
        chunk_bytes=16,
    )
    assert page_mode == "pageable"
    assert torch.equal(out_pageable, src)


def test_scratch_buffer_pool_reuses_fp32_storage() -> None:
    pool = ScratchBufferPool()
    first = pool.get_fp32("merge", (4, 8), torch.device("cpu"))
    assert first.dtype == torch.float32
    assert first.shape == (4, 8)
    first_ptr = first.data_ptr()

    smaller = pool.get_fp32("merge", (2, 4), torch.device("cpu"))
    assert smaller.data_ptr() == first_ptr
    assert smaller.shape == (2, 4)

    larger = pool.get_fp32("merge", (8, 16), torch.device("cpu"))
    assert larger.shape == (8, 16)
    larger_ptr = larger.data_ptr()
    again = pool.get_fp32("merge", (8, 16), torch.device("cpu"))
    assert again.data_ptr() == larger_ptr

    pool.clear()
    assert pool.buffers == {}


def test_merged_linear_weights_lifecycle_and_data_ptr_invariance() -> None:
    torch.manual_seed(0)
    w1 = nn.Parameter(torch.randn(8, 6, dtype=torch.bfloat16), requires_grad=False)
    w2 = nn.Parameter(torch.randn(4, 6, dtype=torch.float32), requires_grad=False)
    base1 = w1.detach().clone()
    base2 = w2.detach().clone()
    ptr1 = w1.data_ptr()
    ptr2 = w2.data_ptr()

    bank = MergedLinearWeights({"l1.weight": w1, "l2.weight": w2}, prefer_pinned=False)

    delta1 = torch.full((8, 6), 0.25, dtype=torch.float32)
    delta2 = torch.full((4, 6), -0.5, dtype=torch.float32)
    location = bank.prepare({"l1.weight": delta1, "l2.weight": delta2})
    assert location == "cpu"
    assert len(bank.prepared_weights) == 2
    # Base bank is an isolated host copy captured lazily on first prepare.
    assert bank.base_bank["l1.weight"].data_ptr() != ptr1
    # Preparing does not mutate live parameters before commit.
    assert torch.equal(w1.detach(), base1)
    assert torch.equal(w2.detach(), base2)

    updated = bank.commit()
    assert updated == 2
    assert not bank.prepared_weights
    assert w1.data_ptr() == ptr1
    assert w2.data_ptr() == ptr2
    expected1 = (base1.float() + delta1).to(dtype=torch.bfloat16)
    expected2 = base2 + delta2
    assert torch.equal(w1.detach(), expected1)
    assert torch.allclose(w2.detach(), expected2, atol=1e-6)

    # Switching to an adapter that omits l2 restores l2 to pristine base when restore_omitted=True.
    delta1_v2 = torch.full((8, 6), -0.125, dtype=torch.float32)
    bank.prepare({"l1.weight": delta1_v2}, restore_omitted=True)
    updated = bank.commit()
    assert updated == 2
    assert w1.data_ptr() == ptr1
    assert w2.data_ptr() == ptr2
    assert torch.equal(
        w1.detach(), (base1.float() + delta1_v2).to(dtype=torch.bfloat16)
    )
    assert torch.equal(w2.detach(), base2)

    # restore_base resets remaining modified parameter without reallocating storage.
    restored = bank.restore_base()
    assert restored == 1
    assert w1.data_ptr() == ptr1
    assert w2.data_ptr() == ptr2
    assert torch.equal(w1.detach(), base1)
    assert torch.equal(w2.detach(), base2)
    assert bank.restore_base() == 0


def test_merged_linear_weights_rollback_and_mid_commit_recovery() -> None:
    torch.manual_seed(1)
    w1 = nn.Parameter(torch.randn(4, 4, dtype=torch.float32), requires_grad=False)
    w2 = nn.Parameter(torch.randn(4, 4, dtype=torch.float32), requires_grad=False)
    orig1 = w1.detach().clone()
    orig2 = w2.detach().clone()

    bank = MergedLinearWeights({"a": w1, "b": w2}, prefer_pinned=False)
    bank.prepare({"a": torch.ones(4, 4)})
    bank.rollback()
    assert not bank.prepared_weights
    assert torch.equal(w1.detach(), orig1)

    # Failed prepare clears staging and rejects unknown parameters or wrong shapes.
    with pytest.raises(KeyError, match="unknown target"):
        bank.prepare({"missing": torch.ones(4, 4)})
    assert not bank.prepared_weights

    with pytest.raises(ValueError, match="delta shape"):
        bank.prepare({"a": torch.ones(2, 2)})
    assert not bank.prepared_weights

    # Simulate an error halfway through commit: parameter "a" is written, then "b" raises.
    bank.prepare({"a": torch.ones(4, 4), "b": torch.full((4, 4), 2.0)})
    bank.prepared_weights["b"] = torch.zeros(2, 2)
    with pytest.raises(RuntimeError):
        bank.commit()

    # Automatic rollback restored parameter "a" and left "b" untouched.
    assert not bank.prepared_weights
    assert torch.allclose(w1.detach(), orig1)
    assert torch.allclose(w2.detach(), orig2)
