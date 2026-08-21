# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the persistent-state registry and the stateful/stateless
registration contract. No GPU required (data_ptr works on CPU tensors)."""
from __future__ import annotations

import pytest
import torch

from sglang_omni.cuda_graph import PersistentStateError, PersistentStateRegistry


def test_declare_and_names():
    reg = PersistentStateRegistry()
    buf = torch.zeros(2, 3)
    reg.declare("k", lambda: buf)
    assert reg.declared_names() == ["k"]
    assert not reg.is_empty()


def test_declare_twice_raises():
    reg = PersistentStateRegistry()
    buf = torch.zeros(2, 3)
    reg.declare("k", lambda: buf)
    with pytest.raises(PersistentStateError, match="already declared"):
        reg.declare("k", lambda: buf)


def test_shape_mismatch_raises():
    reg = PersistentStateRegistry()
    buf = torch.zeros(2, 3)
    with pytest.raises(PersistentStateError, match="shape mismatch"):
        reg.declare("k", lambda: buf, expected_shape=(2, 4))


def test_dtype_mismatch_raises():
    reg = PersistentStateRegistry()
    buf = torch.zeros(2, 3, dtype=torch.float32)
    with pytest.raises(PersistentStateError, match="dtype mismatch"):
        reg.declare("k", lambda: buf, expected_dtype=torch.long)


def test_non_tensor_accessor_raises():
    reg = PersistentStateRegistry()
    with pytest.raises(PersistentStateError, match="must return a tensor"):
        reg.declare("k", lambda: 123)


def test_empty_registry_is_noop():
    reg = PersistentStateRegistry()
    assert reg.is_empty()
    reg.reset()  # no hooks -> no-op
    reg.snapshot_addresses()
    reg.assert_addresses_stable()  # nothing declared -> trivially holds


def test_reset_runs_hooks():
    reg = PersistentStateRegistry()
    calls = []
    reg.add_reset_hook(lambda: calls.append(1))
    reg.add_reset_hook(lambda: calls.append(2))
    assert not reg.is_empty()
    reg.reset()
    assert calls == [1, 2]


def test_assert_stable_holds_for_inplace_write():
    reg = PersistentStateRegistry()
    holder = {"buf": torch.zeros(4, 5)}
    reg.declare("k", lambda: holder["buf"])
    reg.snapshot_addresses()
    holder["buf"].copy_(torch.ones(4, 5))  # in-place: same storage, same address
    reg.assert_addresses_stable()  # must not raise


def test_assert_trips_on_reassignment():
    reg = PersistentStateRegistry()
    holder = {"buf": torch.zeros(4, 5)}
    reg.declare("k", lambda: holder["buf"])
    reg.snapshot_addresses()
    holder["buf"] = torch.zeros(4, 5)  # reassigned: new storage, new address
    with pytest.raises(PersistentStateError, match="moved"):
        reg.assert_addresses_stable()


def test_assert_requires_snapshot():
    reg = PersistentStateRegistry()
    buf = torch.zeros(2)
    reg.declare("k", lambda: buf)
    with pytest.raises(PersistentStateError, match="no address snapshot"):
        reg.assert_addresses_stable()


# --- stateful/stateless registration contract on the base class (no GPU) ---
