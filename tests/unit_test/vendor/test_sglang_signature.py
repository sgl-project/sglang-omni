# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from sglang_omni.vendor.sglang.signature import supported_kwargs


class _LegacyComponent:
    def __init__(self, *, stable: int):
        self.stable = stable


class _CurrentComponent:
    def __init__(self, *, stable: int, added: int):
        self.stable = stable
        self.added = added


def test_supported_kwargs_filters_new_parameter_for_legacy_component() -> None:
    assert supported_kwargs(_LegacyComponent, stable=1, added=2) == {"stable": 1}


def test_supported_kwargs_keeps_new_parameter_for_current_component() -> None:
    assert supported_kwargs(_CurrentComponent, stable=1, added=2) == {
        "stable": 1,
        "added": 2,
    }
