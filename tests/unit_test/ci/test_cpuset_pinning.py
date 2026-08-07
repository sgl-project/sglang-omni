# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OMNI_CI_CPUSET pinning hook in test_model/conftest."""

from __future__ import annotations

import os

import pytest

from tests.test_model.conftest import _apply_omni_ci_cpuset, _parse_cpuset


def test_parse_cpuset_ranges_and_singles():
    assert _parse_cpuset("0-3,8,64-65") == {0, 1, 2, 3, 8, 64, 65}


def test_parse_cpuset_single_cpu():
    assert _parse_cpuset("7") == {7}


@pytest.mark.parametrize("spec", ["", " , ", "0,,1", ",0-3", "0-3,", ","])
def test_parse_cpuset_rejects_empty_components(spec):
    with pytest.raises(ValueError):
        _parse_cpuset(spec)


def test_parse_cpuset_rejects_inverted_range():
    with pytest.raises(ValueError):
        _parse_cpuset("5-2")


def test_parse_cpuset_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_cpuset("0-a")


def test_apply_noop_without_env(monkeypatch):
    monkeypatch.delenv("OMNI_CI_CPUSET", raising=False)
    assert _apply_omni_ci_cpuset() is None


def test_apply_fails_when_kernel_narrows_affinity(monkeypatch):
    monkeypatch.setenv("OMNI_CI_CPUSET", "0-1")
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None, raising=False)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0}, raising=False)
    with pytest.raises(RuntimeError, match=r"requested \[0, 1\].*effective \[0\]"):
        _apply_omni_ci_cpuset()


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity"), reason="requires sched_setaffinity"
)
def test_apply_pins_and_children_would_inherit(monkeypatch):
    original = os.sched_getaffinity(0)
    target = set(sorted(original)[:2])
    spec = ",".join(str(c) for c in sorted(target))
    monkeypatch.setenv("OMNI_CI_CPUSET", spec)
    try:
        assert _apply_omni_ci_cpuset() == target
        assert os.sched_getaffinity(0) == target
    finally:
        os.sched_setaffinity(0, original)
