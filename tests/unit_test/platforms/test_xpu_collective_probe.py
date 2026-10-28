# SPDX-License-Identifier: Apache-2.0
"""The XPU collective probe must reject a dropped collective and agree across every rank.

A regression here would let a TP decode graph capture with collectives that never replay, which
serves wrong tokens silently, so the replay checks and the all-rank MIN are covered directly.
The device plumbing runs against a fake torch.xpu on CPU tensors; the collectives themselves are
covered by the value checks below and, on hardware, by the capture the probe performs.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms.xpu as xpu_platform
from sglang_omni.platforms.xpu import (
    PROBE_ROWS,
    XPUOmniPlatform,
    replayed_collectives_match,
)

WORLD_SIZE = 4


def replayed_tensors(world_size: int = WORLD_SIZE):
    """What a recorded all-reduce and all-gather leave behind, given every rank staged 10*(r+1)."""
    staged = [10.0 * (rank + 1) for rank in range(world_size)]
    reduced = torch.full((PROBE_ROWS,), sum(staged))
    gathered = torch.cat([torch.full((PROBE_ROWS,), value) for value in staged])
    return reduced, gathered


def test_a_recorded_replay_matches():
    reduced, gathered = replayed_tensors()

    assert replayed_collectives_match(reduced, gathered, WORLD_SIZE) is True


def test_a_dropped_all_reduce_is_rejected():
    reduced, gathered = replayed_tensors()
    reduced.fill_(
        10.0
    )  # this rank's staged fill, left untouched by a dropped all-reduce

    assert replayed_collectives_match(reduced, gathered, WORLD_SIZE) is False


def test_a_dropped_all_gather_is_rejected():
    reduced, gathered = replayed_tensors()
    gathered.zero_()  # the pre-replay state, left untouched by a dropped all-gather

    assert replayed_collectives_match(reduced, gathered, WORLD_SIZE) is False


def install_fake_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the probe's device half on CPU tensors: XPU graph capture and replay are no-ops."""

    @contextmanager
    def graph(*args, **kwargs):
        yield

    fake_xpu = SimpleNamespace(
        current_device=lambda: 0,
        synchronize=lambda: None,
        current_stream=lambda: object(),
        Stream=lambda: SimpleNamespace(wait_stream=lambda stream: None),
        XPUGraph=lambda: SimpleNamespace(replay=lambda: None),
        graph=graph,
    )
    fake_torch = SimpleNamespace(
        **{
            name: getattr(torch, name)
            for name in ("full", "zeros", "arange", "tensor", "all", "equal")
        },
        device=lambda *args: torch.device("cpu"),
        xpu=fake_xpu,
    )
    monkeypatch.setattr(xpu_platform, "torch", fake_torch)


def probe_with(monkeypatch: pytest.MonkeyPatch, *, local: bool, peer: bool) -> bool:
    """Run the probe with a fixed local verdict and one peer reporting its own."""
    install_fake_device(monkeypatch)
    monkeypatch.setattr(
        xpu_platform,
        "replayed_collectives_match",
        lambda reduced, gathered, world_size: local,
    )

    def all_reduce(tensor, op=None, group=None):
        if op is None:  # the probe's own captured collective, not the verdict reduction
            return
        assert op is xpu_platform.dist.ReduceOp.MIN
        tensor.fill_(min(int(tensor.item()), int(peer)))

    monkeypatch.setattr(
        xpu_platform,
        "dist",
        SimpleNamespace(
            all_reduce=all_reduce,
            all_gather_into_tensor=lambda output, input, group=None: None,
            barrier=lambda group=None: None,
            ReduceOp=xpu_platform.dist.ReduceOp,
        ),
    )
    group = SimpleNamespace(
        world_size=WORLD_SIZE,
        rank_in_group=0,
        device_group=object(),
        cpu_group=object(),
    )
    return XPUOmniPlatform().graph_capture_records_collectives(group)


def test_every_rank_accepts_when_all_of_them_replayed(monkeypatch) -> None:
    assert probe_with(monkeypatch, local=True, peer=True) is True


def test_one_rank_failing_makes_every_rank_reject(monkeypatch) -> None:
    """The verdict is MIN-reduced, so a peer that saw a dropped collective vetoes capture."""
    assert probe_with(monkeypatch, local=True, peer=False) is False


def test_a_local_failure_rejects(monkeypatch) -> None:
    assert probe_with(monkeypatch, local=False, peer=True) is False


def test_a_single_rank_group_skips_the_probe() -> None:
    """One rank issues no collective, so there is nothing to verify and nothing to capture."""
    group = SimpleNamespace(world_size=1, rank_in_group=0)

    assert XPUOmniPlatform().graph_capture_records_collectives(group) is True
