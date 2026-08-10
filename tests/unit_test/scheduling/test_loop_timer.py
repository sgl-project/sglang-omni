# SPDX-License-Identifier: Apache-2.0
"""Behavior of the env-gated scheduler loop segment timer."""

from __future__ import annotations

import json
import logging

import pytest

from sglang_omni.scheduling.loop_timer import LoopSegmentTimer, maybe_loop_timer


class _FakeClock:
    """Deterministic monotonic clock advanced explicitly by the test."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _reports(caplog):
    out = []
    for rec in caplog.records:
        msg = rec.getMessage()
        idx = msg.find("LOOP_TIMER ")
        if idx >= 0:
            out.append(json.loads(msg[idx + len("LOOP_TIMER ") :]))
    return out


def test_accumulates_within_a_window_without_reporting(caplog):
    clock = _FakeClock()
    t = LoopSegmentTimer(interval_s=10.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="sglang_omni.scheduling.loop_timer"):
        clock.t = 1.0
        t.add("launch", 0.1)
        clock.t = 2.0
        t.add("launch", 0.2)
    assert _reports(caplog) == [], "must not report before the interval elapses"


def test_reports_shares_and_counts_after_interval(caplog):
    clock = _FakeClock()
    t = LoopSegmentTimer(interval_s=10.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="sglang_omni.scheduling.loop_timer"):
        clock.t = 4.0
        t.add("launch", 4.0)
        clock.t = 8.0
        t.add("resolve", 2.0)
        clock.t = 11.0  # crosses the 10 s interval
        t.add("launch", 1.0)
    reports = _reports(caplog)
    assert len(reports) == 1
    segs = reports[0]["segments"]
    assert segs["launch"]["n"] == 2
    assert segs["launch"]["s"] == pytest.approx(5.0)
    assert segs["resolve"]["n"] == 1
    # share is against window wall time (11 s), not summed segment time.
    assert segs["launch"]["share"] == pytest.approx(5.0 / 11.0, abs=1e-3)


def test_window_resets_after_report(caplog):
    clock = _FakeClock()
    t = LoopSegmentTimer(interval_s=10.0, clock=clock)
    with caplog.at_level(logging.INFO, logger="sglang_omni.scheduling.loop_timer"):
        clock.t = 11.0
        t.add("launch", 1.0)  # first report fires, window resets to t=11
        clock.t = 12.0
        t.add("resolve", 0.5)
        clock.t = 22.0
        t.add("resolve", 0.5)  # second report
    reports = _reports(caplog)
    assert len(reports) == 2
    # second window must not carry the first window's launch time.
    assert "launch" not in reports[1]["segments"]
    assert reports[1]["segments"]["resolve"]["n"] == 2


def test_env_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_LOOP_TIMER", raising=False)
    assert maybe_loop_timer() is None


@pytest.mark.parametrize("val", ["", "0", "false", "False"])
def test_env_gate_falsy_values_disable(monkeypatch, val):
    monkeypatch.setenv("SGLANG_OMNI_LOOP_TIMER", val)
    assert maybe_loop_timer() is None


def test_env_gate_on_returns_timer(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_LOOP_TIMER", "1")
    monkeypatch.setenv("SGLANG_OMNI_LOOP_TIMER_INTERVAL_S", "5")
    t = maybe_loop_timer()
    assert isinstance(t, LoopSegmentTimer)
    assert t._interval == 5.0


@pytest.mark.parametrize("val", ["not-a-number", "NaN", "Infinity", "0", "-1", "-0.5"])
def test_invalid_interval_falls_back_to_default(monkeypatch, val):
    """Unparseable, non-finite, and non-positive intervals all fall back:
    a nan/inf interval never reports; a <=0 interval reports every add()."""
    monkeypatch.setenv("SGLANG_OMNI_LOOP_TIMER", "1")
    monkeypatch.setenv("SGLANG_OMNI_LOOP_TIMER_INTERVAL_S", val)
    t = maybe_loop_timer()
    assert t._interval == 10.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_direct_construction_invalid_interval_defaults(bad, caplog):
    """0 would report on every add; nan/inf would never report. The class
    itself (not just the env factory) falls back to the 10 s default."""
    clock = _FakeClock()
    t = LoopSegmentTimer(interval_s=bad, clock=clock)
    with caplog.at_level(logging.INFO, logger="sglang_omni.scheduling.loop_timer"):
        clock.t = 1.0
        t.add("launch", 0.1)
    assert _reports(caplog) == [], "invalid interval must behave like 10 s"
    with caplog.at_level(logging.INFO, logger="sglang_omni.scheduling.loop_timer"):
        clock.t = 11.0
        t.add("launch", 0.1)
    assert len(_reports(caplog)) == 1
