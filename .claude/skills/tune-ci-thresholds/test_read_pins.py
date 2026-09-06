"""Pin parsing must honor PEP 508 environment markers (Jiaxin Deng)."""

from __future__ import annotations

import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_DIR))

import tune  # noqa: E402


def test_parse_pins_strips_marker_from_version(monkeypatch):
    monkeypatch.setattr(tune, "_marker_applies", lambda marker: True)
    pins = tune.parse_pins(['sglang==0.5.18; sys_platform != "darwin"'])
    assert pins == {"sglang": "0.5.18"}


def test_parse_pins_skips_pins_whose_marker_does_not_hold(monkeypatch):
    monkeypatch.setattr(
        tune, "_marker_applies", lambda marker: "!=" in marker
    )
    pins = tune.parse_pins(
        [
            'torch==2.13.0; sys_platform != "darwin"',
            'torch==2.11.0; sys_platform == "darwin"',
        ]
    )
    assert pins == {"torch": "2.13.0"}


def test_parse_pins_keeps_unmarked_pins():
    assert tune.parse_pins(["numpy==2.1.0", "not a pin"]) == {"numpy": "2.1.0"}
