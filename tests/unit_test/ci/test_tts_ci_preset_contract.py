# SPDX-License-Identifier: Apache-2.0
"""Contracts the TTS CI preset registry has to satisfy."""

from __future__ import annotations

import pytest

from tests.test_model.tts_ci_config import TTS_CI_PRESETS


@pytest.mark.parametrize("name", sorted(TTS_CI_PRESETS))
def test_only_calibrated_presets_gate_thresholds(name: str) -> None:
    preset = TTS_CI_PRESETS[name]
    if preset.model.gate_thresholds:
        assert preset.thresholds.calibrated, (
            f"{name} gates its thresholds while they are still seeds; either "
            "calibrate them on the CI host or set gate_thresholds=False"
        )


@pytest.mark.parametrize("name", sorted(TTS_CI_PRESETS))
def test_a_named_voice_preset_carries_a_voice(name: str) -> None:
    model = TTS_CI_PRESETS[name].model
    if model.voice_clone:
        assert model.voice is None, (
            f"{name} clones a reference, so a server-side voice preset would "
            "be ignored"
        )
    else:
        assert model.voice, f"{name} sends no reference, so it needs a voice"


def test_the_workflow_rotation_covers_every_preset() -> None:
    """The random rotation and the preset registry name the same models."""
    from pathlib import Path

    import yaml

    workflow = Path(__file__).resolve().parents[3] / ".github/workflows/omni-ci.yaml"
    jobs = yaml.load(workflow.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)[
        "jobs"
    ]
    script = next(
        step["run"]
        for step in jobs["pick-tts-model"]["steps"]
        if step.get("id") == "tts"
    )
    line = next(
        stripped
        for stripped in (raw.strip() for raw in script.splitlines())
        if stripped.startswith("models=(")
    )
    rotation = set(line[len("models=(") : line.rindex(")")].split())

    assert rotation == set(TTS_CI_PRESETS), (
        "the rotation and tts_ci_config.py disagree: "
        f"rotation only {sorted(rotation - set(TTS_CI_PRESETS))}, "
        f"registry only {sorted(set(TTS_CI_PRESETS) - rotation)}"
    )
