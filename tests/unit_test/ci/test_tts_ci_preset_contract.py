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


def test_the_workflow_rotation_draws_registered_presets_only() -> None:
    """Every model the random rotation can draw is a registered preset, and
    the CustomVoice arm is not in the draw: it gates nothing until it is
    calibrated on the CI host, so a random draw of it would spend a CI slot
    without deciding anything. It runs by label or dispatch until then.
    """
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

    assert rotation <= set(TTS_CI_PRESETS), (
        "the rotation names models tts_ci_config.py does not know: "
        f"{sorted(rotation - set(TTS_CI_PRESETS))}"
    )
    assert "qwen3-tts-custom-voice" not in rotation


@pytest.mark.parametrize("name", sorted(TTS_CI_PRESETS))
def test_the_router_profile_matches_the_preset_request_shape(name: str) -> None:
    """A preset's requests have to match a profile the CI router advertises."""
    from tests.test_model.rust_router_config import (
        CiRouterTopology,
        render_router_config,
    )

    model = TTS_CI_PRESETS[name].model
    config = render_router_config(
        topology=CiRouterTopology.TTS,
        router_port=1,
        worker_urls=["http://127.0.0.1:2"],
        model_name=model.model_path,
        named_voice=not model.voice_clone,
    )

    # note (luojiaxuan): the benchmark sends the text plus a voice name for a
    # named-voice preset and a reference clip for a cloning one; the router
    # answers 422 for a request no profile row covers.
    if model.voice_clone:
        assert 'tasks = ["voice_clone"]' in config
        assert 'reference_forms = ["direct", "list"]' in config
        assert 'voice_name_policy = "uploaded"' in config
    else:
        assert 'tasks = ["text_to_speech"]' in config
        assert 'reference_forms = ["none"]' in config
        assert 'voice_name_policy = "preset"' in config
