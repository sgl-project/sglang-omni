# SPDX-License-Identifier: Apache-2.0
"""Contracts the TTS CI preset registry has to satisfy."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from tests.test_model.tts_ci_config import TTS_CI_PRESETS

CI_LABEL_RULES_PATH = (
    Path(__file__).resolve().parents[3] / ".github/scripts/resolve_ci_labels.py"
)


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


def test_every_tts_ci_label_selects_a_registered_preset() -> None:
    """A TTS model label or the random draw can only select a registered preset."""
    spec = spec_from_file_location("resolve_ci_labels", CI_LABEL_RULES_PATH)
    assert spec is not None and spec.loader is not None
    ci_label_rules = module_from_spec(spec)
    sys.modules[spec.name] = ci_label_rules
    spec.loader.exec_module(ci_label_rules)
    tts = next(stage for stage in ci_label_rules.CI_STAGES if stage.name == "tts")
    selectable_presets = set(tts.preset_by_model_label.values())

    assert selectable_presets <= set(TTS_CI_PRESETS), (
        "the TTS CI labels name models tts_ci_config.py does not know: "
        f"{sorted(selectable_presets - set(TTS_CI_PRESETS))}"
    )


@pytest.mark.parametrize("name", sorted(TTS_CI_PRESETS))
def test_the_latency_stage_covers_exactly_the_qwen3_tts_arms(name: str) -> None:
    """The first-audio latency work is on Qwen3-TTS, so those arms carry the
    latency points and no other arm does."""
    preset = TTS_CI_PRESETS[name]
    is_qwen3_tts = preset.model.model_path.startswith("Qwen/Qwen3-TTS")
    assert (preset.latency is not None) == is_qwen3_tts


@pytest.mark.parametrize("name", sorted(TTS_CI_PRESETS))
def test_a_latency_preset_gates_every_point_or_none(name: str) -> None:
    """A calibrated preset carries a median threshold at every point, so no
    point is silently left ungated; an uncalibrated one carries none, so the
    stage can only print."""
    latency = TTS_CI_PRESETS[name].latency
    if latency is None:
        return
    assert latency.points
    assert all(point.samples > 0 and point.request_rate > 0 for point in latency.points)
    thresholds = [point.ttfp_median_max_s for point in latency.points]
    if latency.calibrated:
        assert all(value is not None and value > 0 for value in thresholds)
    else:
        assert all(value is None for value in thresholds)


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
