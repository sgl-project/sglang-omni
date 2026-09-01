# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

from tests.test_model.tts_ci_config import TTS_CI_PRESETS, select_tts_ci_preset

REPO_ROOT = Path(__file__).resolve().parents[3]
OMNI_WORKFLOW = REPO_ROOT / ".github/workflows/omni-ci.yaml"
TTS_WORKFLOW = REPO_ROOT / ".github/workflows/test-tts-ci.yaml"


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _select_step_script() -> str:
    jobs = _workflow(OMNI_WORKFLOW)["jobs"]
    preflight = jobs["preflight"]
    step = next(s for s in preflight["steps"] if s.get("id") == "tts")
    return step["run"]


def test_rotation_offers_every_registered_preset() -> None:
    """The workflow must be able to select every model the presets define."""
    script = _select_step_script()
    models_line = next(
        line for line in script.splitlines() if line.strip().startswith("models=(")
    )
    rotation = set(models_line.split("(", 1)[1].split(")", 1)[0].split())
    assert rotation == set(TTS_CI_PRESETS), (
        f"workflow rotation {sorted(rotation)} does not match registered presets "
        f"{sorted(TTS_CI_PRESETS)}"
    )
    for model in rotation:
        # A bare substring check is not enough: the label name also appears in
        # the conflict error text, so deleting the selection branch entirely
        # would still pass.
        env_name = f"RUN_{model.upper().replace('-', '_')}_LABEL"
        assert env_name in script, f"no label env for {model}"
        assert (
            f'selection_reason="run-{model}_label"' in script
        ), f"no selection branch for {model}"
        assert (
            model in script.partition("case ")[2].partition(")")[0]
        ), f"dispatch override does not accept {model}"


def test_every_rotation_model_resolves_an_mps_pool() -> None:
    """A model without a pool of its own must borrow one, not drop stage 5.

    Skipping left the MPS path unexercised on a third of the runs.
    """
    script = _select_step_script()
    assert 'resolved_config=""' not in script, "a rotation model still skips stage 5"
    assert 'mps_model="moss"' in script, "no fallback pool in preflight"

    mps = _workflow(TTS_WORKFLOW)["jobs"]["stage-5-mps"]
    condition = mps["if"]
    assert (
        "inputs.tts_mps_config != ''" in condition
    ), "stage 5 must be conditional on an MPS config being resolved"
    # A trailing `|| true` keeps the substring while making the guard inert.
    assert "||" not in condition, f"stage 5 condition is bypassable: {condition}"


def test_qwen3_tts_gates_on_calibrated_thresholds() -> None:
    """Gating is only valid because the values came from a calibration run."""
    _, preset = select_tts_ci_preset("qwen3-tts")
    assert preset.model.model_path == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert preset.model.gate_thresholds is True
    assert preset.thresholds.calibrated is True
    # Calibrated worst-of-N floors (17.66 and 15.324 pre-slack), not the
    # pre-calibration seed of 11.309.
    assert preset.thresholds.non_stream_speed[16]["throughput_qps_min"] > 11.309
    assert preset.thresholds.stream_speed[16]["throughput_qps_min"] > 11.309
    # The tuned operating point, not the shipped defaults, is what CI measures.
    worker_args = preset.model.worker_extra_args
    assert "--tts_engine.engine.max_running_requests 64" in worker_args
    assert "--isolate-stage" not in worker_args
    assert "--vocoder.process vocoder" in worker_args
    assert "--tts_engine.gpu_memory_fraction 0.85" in worker_args
    assert "--vocoder.gpu_memory_fraction 0.10" in worker_args


def test_every_registered_model_is_reachable_from_every_entry_point() -> None:
    """Registration is only complete when all three entry points agree.

    A model registered in the presets but missing from the slash-command map or
    the calibration config is selectable by CI yet impossible to request or
    calibrate, which is exactly how a model ends up with stale thresholds.
    """
    import ast

    import yaml as _yaml

    # Parsed rather than imported: the handler pulls in the GitHub SDK, which
    # only exists on the CI runner.
    handler = ast.parse(
        (REPO_ROOT / "scripts/ci/utils/slash_command_handler.py").read_text(
            encoding="utf-8"
        )
    )
    labels = next(
        ast.literal_eval(node.value)
        for node in handler.body
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "TTS_MODEL_LABELS"
    )
    assert set(labels) == set(TTS_CI_PRESETS), (
        f"slash-command targets {sorted(labels)} do not match presets "
        f"{sorted(TTS_CI_PRESETS)}"
    )

    calibration = _yaml.safe_load(
        (
            REPO_ROOT / ".claude/skills/tune-ci-thresholds/models/tts/config.yaml"
        ).read_text(encoding="utf-8")
    )
    presets = calibration["metric_sources"]["test_tts_ci.py"]["calibration_presets"]
    assert set(presets) == set(TTS_CI_PRESETS), (
        f"calibration presets {sorted(presets)} do not match CI presets "
        f"{sorted(TTS_CI_PRESETS)}"
    )


def test_seed_thresholds_are_never_used_as_gates() -> None:
    """Uncalibrated numbers must not be able to fail somebody else's build."""
    for name, preset in TTS_CI_PRESETS.items():
        if preset.model.gate_thresholds:
            assert preset.thresholds.calibrated, (
                f"{name} gates on thresholds that are still seeds; calibrate "
                "before enabling gate_thresholds"
            )


def test_calibration_filters_can_claim_the_presets_constants() -> None:
    """A filter matching nothing leaves a model silently uncalibratable."""
    import re

    import yaml as _yaml

    config = _yaml.safe_load(
        (
            REPO_ROOT / ".claude/skills/tune-ci-thresholds/models/tts/config.yaml"
        ).read_text(encoding="utf-8")
    )
    source = (REPO_ROOT / "tests/test_model/tts_ci_config.py").read_text(
        encoding="utf-8"
    )
    constants = set(re.findall(r"^_?([A-Z][A-Z0-9_]+)\s*[:=]", source, re.MULTILINE))
    presets = config["metric_sources"]["test_tts_ci.py"]["calibration_presets"]
    for name, preset in presets.items():
        pattern = re.compile(preset["constant_filter"])
        assert [
            c for c in constants if pattern.match(c)
        ], f"calibration filter for {name} claims no constant in tts_ci_config"
