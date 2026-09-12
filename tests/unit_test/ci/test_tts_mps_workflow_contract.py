# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OMNI_WORKFLOW = REPO_ROOT / ".github/workflows/omni-ci.yaml"
TTS_WORKFLOW = REPO_ROOT / ".github/workflows/test-tts-ci.yaml"


def _workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_workflow_keeps_ordinary_stage_and_adds_isolated_mps_validation() -> None:
    workflow = _workflow(TTS_WORKFLOW)
    jobs = workflow["jobs"]
    ordinary = jobs["stage-1-non-streaming"]
    mps = jobs["stage-5-mps"]
    assert mps["name"] == "stage 5 - MPS"

    ordinary_run = _step(ordinary, "Run TTS non-streaming benchmark stage")
    assert "test_tts_ci.py" in ordinary_run["run"]
    assert "TTS_MPS_CONFIG" not in ordinary_run.get("env", {})
    assert ordinary["container"]["options"].find("--cap-add SYS_NICE") == -1
    assert (
        _step(ordinary, "Upload non-streaming speed artifact")["with"]["name"]
        == "tts-stage-nonstream-speed-results"
    )

    assert mps["needs"] == "stage-4-serving"
    assert "always()" in mps["if"]
    assert (
        "tests/test_ci/test_tts_mps_dp2.py"
        in _step(mps, "Run TTS MPS non-streaming validation")["run"]
    )
    assert "--cap-add SYS_NICE" in mps["container"]["options"]
    # Same shape as the other TTS stages: fixed name, overwritten per rerun,
    # evidence paths only, and the run outputs removed from the CI home.
    upload = _step(mps, "Upload TTS MPS evidence")["with"]
    assert upload["name"] == "tts-stage-mps-nonstream-evidence"
    assert str(upload["overwrite"]).lower() == "true"
    assert upload["if-no-files-found"] == "error"
    assert "canonical/*.json" in upload["path"]
    assert "rm -rf" in _step(mps, "Remove TTS MPS run outputs")["run"]
    assert "${{ env.OMNI_CI_HOME }}" not in mps["env"]["TTS_MPS_OUTPUT_ROOT"]
    assert "${{ env.OMNI_CI_HOME }}" not in mps["env"]["TTS_MPS_STATE_ROOT"]
    assert mps["env"]["TTS_MPS_OUTPUT_ROOT"].startswith("${{ inputs.omni_ci_home")
    assert mps["env"]["TTS_MPS_STATE_ROOT"].startswith("${{ inputs.omni_ci_home")
    assert "run-${{ github.run_id }}" not in mps["env"]["TTS_MPS_STATE_ROOT"]
    assert "TTS_MPS_BASE_PORT" not in mps["env"]

    assert jobs["stage-2-streaming"]["needs"] == "stage-1-non-streaming"
    assert jobs["stage-3-consistency"]["needs"] == "stage-2-streaming"
    assert jobs["stage-4-serving"]["needs"] == "stage-3-consistency"
    assert list(jobs).index("stage-5-mps") > list(jobs).index("stage-4-serving")


def test_mps_artifacts_cannot_be_consumed_by_canonical_consistency() -> None:
    workflow = _workflow(TTS_WORKFLOW)
    jobs = workflow["jobs"]
    mps = jobs["stage-5-mps"]
    mps_env = mps["env"]
    assert "/mps-nonstream/" in mps_env["TTS_MPS_OUTPUT_ROOT"]
    assert "/nonstream/" not in mps_env["TTS_MPS_OUTPUT_ROOT"]

    canonical_jobs = [
        jobs["stage-2-streaming"],
        jobs["stage-3-consistency"],
        jobs["stage-4-serving"],
    ]
    canonical_text = yaml.safe_dump(canonical_jobs)
    assert "tts-mps-" not in canonical_text
    assert "mps-nonstream" not in canonical_text


def test_mps_stage_measures_the_pool_model_not_the_rotation_model() -> None:
    """Stage 5 can run a different model than stages 1-4, so it must say so.

    The evidence writer and the threshold lookup both key on the model name,
    and both reject a name they have no MPS references for.
    """
    mps = _workflow(TTS_WORKFLOW)["jobs"]["stage-5-mps"]
    assert (
        _step(mps, "Run TTS MPS non-streaming validation")["env"]["TTS_CI_MODEL"]
        == "${{ inputs.tts_mps_model }}"
    )
    assert (
        '--selected-model "${{ inputs.tts_mps_model }}"'
        in _step(mps, "Initialize TTS MPS evidence")["run"]
    )
    assert "inputs.tts_ci_model" not in yaml.safe_dump(mps)
    # Passing the config without the model would gate a moss pool on whichever
    # model the test defaults to.
    tts_ci = _workflow(OMNI_WORKFLOW)["jobs"]["tts-ci"]
    assert "pick-tts-model" in tts_ci["needs"]
    assert (
        tts_ci["with"]["tts_mps_model"]
        == "${{ needs.pick-tts-model.outputs.resolved_mps_model }}"
    )
    assert (
        tts_ci["with"]["tts_ci_model"]
        == "${{ needs.pick-tts-model.outputs.selected_model }}"
    )
