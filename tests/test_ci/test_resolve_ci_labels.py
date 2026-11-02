import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/scripts/resolve_ci_labels.py"
SPEC = importlib.util.spec_from_file_location("resolve_ci_labels", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# The script's dataclasses resolve their module through sys.modules while the class body runs.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

SPEC_STAGE = next(stage for stage in MODULE.CI_STAGES if stage.name == "spec")
EVERY_SPEC_MODEL = list(dict.fromkeys(SPEC_STAGE.preset_by_model_label.values()))


@pytest.mark.parametrize(
    ("labels", "spec_models", "asr_model"),
    [
        pytest.param({"run-spec-ci"}, EVERY_SPEC_MODEL, "", id="every-draft"),
        pytest.param(
            {"run-all-ci", "run-qwen3-asr-1.7b"},
            ["qwen3-asr"],
            "qwen3",
            id="all-ci-with-model",
        ),
        pytest.param(
            {"run-spec-ci", "run-qwen3-asr-1.7b"}, ["qwen3-asr"], "", id="one-draft"
        ),
        pytest.param({"run-asr-ci", "run-qwen3-asr-1.7b"}, [], "qwen3", id="asr-only"),
    ],
)
def test_speculative_stage_serves_every_draft_unless_a_model_label_picks_one(
    labels: set[str], spec_models: list[str], asr_model: str
) -> None:
    """Without a model label, Speculative CI checks the shared draft path on every model."""
    plan = MODULE.resolve_ci_plan(frozenset(labels), seed="1")
    outputs = dict(line.split("=", 1) for line in MODULE.format_github_outputs(plan))

    assert outputs["run_spec"] == str(bool(spec_models)).lower()
    assert json.loads(outputs["spec_ci_models"]) == spec_models
    assert outputs["asr_ci_model"] == asr_model
    assert plan.ignored_model_labels == ()
