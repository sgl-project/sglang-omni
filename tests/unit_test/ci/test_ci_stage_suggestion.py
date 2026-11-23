# SPDX-License-Identifier: Apache-2.0
"""The stage suggestion names /tag-and-rerun-ci targets that test what a pull request changes."""

import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS = "sglang_omni/models"
MINICPM_PATH = f"{MODELS}/minicpm_o/config.py"
MARKER = "<!-- ci-stage-suggestion -->"
REPOSITORY = "example/example"
PULL_REQUEST_NUMBER = 42


@pytest.fixture(scope="module")
def resolver() -> Iterator[ModuleType]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.syspath_prepend(str(REPO_ROOT / ".github/scripts"))
        monkeypatch.syspath_prepend(str(REPO_ROOT / "scripts/ci/utils"))

        yield importlib.import_module("resolve_ci_labels")

        sys.modules.pop("resolve_ci_labels")
        sys.modules.pop("suggest_ci_stages", None)


@pytest.fixture(scope="module")
def suggester(resolver: ModuleType) -> ModuleType:
    return importlib.import_module("suggest_ci_stages")


class RecordingGithub:
    """Answers the three reads of a pull request that changes MiniCPM-o and records every write."""

    def __init__(self, comments: list[dict[str, str | int]]) -> None:
        self.comments = comments
        self.writes: list[tuple[str, str, str]] = []

    def __call__(self, arguments: list[str]) -> str:
        endpoint = next(
            argument for argument in arguments if argument.startswith("repos/")
        )
        if "--method" in arguments:
            self.writes.append(
                (arguments[1], endpoint, arguments[-1].removeprefix("body="))
            )
            return ""
        elif endpoint.endswith("/files"):
            return f"{MINICPM_PATH}\n"
        elif endpoint.endswith("/comments"):
            return "".join(f"{json.dumps(comment)}\n" for comment in self.comments)
        else:
            return json.dumps({"draft": False, "labels": []})


def test_the_handler_accepts_the_command_and_it_makes_every_stage_test_the_model(
    resolver: ModuleType,
) -> None:
    labels_by_target = resolver.build_labels_by_target()
    model_directories = sorted(
        {
            model_directory
            for preset_directories in resolver.MODEL_DIRECTORIES_BY_PRESET.values()
            for model_directory in preset_directories
        }
        | {
            model_directory
            for stage in resolver.CI_STAGES
            for model_directory in stage.model_directories
        }
    )

    for model_directory in model_directories:
        model_file = REPO_ROOT / MODELS / model_directory / "__init__.py"
        assert model_file.is_file(), model_directory
        changed_paths = [str(model_file.relative_to(REPO_ROOT))]
        suggestions = resolver.suggest_stage_targets(changed_paths, frozenset())
        tagged_labels = resolver.labels_after_tagging(
            frozenset(),
            frozenset(
                label
                for suggestion in suggestions
                for target in suggestion.targets
                for label in labels_by_target[target]
            ),
        )

        resolver.resolve_ci_plan(tagged_labels, seed="1")
        assert suggestions, model_directory
        assert resolver.suggest_stage_targets(changed_paths, tagged_labels) == []


def test_every_preset_a_stage_runs_names_existing_model_directories(
    resolver: ModuleType,
) -> None:
    presets = {
        preset
        for stage in resolver.CI_STAGES
        for preset in (*stage.preset_by_model_label.values(), *stage.random_presets)
    }

    assert presets <= set(resolver.MODEL_DIRECTORIES_BY_PRESET)
    for preset in presets:
        for model_directory in resolver.MODEL_DIRECTORIES_BY_PRESET[preset]:
            assert (REPO_ROOT / MODELS / model_directory).is_dir(), (
                preset,
                model_directory,
            )


def test_a_change_to_ark_asr_gets_the_asr_stage_suggested(
    resolver: ModuleType,
) -> None:
    suggestions = resolver.suggest_stage_targets(
        [f"{MODELS}/arkasr/config.py"], frozenset()
    )

    assert [suggestion.model_directory for suggestion in suggestions] == ["arkasr"]


def test_a_model_label_that_mps_took_still_gets_the_tts_stage_suggested(
    resolver: ModuleType,
) -> None:
    suggestions = resolver.suggest_stage_targets(
        [f"{MODELS}/higgs_tts/config.py"], frozenset({"run-mps-ci"})
    )

    assert [suggestion.targets for suggestion in suggestions] == [("higgs", "tts")]


def test_another_changed_model_of_a_selected_stage_is_listed_once_with_its_target(
    resolver: ModuleType,
) -> None:
    suggestions = resolver.suggest_stage_targets(
        [
            f"{MODELS}/fun_cosyvoice3/config.py",
            f"{MODELS}/moss_tts/audio_tokenizer.py",
            f"{MODELS}/moss_tts_local/config.py",
        ],
        frozenset({"run-fun-cosyvoice3-0.5b-2512"}),
    )

    assert [
        (suggestion.model_directory, suggestion.targets, suggestion.is_in_command)
        for suggestion in suggestions
    ] == [("moss_tts", ("moss",), False)]


def test_shared_code_gets_the_stage_targets_until_a_model_stage_is_selected(
    resolver: ModuleType,
) -> None:
    changed_paths = ["sglang_omni/serve/launcher.py"]

    assert resolver.suggest_shared_code_targets(
        changed_paths, frozenset({"run-unit-test-ci-only"})
    ) == ["asr", "tts", "omni"]
    assert (
        resolver.suggest_shared_code_targets(changed_paths, frozenset({"run-tts-ci"}))
        == []
    )


def test_the_bot_rewrites_its_own_comment_and_leaves_one_a_person_wrote(
    suggester: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording_github = RecordingGithub(
        [
            {"id": 1, "body": f"{MARKER}\npasted by hand", "user_type": "User"},
            {"id": 2, "body": f"{MARKER}\nstale", "user_type": "Bot"},
        ]
    )
    monkeypatch.setattr(suggester, "call_github", recording_github)

    suggester.update_stage_suggestion(REPOSITORY, PULL_REQUEST_NUMBER)

    assert [
        (method, endpoint) for method, endpoint, body in recording_github.writes
    ] == [("PATCH", f"repos/{REPOSITORY}/issues/comments/2")]


def test_an_unchanged_suggestion_is_not_written_again(
    suggester: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_run = RecordingGithub([])
    monkeypatch.setattr(suggester, "call_github", first_run)
    suggester.update_stage_suggestion(REPOSITORY, PULL_REQUEST_NUMBER)
    written_body = first_run.writes[0][2]
    second_run = RecordingGithub([{"id": 2, "body": written_body, "user_type": "Bot"}])
    monkeypatch.setattr(suggester, "call_github", second_run)

    suggester.update_stage_suggestion(REPOSITORY, PULL_REQUEST_NUMBER)

    assert "/tag-and-rerun-ci minicpmo full-duplex" in written_body
    assert second_run.writes == []
