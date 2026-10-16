# SPDX-License-Identifier: Apache-2.0
"""CI comment commands update selectors and restart the current PR runs."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def slash_handler(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(REPO_ROOT / ".github/scripts"))
    monkeypatch.setitem(
        sys.modules,
        "github",
        SimpleNamespace(Auth=Mock(), Github=Mock(), GithubException=RuntimeError),
    )
    spec = spec_from_file_location(
        "slash_command_handler", REPO_ROOT / "scripts/ci/utils/slash_command_handler.py"
    )
    assert spec is not None and spec.loader is not None
    handler = module_from_spec(spec)
    spec.loader.exec_module(handler)
    environment = {
        "GITHUB_TOKEN": "test-token",
        "REPO_FULL_NAME": "owner/repo",
        "PR_NUMBER": "1",
        "COMMENT_ID": "2",
        "COMMENT_BODY": "/tag-and-rerun-ci",
        "USER_LOGIN": "contributor",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    handler.load_permissions = Mock(
        return_value={"can_tag_run_ci_label": True, "can_rerun_failed_ci": True}
    )
    repository = handler.Github.return_value.get_repo.return_value
    repository.get_workflow_runs.return_value = []
    pull_request = repository.get_pull.return_value
    pull_request.number = 1
    pull_request.user.login = "author"
    pull_request.head.sha = "head-sha"
    pull_request.head.ref = "feature"
    pull_request.head.repo.full_name = "owner/repo"
    pull_request.get_labels.return_value = []
    monkeypatch.setattr(handler.time, "sleep", Mock())
    return handler


def make_ci_run(
    *,
    workflow_name: str = "Omni CI",
    run_id: int = 10,
    status: str = "completed",
    conclusion: str | None = "failure",
) -> Mock:
    run = Mock(
        id=run_id,
        workflow_id=workflow_name,
        event="pull_request",
        created_at=run_id,
        status=status,
        conclusion=conclusion,
        pull_requests=[SimpleNamespace(number=1)],
        html_url=f"https://github.com/owner/repo/actions/runs/{run_id}",
    )
    run.name = workflow_name
    run.rerun.return_value = True
    run.cancel.return_value = True
    return run


@pytest.mark.parametrize("command", ["/tag-run-ci-label", "/tag-and-rerun-ci"])
def test_bare_commands_add_run_all_ci(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setenv("COMMENT_BODY", command)
    repository = slash_handler.Github.return_value.get_repo.return_value
    run = make_ci_run()
    repository.get_workflow_runs.return_value = [run]
    slash_handler.main()
    repository.get_pull.return_value.add_to_labels.assert_called_once_with("run-all-ci")
    if command == "/tag-run-ci-label":
        repository.get_workflow_runs.assert_not_called()
    else:
        run.rerun.assert_called_once_with()
    repository.get_issue.return_value.get_comment.return_value.create_reaction.assert_called_once_with(
        "+1"
    )


def test_model_replacement_preserves_other_stage_selections(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "COMMENT_BODY",
        "/tag-run-ci-label moss mps",
    )
    pull_request = (
        slash_handler.Github.return_value.get_repo.return_value.get_pull.return_value
    )
    pull_request.get_labels.return_value = [
        SimpleNamespace(name=name)
        for name in ["run-all-ci", "run-tts-higgs-tts-3-4b", "run-mps-higgs-tts-3-4b"]
    ]
    slash_handler.main()
    pull_request.remove_from_labels.assert_called_once_with("run-tts-higgs-tts-3-4b")
    pull_request.add_to_labels.assert_called_once_with(
        "run-mps-ci", "run-tts-ci", "run-tts-moss-tts-local-transformer-v1.5"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        "unknown-model",
        "run-ci",
        "run-tts-moss-tts-local-transformer-v1.5",
        "higgs moss",
        "fun-asr whisper-asr",
        "qwen3-omni minicpmo",
        "mps-higgs mps-moss",
        "run-tts-ci run-tts-higgs-tts-3-4b run-tts-moss-tts-local-transformer-v1.5",
    ],
)
def test_invalid_selections_do_not_change_labels(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch, arguments: str
) -> None:
    monkeypatch.setenv("COMMENT_BODY", f"/tag-and-rerun-ci {arguments}")
    repository = slash_handler.Github.return_value.get_repo.return_value
    slash_handler.main()
    pull_request = repository.get_pull.return_value
    pull_request.add_to_labels.assert_not_called()
    pull_request.remove_from_labels.assert_not_called()
    repository.get_workflow_runs.assert_not_called()
    pull_request.create_issue_comment.assert_called_once()


@pytest.mark.parametrize("command", ["/tag-run-ci-label", "/tag-and-rerun-ci"])
@pytest.mark.parametrize(
    "arguments",
    ["tts", "asr", "omni", "tts asr", "tts omni", "asr omni", "tts asr omni"],
)
def test_short_stage_combinations(
    slash_handler: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    arguments: str,
) -> None:
    monkeypatch.setenv("COMMENT_BODY", f"{command} {arguments}")
    repository = slash_handler.Github.return_value.get_repo.return_value
    run = make_ci_run()
    repository.get_workflow_runs.return_value = [run]
    slash_handler.main()
    repository.get_pull.return_value.add_to_labels.assert_called_once_with(
        *sorted(f"run-{stage}-ci" for stage in arguments.split())
    )
    if command == "/tag-and-rerun-ci":
        run.rerun.assert_called_once_with()
    else:
        repository.get_workflow_runs.assert_not_called()


@pytest.mark.parametrize("stage", ["asr", "tts", "mps", "omni"])
@pytest.mark.parametrize("extra_stage", ["", "asr", "omni"])
def test_short_model_targets_enable_their_stage(
    slash_handler: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    extra_stage: str,
) -> None:
    selected_stage = next(
        entry for entry in slash_handler.CI_STAGES if entry.name == stage
    )
    for label, preset in selected_stage.preset_by_model_label.items():
        target = f"{preset}-asr" if stage == "asr" else preset
        target = f"mps-{preset}" if stage == "mps" else target
        monkeypatch.setenv("COMMENT_BODY", f"/tag-run-ci-label {target} {extra_stage}")
        pull_request = (
            slash_handler.Github.return_value.get_repo.return_value.get_pull.return_value
        )
        pull_request.add_to_labels.reset_mock()
        slash_handler.main()
        expected_labels = {selected_stage.label, label}
        if extra_stage:
            expected_labels.add(f"run-{extra_stage}-ci")
        else:
            pass
        pull_request.add_to_labels.assert_called_once_with(*sorted(expected_labels))


@pytest.mark.parametrize("permission", ["can_tag_run_ci_label", "can_rerun_failed_ci"])
def test_combined_command_checks_both_permissions_before_label_changes(
    slash_handler: ModuleType, permission: str
) -> None:
    slash_handler.load_permissions.return_value[permission] = False
    repository = slash_handler.Github.return_value.get_repo.return_value
    slash_handler.main()
    repository.get_pull.return_value.add_to_labels.assert_not_called()
    repository.get_workflow_runs.assert_not_called()


def test_pr_author_does_not_gain_tagging_permission(slash_handler: ModuleType) -> None:
    slash_handler.load_permissions.return_value = None
    repository = slash_handler.Github.return_value.get_repo.return_value
    repository.get_pull.return_value.user.login = "contributor"
    slash_handler.main()
    repository.get_pull.return_value.add_to_labels.assert_not_called()


def test_run_all_restarts_all_selected_parents_but_not_hosted_checks(
    slash_handler: ModuleType,
) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    runs = [
        make_ci_run(workflow_name=name)
        for name in ["Omni CI", "CPU CI", "XPU CI", "Voxt Mac CI", "Lint"]
    ]
    repository.get_workflow_runs.return_value = runs
    slash_handler.main()
    for run in runs[:-1]:
        run.rerun.assert_called_once_with()
    runs[-1].rerun.assert_not_called()
    assert all(
        invocation == call(head_sha="head-sha", event="pull_request")
        for invocation in repository.get_workflow_runs.call_args_list
    )


@pytest.mark.parametrize("workflow_name", ["CPU CI", "XPU CI", "Voxt Mac CI"])
def test_hardware_selection_also_restarts_unit_tests(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch, workflow_name: str
) -> None:
    stage = next(
        stage for stage in slash_handler.CI_STAGES if stage.workflow == workflow_name
    )
    monkeypatch.setenv("COMMENT_BODY", f"/tag-and-rerun-ci {stage.label}")
    repository = slash_handler.Github.return_value.get_repo.return_value
    runs = [
        make_ci_run(workflow_name=name) for name in ["Omni CI", workflow_name, "Lint"]
    ]
    repository.get_workflow_runs.return_value = runs
    slash_handler.main()
    runs[0].rerun.assert_called_once_with()
    runs[1].rerun.assert_called_once_with()
    runs[2].rerun.assert_not_called()


@pytest.mark.parametrize(
    "conclusion", ["success", "failure", "cancelled", "timed_out", "skipped"]
)
def test_combined_command_full_reruns_the_newest_completed_run(
    slash_handler: ModuleType, conclusion: str
) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    older = make_ci_run(run_id=1)
    latest = make_ci_run(conclusion=conclusion)
    repository.get_workflow_runs.return_value = [older, latest]
    slash_handler.main()
    latest.rerun.assert_called_once_with()
    latest.rerun_failed_jobs.assert_not_called()
    older.rerun.assert_not_called()


@pytest.mark.parametrize(
    "status", ["queued", "in_progress", "waiting", "pending", "requested"]
)
def test_active_run_is_cancelled_before_full_rerun(
    slash_handler: ModuleType, status: str
) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    active = make_ci_run(status=status, conclusion=None)
    completed = make_ci_run(conclusion="cancelled")
    repository.get_workflow_runs.return_value = [active]
    repository.get_workflow_run.return_value = completed
    operations = Mock()
    operations.attach_mock(active.cancel, "cancel")
    operations.attach_mock(completed.rerun, "rerun")
    slash_handler.main()
    assert operations.mock_calls == [call.cancel(), call.rerun()]
    active.rerun.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "approval",
        "cancel-rejected",
        "rerun-rejected",
        "api-error",
        "cancel-timeout",
        "new-head",
    ],
)
def test_failed_restart_does_not_report_success(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    run = make_ci_run()
    if failure == "approval":
        run.conclusion = "action_required"
    elif failure == "cancel-rejected":
        run.status = "in_progress"
        run.cancel.return_value = False
    elif failure == "rerun-rejected":
        run.rerun.return_value = False
    elif failure == "api-error":
        run.rerun.side_effect = RuntimeError("GitHub rejected the rerun")
    elif failure == "cancel-timeout":
        run.status = "in_progress"
        monkeypatch.setattr(slash_handler.time, "monotonic", Mock(side_effect=[0, 121]))
    else:
        repository.get_pull.return_value.update.side_effect = lambda: setattr(
            repository.get_pull.return_value.head, "sha", "new-head"
        )
    repository.get_workflow_runs.return_value = [run]
    slash_handler.main()
    comment = repository.get_issue.return_value.get_comment.return_value
    comment.create_reaction.assert_called_once_with("confused")
    repository.get_pull.return_value.create_issue_comment.assert_called_once()
    if failure in {"approval", "cancel-rejected", "cancel-timeout", "new-head"}:
        run.rerun.assert_not_called()
    else:
        pass


def test_missing_runs_do_not_dispatch_main(slash_handler: ModuleType) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    slash_handler.main()
    repository.get_workflows.assert_not_called()
    repository.get_pull.return_value.create_issue_comment.assert_called_once()
    repository.get_issue.return_value.get_comment.return_value.create_reaction.assert_called_once_with(
        "confused"
    )


def test_fork_run_with_empty_pr_metadata_matches_repository_and_branch(
    slash_handler: ModuleType,
) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    repository.get_pull.return_value.head.repo.full_name = "fork/repo"
    run = make_ci_run()
    run.pull_requests = []
    run.head_repository.full_name = "fork/repo"
    run.head_branch = "feature"
    repository.get_workflow_runs.return_value = [run]
    slash_handler.main()
    run.rerun.assert_called_once_with()


def test_other_pr_at_same_commit_is_not_restarted(slash_handler: ModuleType) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    run = make_ci_run()
    run.pull_requests = [SimpleNamespace(number=2)]
    repository.get_workflow_runs.return_value = [run]
    slash_handler.main()
    run.cancel.assert_not_called()
    run.rerun.assert_not_called()


def test_newer_run_created_during_cancellation_is_not_displaced(
    slash_handler: ModuleType,
) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    active = make_ci_run(status="in_progress", conclusion=None)
    completed = make_ci_run(conclusion="cancelled")
    replacement = make_ci_run(run_id=11, status="in_progress", conclusion=None)
    repository.get_workflow_runs.side_effect = [[active], [replacement]]
    repository.get_workflow_run.return_value = completed
    slash_handler.main()
    completed.rerun.assert_not_called()
    replacement.cancel.assert_not_called()
    replacement.rerun.assert_not_called()
    repository.get_issue.return_value.get_comment.return_value.create_reaction.assert_called_once_with(
        "confused"
    )


@pytest.mark.parametrize("completed_before_cancel", [True, False])
def test_cancellation_exception_only_allows_rerun_after_completion(
    slash_handler: ModuleType, completed_before_cancel: bool
) -> None:
    repository = slash_handler.Github.return_value.get_repo.return_value
    active = make_ci_run(status="in_progress", conclusion=None)
    active.cancel.side_effect = RuntimeError("Cancellation rejected")
    refreshed = make_ci_run(
        status="completed" if completed_before_cancel else "in_progress",
        conclusion="success" if completed_before_cancel else None,
    )
    repository.get_workflow_runs.return_value = [active]
    repository.get_workflow_run.return_value = refreshed
    slash_handler.main()
    if completed_before_cancel:
        refreshed.rerun.assert_called_once_with()
    else:
        refreshed.rerun.assert_not_called()
        repository.get_pull.return_value.create_issue_comment.assert_called_once()
    repository.get_issue.return_value.get_comment.return_value.create_reaction.assert_called_once_with(
        "+1" if completed_before_cancel else "confused"
    )


def test_rerun_failed_ci_still_reruns_failed_jobs(
    slash_handler: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COMMENT_BODY", "/rerun-failed-ci")
    repository = slash_handler.Github.return_value.get_repo.return_value
    run = make_ci_run()
    repository.get_workflow_runs.return_value = [run]
    slash_handler.main()
    run.rerun_failed_jobs.assert_called_once_with()
    run.rerun.assert_not_called()
    repository.get_issue.return_value.get_comment.return_value.create_reaction.assert_called_once_with(
        "+1"
    )
