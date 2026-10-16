from __future__ import annotations

import json
import os
import sys
import time

import github
from github import Auth, Github
from resolve_ci_labels import (
    CI_STAGES,
    RUN_ALL_LABEL,
    resolve_ci_plan,
    select_workflows,
)

PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"


def get_env_var(name):
    val = os.getenv(name)
    if not val:
        print(f"Error: Environment variable {name} not set.")
        sys.exit(1)
    return val


def load_permissions(user_login):
    """
    Reads the permissions JSON from the local file system and returns
    the permissions dict for the specific user.
    """
    try:
        print(f"Loading permissions from {PERMISSIONS_FILE_PATH}...")
        if not os.path.exists(PERMISSIONS_FILE_PATH):
            print(f"Error: Permissions file not found at {PERMISSIONS_FILE_PATH}")
            return None

        with open(PERMISSIONS_FILE_PATH, "r") as f:
            data = json.load(f)

        user_perms = data.get(user_login)

        if not user_perms:
            print(f"User '{user_login}' not found in permissions file.")
            return None

        return user_perms

    except Exception as e:
        print(f"Failed to load or parse permissions file: {e}")
        sys.exit(1)


def latest_workflow_runs(
    runs: list[github.WorkflowRun.WorkflowRun],
) -> list[github.WorkflowRun.WorkflowRun]:
    """Select the newest run of each workflow and event."""
    latest_runs: dict[tuple[int, str], github.WorkflowRun.WorkflowRun] = {}
    for run in sorted(runs, key=lambda run: (run.created_at, run.id), reverse=True):
        latest_runs.setdefault((run.workflow_id, run.event), run)
    return list(latest_runs.values())


def latest_pr_workflow_runs(
    repository: github.Repository.Repository,
    pull_request: github.PullRequest.PullRequest,
) -> list[github.WorkflowRun.WorkflowRun]:
    """Filter current-commit runs to this PR before choosing the newest ones."""
    runs: list[github.WorkflowRun.WorkflowRun] = []
    for run in repository.get_workflow_runs(
        head_sha=pull_request.head.sha, event="pull_request"
    ):
        if run.pull_requests:
            matches_pr = any(
                pull.number == pull_request.number for pull in run.pull_requests
            )
        else:
            matches_pr = (
                pull_request.head.repo is not None
                and run.head_repository is not None
                and run.head_repository.full_name == pull_request.head.repo.full_name
                and run.head_branch == pull_request.head.ref
            )
        if matches_pr:
            runs.append(run)
        else:
            pass
    return latest_workflow_runs(runs)


def restart_ci_workflows(
    repository: github.Repository.Repository,
    pull_request: github.PullRequest.PullRequest,
    workflow_names: frozenset[str],
) -> bool:
    """Restart the selected parent workflows on the current PR commit."""
    head_sha = pull_request.head.sha
    runs = [
        run
        for run in latest_pr_workflow_runs(repository, pull_request)
        if run.name in workflow_names
    ]

    rerun_count = 0
    errors: list[str] = []
    for run in runs:
        if run.conclusion == "action_required":
            errors.append(
                f"{run.name} needs GitHub workflow approval before it can run."
            )
            continue
        else:
            pass

        try:
            if run.status != "completed":
                try:
                    cancelled = run.cancel()
                except github.GithubException:
                    run = repository.get_workflow_run(run.id)
                    if run.status != "completed":
                        raise
                    else:
                        pass
                else:
                    if not cancelled:
                        errors.append(
                            f"GitHub did not accept cancellation of {run.name} ({run.id})."
                        )
                        continue
                    else:
                        pass

                deadline = time.monotonic() + 120
                while run.status != "completed" and time.monotonic() < deadline:
                    time.sleep(5)
                    run = repository.get_workflow_run(run.id)
                if run.status != "completed":
                    errors.append(
                        f"Timed out waiting for {run.name} ({run.id}) to stop."
                    )
                    continue
                else:
                    pass
            else:
                pass

            pull_request.update()
            if pull_request.head.sha != head_sha:
                errors.append(
                    "The PR commit changed during the command. Run the command again."
                )
                break
            else:
                pass
            current_run = next(
                (
                    candidate
                    for candidate in latest_pr_workflow_runs(repository, pull_request)
                    if candidate.workflow_id == run.workflow_id
                ),
                None,
            )
            if current_run is None or current_run.id != run.id:
                errors.append(
                    f"A newer {run.name} run replaced {run.id}. Run the command again to restart it."
                )
                continue
            elif run.rerun():
                rerun_count += 1
                print(f"Restarted {run.name}: {run.html_url}")
            else:
                errors.append(
                    f"GitHub did not accept the rerun of {run.name} ({run.id})."
                )
        except github.GithubException as error:
            errors.append(f"Could not restart {run.name} ({run.id}): {error}")

    missing_workflows = workflow_names - {run.name for run in runs}
    if missing_workflows:
        print(
            f"No current-commit PR run found for: {', '.join(sorted(missing_workflows))}. Path filters may exclude these workflows."
        )
    else:
        pass
    if not runs:
        errors.append(
            "No selected CI workflow run exists for this PR commit. Update the branch from main and push before retrying."
        )
    else:
        pass
    if errors:
        pull_request.create_issue_comment("\n\n".join(errors))
    else:
        pass
    return rerun_count > 0 and not errors


def handle_tag_run_ci(
    repository: github.Repository.Repository,
    pull_request: github.PullRequest.PullRequest,
    comment: github.IssueComment.IssueComment,
    user_permissions: dict[str, bool | str],
    requested_targets: frozenset[str],
    restart_workflows: bool,
) -> bool:
    """Apply CI selectors and optionally restart their parent workflows."""
    if not user_permissions.get("can_tag_run_ci_label", False):
        print("Permission denied: can_tag_run_ci_label is false.")
        comment.create_reaction("confused")
        return False
    elif restart_workflows and not user_permissions.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        comment.create_reaction("confused")
        return False
    else:
        pass

    labels_by_target = {RUN_ALL_LABEL: frozenset({RUN_ALL_LABEL})}
    for stage in CI_STAGES:
        labels_by_target[stage.name.replace("_", "-")] = frozenset({stage.label})
        labels_by_target[stage.label] = frozenset({stage.label})
        for label, preset in stage.preset_by_model_label.items():
            if stage.name == "asr":
                target = f"{preset}-asr"
            elif stage.name == "mps":
                target = f"mps-{preset}"
            else:
                target = preset
            labels_by_target[target] = frozenset({stage.label, label})
            labels_by_target[label] = frozenset({label})
    unknown_targets = requested_targets - labels_by_target.keys()
    if unknown_targets:
        pull_request.create_issue_comment(
            f"Unknown CI targets: {', '.join(sorted(unknown_targets))}."
        )
        comment.create_reaction("confused")
        return False
    else:
        pass
    requested_labels = frozenset(
        label for target in requested_targets for label in labels_by_target[target]
    )
    current_labels = frozenset(label.name for label in pull_request.get_labels())
    removed_labels: set[str] = set()
    for stage in CI_STAGES:
        model_labels = stage.preset_by_model_label.keys()
        if requested_labels & model_labels:
            removed_labels.update((current_labels & model_labels) - requested_labels)
        else:
            pass
    final_labels = (current_labels - removed_labels) | requested_labels
    try:
        plan = resolve_ci_plan(final_labels, seed=str(pull_request.number))
    except ValueError as error:
        pull_request.create_issue_comment(str(error))
        comment.create_reaction("confused")
        return False

    for label in sorted(removed_labels):
        pull_request.remove_from_labels(label)
    pull_request.add_to_labels(*sorted(requested_labels))
    for label in plan.ignored_model_labels:
        print(f"Ignoring {label}: its stage is not selected.")

    if restart_workflows:
        succeeded = restart_ci_workflows(
            repository, pull_request, select_workflows(final_labels)
        )
    else:
        succeeded = True
    comment.create_reaction("+1" if succeeded else "confused")
    return succeeded


def handle_rerun_failed_ci(gh_repo, pr, comment, user_perms):
    """
    Handles the /rerun-failed-ci command.
    Reruns workflows with 'failure' or 'skipped' conclusions.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return False

    print("Permission granted. Triggering CI workflow reruns.")

    head_sha = pr.head.sha
    print(f"Checking workflows for commit: {head_sha}")

    latest_runs = latest_workflow_runs(
        list(gh_repo.get_workflow_runs(head_sha=head_sha))
    )

    rerun_count = 0
    for run in latest_runs:
        if run.status != "completed":
            print(
                f"Skipping latest workflow because it is still {run.status}: "
                f"{run.name} (ID: {run.id})"
            )
            continue
        if run.conclusion not in ("failure", "skipped"):
            print(
                f"Skipping latest workflow with conclusion {run.conclusion}: "
                f"{run.name} (ID: {run.id})"
            )
            continue

        print(f"Processing {run.conclusion} workflow: {run.name} (ID: {run.id})")
        try:
            if run.conclusion == "skipped":
                print("  Full rerun")
                run.rerun()
            else:
                print("  rerun_failed_jobs")
                run.rerun_failed_jobs()
            rerun_count += 1
        except Exception as e:
            print(f"Failed to rerun workflow {run.id}: {e}")

    if rerun_count > 0:
        print(f"Triggered rerun for {rerun_count} workflows.")
        comment.create_reaction("+1")
    else:
        print("No failed or skipped workflows found to rerun.")
    return rerun_count > 0


def main():
    token = get_env_var("GITHUB_TOKEN")
    repo_name = get_env_var("REPO_FULL_NAME")
    pr_number = int(get_env_var("PR_NUMBER"))
    comment_id = int(get_env_var("COMMENT_ID"))
    comment_body = get_env_var("COMMENT_BODY").strip()
    user_login = get_env_var("USER_LOGIN")

    user_perms = load_permissions(user_login)

    auth = Auth.Token(token)
    g = Github(auth=auth)

    repo = g.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    comment = repo.get_issue(pr_number).get_comment(comment_id)

    # PR authors can always rerun failed CI on their own PRs, even if they are
    # not listed in CI_PERMISSIONS.json.
    if pr.user.login == user_login:
        if user_perms is None:
            print(
                f"User {user_login} is the PR author (not in CI_PERMISSIONS.json). "
                "Granting CI rerun permissions."
            )
            user_perms = {}
        else:
            print(
                f"User {user_login} is the PR author and has existing CI permissions."
            )
        user_perms["can_rerun_failed_ci"] = True

    if not user_perms:
        print(f"User {user_login} does not have any configured permissions. Exiting.")
        return

    command, *label_arguments = comment_body.split("\n")[0].split()

    if command in ("/tag-run-ci-label", "/tag-and-rerun-ci"):
        handle_tag_run_ci(
            repo,
            pr,
            comment,
            user_perms,
            frozenset(label_arguments or [RUN_ALL_LABEL]),
            restart_workflows=command == "/tag-and-rerun-ci",
        )

    elif command == "/rerun-failed-ci":
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    else:
        print(f"Unknown or ignored command: {command}")


if __name__ == "__main__":
    main()
