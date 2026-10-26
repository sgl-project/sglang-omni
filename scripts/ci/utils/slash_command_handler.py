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
    RUN_LINT_ONLY_LABEL,
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


def handle_tag_run_ci(
    pull_request: github.PullRequest.PullRequest,
    comment: github.IssueComment.IssueComment,
    user_permissions: dict[str, bool | str],
    requested_targets: frozenset[str],
    react_on_success: bool = True,
    is_full_ci: bool = False,
) -> bool:
    """Apply CI stage and model labels."""
    if not user_permissions.get("can_tag_run_ci_label", False):
        print("Permission denied: can_tag_run_ci_label is false.")
        comment.create_reaction("confused")
        return False
    else:
        pass

    labels_by_target: dict[str, frozenset[str]] = {
        "lint": frozenset({RUN_LINT_ONLY_LABEL}),
    }
    for stage in CI_STAGES:
        labels_by_target[stage.name.replace("_", "-")] = frozenset({stage.label})
        for label, preset in stage.preset_by_model_label.items():
            if stage.name == "asr":
                target = f"{preset}-asr"
            elif stage.name == "mps":
                target = f"mps-{preset}"
            else:
                target = preset
            labels_by_target[target] = frozenset({stage.label, label})
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
    if is_full_ci or not requested_targets:
        requested_labels |= {RUN_ALL_LABEL}
    else:
        pass
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

    if react_on_success:
        comment.create_reaction("+1")
    else:
        pass
    return True


def handle_rerun_failed_ci(
    gh_repo: github.Repository.Repository,
    pr: github.PullRequest.PullRequest,
    comment: github.IssueComment.IssueComment,
    user_perms: dict[str, bool | str],
    react_on_success: bool = True,
    full_rerun_workflows: frozenset[str] | None = None,
) -> bool:
    """Retry failed jobs, or fully rerun selected completed PR workflows."""
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return False
    else:
        pass

    print("Permission granted. Triggering CI workflow reruns.")

    head_sha = pr.head.sha
    print(f"Checking workflows for commit: {head_sha}")

    runs = gh_repo.get_workflow_runs(head_sha=head_sha)
    latest_runs: dict[tuple[int, str], github.WorkflowRun.WorkflowRun] = {}
    for run in sorted(runs, key=lambda run: (run.created_at, run.id), reverse=True):
        latest_runs.setdefault((run.workflow_id, run.event), run)

    rerun_count = 0
    for run in latest_runs.values():
        if full_rerun_workflows is not None and (
            run.event != "pull_request" or run.name not in full_rerun_workflows
        ):
            continue
        else:
            pass
        if run.status != "completed":
            print(
                f"Skipping latest workflow because it is still {run.status}: "
                f"{run.name} (ID: {run.id})"
            )
            continue
        elif run.conclusion == "action_required":
            print(f"GitHub workflow approval is required: {run.html_url}")
            continue
        elif full_rerun_workflows is None and run.conclusion not in (
            "failure",
            "skipped",
        ):
            print(
                f"Skipping latest workflow with conclusion {run.conclusion}: "
                f"{run.name} (ID: {run.id})"
            )
            continue
        else:
            pass

        print(f"Processing {run.conclusion} workflow: {run.name} (ID: {run.id})")
        try:
            if full_rerun_workflows is not None or run.conclusion == "skipped":
                print("  Full rerun")
                rerun_started = run.rerun()
            else:
                print("  rerun_failed_jobs")
                rerun_started = run.rerun_failed_jobs()
            if rerun_started:
                rerun_count += 1
            else:
                print(f"GitHub did not accept the rerun of workflow {run.id}.")
        except github.GithubException as e:
            print(f"Failed to rerun workflow {run.id}: {e}")

    if rerun_count > 0:
        print(f"Triggered rerun for {rerun_count} workflows.")
        if react_on_success:
            comment.create_reaction("+1")
        else:
            pass
    else:
        print("No eligible completed workflows were rerun.")
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

    if command == "/tag-run-ci-label":
        handle_tag_run_ci(
            pr,
            comment,
            user_perms,
            frozenset(label_arguments),
        )

    elif command == "/rerun-failed-ci":
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    elif command in ("/tag-and-rerun-ci", "/tag-and-rerun-full-ci"):
        if command == "/tag-and-rerun-ci" and not label_arguments:
            pr.create_issue_comment(
                "Specify CI targets, for example /tag-and-rerun-ci tts asr, "
                "or use /tag-and-rerun-full-ci for all stages."
            )
            comment.create_reaction("confused")
            return
        elif not user_perms.get("can_rerun_failed_ci", False):
            print("Permission denied: can_rerun_failed_ci is false.")
            comment.create_reaction("confused")
            return
        else:
            pass
        tagged = handle_tag_run_ci(
            pr,
            comment,
            user_perms,
            frozenset(label_arguments),
            react_on_success=False,
            is_full_ci=command == "/tag-and-rerun-full-ci",
        )
        if not tagged:
            return
        else:
            pass
        time.sleep(5)
        pr.update()
        labels = frozenset(label.name for label in pr.get_labels())
        workflows_to_rerun = select_workflows(labels)
        if not workflows_to_rerun:
            comment.create_reaction("+1")
            return
        else:
            pass
        rerun = handle_rerun_failed_ci(
            repo,
            pr,
            comment,
            user_perms,
            react_on_success=False,
            full_rerun_workflows=workflows_to_rerun,
        )
        if rerun:
            comment.create_reaction("+1")
        else:
            comment.create_reaction("confused")
            pr.create_issue_comment(
                "Labels updated, but no workflow rerun was accepted. "
                "Active workflows are left running; retry after they finish. "
                "If GitHub requires workflow approval, approve it first. "
                "If no PR run exists for this commit, push a commit to create one."
            )

    else:
        print(f"Unknown or ignored command: {command}")


if __name__ == "__main__":
    main()
