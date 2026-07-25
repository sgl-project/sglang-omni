import json
import os
import sys
import time

from github import Auth, Github
from github.GithubException import GithubException

PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"
TTS_MODEL_LABELS = {
    "higgs": "run-higgs",
    "moss": "run-moss",
}


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


def parse_tts_model_target(tokens):
    targets = [token for token in tokens[1:] if token in TTS_MODEL_LABELS]
    if len(set(targets)) > 1:
        return None, "Specify only one TTS CI model target: higgs or moss."
    return (targets[0] if targets else None), None


def handle_tag_run_ci(
    pr, comment, user_perms, react_on_success=True, tts_model_target=None
):
    """
    Handles the /tag-run-ci-label command.

    When tts_model_target is set, also applies the matching TTS model label.
    The TTS model labels are mutually exclusive, so remove the opposite label
    before adding the selected one.

    The combined /tag-and-rerun-ci command restarts Omni CI after updating
    labels, because labels added by GITHUB_TOKEN do not cascade-trigger a
    new pull_request.labeled workflow run.

    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_tag_run_ci_label", False):
        print("Permission denied: can_tag_run_ci_label is false.")
        return False

    labels = ["run-ci"]
    if tts_model_target:
        selected_label = TTS_MODEL_LABELS[tts_model_target]
        opposite_labels = [
            label
            for model, label in TTS_MODEL_LABELS.items()
            if model != tts_model_target
        ]
        current_labels = {label.name for label in pr.get_labels()}
        for label in opposite_labels:
            if label in current_labels:
                print(f"Removing mutually exclusive label: {label}.")
                pr.remove_from_labels(label)
        labels.append(selected_label)

    print(f"Permission granted. Adding labels: {labels}.")
    for label in labels:
        pr.add_to_labels(label)

    if react_on_success:
        comment.create_reaction("+1")
        print("Labels added and comment reacted.")
    else:
        print("Labels added (reaction suppressed).")

    return True


def wait_for_workflow_run_completed(gh_repo, run_id):
    deadline = time.time() + 120
    while time.time() < deadline:
        latest_run = gh_repo.get_workflow_run(run_id)
        if latest_run.status == "completed":
            return True
        print(
            f"Waiting for workflow {run_id} to complete cancellation: "
            f"{latest_run.status}"
        )
        time.sleep(5)

    print(f"Timed out waiting for workflow {run_id} to complete cancellation.")
    return False


def cancel_and_rerun_workflow(gh_repo, run):
    print(f"Cancelling active workflow before full rerun: {run.name} (ID: {run.id})")
    try:
        cancelled = run.cancel()
    except GithubException as e:
        latest_run = gh_repo.get_workflow_run(run.id)
        if latest_run.status != "completed":
            print(f"Failed to cancel workflow {run.id}: {e}")
            return False
    else:
        if not cancelled:
            print(f"Failed to cancel workflow {run.id}.")
            return False

    if not wait_for_workflow_run_completed(gh_repo, run.id):
        return False

    latest_run = gh_repo.get_workflow_run(run.id)
    try:
        rerun_started = latest_run.rerun()
    except GithubException as e:
        print(f"Failed to rerun workflow {latest_run.id}: {e}")
        return False
    if not rerun_started:
        print(f"Failed to rerun workflow {latest_run.id}.")
        return False
    print(f"Triggered full rerun for workflow {latest_run.name} (ID: {run.id}).")
    return True


def handle_rerun_failed_ci(
    gh_repo,
    pr,
    comment,
    user_perms,
    react_on_success=True,
    force_full_omni_ci_rerun=False,
):
    """
    Handles the /rerun-failed-ci command.
    Reruns workflows with 'failure' or 'skipped' conclusions.
    When force_full_omni_ci_rerun is set, full-reruns the latest Omni CI
    pull_request workflow even if it is active or successful.
    Returns True if action was taken, False otherwise.
    """
    if not user_perms.get("can_rerun_failed_ci", False):
        print("Permission denied: can_rerun_failed_ci is false.")
        return False

    print("Permission granted. Triggering CI workflow reruns.")

    head_sha = pr.head.sha
    print(f"Checking workflows for commit: {head_sha}")

    runs = gh_repo.get_workflow_runs(head_sha=head_sha)

    latest_runs = []
    seen_workflows = set()
    for run in sorted(runs, key=lambda run: (run.created_at, run.id), reverse=True):
        workflow_key = (run.workflow_id, run.event)
        if workflow_key in seen_workflows:
            continue
        seen_workflows.add(workflow_key)
        latest_runs.append(run)

    rerun_count = 0
    force_rerun_target_found = False
    force_rerun_succeeded = False
    for run in latest_runs:
        force_full_rerun = (
            run.event == "pull_request"
            and run.name == "Omni CI"
            and force_full_omni_ci_rerun
        )
        if force_full_rerun:
            force_rerun_target_found = True
        if run.status in {"queued", "in_progress", "waiting", "pending", "requested"}:
            if not force_full_rerun:
                print(
                    f"Skipping latest workflow because it is still {run.status}: "
                    f"{run.name} (ID: {run.id})"
                )
                continue
            if cancel_and_rerun_workflow(gh_repo, run):
                rerun_count += 1
                force_rerun_succeeded = True
            continue

        if run.status != "completed":
            print(
                f"Skipping latest workflow because it is still {run.status}: "
                f"{run.name} (ID: {run.id})"
            )
            continue
        if force_full_rerun:
            if run.conclusion == "action_required":
                print(f"Cannot rerun workflow {run.id}: GitHub approval is required.")
                continue
            print(f"Processing full workflow rerun: {run.name} (ID: {run.id})")
            try:
                rerun_started = run.rerun()
            except GithubException as e:
                print(f"Failed to rerun workflow {run.id}: {e}")
                continue
            if not rerun_started:
                print(f"Failed to rerun workflow {run.id}.")
                continue
            rerun_count += 1
            force_rerun_succeeded = True
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

    if force_full_omni_ci_rerun and not force_rerun_target_found:
        print(
            "No Omni CI pull_request workflow run exists for the current PR head "
            f"{head_sha}."
        )

    action_succeeded = (
        force_rerun_succeeded if force_full_omni_ci_rerun else rerun_count > 0
    )
    if rerun_count > 0:
        print(f"Triggered rerun for {rerun_count} workflows.")
    elif force_full_omni_ci_rerun:
        print("No eligible workflows were rerun.")
    else:
        print("No failed or skipped workflows found to rerun.")
    if action_succeeded and react_on_success:
        comment.create_reaction("+1")
    return action_succeeded


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
    # not listed in CI_PERMISSIONS.json. Tagging still requires explicit
    # CI_PERMISSIONS.json access.
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

    first_line = comment_body.split("\n")[0].strip()
    tokens = first_line.split()

    if first_line.startswith("/tag-run-ci-label"):
        tts_model_target, parse_error = parse_tts_model_target(tokens)
        if parse_error:
            print(parse_error)
            comment.create_reaction("confused")
            return
        handle_tag_run_ci(pr, comment, user_perms, tts_model_target=tts_model_target)

    elif first_line.startswith("/rerun-failed-ci"):
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    elif first_line.startswith("/tag-and-rerun-ci"):
        tts_model_target, parse_error = parse_tts_model_target(tokens)
        if parse_error:
            print(parse_error)
            comment.create_reaction("confused")
            return
        print(
            "Processing combined command: "
            f"/tag-and-rerun-ci (tts_model_target={tts_model_target})"
        )

        tagged = handle_tag_run_ci(
            pr,
            comment,
            user_perms,
            react_on_success=False,
            tts_model_target=tts_model_target,
        )

        if tagged:
            print("Waiting 5 seconds for label to propagate...")
            time.sleep(5)

        rerun = handle_rerun_failed_ci(
            repo,
            pr,
            comment,
            user_perms,
            react_on_success=False,
            force_full_omni_ci_rerun=tagged,
        )

        if rerun:
            comment.create_reaction("+1")
            print("Combined command processed successfully; reaction added.")
        elif tagged:
            comment.create_reaction("confused")
            print("Labels were updated, but Omni CI was not restarted.")
        else:
            print("Combined command finished, but no actions were taken.")

    else:
        print(f"Unknown or ignored command: {first_line}")


if __name__ == "__main__":
    main()
