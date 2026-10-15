import json
import os
import sys

from github import Auth, Github

PERMISSIONS_FILE_PATH = ".github/CI_PERMISSIONS.json"
DISABLED_LABEL_COMMANDS = ("/tag-run-ci-label", "/tag-and-rerun-ci")
DISABLED_LABEL_COMMAND_REPLY = (
    "This command is disabled while it is redesigned for the new CI labels. "
    "Add `run-all-ci`, or the stage and model labels, to the pull request "
    "directly; see the CI section of the pull request template."
)


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

    first_line = comment_body.split("\n")[0].strip()

    if first_line.startswith(DISABLED_LABEL_COMMANDS):
        print(f"Disabled command: {first_line}")
        pr.create_issue_comment(DISABLED_LABEL_COMMAND_REPLY)
        comment.create_reaction("confused")

    elif first_line.startswith("/rerun-failed-ci"):
        handle_rerun_failed_ci(repo, pr, comment, user_perms)

    else:
        print(f"Unknown or ignored command: {first_line}")


if __name__ == "__main__":
    main()
