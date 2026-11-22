# SPDX-License-Identifier: Apache-2.0
"""Keep one pull request comment with the /tag-and-rerun-ci targets that test what it changes."""

from __future__ import annotations

import json
import os
import subprocess

from resolve_ci_labels import (
    MODEL_DIRECTORY_ROOT,
    StageSuggestion,
    suggest_shared_code_targets,
    suggest_stage_targets,
)

COMMENT_MARKER = "<!-- ci-stage-suggestion -->"


def call_github(arguments: list[str]) -> str:
    return subprocess.run(
        ["gh", "api", *arguments], check=True, capture_output=True, text=True
    ).stdout


def build_comment_body(
    suggestions: list[StageSuggestion], shared_code_targets: list[str]
) -> str:
    lines = [COMMENT_MARKER]
    command_suggestions = [
        suggestion for suggestion in suggestions if suggestion.is_in_command
    ]
    if command_suggestions:
        command_targets = dict.fromkeys(
            target
            for suggestion in command_suggestions
            for target in suggestion.targets
        )
        lines += [
            "This pull request changes model code. A maintainer can test it on GPU with:",
            "",
            f"`/tag-and-rerun-ci {' '.join(command_targets)}`",
            "",
            *(
                f"- `{' '.join(suggestion.targets)}` tests `{MODEL_DIRECTORY_ROOT}{suggestion.model_directory}/`"
                for suggestion in command_suggestions
            ),
            "",
        ]
    else:
        pass
    lines += [
        f"Note: a stage runs only one model per run, and this pull request changes several models of one stage: {', '.join(f'`{model_directory}`' for model_directory in suggestion.stage_model_directories)}. `/tag-and-rerun-ci {' '.join(suggestion.targets)}` runs that stage on `{suggestion.model_directory}`."
        for suggestion in suggestions
        if not suggestion.is_in_command
    ]
    if shared_code_targets:
        quoted_targets = [f"`{target}`" for target in shared_code_targets]
        lines.append(
            f"This pull request changes code that every model shares. A maintainer can test it on GPU with `/tag-and-rerun-ci` followed by {', '.join(quoted_targets[:-1])} or {quoted_targets[-1]}, whichever it can affect. Each runs that stage on one of its models."
        )
    else:
        pass
    if len(lines) == 1:
        lines.append(
            "All set. The code this pull request changes needs no further CI targets."
        )
    else:
        pass
    return "\n".join(lines).rstrip()


def update_stage_suggestion(repository: str, pull_request_number: int) -> None:
    """Create the comment when there is something to run, and rewrite it only when its text changes."""
    pull_request = json.loads(
        call_github([f"repos/{repository}/pulls/{pull_request_number}"])
    )
    if pull_request["draft"]:
        return
    else:
        pass
    changed_paths = call_github(
        [
            "--paginate",
            f"repos/{repository}/pulls/{pull_request_number}/files",
            "--jq",
            ".[] | .filename, (.previous_filename // empty)",
        ]
    ).splitlines()
    comments = [
        json.loads(line)
        for line in call_github(
            [
                "--paginate",
                f"repos/{repository}/issues/{pull_request_number}/comments",
                "--jq",
                ".[] | {id, body, user_type: .user.type} | @json",
            ]
        ).splitlines()
    ]
    labels = frozenset(label["name"] for label in pull_request["labels"])
    suggestions = suggest_stage_targets(changed_paths, labels)
    shared_code_targets = (
        [] if suggestions else suggest_shared_code_targets(changed_paths, labels)
    )
    body = build_comment_body(suggestions, shared_code_targets)
    # note (0xtoward): a person can paste the marker, so only a comment the bot wrote is ever rewritten.
    own_comments = [
        comment
        for comment in comments
        if comment["user_type"] == "Bot" and COMMENT_MARKER in comment["body"]
    ]
    if own_comments and own_comments[0]["body"] != body:
        call_github(
            [
                "--method",
                "PATCH",
                f"repos/{repository}/issues/comments/{own_comments[0]['id']}",
                "-f",
                f"body={body}",
            ]
        )
    elif not own_comments and (suggestions or shared_code_targets):
        call_github(
            [
                "--method",
                "POST",
                f"repos/{repository}/issues/{pull_request_number}/comments",
                "-f",
                f"body={body}",
            ]
        )
    else:
        pass


if __name__ == "__main__":
    update_stage_suggestion(
        os.environ["GITHUB_REPOSITORY"], int(os.environ["PR_NUMBER"])
    )
