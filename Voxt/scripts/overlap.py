#!/usr/bin/env python3
"""Count changed Voxt view and logic files in GitHub pull requests."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal, Protocol

MYGO_IMPORT_PATTERN = re.compile(
    r'(?m)^\s*(?:import\s+)?(?:[._A-Za-z][\w]*\s+)?"github\.com/egoist/mygo(?:/[^"]*)?"'
)

FileCategory = Literal["view", "logic", "other", "unknown"]


class CommandRunner(Protocol):
    def __call__(
        self, command: list[str], working_directory: Path
    ) -> subprocess.CompletedProcess[str]: ...


def run_command(
    command: list[str], working_directory: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=working_directory,
        capture_output=True,
        text=True,
        check=False,
    )


def find_repository_root(command_runner: CommandRunner) -> Path:
    command_result = command_runner(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    if command_result.returncode == 0:
        return Path(command_result.stdout.strip())
    else:
        raise RuntimeError(
            f"git rev-parse --show-toplevel failed: {command_result.stderr.strip()}"
        )


def classify_file(
    repository_root: Path, file_path: str, command_runner: CommandRunner
) -> FileCategory:
    if (
        file_path.endswith("_test.go")
        or file_path.endswith(".md")
        or file_path.startswith("Voxt/docs/")
        or file_path.startswith("Voxt/helper/")
        or file_path.startswith("Voxt/testdata/")
        or file_path.startswith("Voxt/internal/i18n/locales/")
    ):
        return "other"
    elif (
        file_path.startswith("Voxt/")
        and file_path.count("/") == 1
        and file_path.endswith(".go")
    ):
        try:
            source_text = (repository_root / file_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            command_result = command_runner(
                ["git", "show", f"origin/main:{file_path}"], repository_root
            )
            if command_result.returncode == 0:
                source_text = command_result.stdout
            else:
                return "unknown"
        if MYGO_IMPORT_PATTERN.search(source_text):
            return "view"
        else:
            return "logic"
    elif file_path.startswith("Voxt/internal/") and file_path.endswith(".go"):
        return "logic"
    else:
        return "other"


def classify_file_at_revision(
    repository_root: Path, file_path: str, revision: str, command_runner: CommandRunner
) -> FileCategory:
    if (
        file_path.endswith("_test.go")
        or file_path.endswith(".md")
        or file_path.startswith("Voxt/docs/")
        or file_path.startswith("Voxt/helper/")
        or file_path.startswith("Voxt/testdata/")
        or file_path.startswith("Voxt/internal/i18n/locales/")
    ):
        return "other"
    elif (
        file_path.startswith("Voxt/")
        and file_path.count("/") == 1
        and file_path.endswith(".go")
    ):
        command_result = command_runner(
            [
                "gh",
                "api",
                f"repos/:owner/:repo/contents/{file_path}?ref={revision}",
                "-H",
                "Accept: application/vnd.github.raw+json",
            ],
            repository_root,
        )
        if command_result.returncode != 0:
            return "unknown"
        if MYGO_IMPORT_PATTERN.search(command_result.stdout):
            return "view"
        else:
            return "logic"
    elif file_path.startswith("Voxt/internal/") and file_path.endswith(".go"):
        return "logic"
    else:
        return "other"


def count_pull_request_files(
    pull_request_number: int,
    repository_root: Path,
    command_runner: CommandRunner,
) -> dict[str, int]:
    command_result = command_runner(
        [
            "gh",
            "pr",
            "view",
            str(pull_request_number),
            "--json",
            "changedFiles,headRefOid,baseRefOid",
        ],
        repository_root,
    )
    if command_result.returncode != 0:
        raise RuntimeError(
            f"gh pr view {pull_request_number} failed: {command_result.stderr.strip()}"
        )
    else:
        pass

    category_counts = {"view": 0, "logic": 0, "other": 0, "unknown": 0}
    pull_request_payload = json.loads(command_result.stdout)
    file_entries = []
    page = 1
    while True:
        files_result = command_runner(
            [
                "gh",
                "api",
                f"repos/:owner/:repo/pulls/{pull_request_number}/files?per_page=100&page={page}",
            ],
            repository_root,
        )
        if files_result.returncode != 0:
            raise RuntimeError(
                f"gh api pulls/{pull_request_number}/files failed: {files_result.stderr.strip()}"
            )
        page_entries = json.loads(files_result.stdout)
        file_entries.extend(page_entries)
        if len(page_entries) < 100:
            break
        page += 1

    if len(file_entries) != pull_request_payload["changedFiles"]:
        raise RuntimeError(
            f"PR {pull_request_number} returned {len(file_entries)} files, "
            f"want {pull_request_payload['changedFiles']}; refusing an incomplete count"
        )

    for file_entry in file_entries:
        revision = pull_request_payload["headRefOid"]
        if file_entry["status"] == "removed":
            revision = pull_request_payload["baseRefOid"]
        file_category = classify_file_at_revision(
            repository_root, file_entry["filename"], revision, command_runner
        )
        category_counts[file_category] += 1
    return category_counts


def main(argument_list: list[str] | None = None) -> int:
    argument_parser = argparse.ArgumentParser(
        description="Count changed Voxt view and logic Go files for each GitHub PR.",
        epilog="Example: python3 Voxt/scripts/overlap.py 3949 3997",
    )
    argument_parser.add_argument(
        "pull_request_numbers",
        metavar="PR",
        type=int,
        nargs="+",
        help="GitHub pull request number",
    )
    arguments = argument_parser.parse_args(argument_list)

    try:
        repository_root = find_repository_root(run_command)
        print("PR\tVIEW\tLOGIC\tOTHER\tUNKNOWN")
        for pull_request_number in arguments.pull_request_numbers:
            category_counts = count_pull_request_files(
                pull_request_number, repository_root, run_command
            )
            print(
                f"{pull_request_number}\t{category_counts['view']}\t"
                f"{category_counts['logic']}\t{category_counts['other']}\t"
                f"{category_counts['unknown']}"
            )
    except (json.JSONDecodeError, OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
