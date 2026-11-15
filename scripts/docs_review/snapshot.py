# SPDX-License-Identifier: Apache-2.0
"""Read and cite files from frozen Git revisions."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.docs_review.models import ReviewBundle

MAX_COLLECTED_BYTES = 200_000
GIT_COMMAND_PREFIX = ["git", "-c", "core.hooksPath=/dev/null"]


def clip_content(content: str) -> tuple[str, bool]:
    encoded = content.encode()
    if len(encoded) > MAX_COLLECTED_BYTES:
        return encoded[:MAX_COLLECTED_BYTES].decode(errors="ignore"), True
    else:
        return content, False


def git_show(repository: Path, commit_sha: str, file_path: str) -> str | None:
    completed = run_git(repository, ["show", f"{commit_sha}:{file_path}"])
    if completed.returncode != 0:
        return None
    else:
        return completed.stdout


def require_git(repository: Path, arguments: list[str]) -> str:
    completed = run_git(repository, arguments)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "git command failed"
        raise RuntimeError(message)
    else:
        return completed.stdout


def run_git(repository: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    if arguments and arguments[0] == "diff":
        git_arguments = ["diff", "--no-ext-diff", "--no-textconv", *arguments[1:]]
    else:
        git_arguments = arguments
    return subprocess.run(
        [*GIT_COMMAND_PREFIX, "-C", str(repository), *git_arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def repository_identity(repository: Path) -> str:
    completed = run_git(repository, ["remote", "get-url", "origin"])
    if completed.returncode == 0:
        origin = completed.stdout.strip()
    else:
        origin = ""
    return f"{repository.resolve().as_posix()}\n{origin}"


def citation_matches(
    bundle: ReviewBundle, file_path: str, line_start: int, line_end: int, quote: str
) -> bool:
    if quote == "":
        return False
    else:
        pass
    for collected in bundle.files:
        if collected.file_path == file_path:
            lines = collected.content.splitlines()
            if line_start < 1 or line_end > len(lines) or line_end < line_start:
                return False
            else:
                excerpt = "\n".join(lines[line_start - 1 : line_end])
                return quote in excerpt
        else:
            pass
    return False
