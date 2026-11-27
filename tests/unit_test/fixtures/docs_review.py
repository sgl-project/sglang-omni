# SPDX-License-Identifier: Apache-2.0
"""Temporary repositories and budgets for documentation review tests."""

import subprocess
from pathlib import Path

from scripts.docs_review.models import BudgetConfig


def git_command(repository: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    else:
        return completed.stdout


def start_repository(root: Path) -> str:
    root.mkdir()
    hooks = root.parent / f"{root.name}-hooks"
    hooks.mkdir()
    git_command(root, ["init"])
    git_command(root, ["config", "user.email", "docs-review@example.com"])
    git_command(root, ["config", "user.name", "Docs Review Test"])
    git_command(root, ["config", "commit.gpgsign", "false"])
    git_command(root, ["config", "core.hooksPath", str(hooks)])
    (root / "README.md").write_text("Repository.\n", encoding="utf-8")
    return commit_all(root, "base")


def commit_all(repository: Path, message: str) -> str:
    git_command(repository, ["add", "."])
    git_command(repository, ["commit", "-m", message])
    return git_command(repository, ["rev-parse", "HEAD"]).strip()


def write_file(repository: Path, relative: str, content: str) -> None:
    path = repository / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def sample_budget() -> BudgetConfig:
    return BudgetConfig(
        max_cost_cents_per_call=10,
        max_cost_cents_per_pull_request=50,
        max_cost_cents_per_day=50,
        call_cost_cents=10,
    )
