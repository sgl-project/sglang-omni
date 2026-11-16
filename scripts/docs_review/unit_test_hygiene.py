# SPDX-License-Identifier: Apache-2.0
"""Deterministic unit-test hygiene notes that need no model call."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from scripts.docs_review.models import UnitTestCase
from scripts.docs_review.snapshot import git_show, require_git
from scripts.docs_review.unit_tests import CONTRACT_MODULES

INTERNAL_MOCK_RULE_ID = "UNIT_TEST_INTERNAL_MOCK"
TEST_VOLUME_RULE_ID = "UNIT_TEST_VOLUME"
TEST_VOLUME_RATIO = 2
TEST_VOLUME_MINIMUM_LINES = 100
TEST_VOLUME_JUSTIFICATION = re.compile(r"^\s*Test volume:\s*\S", re.MULTILINE)
INTERNAL_ROOTS = CONTRACT_MODULES | {"scripts"}
BOUNDARY_NAMES = {
    "aiohttp",
    "builtins",
    "datetime",
    "http",
    "httpx",
    "os",
    "pathlib",
    "random",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "tempfile",
    "time",
    "urllib",
    "uuid",
}
RENAME_DESTINATION = re.compile(r"\{(.*) => (.*)\}")
ADDED_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)


def dotted_name(expression: ast.expr) -> str:
    if isinstance(expression, ast.Name):
        return expression.id
    elif isinstance(expression, ast.Attribute):
        owner = dotted_name(expression.value)
        return f"{owner}.{expression.attr}" if owner else ""
    else:
        return ""


def imported_names(module: ast.Module) -> dict[str, str]:
    names: dict[str, str] = {}
    for statement in ast.walk(module):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.asname is None:
                    root = alias.name.split(".")[0]
                    names[root] = root
                else:
                    names[alias.asname] = alias.name
        elif isinstance(statement, ast.ImportFrom) and statement.module is not None:
            for alias in statement.names:
                names[alias.asname or alias.name] = f"{statement.module}.{alias.name}"
        else:
            pass
    return names


def resolve_name(expression: ast.expr, names: dict[str, str]) -> str:
    local_name = dotted_name(expression)
    root, separator, rest = local_name.partition(".")
    if root in names:
        return f"{names[root]}{separator}{rest}"
    else:
        return local_name


def is_string_patch(callee: str) -> bool:
    return callee in {"patch", "mocker.patch"} or callee.endswith(".mock.patch")


def is_object_patch(callee: str) -> bool:
    return callee == "patch.object" or callee.endswith(".patch.object")


def patch_target(call: ast.Call, names: dict[str, str]) -> str:
    callee = resolve_name(call.func, names)
    arguments = call.args
    if (
        callee == "monkeypatch.setattr"
        and arguments
        and isinstance(arguments[0], ast.Constant)
        and isinstance(arguments[0].value, str)
    ):
        return arguments[0].value
    elif (
        (callee == "monkeypatch.setattr" or is_object_patch(callee))
        and len(arguments) >= 2
        and isinstance(arguments[1], ast.Constant)
        and isinstance(arguments[1].value, str)
    ):
        owner = resolve_name(arguments[0], names)
        return f"{owner}.{arguments[1].value}" if owner else ""
    elif (
        is_string_patch(callee)
        and arguments
        and isinstance(arguments[0], ast.Constant)
        and isinstance(arguments[0].value, str)
    ):
        return arguments[0].value
    else:
        return ""


def is_internal_target(target: str) -> bool:
    parts = [part for part in target.split(".") if part]
    if not parts or parts[0] not in INTERNAL_ROOTS:
        return False
    else:
        return not any(part in BOUNDARY_NAMES for part in parts[1:])


def added_line_numbers(
    repository: Path, merge_base: str, head_commit_sha: str, file_path: str
) -> set[int]:
    diff = require_git(
        repository,
        ["diff", "-U0", merge_base, head_commit_sha, "--", file_path],
    )
    added: set[int] = set()
    for match in ADDED_HUNK.finditer(diff):
        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count == 0:
            continue
        else:
            added.update(range(start, start + count))
    return added


def internal_mock_notes(
    repository: Path,
    merge_base: str,
    head_commit_sha: str,
    cases: list[UnitTestCase],
) -> list[str]:
    notes: list[str] = []
    for file_path in dict.fromkeys(case.file_path for case in cases):
        source = git_show(repository, head_commit_sha, file_path)
        if source is None:
            continue
        else:
            pass
        try:
            module = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        else:
            pass
        names = imported_names(module)
        added = added_line_numbers(repository, merge_base, head_commit_sha, file_path)
        targets: set[tuple[int, str]] = set()
        for call in ast.walk(module):
            if isinstance(call, ast.Call) and call.lineno in added:
                target = patch_target(call, names)
                if is_internal_target(target):
                    targets.add((call.lineno, target))
                else:
                    pass
            else:
                pass
        notes.extend(
            f"{INTERNAL_MOCK_RULE_ID}: {file_path}:{line} patches `{target}`, "
            "which is repository code. Mock network, subprocess, time, or "
            "filesystem boundaries and assert the caller-visible result."
            for line, target in sorted(targets)
        )
    return notes


def renamed_destination(file_path: str) -> str:
    rename = RENAME_DESTINATION.search(file_path)
    if rename is not None:
        return file_path[: rename.start()] + rename.group(2) + file_path[rename.end() :]
    elif " => " in file_path:
        return file_path.split(" => ", 1)[1]
    else:
        return file_path


def unit_test_volume_notes(
    repository: Path, merge_base: str, head_commit_sha: str, pull_request_text: str
) -> list[str]:
    numstat = require_git(
        repository, ["diff", "--numstat", merge_base, head_commit_sha]
    )
    test_lines = 0
    product_lines = 0
    for line in numstat.splitlines():
        added, _deleted, file_path = line.split("\t", 2)
        destination = renamed_destination(file_path)
        if added == "-" or not destination.endswith(".py"):
            continue
        elif destination.startswith("tests/unit_test/"):
            test_lines += int(added)
        else:
            product_lines += int(added)
    if (
        test_lines >= TEST_VOLUME_MINIMUM_LINES
        and test_lines > TEST_VOLUME_RATIO * product_lines
        and not TEST_VOLUME_JUSTIFICATION.search(pull_request_text)
    ):
        return [
            f"{TEST_VOLUME_RULE_ID}: this pull request adds {test_lines} Python "
            f"lines under tests/unit_test and {product_lines} Python product "
            "lines. Remove tests that do not guard a caller-visible regression, "
            "or explain the volume in a `Test volume:` line of the pull request "
            "description."
        ]
    else:
        return []
