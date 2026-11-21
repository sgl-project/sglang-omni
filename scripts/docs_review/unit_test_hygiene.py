# SPDX-License-Identifier: Apache-2.0
"""Deterministic unit-test hygiene notes that need no model call."""

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

from scripts.docs_review.snapshot import git_show, require_git
from scripts.docs_review.unit_tests import CONTRACT_MODULES, changed_test_paths

INTERNAL_MOCK_RULE_ID = "UNIT_TEST_INTERNAL_MOCK"
TEST_VOLUME_RULE_ID = "UNIT_TEST_VOLUME"
TEST_VOLUME_RATIO = 2
TEST_VOLUME_MINIMUM_LINES = 100
TEST_VOLUME_JUSTIFICATION = re.compile(r"^[ \t]*Test volume:[ \t]*\S", re.MULTILINE)
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
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    first = call.args[0] if call.args else keywords.get("target")
    second = call.args[1] if len(call.args) > 1 else keywords.get("attribute")
    if callee == "monkeypatch.setattr" and second is None:
        second = keywords.get("name")
    else:
        pass
    if (
        callee == "monkeypatch.setattr"
        and isinstance(first, ast.Constant)
        and isinstance(first.value, str)
    ):
        return first.value
    elif (
        (callee == "monkeypatch.setattr" or is_object_patch(callee))
        and first is not None
        and isinstance(second, ast.Constant)
        and isinstance(second.value, str)
    ):
        owner = resolve_name(first, names)
        return f"{owner}.{second.value}" if owner else ""
    elif (
        is_string_patch(callee)
        and isinstance(first, ast.Constant)
        and isinstance(first.value, str)
    ):
        return first.value
    else:
        return ""


def is_internal_target(target: str) -> bool:
    parts = [part for part in target.split(".") if part]
    return bool(parts and parts[0] in INTERNAL_ROOTS)


def is_boundary_target(repository: Path, head_commit_sha: str, target: str) -> bool:
    parts = target.split(".")
    for index, name in enumerate(parts[1:], 1):
        child = "/".join(parts[: index + 1])
        if (
            git_show(repository, head_commit_sha, f"{child}.py") is not None
            or git_show(repository, head_commit_sha, f"{child}/__init__.py") is not None
        ):
            continue
        else:
            pass
        owner = "/".join(parts[:index])
        source = git_show(repository, head_commit_sha, f"{owner}.py")
        if source is None:
            source = git_show(repository, head_commit_sha, f"{owner}/__init__.py")
        else:
            pass
        if source is None:
            continue
        else:
            pass
        try:
            imports = imported_names(ast.parse(source))
        except (SyntaxError, ValueError):
            continue
        else:
            pass
        if imports.get(name, "").split(".")[0] in BOUNDARY_NAMES:
            return True
        else:
            pass
    return False


def internal_mock_notes(
    repository: Path,
    merge_base: str,
    head_commit_sha: str,
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    limitations: list[str] = []
    for previous_path, file_path in changed_test_paths(
        repository, merge_base, head_commit_sha
    ):
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
        previous_source = (
            git_show(repository, merge_base, previous_path)
            if previous_path is not None
            else ""
        )
        try:
            previous_module = ast.parse(previous_source or "")
        except (SyntaxError, ValueError):
            continue
        else:
            pass
        previous_names = imported_names(previous_module)
        previous_calls = Counter(
            (patch_target(call, previous_names), ast.dump(call))
            for call in ast.walk(previous_module)
            if isinstance(call, ast.Call)
        )
        targets: set[tuple[int, str]] = set()
        for call in ast.walk(module):
            if not isinstance(call, ast.Call):
                continue
            else:
                pass
            target = patch_target(call, names)
            signature = (target, ast.dump(call))
            if previous_calls[signature]:
                previous_calls[signature] -= 1
            elif not target and (
                resolve_name(call.func, names) == "monkeypatch.setattr"
                or is_string_patch(resolve_name(call.func, names))
                or is_object_patch(resolve_name(call.func, names))
            ):
                limitations.append(
                    f"{file_path}:{call.lineno}: mock target could not be resolved statically"
                )
            elif is_internal_target(target) and not is_boundary_target(
                repository, head_commit_sha, target
            ):
                targets.add((call.lineno, target))
            else:
                pass
        notes.extend(
            f"{INTERNAL_MOCK_RULE_ID}: {file_path}:{line} patches `{target}`, "
            "which is repository code. Mock network, subprocess, time, or "
            "filesystem boundaries and assert the caller-visible result."
            for line, target in sorted(targets)
        )
    return notes, limitations


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
        elif destination.split("/", 1)[0] in INTERNAL_ROOTS:
            product_lines += int(added)
        else:
            pass
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
