# SPDX-License-Identifier: Apache-2.0
"""Resolve static pytest fixture context for selected test cases."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

from scripts.docs_review.snapshot import git_show
from scripts.docs_review.unit_test_ast import (
    UNIT_TEST_ROOT,
    TestFunction,
    fixture_settings,
    functions,
    referenced_names,
)

PYTEST_FIXTURES = {
    "cache",
    "capfd",
    "capfdbinary",
    "caplog",
    "capsys",
    "capsysbinary",
    "doctest_namespace",
    "monkeypatch",
    "mocker",
    "pytestconfig",
    "record_property",
    "record_testsuite_property",
    "record_xml_attribute",
    "recwarn",
    "request",
    "tmp_path",
    "tmp_path_factory",
    "tmpdir",
    "tmpdir_factory",
}


def case_dependencies(
    repository: Path,
    head_commit_sha: str,
    test_path: str,
    module: ast.Module,
    definition: TestFunction,
    module_cache: dict[str, ast.Module | None],
) -> tuple[list[tuple[str, ast.Module, set[str]]], list[str]]:
    visible: dict[str, tuple[str, ast.Module, TestFunction]] = {}
    limitations: list[str] = []
    parent = PurePosixPath(test_path).parent
    for directory in reversed([parent, *parent.parents]):
        conftest = (directory / "conftest.py").as_posix()
        if not conftest.startswith(UNIT_TEST_ROOT) and conftest not in {
            "conftest.py",
            "tests/conftest.py",
        }:
            continue
        else:
            pass
        if conftest not in module_cache:
            source = git_show(repository, head_commit_sha, conftest)
            if source is None:
                module_cache[conftest] = None
            else:
                try:
                    module_cache[conftest] = ast.parse(source)
                except (SyntaxError, ValueError):
                    module_cache[conftest] = None
                    limitations.append(
                        f"{conftest}: fixture source could not be parsed"
                    )
                else:
                    pass
        else:
            pass
        conftest_module = module_cache[conftest]
        if conftest_module is not None:
            for fixture in functions(conftest_module).values():
                fixture_name, _autouse = fixture_settings(fixture)
                if fixture_name:
                    visible[fixture_name] = (conftest, conftest_module, fixture)
                else:
                    pass
        else:
            pass
    for fixture in functions(module).values():
        fixture_name, _autouse = fixture_settings(fixture)
        if fixture_name:
            visible[fixture_name] = (test_path, module, fixture)
        else:
            pass

    parameters: set[str] = set()
    for decorator in definition.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
            and decorator.args
        ):
            names = decorator.args[0]
            if isinstance(names, ast.Constant) and isinstance(names.value, str):
                parameters.update(part.strip() for part in names.value.split(","))
            elif isinstance(names, (ast.Tuple, ast.List)) and all(
                isinstance(part, ast.Constant) and isinstance(part.value, str)
                for part in names.elts
            ):
                parameters.update(part.value for part in names.elts)
            else:
                limitations.append(
                    "parametrized test inputs could not be resolved statically"
                )
        else:
            pass

    pending: list[tuple[str, ast.Module, TestFunction, bool]] = [
        (test_path, module, definition, True)
    ]
    pending.extend(
        (file_path, fixture_module, fixture, True)
        for file_path, fixture_module, fixture in visible.values()
        if fixture_settings(fixture)[1]
    )
    seen: set[tuple[str, str]] = set()
    contexts: list[tuple[str, ast.Module, set[str]]] = []
    while pending:
        file_path, current_module, current, is_fixture = pending.pop()
        key = (file_path, current.name)
        if key in seen:
            continue
        else:
            seen.add(key)
        names = referenced_names(current)
        contexts.append((file_path, current_module, names))
        if is_fixture:
            required = (
                {
                    argument.arg
                    for argument in [*current.args.args, *current.args.kwonlyargs]
                }
                - {"self", "cls"}
                - parameters
            )
            for fixture_name in sorted(required - PYTEST_FIXTURES):
                if fixture_name in visible:
                    fixture_path, fixture_module, fixture = visible[fixture_name]
                    pending.append((fixture_path, fixture_module, fixture, True))
                else:
                    limitations.append(
                        f"fixture {fixture_name} could not be resolved statically"
                    )
        else:
            pass
        for name, helper in functions(current_module).items():
            if name in names and name != current.name:
                pending.append(
                    (
                        file_path,
                        current_module,
                        helper,
                        bool(fixture_settings(helper)[0]),
                    )
                )
            else:
                pass
        for name in sorted(names & visible.keys()):
            fixture_path, fixture_module, fixture = visible[name]
            pending.append((fixture_path, fixture_module, fixture, True))
    return contexts, list(dict.fromkeys(limitations))
