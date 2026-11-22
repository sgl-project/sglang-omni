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
    has_dynamic_fixture_settings,
    referenced_names,
)
from scripts.docs_review.unit_test_imports import imported_test_helpers

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


def fixture_module_limitations(file_path: str, module: ast.Module) -> list[str]:
    limitations: list[str] = []
    for statement in module.body:
        if isinstance(
            statement, (ast.Assign, ast.AnnAssign)
        ) and "pytest_plugins" in ast.unparse(statement):
            limitations.append(
                f"{file_path}: pytest plugin dependencies are unresolved"
            )
        elif (
            isinstance(statement, ast.ImportFrom)
            and any(alias.name == "*" for alias in statement.names)
            and (
                statement.level
                or (statement.module or "").split(".")[0]
                in {
                    "tests",
                    "sglang_omni",
                    "sglang_omni_router",
                    "sglang_omni_mlx",
                    "examples",
                }
            )
        ):
            limitations.append(
                f"{file_path}: wildcard test dependencies are unresolved"
            )
        else:
            pass
    return limitations


def module_fixtures(
    repository: Path,
    head_commit_sha: str,
    file_path: str,
    module: ast.Module,
    module_cache: dict[str, ast.Module | None],
) -> tuple[dict[str, tuple[str, ast.Module, TestFunction]], list[str]]:
    imports = [
        statement for statement in module.body if isinstance(statement, ast.ImportFrom)
    ]
    imported, import_limitations = imported_test_helpers(
        repository,
        head_commit_sha,
        file_path,
        ast.Module(body=imports, type_ignores=[]),
        {
            alias.asname or alias.name
            for statement in imports
            for alias in statement.names
        },
        module_cache,
        resolving_fixtures=True,
    )
    bindings = imported | {
        name: (file_path, module, definition)
        for name, definition in functions(module).items()
    }
    visible: dict[str, tuple[str, ast.Module, TestFunction]] = {}
    limitations = fixture_module_limitations(file_path, module)
    if PurePosixPath(file_path).name == "conftest.py":
        limitations.extend(import_limitations)
    else:
        pass
    for binding_name, (source_path, source_module, fixture) in bindings.items():
        if has_dynamic_fixture_settings(fixture):
            limitations.append(
                f"{source_path}:{fixture.lineno}: dynamic fixture settings could not be resolved statically"
            )
        else:
            pass
        fixture_name, _autouse = fixture_settings(fixture, binding_name=binding_name)
        if fixture_name:
            visible[fixture_name] = (source_path, source_module, fixture)
        else:
            pass
    return visible, limitations


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
            conftest_fixtures, conftest_limitations = module_fixtures(
                repository, head_commit_sha, conftest, conftest_module, module_cache
            )
            visible.update(conftest_fixtures)
            limitations.extend(conftest_limitations)
        else:
            if git_show(repository, head_commit_sha, conftest) is not None:
                limitations.append(f"{conftest}: fixture source could not be parsed")
            else:
                pass
    test_fixtures, test_limitations = module_fixtures(
        repository, head_commit_sha, test_path, module, module_cache
    )
    visible.update(test_fixtures)
    limitations.extend(test_limitations)
    class_helpers: dict[str, TestFunction] = {}
    for test_class in ast.walk(module):
        if (
            not isinstance(test_class, ast.ClassDef)
            or definition not in test_class.body
        ):
            continue
        else:
            pass
        if test_class.decorator_list or any(
            isinstance(member, (ast.Assign, ast.AnnAssign))
            for member in test_class.body
        ):
            limitations.append(
                f"{test_path}:{test_class.lineno}: class-level test dependencies require review"
            )
        else:
            pass
        for member in test_class.body:
            if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            else:
                pass
            fixture_name, _autouse = fixture_settings(member)
            if has_dynamic_fixture_settings(member):
                limitations.append(
                    f"{test_path}:{member.lineno}: dynamic fixture settings could not be resolved statically"
                )
            else:
                pass
            if fixture_name:
                visible[fixture_name] = (test_path, module, member)
            elif not member.name.startswith("test_"):
                class_helpers[member.name] = member
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
        contexts.append((file_path, current_module, names | {current.name}))
        limitations.extend(fixture_module_limitations(file_path, current_module))
        for call in ast.walk(current):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in {"getfixturevalue", "usefixtures"}
            ):
                continue
            else:
                pass
            fixture_arguments = (
                call.args[:1] if call.func.attr == "getfixturevalue" else call.args
            )
            if (call.func.attr == "getfixturevalue" and not fixture_arguments) or any(
                not isinstance(argument, ast.Constant)
                or not isinstance(argument.value, str)
                for argument in fixture_arguments
            ):
                limitations.append(
                    f"{file_path}:{call.lineno}: dynamic fixture lookup cannot be resolved statically"
                )
            else:
                for argument in fixture_arguments:
                    assert isinstance(argument, ast.Constant)
                    fixture_name = argument.value
                    if fixture_name in visible:
                        fixture_path, fixture_module, fixture = visible[fixture_name]
                        pending.append((fixture_path, fixture_module, fixture, True))
                    elif fixture_name not in PYTEST_FIXTURES:
                        limitations.append(
                            f"fixture {fixture_name} could not be resolved statically"
                        )
                    else:
                        pass
        imported, import_limitations = imported_test_helpers(
            repository, head_commit_sha, file_path, current_module, names, module_cache
        )
        limitations.extend(import_limitations)
        for imported_path, imported_module, helper in imported.values():
            pending.append((imported_path, imported_module, helper, False))
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
        for helper_class in current_module.body:
            if isinstance(helper_class, ast.ClassDef) and helper_class.name in names:
                pending.extend(
                    (file_path, current_module, member, False)
                    for member in helper_class.body
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
            else:
                pass
        for node in ast.walk(current):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"self", "cls"}
                and node.attr in class_helpers
            ):
                pending.append((test_path, module, class_helpers[node.attr], False))
            else:
                pass
        for name in sorted(names & visible.keys()):
            fixture_path, fixture_module, fixture = visible[name]
            pending.append((fixture_path, fixture_module, fixture, True))
    return contexts, list(dict.fromkeys(limitations))
