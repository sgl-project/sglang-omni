# SPDX-License-Identifier: Apache-2.0
"""Static Python test definitions and fixture references."""

from __future__ import annotations

import ast
from pathlib import PurePosixPath

UNIT_TEST_ROOT = "tests/unit_test/"
TestFunction = ast.FunctionDef | ast.AsyncFunctionDef


def is_unit_test_file(file_path: str) -> bool:
    name = PurePosixPath(file_path).name
    return (
        file_path.startswith(UNIT_TEST_ROOT)
        and name.endswith(".py")
        and (name.startswith("test_") or name.endswith("_test.py"))
    )


def test_definitions(
    suite: ast.Module | ast.ClassDef, prefix: str = ""
) -> dict[str, TestFunction]:
    definitions: dict[str, TestFunction] = {}
    for statement in suite.body:
        if isinstance(statement, ast.ClassDef) and (
            statement.name.startswith("Test")
            or any(
                ast.unparse(base) in {"TestCase", "unittest.TestCase"}
                for base in statement.bases
            )
        ):
            definitions.update(
                test_definitions(statement, f"{prefix}{statement.name}.")
            )
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            statement.name.startswith("test_")
        ):
            definitions[f"{prefix}{statement.name}"] = statement
        else:
            pass
    return definitions


def case_start(definition: TestFunction) -> int:
    return min(
        [
            definition.lineno,
            *(decorator.lineno for decorator in definition.decorator_list),
        ]
    )


def class_test_metadata(module: ast.Module) -> dict[str, tuple[str, ...]]:
    metadata: dict[str, tuple[str, ...]] = {}
    for statement in module.body:
        if not isinstance(statement, ast.ClassDef) or not test_definitions(statement):
            continue
        else:
            pass
        class_nodes = [
            *statement.decorator_list,
            *statement.bases,
            *statement.keywords,
            *[
                member
                for member in statement.body
                if not isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            ],
        ]
        if class_nodes:
            metadata[statement.name] = tuple(ast.dump(node) for node in class_nodes)
        else:
            pass
    return metadata


def functions(module: ast.Module) -> dict[str, TestFunction]:
    return {
        statement.name: statement
        for statement in module.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def fixture_settings(
    definition: TestFunction, *, binding_name: str = ""
) -> tuple[str, bool]:
    for decorator in definition.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if (
            isinstance(target, ast.Name)
            and target.id == "fixture"
            or (isinstance(target, ast.Attribute) and target.attr == "fixture")
        ):
            fixture_name = binding_name or definition.name
            autouse = False
            if isinstance(decorator, ast.Call):
                for keyword in decorator.keywords:
                    if keyword.arg == "name" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        if isinstance(keyword.value.value, str):
                            fixture_name = keyword.value.value
                        else:
                            pass
                    elif keyword.arg == "autouse" and isinstance(
                        keyword.value, ast.Constant
                    ):
                        autouse = keyword.value.value is True
                    else:
                        pass
            else:
                pass
            return fixture_name, autouse
        else:
            pass
    return "", False


def has_dynamic_fixture_settings(definition: TestFunction) -> bool:
    if not fixture_settings(definition)[0]:
        return False
    else:
        pass
    for decorator in definition.decorator_list:
        if isinstance(decorator, ast.Call) and any(
            keyword.arg in {"autouse", "name"}
            and (
                not isinstance(keyword.value, ast.Constant)
                or (
                    keyword.arg == "autouse"
                    and not isinstance(keyword.value.value, bool)
                )
                or (keyword.arg == "name" and not isinstance(keyword.value.value, str))
            )
            for keyword in decorator.keywords
        ):
            return True
        else:
            pass
    return False


def referenced_names(definition: TestFunction) -> set[str]:
    names = {node.id for node in ast.walk(definition) if isinstance(node, ast.Name)}
    names.update(argument.arg for argument in definition.args.args)
    names.update(argument.arg for argument in definition.args.kwonlyargs)
    for node in ast.walk(definition):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"usefixtures", "getfixturevalue"}
        ):
            names.update(
                argument.value
                for argument in node.args
                if isinstance(argument, ast.Constant)
                and isinstance(argument.value, str)
            )
        else:
            pass
    return names
