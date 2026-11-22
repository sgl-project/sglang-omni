# SPDX-License-Identifier: Apache-2.0
"""Resolve test helper imports from frozen Git revisions."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath

from scripts.docs_review.snapshot import git_show
from scripts.docs_review.unit_test_ast import TestFunction, functions


def read_test_module(
    repository: Path,
    head_commit_sha: str,
    module_path: str,
    module_cache: dict[str, ast.Module | None],
) -> tuple[str, ast.Module | None]:
    for source_path in (f"{module_path}.py", f"{module_path}/__init__.py"):
        cached_module = module_cache.get(source_path)
        if cached_module is not None:
            return source_path, cached_module
        else:
            pass
        source = git_show(repository, head_commit_sha, source_path)
        if source is None:
            continue
        elif source_path not in module_cache:
            try:
                module_cache[source_path] = ast.parse(source)
            except (SyntaxError, ValueError):
                module_cache[source_path] = None
            else:
                pass
        else:
            pass
        return source_path, module_cache[source_path]
    return "", None


def imported_test_helpers(
    repository: Path,
    head_commit_sha: str,
    source_path: str,
    module: ast.Module,
    referenced: set[str],
    module_cache: dict[str, ast.Module | None],
    *,
    resolving_fixtures: bool = False,
) -> tuple[dict[str, tuple[str, ast.Module, TestFunction]], list[str]]:
    helpers: dict[str, tuple[str, ast.Module, TestFunction]] = {}
    limitations: list[str] = []
    for statement in ast.walk(module):
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                local_name = alias.asname or alias.name.split(".")[0]
                if not (
                    (alias.name == "tests" or alias.name.startswith("tests."))
                    and local_name in referenced
                ):
                    continue
                else:
                    pass
                imported_path = alias.name.replace(".", "/")
                module_file, imported = read_test_module(
                    repository, head_commit_sha, imported_path, module_cache
                )
                if imported is None:
                    limitations.append(
                        f"{imported_path}: imported test helper is unavailable"
                    )
                    continue
                else:
                    pass
                prefix = alias.asname or alias.name
                members = {
                    ast.unparse(node)[len(prefix) + 1 :].split(".")[0]
                    for node in ast.walk(module)
                    if isinstance(node, ast.Attribute)
                    and ast.unparse(node).startswith(f"{prefix}.")
                }
                definitions = functions(imported)
                for name in members & definitions.keys():
                    helpers[f"{local_name}.{name}"] = (
                        module_file,
                        imported,
                        definitions[name],
                    )
                if not members & definitions.keys():
                    limitations.append(
                        f"{module_file}: imported test helper could not be resolved"
                    )
                else:
                    pass
            continue
        elif not isinstance(statement, ast.ImportFrom):
            continue
        else:
            pass
        if statement.level:
            package = list(PurePosixPath(source_path).parent.parts)
            package = package[: len(package) - statement.level + 1]
            imported_module = "/".join(
                [*package, *(statement.module or "").split(".")]
            ).rstrip("/")
        else:
            imported_module = (statement.module or "").replace(".", "/")
        if imported_module != "tests" and not imported_module.startswith("tests/"):
            continue
        else:
            pass
        aliases = [
            alias
            for alias in statement.names
            if (alias.asname or alias.name) in referenced
        ]
        if not aliases:
            continue
        else:
            pass
        module_file, imported = read_test_module(
            repository, head_commit_sha, imported_module, module_cache
        )
        if not module_file:
            limitations.append(
                f"{imported_module}: imported test helper is unavailable"
            )
            continue
        elif imported is None:
            limitations.append(
                f"{module_file}: imported test helper could not be parsed"
            )
            continue
        else:
            pass
        definitions = functions(imported)
        for alias in aliases:
            if alias.name in definitions:
                helpers[alias.asname or alias.name] = (
                    module_file,
                    imported,
                    definitions[alias.name],
                )
            elif resolving_fixtures and any(
                (
                    isinstance(declaration, ast.ClassDef)
                    and declaration.name == alias.name
                )
                or (
                    isinstance(declaration, ast.AnnAssign)
                    and isinstance(declaration.target, ast.Name)
                    and declaration.target.id == alias.name
                    and ast.unparse(declaration.annotation)
                    in {"TypeAlias", "typing.TypeAlias"}
                )
                for declaration in imported.body
            ):
                pass
            else:
                limitations.append(
                    f"{module_file}: imported helper {alias.name} could not be resolved"
                )
    return helpers, limitations
