# SPDX-License-Identifier: Apache-2.0
"""Find test cases affected by fixture and helper changes."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from scripts.docs_review.models import SourceCitation
from scripts.docs_review.snapshot import git_show, require_git
from scripts.docs_review.unit_test_ast import (
    UNIT_TEST_ROOT,
    TestFunction,
    case_start,
    class_test_metadata,
    fixture_settings,
    functions,
    has_dynamic_fixture_settings,
    is_unit_test_file,
    test_definitions,
)
from scripts.docs_review.unit_test_fixtures import case_dependencies


@dataclass(frozen=True, kw_only=True)
class ChangedHelper:
    file_path: str
    definition: TestFunction
    citation: SourceCitation


def changed_helpers(
    repository: Path,
    merge_base: str,
    head_commit_sha: str,
    changed_paths: list[tuple[str | None, str]],
) -> tuple[list[ChangedHelper], list[str]]:
    changed: list[ChangedHelper] = []
    limitations: list[str] = []
    for previous_path, head_path in changed_paths:
        head_source = git_show(repository, head_commit_sha, head_path)
        previous_source = (
            git_show(repository, merge_base, previous_path)
            if previous_path is not None
            else ""
        )
        if head_source is None or previous_source is None:
            limitations.append(
                f"{head_path}: test dependency revision could not be read"
            )
            continue
        else:
            pass
        try:
            head_module = ast.parse(head_source)
            previous_module = ast.parse(previous_source)
        except (SyntaxError, ValueError):
            limitations.append(f"{head_path}: test dependency could not be parsed")
            continue
        else:
            pass
        previous_functions = functions(previous_module)
        head_functions = functions(head_module)
        if previous_path is not None:
            previous_module_statements = [
                ast.dump(statement)
                for statement in previous_module.body
                if not isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            ]
            current_module_statements = [
                ast.dump(statement)
                for statement in head_module.body
                if not isinstance(
                    statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                )
            ]
            previous_tests = test_definitions(previous_module)
            current_tests = test_definitions(head_module)
            if class_test_metadata(previous_module) != class_test_metadata(head_module):
                limitations.append(
                    f"{head_path}: class-level test dependencies require review"
                )
            else:
                pass
            unchanged_tests = any(
                name in previous_tests
                and ast.dump(previous_tests[name]) == ast.dump(definition)
                for name, definition in current_tests.items()
            )
            if previous_module_statements != current_module_statements and (
                not is_unit_test_file(head_path) or unchanged_tests
            ):
                limitations.append(
                    f"{head_path}: module-level test dependencies changed and need review"
                )
            else:
                pass
            previous_classes = {
                (statement.name, member.name): ast.dump(member)
                for statement in previous_module.body
                if isinstance(statement, ast.ClassDef)
                for member in statement.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and fixture_settings(member)[0]
            }
            current_classes = {
                (statement.name, member.name): ast.dump(member)
                for statement in head_module.body
                if isinstance(statement, ast.ClassDef)
                for member in statement.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and fixture_settings(member)[0]
            }
            if previous_classes != current_classes:
                limitations.append(
                    f"{head_path}: class fixture dependencies require review"
                )
            else:
                pass
            previous_class_helpers = {
                (statement.name, member.name): ast.dump(member)
                for statement in previous_module.body
                if isinstance(statement, ast.ClassDef)
                for member in statement.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not member.name.startswith("test_")
                and not fixture_settings(member)[0]
            }
            current_class_helpers = {
                (statement.name, member.name): ast.dump(member)
                for statement in head_module.body
                if isinstance(statement, ast.ClassDef)
                for member in statement.body
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not member.name.startswith("test_")
                and not fixture_settings(member)[0]
            }
            if previous_class_helpers != current_class_helpers and (
                not is_unit_test_file(head_path) or unchanged_tests
            ):
                limitations.append(
                    f"{head_path}: class helper dependencies require review"
                )
            else:
                pass
        else:
            pass
        for name, definition in head_functions.items():
            if name.startswith("test_"):
                continue
            else:
                pass
            previous = previous_functions.get(name)
            if previous is not None and ast.dump(previous) == ast.dump(definition):
                continue
            else:
                pass
            if has_dynamic_fixture_settings(definition):
                limitations.append(
                    f"{head_path}:{definition.lineno}: dynamic fixture settings "
                    "could not be resolved statically"
                )
            else:
                pass
            assert definition.end_lineno is not None
            head_lines = head_source.splitlines()
            first_line = case_start(definition)
            if previous is not None and previous.end_lineno is not None:
                previous_lines = previous_source.splitlines()[
                    case_start(previous) - 1 : previous.end_lineno
                ]
                current_lines = head_lines[first_line - 1 : definition.end_lineno]
                changes = SequenceMatcher(
                    None, previous_lines, current_lines, autojunk=False
                ).get_opcodes()
                changed_lines = [
                    first_line + min(start, len(current_lines) - 1)
                    for tag, _, _, start, _ in changes
                    if tag != "equal"
                ]
                line_number = changed_lines[0] if changed_lines else first_line
            else:
                line_number = first_line
            changed.append(
                ChangedHelper(
                    file_path=head_path,
                    definition=definition,
                    citation=SourceCitation(
                        file_path=head_path,
                        line_start=line_number,
                        line_end=line_number,
                        quote=head_lines[line_number - 1].strip(),
                    ),
                )
            )
        removed = previous_functions.keys() - head_functions.keys()
        removed = {name for name in removed if not name.startswith("test_")}
        if removed:
            limitations.append(
                f"{head_path}: removed test dependencies could not be resolved: "
                + ", ".join(sorted(removed))
            )
        else:
            pass
        if any(
            isinstance(statement, (ast.Assign, ast.AnnAssign))
            and "pytest_plugins" in ast.unparse(statement)
            for statement in head_module.body
        ) and ast.dump(previous_module) != ast.dump(head_module):
            limitations.append(
                f"{head_path}: pytest plugin dependencies are unresolved"
            )
        else:
            pass
    return changed, limitations


def affected_test_cases(
    repository: Path,
    merge_base: str,
    head_commit_sha: str,
    changed_paths: list[tuple[str | None, str]],
) -> tuple[dict[tuple[str, str], list[SourceCitation]], list[str]]:
    helpers, limitations = changed_helpers(
        repository, merge_base, head_commit_sha, changed_paths
    )
    if not helpers:
        return {}, limitations
    else:
        pass
    files = require_git(
        repository,
        ["ls-tree", "-r", "--name-only", "-z", head_commit_sha, "--", UNIT_TEST_ROOT],
    ).split("\0")
    affected: dict[tuple[str, str], list[SourceCitation]] = {}
    module_cache: dict[str, ast.Module | None] = {}
    for test_path in filter(is_unit_test_file, files):
        source = git_show(repository, head_commit_sha, test_path)
        if source is None:
            limitations.append(f"{test_path}: affected test source could not be read")
            continue
        else:
            pass
        try:
            module = ast.parse(source)
        except (SyntaxError, ValueError):
            limitations.append(f"{test_path}: affected test source could not be parsed")
            continue
        else:
            pass
        for case_name, definition in test_definitions(module).items():
            contexts, case_limitations = case_dependencies(
                repository, head_commit_sha, test_path, module, definition, module_cache
            )
            citations = [
                helper.citation
                for helper in helpers
                if any(
                    helper.file_path == file_path and helper.definition.name in names
                    for file_path, _module, names in contexts
                )
            ]
            if citations:
                affected[(test_path, case_name)] = sorted(
                    {
                        citation.model_dump_json(): citation for citation in citations
                    }.values(),
                    key=lambda citation: (citation.file_path, citation.line_start),
                )
            else:
                pass
            limitations.extend(
                f"{test_path}:{definition.lineno}: {limitation}"
                for limitation in case_limitations
                if any(
                    marker in limitation
                    for marker in (
                        "dynamic fixture",
                        "imported test helper",
                        "imported helper",
                        "plugin dependencies",
                        "wildcard test dependencies",
                    )
                )
            )
    matched_citations = {
        citation.model_dump_json()
        for citations in affected.values()
        for citation in citations
    }
    for helper in helpers:
        if helper.citation.model_dump_json() not in matched_citations:
            limitations.append(
                f"{helper.file_path}:{helper.citation.line_start}: consumers of "
                f"{helper.definition.name} could not be resolved statically"
            )
        else:
            pass
    return affected, list(dict.fromkeys(limitations))
