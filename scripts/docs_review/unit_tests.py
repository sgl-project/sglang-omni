# SPDX-License-Identifier: Apache-2.0
"""Collect changed test cases and verify contract-review evidence."""

from __future__ import annotations

import ast
from difflib import SequenceMatcher
from pathlib import Path, PurePosixPath

from scripts.docs_review.models import CollectedFile, UnitTestCase
from scripts.docs_review.snapshot import clip_content, git_show, require_git
from scripts.docs_review.unit_test_ast import (
    case_start,
    is_unit_test_file,
    test_definitions,
)
from scripts.docs_review.unit_test_dependencies import affected_test_cases
from scripts.docs_review.unit_test_fixtures import case_dependencies

UNIT_TEST_POLICY_VERSION = "unit-tests-7"
UNIT_TEST_ROOT = "tests/unit_test/"
AUTOMATIC_CONTEXT_BYTES = 36_000
EXPLICIT_CONTEXT_BYTES = 16_000
CONTRACT_MODULES = {"sglang_omni", "sglang_omni_router", "sglang_omni_mlx", "examples"}
CONTEXT_SUFFIXES = {
    ".py",
    ".md",
    ".rst",
    ".rs",
    ".cpp",
    ".h",
    ".toml",
    ".json",
    ".yaml",
    ".yml",
}
TestFunction = ast.FunctionDef | ast.AsyncFunctionDef


def is_context_path(file_path: str) -> bool:
    path = PurePosixPath(file_path)
    return bool(
        path.parts
        and not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == file_path
        and path.suffix in CONTEXT_SUFFIXES
        and (
            path.parts[0] in CONTRACT_MODULES | {"docs"}
            or file_path.startswith(UNIT_TEST_ROOT)
        )
    )


def changed_test_paths(
    repository: Path, merge_base: str, head_commit_sha: str
) -> list[tuple[str | None, str]]:
    diff = require_git(
        repository,
        [
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=AMRTD",
            merge_base,
            head_commit_sha,
            "--",
            UNIT_TEST_ROOT,
            "tests/utils.py",
            "tests/utils/",
            "tests/conftest.py",
            "conftest.py",
        ],
    )
    entries = iter(diff.rstrip("\0").split("\0") if diff else [])
    paths: list[tuple[str | None, str]] = []
    for status in entries:
        previous_path = next(entries)
        if status.startswith("R"):
            head_path = next(entries)
        else:
            head_path = previous_path
        if head_path.endswith(".py"):
            paths.append((None if status == "A" else previous_path, head_path))
        else:
            pass
    return paths


def changed_case_ranges(
    before: TestFunction | None,
    after: TestFunction,
    before_lines: list[str],
    after_lines: list[str],
) -> list[tuple[int, int]]:
    after_start = case_start(after)
    assert after.end_lineno is not None
    if before is None:
        return [(after_start, after.end_lineno)]
    else:
        assert before.end_lineno is not None
        before_case = before_lines[case_start(before) - 1 : before.end_lineno]
        after_case = after_lines[after_start - 1 : after.end_lineno]
        changes = SequenceMatcher(None, before_case, after_case, autojunk=False)
        # A deletion has no added line, so cite the nearest surviving line in the case.
        return [
            (
                after_start + min(head_start, len(after_case) - 1),
                after_start + max(min(head_start, len(after_case) - 1), head_end - 1),
            )
            for tag, _, _, head_start, head_end in changes.get_opcodes()
            if tag != "equal"
        ]


def referenced_contract_modules(
    test_module: ast.Module, changed_names: set[str]
) -> list[str]:
    modules: list[str] = []
    for imported in ast.walk(test_module):
        if isinstance(imported, ast.ImportFrom) and imported.level == 0:
            if imported.module is not None and any(
                (alias.asname or alias.name) in changed_names
                for alias in imported.names
            ):
                modules.append(imported.module)
            else:
                pass
        elif isinstance(imported, ast.Import):
            modules.extend(
                alias.name
                for alias in imported.names
                if (alias.asname or alias.name.split(".")[0]) in changed_names
            )
        else:
            pass
    return [
        module
        for module in dict.fromkeys(modules)
        if module.split(".")[0] in CONTRACT_MODULES
    ]


def collect_unit_test_inputs(
    repository: Path, merge_base: str, head_commit_sha: str, include_paths: list[str]
) -> tuple[list[CollectedFile], list[UnitTestCase], list[str]]:
    files: dict[str, CollectedFile] = {}
    cases: list[UnitTestCase] = []
    limitations: list[str] = []
    contract_modules: list[str] = []
    changed_paths = changed_test_paths(repository, merge_base, head_commit_sha)
    affected, dependency_limitations = affected_test_cases(
        repository, merge_base, head_commit_sha, changed_paths
    )
    limitations.extend(dependency_limitations)
    candidate_paths = dict(
        (head_path, previous_path) for previous_path, head_path in changed_paths
    )
    for head_path, _case_name in affected:
        candidate_paths.setdefault(head_path, head_path)
    case_contract_modules: dict[tuple[str, str], list[str]] = {}
    fixture_module_cache: dict[str, ast.Module | None] = {}
    for head_path, previous_path in candidate_paths.items():
        head_source = git_show(repository, head_commit_sha, head_path)
        before_source = (
            git_show(repository, merge_base, previous_path)
            if previous_path is not None
            else ""
        )
        if head_source is None or before_source is None:
            limitations.append(f"{head_path}: test revision could not be read")
            continue
        else:
            pass
        content, truncated = clip_content(head_source)
        if truncated:
            limitations.append(f"{head_path}: test source exceeds the collection limit")
            continue
        else:
            pass
        try:
            before_module = ast.parse(before_source)
            before_definitions = test_definitions(before_module)
            head_module = ast.parse(head_source)
        except (SyntaxError, ValueError):
            limitations.append(f"{head_path}: test source could not be parsed")
            continue
        else:
            pass
        if not is_unit_test_file(head_path):
            content, truncated = clip_content(head_source)
            files[head_path] = CollectedFile(
                file_path=head_path,
                content=content,
                truncated=truncated,
                change="added" if previous_path is None else "modified",
            )
            continue
        else:
            pass
        before_lines = before_source.splitlines()
        head_lines = head_source.splitlines()
        has_changed_case = False
        head_definitions = test_definitions(head_module)
        removed_cases = sorted(before_definitions.keys() - head_definitions.keys())
        if removed_cases:
            limitations.append(
                f"{head_path}: deleted test cases were not reviewed: "
                + ", ".join(removed_cases)
            )
        else:
            pass
        for name, definition in head_definitions.items():
            previous = before_definitions.get(name)
            citations = affected.get((head_path, name), [])
            if (
                previous is not None
                and ast.dump(previous) == ast.dump(definition)
                and not citations
            ):
                continue
            else:
                pass
            assert definition.end_lineno is not None
            dependency_contexts, case_limitations = case_dependencies(
                repository,
                head_commit_sha,
                head_path,
                head_module,
                definition,
                fixture_module_cache,
            )
            modules = [
                module
                for _context_path, context_module, context_names in dependency_contexts
                for module in referenced_contract_modules(context_module, context_names)
            ]
            for context_path, _context_module, _context_names in dependency_contexts:
                if context_path == head_path:
                    continue
                elif context_path in files:
                    if files[context_path].truncated or files[context_path].missing:
                        case_limitations.append(
                            f"{context_path}: fixture source is unavailable"
                        )
                    else:
                        pass
                    continue
                else:
                    dependency_source = git_show(
                        repository, head_commit_sha, context_path
                    )
                if dependency_source is None:
                    case_limitations.append(
                        f"{context_path}: fixture source is missing"
                    )
                else:
                    dependency_content, dependency_truncated = clip_content(
                        dependency_source
                    )
                    files[context_path] = CollectedFile(
                        file_path=context_path,
                        content=dependency_content,
                        truncated=dependency_truncated,
                        change="included",
                    )
                    if dependency_truncated:
                        case_limitations.append(
                            f"{context_path}: fixture source is truncated"
                        )
                    else:
                        pass
            previous_dependencies: dict[str, str] = {}
            for citation in citations:
                dependency_source = git_show(
                    repository, head_commit_sha, citation.file_path
                )
                if dependency_source is None:
                    limitations.append(
                        f"{citation.file_path}: changed fixture or helper is missing"
                    )
                else:
                    dependency_module = ast.parse(dependency_source)
                    dependency_names = {
                        node.id
                        for node in ast.walk(dependency_module)
                        if isinstance(node, ast.Name)
                    }
                    modules.extend(
                        referenced_contract_modules(dependency_module, dependency_names)
                    )
                    dependency_content, dependency_truncated = clip_content(
                        dependency_source
                    )
                    files[citation.file_path] = CollectedFile(
                        file_path=citation.file_path,
                        content=dependency_content,
                        truncated=dependency_truncated,
                        change="modified",
                    )
                    if dependency_truncated:
                        case_limitations.append(
                            f"{citation.file_path}: changed dependency is truncated"
                        )
                    else:
                        pass
                    previous_dependency_path = next(
                        (
                            old
                            for old, current in changed_paths
                            if current == citation.file_path
                        ),
                        citation.file_path,
                    )
                    if previous_dependency_path is not None:
                        previous_dependency = git_show(
                            repository, merge_base, previous_dependency_path
                        )
                        if previous_dependency is None:
                            case_limitations.append(
                                f"{previous_dependency_path}: previous dependency is missing"
                            )
                        else:
                            previous_content, previous_truncated = clip_content(
                                previous_dependency
                            )
                            if previous_truncated:
                                case_limitations.append(
                                    f"{previous_dependency_path}: previous dependency is truncated"
                                )
                            else:
                                previous_dependencies[previous_dependency_path] = (
                                    previous_content
                                )
                    else:
                        pass
            case_contract_modules[(head_path, name)] = list(dict.fromkeys(modules))
            cases.append(
                UnitTestCase(
                    file_path=head_path,
                    name=name,
                    line_start=case_start(definition),
                    line_end=definition.end_lineno,
                    previous_source=(
                        "\n".join(
                            before_lines[case_start(previous) - 1 : previous.end_lineno]
                        )
                        if previous is not None
                        else ""
                    ),
                    changed_line_ranges=changed_case_ranges(
                        previous, definition, before_lines, head_lines
                    ),
                    change_evidence=citations,
                    evidence_limitations=case_limitations,
                    previous_dependencies=previous_dependencies,
                )
            )
            has_changed_case = True
        if has_changed_case:
            files[head_path] = CollectedFile(
                file_path=head_path,
                content=content,
                truncated=False,
                change="added" if previous_path is None else "modified",
            )
            contract_modules.extend(
                module
                for case in cases
                if case.file_path == head_path
                for module in case_contract_modules[(head_path, case.name)]
            )
        else:
            pass

    for file_path in dict.fromkeys(include_paths):
        if file_path in files:
            continue
        elif not is_context_path(file_path):
            limitations.append(f"{file_path}: unsupported evidence path")
            continue
        else:
            pass
        evidence = git_show(repository, head_commit_sha, file_path)
        if evidence is None:
            limitations.append(f"{file_path}: evidence was not found at head")
        else:
            content, truncated = clip_content(evidence)
            files[file_path] = CollectedFile(
                file_path=file_path,
                content=content,
                truncated=truncated,
                change="included",
            )
    remaining_context_bytes = (
        EXPLICIT_CONTEXT_BYTES if include_paths else AUTOMATIC_CONTEXT_BYTES
    )
    for module in dict.fromkeys(contract_modules):
        module_path = module.replace(".", "/")
        for file_path in (f"{module_path}.py", f"{module_path}/__init__.py"):
            if file_path in files:
                break
            else:
                pass
            evidence = git_show(repository, head_commit_sha, file_path)
            if evidence is None:
                continue
            else:
                evidence_bytes = len(evidence.encode())
            if evidence_bytes <= remaining_context_bytes:
                files[file_path] = CollectedFile(
                    file_path=file_path,
                    content=evidence,
                    truncated=False,
                    change="included",
                )
                remaining_context_bytes -= evidence_bytes
            else:
                limitations.append(
                    f"{file_path}: contract source exceeds the collection budget "
                    f"({evidence_bytes} bytes, {remaining_context_bytes} bytes left)"
                )
            break
    explicit_contracts = [
        file_path
        for file_path in include_paths
        if file_path in files
        and not file_path.startswith(UNIT_TEST_ROOT)
        and not files[file_path].truncated
    ]
    for case in cases:
        modules = case_contract_modules[(case.file_path, case.name)]
        module_paths = {
            candidate
            for module in modules
            for candidate in (
                f"{module.replace('.', '/')}.py",
                f"{module.replace('.', '/')}/__init__.py",
            )
        }
        if not modules:
            case.evidence_limitations.append(
                "No contract source could be statically linked to this case."
            )
        elif not explicit_contracts and not any(
            path in files and not files[path].truncated for path in module_paths
        ):
            case.evidence_limitations.append(
                "No contract source was collected. Include the relevant source or docs."
            )
        else:
            pass
        for module in modules:
            module_path = module.replace(".", "/")
            if not any(
                path in files and not files[path].truncated
                for path in (f"{module_path}.py", f"{module_path}/__init__.py")
            ):
                case.evidence_limitations.append(
                    f"Contract source for {module} could not be collected."
                )
            else:
                pass
    return list(files.values()), cases, limitations
