# SPDX-License-Identifier: Apache-2.0
"""Discover referenced code files and collect bounded code context from git."""

from __future__ import annotations

import posixpath
import re
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath

from scripts.docs_review.models import CollectedFile, FileChange, FileExcerpt
from scripts.docs_review.reporting import render_collected_file

DOCUMENTATION_SUFFIXES = {".md", ".rst", ".mdx"}
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}
GIT_COMMAND_PREFIX = [
    "git",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.quotePath=true",
]

MAX_FULL_CODE_FILE_BYTES = 16_000
MAX_AUTO_CONTEXT_FILES = 8
MAX_AUTO_CONTEXT_FILE_BYTES = 12_000

PATH_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_/@:-])(?:\.\.?/)*[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.(?:py|ya?ml|toml|json)(?=[#:\s`'\"(),;]|$)"
)
MODULE_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./-])([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)(?![A-Za-z0-9_./-])"
)
BACKTICK_IDENTIFIER_PATTERN = re.compile(r"`([A-Za-z0-9_./-]{3,80})`")
GENERIC_DOCUMENT_STEMS = frozenset(
    {
        "api_reference",
        "asr",
        "basic_usage",
        "benchmarks",
        "companies",
        "cookbook",
        "developer_reference",
        "docs",
        "guide",
        "index",
        "installation",
        "notes",
        "omni",
        "overview",
        "platforms",
        "quickstart",
        "readme",
        "reference",
        "serving",
        "tts",
        "tuning",
        "usage",
        "user_guide",
    }
)
HIGH_SIGNAL_CODE_NAMES = frozenset(
    {"args.py", "cli.py", "config.py", "io.py", "schema.py", "schemas.py", "types.py"}
)


def is_documentation(file_path: str) -> bool:
    return (
        file_path.startswith("docs/")
        and Path(file_path).suffix in DOCUMENTATION_SUFFIXES
    )


def is_code_or_config(file_path: str) -> bool:
    return Path(file_path).suffix in CODE_SUFFIXES and not is_documentation(file_path)


def change_from_status(status: str) -> FileChange:
    kind = status[:1]
    if kind in ("A", "R", "C"):
        return "added"
    elif kind == "D":
        return "deleted"
    else:
        return "modified"


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


def referenced_paths(
    documents: list[CollectedFile], diffs: dict[str, str], tracked: set[str]
) -> list[str]:
    stems: set[str] = set()
    paths: list[str] = []
    for document in documents:
        path = PurePosixPath(document.file_path)
        for raw_stem in (path.stem.lower(), path.parent.name.lower()):
            if len(raw_stem) >= 3 and raw_stem not in GENERIC_DOCUMENT_STEMS:
                stems.add(raw_stem)
            else:
                pass
        text = document.content + "\n" + diffs.get(document.file_path, "")
        for match in PATH_TOKEN_PATTERN.finditer(text):
            token = match.group()
            paths.extend(
                (
                    posixpath.normpath(str(path.parent / token)),
                    posixpath.normpath(token),
                )
            )
        for match in MODULE_TOKEN_PATTERN.finditer(text):
            module = match.group().replace(".", "/")
            paths.extend(
                (
                    module + ".py",
                    module + "/__init__.py",
                    posixpath.dirname(module) + ".py",
                )
            )
    inferred: list[str] = []
    for candidate in tracked:
        path = PurePosixPath(candidate)
        lower_parts = {part.lower() for part in path.parts[:-1]}
        if {"tests", "benchmarks"} & lower_parts:
            continue
        elif path.stem.lower() in stems or stems.intersection(lower_parts):
            inferred.append(candidate)
        else:
            pass
    inferred.sort(
        key=lambda name: (PurePosixPath(name).name not in HIGH_SIGNAL_CODE_NAMES, name)
    )
    return list(dict.fromkeys(name for name in [*paths, *inferred] if name in tracked))


def bounded_file(
    file: CollectedFile,
    budget: int,
    ranges: list[tuple[int, int]],
    *,
    mark_truncated: bool = False,
) -> CollectedFile | None:
    if len(render_collected_file(file).encode()) + 1 <= budget:
        return file
    else:
        pass
    lines = file.content.splitlines()
    candidate = replace(file, content="", truncated=mark_truncated)
    if len(render_collected_file(candidate).encode()) + 1 > budget:
        return None
    else:
        pass
    for start, end in ranges or [(1, len(lines))]:
        low, high = start - 1, min(len(lines), end)
        best = candidate
        while low < high:
            middle = (low + high + 1) // 2
            excerpt = FileExcerpt(
                line_start=start,
                line_end=middle,
                content="\n".join(lines[start - 1 : middle]),
            )
            trial = replace(candidate, excerpts=(*candidate.excerpts, excerpt))
            if len(render_collected_file(trial).encode()) + 1 <= budget:
                best = trial
                low = middle
            else:
                high = middle - 1
        candidate = best
        if low < end:
            break
        else:
            pass
    return replace(
        candidate, content="\n".join(part.content for part in candidate.excerpts)
    )


def collect_code_context(
    repository: Path,
    base: str,
    head: str,
    changed: list[tuple[str, str]],
    documents: list[CollectedFile],
    diffs: dict[str, str],
    files: list[CollectedFile],
    remaining: int | None,
) -> None:
    seen = {file.file_path for file in files}
    changed_entries = [(path, change_from_status(status)) for status, path in changed]
    seen.update(path for path, change in changed_entries)
    included_entries: list[tuple[str, FileChange]] = []
    if documents:
        tracked = {
            path
            for path in require_git(
                repository, ["ls-tree", "-rz", "--name-only", head]
            ).split("\0")
            if is_code_or_config(path)
        }
        references = [
            path
            for path in referenced_paths(documents, diffs, tracked)
            if path not in seen
        ]
        included_entries = [
            (path, "included") for path in references[:MAX_AUTO_CONTEXT_FILES]
        ]
    else:
        pass
    keywords = {
        match.lstrip("-").replace("-", "_")
        for doc in documents
        for match in BACKTICK_IDENTIFIER_PATTERN.findall(
            doc.content + "\n" + diffs.get(doc.file_path, "")
        )
    }
    for group in (changed_entries, included_entries):
        for index, (path, change) in enumerate(group):
            budget = (
                MAX_AUTO_CONTEXT_FILE_BYTES
                if change == "included"
                else MAX_FULL_CODE_FILE_BYTES
            )
            if remaining is not None:
                budget = min(budget, remaining // (len(group) - index))
            else:
                pass
            content = "" if change == "deleted" else git_show(repository, head, path)
            file = CollectedFile(
                file_path=path,
                change=change,
                content=content or "",
                missing=content is None,
                truncated=False,
            )
            ranges: list[tuple[int, int]] = []
            if len(render_collected_file(file).encode()) + 1 > budget:
                lines = file.content.splitlines()
                if change == "modified":
                    diff = require_git(
                        repository, ["diff", "-U0", base, head, "--", path]
                    )
                    for start, count in re.findall(
                        r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M
                    ):
                        first = max(1, int(start))
                        ranges.append((first, first + max(1, int(count or 1)) - 1))
                    context_radius = 16
                else:
                    ranges = [
                        (i, i)
                        for i, line in enumerate(lines, 1)
                        if any(word in line for word in keywords)
                    ]
                    context_radius = 10
                windows: list[tuple[int, int]] = []
                for first, last in ranges:
                    start = max(1, first - context_radius)
                    end = min(len(lines), last + context_radius)
                    if windows and start <= windows[-1][1] + 1:
                        windows[-1] = (windows[-1][0], max(windows[-1][1], end))
                    else:
                        windows.append((start, end))
                file = bounded_file(file, budget, windows)
            else:
                pass
            if file is not None:
                files.append(file)
                if remaining is not None:
                    remaining -= len(render_collected_file(file).encode()) + 1
                else:
                    pass
            else:
                pass
