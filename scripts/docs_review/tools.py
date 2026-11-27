# SPDX-License-Identifier: Apache-2.0
"""Bounded evidence lookup in a pinned Git commit."""

from __future__ import annotations

import fnmatch
import os
import selectors
import subprocess
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from scripts.docs_review.context import (
    GIT_COMMAND_PREFIX,
    GIT_TOOL_TIMEOUT_SECONDS,
    resolve_commit,
)
from scripts.docs_review.models import CollectedFile, FileExcerpt, ReviewBundle

MAX_TOOL_ROUNDS = 4
MAX_TOOL_CALLS = 8
MAX_GREP_MATCHES = 30
MAX_GREP_MATCHES_PER_FILE = 20
MAX_GREP_LINE_CHARS = 300
MAX_READ_LINES = 120
MAX_LIST_FILES = 100
MAX_SINGLE_TOOL_OUTPUT_BYTES = 12_000
MAX_TOTAL_TOOL_OUTPUT_BYTES = 40_000
MAX_GIT_OUTPUT_BYTES = 8 * 1024 * 1024
GIT_READ_CHUNK_BYTES = 64 * 1024
ToolName = Literal["git_grep", "read_file_lines", "list_files"]


class ToolPropertySchema(BaseModel):
    type: Literal["string", "integer"]
    description: str
    minimum: int | None = None


class ToolParametersSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: dict[str, ToolPropertySchema]
    required: list[str]
    additional_properties: Literal[False] = Field(
        default=False, alias="additionalProperties"
    )


class ToolFunctionSchema(BaseModel):
    name: ToolName
    description: str
    parameters: ToolParametersSchema


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: ToolFunctionSchema


class GitGrepArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(
        min_length=1, max_length=200, description="Literal text to find."
    )
    path_prefix: str = Field(
        default="", description="Optional repository-relative file or directory."
    )


class ReadFileLinesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    file_path: str = Field(
        min_length=1, description="Repository-relative tracked file."
    )
    line_start: int = Field(ge=1, description="First line (inclusive).")
    line_end: int = Field(ge=1, description="Last line (inclusive).")


class ListFilesArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    glob_pattern: str = Field(
        min_length=1, max_length=200, description="Path glob, for example models/*.py."
    )


class ReadOnlyToolExecutor(Protocol):
    def tool_definitions(self) -> list[ToolDefinition]: ...
    def execute(
        self,
        tool_name: str,
        raw_arguments: str,
        *,
        deadline_seconds: float | None = None,
    ) -> str: ...
    def budget_exhausted(self) -> bool: ...
    def calls_used(self) -> int: ...
    def files_read(self) -> list[str]: ...


def tool_definition(
    name: ToolName,
    description: str,
    arguments: (
        type[GitGrepArguments] | type[ReadFileLinesArguments] | type[ListFilesArguments]
    ),
) -> ToolDefinition:
    return ToolDefinition(
        function=ToolFunctionSchema(
            name=name,
            description=description,
            parameters=ToolParametersSchema.model_validate(
                arguments.model_json_schema()
            ),
        )
    )


def safe_path(file_path: str) -> bool:
    return bool(file_path) and not (
        file_path.startswith(("/", "-"))
        or "\\" in file_path
        or any(ord(character) < 32 for character in file_path)
        or ".." in PurePosixPath(file_path).parts
    )


class GitOutputLimitError(ValueError):
    """A Git operation produced more data than the lookup can inspect."""


class GitReadOnlyToolbox:
    def __init__(
        self, repository: Path, head_commit_sha: str, bundle: ReviewBundle
    ) -> None:
        self.deadline_seconds: float | None = None
        self.repository = repository
        self.bundle = bundle
        self.commit_sha = resolve_commit(repository, head_commit_sha)
        self.tracked_paths: tuple[str, ...] | None = None
        self.tracked_path_set: frozenset[str] = frozenset()
        self.executed_call_count = 0
        self.total_output_bytes = 0
        self.read_paths: list[str] = []

    def git(
        self,
        arguments: list[str],
        *,
        decode_errors: Literal["strict", "replace"] = "strict",
    ) -> subprocess.CompletedProcess[str]:
        command = [
            *GIT_COMMAND_PREFIX,
            "--no-replace-objects",
            "--literal-pathspecs",
            "-C",
            str(self.repository),
            *arguments,
        ]
        deadline = time.monotonic() + GIT_TOOL_TIMEOUT_SECONDS
        if self.deadline_seconds is not None:
            deadline = min(deadline, self.deadline_seconds)
        else:
            pass
        output = bytearray()
        with subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as process:
            assert process.stdout is not None
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 or not selector.select(remaining):
                            raise subprocess.TimeoutExpired(
                                command, GIT_TOOL_TIMEOUT_SECONDS
                            )
                        else:
                            chunk = os.read(
                                process.stdout.fileno(), GIT_READ_CHUNK_BYTES
                            )
                        if not chunk:
                            break
                        elif len(output) + len(chunk) > MAX_GIT_OUTPUT_BYTES:
                            raise GitOutputLimitError
                        else:
                            output.extend(chunk)
                status = process.wait(timeout=max(0, deadline - time.monotonic()))
            finally:
                process.kill()
        return subprocess.CompletedProcess(
            command, status, stdout=output.decode(errors=decode_errors)
        )

    def budget_exhausted(self) -> bool:
        return (
            self.executed_call_count >= MAX_TOOL_CALLS
            or self.total_output_bytes >= MAX_TOTAL_TOOL_OUTPUT_BYTES
        )

    def calls_used(self) -> int:
        return self.executed_call_count

    def files_read(self) -> list[str]:
        return list(self.read_paths)

    def tool_definitions(self) -> list[ToolDefinition]:
        return [
            tool_definition(
                "git_grep", "Find literal text in the pinned commit.", GitGrepArguments
            ),
            tool_definition(
                "read_file_lines",
                "Read numbered lines as citable evidence.",
                ReadFileLinesArguments,
            ),
            tool_definition(
                "list_files",
                "List tracked files matching a repository-relative POSIX path glob.",
                ListFilesArguments,
            ),
        ]

    def paths(self) -> tuple[str, ...]:
        if self.tracked_paths is None:
            tree = self.git(["ls-tree", "-rz", "--name-only", self.commit_sha])
            tree.check_returncode()
            self.tracked_paths = tuple(filter(None, tree.stdout.split("\0")))
            self.tracked_path_set = frozenset(self.tracked_paths)
        else:
            pass
        return self.tracked_paths

    def execute(
        self,
        tool_name: str,
        raw_arguments: str,
        *,
        deadline_seconds: float | None = None,
    ) -> str:
        self.deadline_seconds = deadline_seconds
        remaining_bytes = min(
            MAX_SINGLE_TOOL_OUTPUT_BYTES,
            MAX_TOTAL_TOOL_OUTPUT_BYTES - self.total_output_bytes,
        )
        if self.budget_exhausted():
            return "Tool budget exhausted. Produce the final JSON finding list using collected evidence."
        else:
            self.executed_call_count += 1
        try:
            if tool_name == "git_grep":
                result = self.git_grep(
                    GitGrepArguments.model_validate_json(raw_arguments)
                )
            elif tool_name == "read_file_lines":
                result = self.read_file_lines(
                    ReadFileLinesArguments.model_validate_json(raw_arguments),
                    remaining_bytes,
                )
            elif tool_name == "list_files":
                pattern = ListFilesArguments.model_validate_json(
                    raw_arguments
                ).glob_pattern
                matches = [
                    path for path in self.paths() if fnmatch.fnmatchcase(path, pattern)
                ]
                visible = matches[:MAX_LIST_FILES]
                if len(matches) > len(visible):
                    visible.append(f"{len(matches) - len(visible)} more paths omitted.")
                else:
                    pass
                result = "\n".join(visible) or "No files found."
            else:
                result = "Unknown tool."
        except ValidationError:
            result = f"Invalid arguments for {tool_name}."
        except GitOutputLimitError:
            result = "Git output exceeds the lookup size limit."
        except subprocess.TimeoutExpired:
            result = f"{tool_name} timed out; narrow path_prefix or query."
        except (subprocess.CalledProcessError, UnicodeError):
            result = f"{tool_name} failed."
        encoded = result.encode()[:remaining_bytes]
        clipped = encoded.decode(errors="ignore")
        self.total_output_bytes += len(encoded)
        return clipped

    def git_grep(self, arguments: GitGrepArguments) -> str:
        prefix = arguments.path_prefix
        command = [
            "-c",
            "core.quotePath=false",
            "grep",
            "--no-textconv",
            "--no-recurse-submodules",
            "--no-color",
            "-n",
            "-F",
            "-I",
            f"--max-count={MAX_GREP_MATCHES_PER_FILE}",
            "-e",
            arguments.query,
            self.commit_sha,
        ]
        if "\0" in arguments.query:
            return "Invalid arguments for git_grep."
        elif prefix and not safe_path(prefix):
            return "Invalid path_prefix."
        elif prefix:
            prefix = str(PurePosixPath(prefix))
            if not any(
                path == prefix or path.startswith(prefix + "/") for path in self.paths()
            ):
                return "No tracked files match path_prefix."
            else:
                command.extend(["--", prefix])
        else:
            pass
        completed = self.git(command, decode_errors="replace")
        if completed.returncode == 1:
            return "No matches found."
        elif completed.returncode != 0:
            return "git_grep failed."
        else:
            matches = [
                line.removeprefix(self.commit_sha + ":")[:MAX_GREP_LINE_CHARS]
                for line in completed.stdout.splitlines()
            ]
            matches.sort(
                key=lambda line: bool(
                    set(PurePosixPath(line.partition(":")[0].lower()).parts[:-1])
                    & {"tests", "benchmarks"}
                )
            )
            visible = matches[:MAX_GREP_MATCHES]
            if len(matches) > MAX_GREP_MATCHES:
                visible.append("More matches omitted.")
            else:
                pass
            return "\n".join(visible)

    def read_file_lines(
        self, arguments: ReadFileLinesArguments, remaining_bytes: int
    ) -> str:
        path = arguments.file_path
        start = arguments.line_start
        if not safe_path(path):
            return "Invalid file_path."
        elif arguments.line_end < start:
            return "line_end must be at least line_start."
        else:
            self.paths()
        if path not in self.tracked_path_set:
            return "File not found on head commit."
        elif any(
            file.file_path == path and file.change in ("trusted_base", "review_input")
            for file in self.bundle.files
        ):
            return "This path is reserved for existing review evidence."
        else:
            try:
                blob = self.git(["cat-file", "blob", f"{self.commit_sha}:{path}"])
            except UnicodeDecodeError:
                return "Binary file cannot be read."
        if blob.returncode != 0:
            return "File not found on head commit."
        elif "\0" in blob.stdout:
            return "Binary file cannot be read."
        else:
            lines = blob.stdout.splitlines()
        if start > len(lines):
            return f"File has only {len(lines)} lines."
        else:
            end = min(arguments.line_end, start + MAX_READ_LINES - 1, len(lines))
        numbered: list[str] = []
        body_bytes = 0
        header = ""
        for number in range(start, end + 1):
            candidate = f"File {path} [Lines {start}-{number} of {len(lines)}]"
            line = f"{number}: {lines[number - 1]}"
            body_bytes += 1 + len(line.encode())
            if len(candidate.encode()) + body_bytes > remaining_bytes:
                break
            else:
                numbered.append(line)
                header = candidate
        if not numbered:
            return "Requested lines exceed the remaining output budget."
        else:
            end = start + len(numbered) - 1
        excerpt = FileExcerpt(
            line_start=start, line_end=end, content="\n".join(lines[start - 1 : end])
        )
        self.register_excerpt(path, excerpt)
        if path not in self.read_paths:
            self.read_paths.append(path)
        else:
            pass
        return "\n".join([header, *numbered])

    def register_excerpt(self, path: str, excerpt: FileExcerpt) -> None:
        for index, file in enumerate(self.bundle.files):
            if file.file_path != path:
                continue
            elif file.excerpts or file.truncated or file.missing:
                parts = list(file.excerpts)
                if not parts and file.content:
                    parts.append(
                        FileExcerpt(
                            line_start=1,
                            line_end=len(file.content.splitlines()),
                            content=file.content,
                        )
                    )
                else:
                    pass
                merged = merge_excerpts([*parts, excerpt])
                self.bundle.files[index] = replace(
                    file,
                    excerpts=merged,
                    content="\n".join(part.content for part in merged),
                    missing=False,
                )
            else:
                pass
            return
        self.bundle.files.append(
            CollectedFile(
                file_path=path,
                change="included",
                truncated=False,
                content=excerpt.content,
                excerpts=(excerpt,),
            )
        )


def merge_excerpts(parts: list[FileExcerpt]) -> tuple[FileExcerpt, ...]:
    lines: dict[int, str] = {}
    for part in parts:
        numbers = range(part.line_start, part.line_end + 1)
        lines.update(zip(numbers, part.content.split("\n")))
    merged: list[FileExcerpt] = []
    for start in sorted(number for number in lines if number - 1 not in lines):
        end = start
        while end + 1 in lines:
            end += 1
        merged.append(
            FileExcerpt(
                line_start=start,
                line_end=end,
                content="\n".join(lines[number] for number in range(start, end + 1)),
            )
        )
    return tuple(merged)
