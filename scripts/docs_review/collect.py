# SPDX-License-Identifier: Apache-2.0
"""Collect a frozen review bundle. This never runs scripts from the pull request."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from scripts.docs_review.context import (
    bounded_file,
    change_from_status,
    collect_code_context,
    git_show,
    is_code_or_config,
    is_documentation,
    require_git,
    run_git,
)
from scripts.docs_review.models import (
    BundlePayload,
    CollectedFile,
    ModelSettings,
    ModelSnapshot,
    ReviewBundle,
    ReviewRequest,
)
from scripts.docs_review.policy import (
    POLICY_VERSION,
    STYLE_GUIDE_PATH,
    classify_document,
)
from scripts.docs_review.reporting import render_bundle, render_collected_file

MAX_COLLECTED_BYTES = 200_000


def collect_review_bundle(request: ReviewRequest) -> ReviewBundle:
    repository = request.repository.resolve()
    merge_base = require_git(
        repository, ["merge-base", request.base_commit_sha, request.head_commit_sha]
    ).strip()
    name_status = require_git(
        repository, ["diff", "--name-status", "-z", merge_base, request.head_commit_sha]
    )
    changed = parse_name_status(name_status)
    reviewed_paths = select_reviewed_paths(changed, request.include_paths)
    files: list[CollectedFile] = []
    changed_code_entries: list[tuple[str, str]] = []
    modified_or_deleted_docs: list[str] = []
    for status, file_path in changed:
        if file_path in reviewed_paths:
            collected = read_changed_file(
                repository, request.head_commit_sha, status, file_path
            )
            files.append(collected)
            if collected.change != "added":
                modified_or_deleted_docs.append(file_path)
            else:
                pass
        elif is_code_or_config(file_path):
            changed_code_entries.append((status, file_path))
        else:
            pass
    for file_path in request.include_paths:
        if any(collected.file_path == file_path for collected in files) or any(
            code_path == file_path for status, code_path in changed_code_entries
        ):
            continue
        else:
            pass
        shown = git_show(repository, request.head_commit_sha, file_path)
        content, truncated = clip_content(shown or "")
        files.append(
            CollectedFile(
                file_path=file_path,
                change="included",
                truncated=truncated,
                content=content,
                missing=shown is None,
            )
        )
    style_content = git_show(repository, request.base_commit_sha, STYLE_GUIDE_PATH)
    if style_content is None:
        style_guide_limitation = (
            f"{STYLE_GUIDE_PATH} was not found on the base commit. "
            "No style standard was invented for this review."
        )
    else:
        style_guide_limitation = ""
        content, truncated = clip_content(style_content)
        files.append(
            CollectedFile(
                file_path=STYLE_GUIDE_PATH,
                change="trusted_base",
                truncated=truncated,
                content=content,
            )
        )
    files.extend(review_input_files(request))
    scope_body = (
        "\n".join(reviewed_paths)
        if reviewed_paths
        else "No documentation files in scope."
    )
    files.append(
        CollectedFile(
            file_path="scope.txt",
            change="review_input",
            truncated=False,
            content=scope_body,
        )
    )
    diffs: dict[str, str] = {}
    if modified_or_deleted_docs:
        diff_text = require_git(
            repository,
            [
                "diff",
                "--no-renames",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "-U3",
                merge_base,
                request.head_commit_sha,
                "--",
                *modified_or_deleted_docs,
            ],
        )
        for section in re.split(r"(?=^diff --git )", diff_text, flags=re.MULTILINE):
            header = re.search(r"^--- (.+)$", section, re.MULTILINE)
            if header:
                source = header.group(1).rstrip("\t")
                if source.startswith('"'):
                    source = (
                        ast.literal_eval(source)
                        .encode("latin-1")
                        .decode("utf-8", "surrogateescape")
                    )
                else:
                    pass
                diffs[source.removeprefix("a/")] = section
            else:
                pass
    else:
        pass
    reviewed_files = [
        file
        for file in files
        if file.file_path in reviewed_paths and file.change != "trusted_base"
    ]
    added_lines = [file.content for file in reviewed_files if file.change == "added"]
    for diff in diffs.values():
        in_hunk = False
        for line in diff.splitlines():
            if line.startswith("@@ "):
                in_hunk = True
            elif in_hunk and line.startswith("+"):
                added_lines.append(line[1:])
            else:
                pass
    added = "\n".join(added_lines)
    bundle = ReviewBundle(
        repository_identity=repository_identity(repository),
        base_commit_sha=request.base_commit_sha,
        head_commit_sha=request.head_commit_sha,
        merge_base_commit_sha=merge_base,
        policy_version=POLICY_VERSION,
        pull_request_text=request.pull_request_text,
        comment_text=request.comment_text,
        evidence_text=request.evidence_text,
        checks=list(request.checks),
        files=files,
        reviewed_paths=reviewed_paths,
        added_documentation_text=added,
        style_guide_limitation=style_guide_limitation,
        document_scopes=[
            classify_document(collected.file_path, collected.content)
            for collected in reviewed_files
        ],
    )
    remaining = None
    if request.max_bundle_bytes is not None:
        remaining = request.max_bundle_bytes - len(render_bundle(bundle).encode())
        if remaining < 0:
            for index in sorted(
                range(len(files)),
                key=lambda i: len(files[i].content.encode()),
                reverse=True,
            ):
                file = files[index]
                size = len(render_collected_file(file).encode()) + 1
                minimum = (
                    len(
                        render_collected_file(
                            replace(file, content="", truncated=True)
                        ).encode()
                    )
                    + 1
                )
                fitted = bounded_file(
                    file, max(minimum, size + remaining), [], mark_truncated=True
                )
                if fitted is not None:
                    files[index] = fitted
                    remaining += size - len(render_collected_file(fitted).encode()) - 1
                else:
                    pass
                if remaining >= 0:
                    break
                else:
                    pass
        else:
            pass
    else:
        pass
    if diffs:
        content, truncated = clip_content("".join(diffs.values()))
        diff = CollectedFile(
            file_path="documentation.diff",
            change="review_input",
            content=content,
            truncated=truncated,
        )
        if remaining is not None:
            diff = bounded_file(diff, remaining, [], mark_truncated=True)
        else:
            pass
        if diff is not None:
            files.append(diff)
            if remaining is not None:
                remaining -= len(render_collected_file(diff).encode()) + 1
            else:
                pass
        else:
            pass
    else:
        pass
    collect_code_context(
        repository,
        merge_base,
        request.head_commit_sha,
        changed_code_entries,
        reviewed_files,
        diffs,
        files,
        remaining,
    )
    return bundle


def review_input_sha256(
    bundle: ReviewBundle, model_settings: ModelSettings | None, prompt_version: str
) -> str:
    model_configuration = (
        model_settings.model_dump() if model_settings is not None else None
    )
    parts = [
        bundle.repository_identity,
        bundle.policy_version,
        prompt_version,
        json.dumps(model_configuration, sort_keys=True),
        json.dumps(
            [scope.model_dump(mode="json") for scope in bundle.document_scopes],
            sort_keys=True,
        ),
        json.dumps(bundle.pull_request_text),
        bundle.comment_text,
        bundle.evidence_text,
        files_canonical(bundle.files),
        bundle.added_documentation_text,
        bundle.style_guide_limitation,
    ]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def bundle_payload(
    bundle: ReviewBundle,
    prompt_version: str,
    input_hash: str,
    model_settings: ModelSettings | None,
) -> BundlePayload:
    return BundlePayload(
        model_configuration=(
            ModelSnapshot.from_settings(model_settings) if model_settings else None
        ),
        repository_identity=bundle.repository_identity,
        base_commit_sha=bundle.base_commit_sha,
        head_commit_sha=bundle.head_commit_sha,
        merge_base_commit_sha=bundle.merge_base_commit_sha,
        policy_version=bundle.policy_version,
        prompt_version=prompt_version,
        review_input_sha256=input_hash,
        reviewed_paths=list(bundle.reviewed_paths),
        added_documentation_text=bundle.added_documentation_text,
        style_guide_limitation=bundle.style_guide_limitation,
        pull_request_text=bundle.pull_request_text,
        comment_text=bundle.comment_text,
        evidence_text=bundle.evidence_text,
        files=list(bundle.files),
        document_scopes=list(bundle.document_scopes),
    )


def repository_identity(repository: Path) -> str:
    completed = run_git(repository, ["remote", "get-url", "origin"])
    if completed.returncode == 0:
        origin = completed.stdout.strip()
    else:
        origin = ""
    return f"{repository.resolve().as_posix()}\n{origin}"


def select_reviewed_paths(
    changed: list[tuple[str, str]], include_paths: list[str]
) -> list[str]:
    reviewed = [
        file_path for status, file_path in changed if is_documentation(file_path)
    ]
    for file_path in include_paths:
        if is_documentation(file_path) and file_path not in reviewed:
            reviewed.append(file_path)
        else:
            pass
    return reviewed


def parse_name_status(raw: str) -> list[tuple[str, str]]:
    entries = iter(raw.rstrip("\0").split("\0"))
    parsed = []
    for status in entries:
        if not status:
            continue
        else:
            pass
        path = next(entries)
        if status.startswith(("R", "C")):
            path = next(entries)
        else:
            pass
        parsed.append((status, path))
    return parsed


def read_changed_file(
    repository: Path, head_commit_sha: str, status: str, file_path: str
) -> CollectedFile:
    change = change_from_status(status)
    if change == "deleted":
        return CollectedFile(
            file_path=file_path, change=change, truncated=False, content=""
        )
    else:
        shown = git_show(repository, head_commit_sha, file_path)
        if shown is None:
            content, truncated = "", False
        else:
            content, truncated = clip_content(shown)
        return CollectedFile(
            file_path=file_path,
            change=change,
            truncated=truncated,
            content=content,
            missing=shown is None,
        )


def review_input_files(request: ReviewRequest) -> list[CollectedFile]:
    checks_text = json.dumps(
        [
            dict(
                check.model_dump(mode="json", exclude={"execution_commit_sha"}),
                matches_head=check.execution_commit_sha == request.head_commit_sha,
            )
            for check in sorted(request.checks, key=lambda item: item.name)
        ],
        sort_keys=True,
        indent=2,
    )
    bodies = {
        "pull_request.md": request.pull_request_text or "",
        "comments.md": request.comment_text,
        "evidence.md": request.evidence_text,
        "checks.json": checks_text,
    }
    return [
        CollectedFile(
            file_path=file_path,
            change="review_input",
            truncated=False,
            content=content,
        )
        for file_path, content in bodies.items()
    ]


def files_canonical(files: list[CollectedFile]) -> str:
    rows: list[str] = []
    for collected in sorted(files, key=lambda collected: collected.file_path):
        if collected.excerpts:
            signature = ",".join(
                f"{excerpt.line_start}-{excerpt.line_end}:{content_sha256(excerpt.content)}"
                for excerpt in collected.excerpts
            )
        else:
            signature = content_sha256(collected.content)
        rows.append(
            f"{collected.file_path}\t{collected.change}\t{collected.truncated}\t{collected.missing}\t{signature}"
        )
    return "\n".join(rows)


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def clip_content(content: str) -> tuple[str, bool]:
    encoded = content.encode()
    if len(encoded) > MAX_COLLECTED_BYTES:
        return encoded[:MAX_COLLECTED_BYTES].decode(errors="ignore"), True
    else:
        return content, False
