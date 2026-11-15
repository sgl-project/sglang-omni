# SPDX-License-Identifier: Apache-2.0
"""Collect a frozen review bundle. This never runs scripts from the pull request."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.docs_review.models import (
    BundleFileRecord,
    BundlePayload,
    CheckRecord,
    CollectedFile,
    FileChange,
    ModelSettings,
    ReviewBundle,
    ReviewRequest,
)
from scripts.docs_review.policy import (
    POLICY_VERSION,
    STYLE_GUIDE_PATH,
    classify_document,
)
from scripts.docs_review.snapshot import (
    clip_content,
    git_show,
    repository_identity,
    require_git,
)
from scripts.docs_review.unit_tests import (
    UNIT_TEST_POLICY_VERSION,
    collect_unit_test_inputs,
)

DOCUMENTATION_SUFFIXES = {".md", ".rst", ".mdx"}
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}


def collect_review_bundle(request: ReviewRequest) -> ReviewBundle:
    repository = request.repository.resolve()
    merge_base = require_git(
        repository, ["merge-base", request.base_commit_sha, request.head_commit_sha]
    ).strip()
    if request.review_scope == "unit_tests":
        test_files, cases, limitations = collect_unit_test_inputs(
            repository, merge_base, request.head_commit_sha, request.include_paths
        )
        test_files.extend(review_input_files(request))
        return ReviewBundle(
            repository_identity=repository_identity(repository),
            base_commit_sha=request.base_commit_sha,
            head_commit_sha=request.head_commit_sha,
            merge_base_commit_sha=merge_base,
            policy_version=UNIT_TEST_POLICY_VERSION,
            pull_request_text=request.pull_request_text,
            comment_text=request.comment_text,
            evidence_text=request.evidence_text,
            checks=list(request.checks),
            files=test_files,
            reviewed_paths=list(dict.fromkeys(case.file_path for case in cases)),
            added_documentation_text="",
            style_guide_limitation="",
            review_scope=request.review_scope,
            unit_test_cases=cases,
            collection_limitations=[*request.collection_limitations, *limitations],
        )
    else:
        pass
    name_status = require_git(
        repository, ["diff", "--name-status", merge_base, request.head_commit_sha]
    )
    changed = parse_name_status(name_status)
    reviewed_paths = select_reviewed_paths(changed, request.include_paths)
    files: list[CollectedFile] = []
    for status, file_path in changed:
        if file_path in reviewed_paths or is_code_or_config(file_path):
            files.append(
                read_changed_file(
                    repository, request.head_commit_sha, status, file_path
                )
            )
        else:
            pass
    for file_path in request.include_paths:
        if any(collected.file_path == file_path for collected in files):
            pass
        else:
            shown = git_show(repository, request.head_commit_sha, file_path)
            if shown is None:
                files.append(
                    CollectedFile(
                        file_path=file_path,
                        change="included",
                        truncated=False,
                        content="",
                        missing=True,
                    )
                )
            else:
                content, truncated = clip_content(shown)
                files.append(
                    CollectedFile(
                        file_path=file_path,
                        change="included",
                        truncated=truncated,
                        content=content,
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
    added = added_documentation_text(
        repository, merge_base, request.head_commit_sha, reviewed_paths
    )
    return ReviewBundle(
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
            for collected in files
            if collected.file_path in reviewed_paths
            and collected.change != "trusted_base"
        ],
    )


def review_input_sha256(
    bundle: ReviewBundle, model_settings: ModelSettings | None, prompt_version: str
) -> str:
    model_configuration = (
        model_settings.model_dump() if model_settings is not None else None
    )
    if model_configuration is not None and model_settings is not None:
        if model_settings.token_limit_field is None:
            model_configuration.pop("token_limit_field")
        else:
            pass
        if model_settings.response_format is None:
            model_configuration.pop("response_format")
        else:
            pass
    else:
        pass
    parts = [
        bundle.repository_identity,
        bundle.base_commit_sha,
        bundle.head_commit_sha,
        bundle.merge_base_commit_sha,
        bundle.policy_version,
        prompt_version,
        json.dumps(model_configuration, sort_keys=True),
        json.dumps(
            [scope.model_dump(mode="json") for scope in bundle.document_scopes],
            sort_keys=True,
        ),
        bundle.pull_request_text,
        bundle.comment_text,
        bundle.evidence_text,
        checks_canonical(bundle.checks),
        files_canonical(bundle.files),
        bundle.added_documentation_text,
        bundle.style_guide_limitation,
    ]
    if bundle.review_scope == "unit_tests":
        parts.extend(
            [
                bundle.review_scope,
                json.dumps(
                    [case.model_dump(mode="json") for case in bundle.unit_test_cases],
                    sort_keys=True,
                ),
                json.dumps(bundle.collection_limitations),
            ]
        )
    else:
        pass
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def bundle_payload(
    bundle: ReviewBundle, prompt_version: str, input_hash: str
) -> BundlePayload:
    records = [
        BundleFileRecord(
            file_path=collected.file_path,
            change=collected.change,
            truncated=collected.truncated,
            content=collected.content,
            missing=collected.missing,
        )
        for collected in bundle.files
    ]
    return BundlePayload(
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
        files=records,
        document_scopes=list(bundle.document_scopes),
        review_scope=bundle.review_scope,
        unit_test_cases=list(bundle.unit_test_cases),
        collection_limitations=list(bundle.collection_limitations),
    )


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


def is_documentation(file_path: str) -> bool:
    return (
        file_path.startswith("docs/")
        and Path(file_path).suffix in DOCUMENTATION_SUFFIXES
    )


def is_code_or_config(file_path: str) -> bool:
    suffix = Path(file_path).suffix
    if suffix in CODE_SUFFIXES and not is_documentation(file_path):
        return True
    else:
        return False


def parse_name_status(raw: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if line:
            parts = line.split("\t")
            parsed.append((parts[0], parts[-1]))
        else:
            pass
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


def change_from_status(status: str) -> FileChange:
    kind = status[:1]
    if kind == "A" or kind == "R" or kind == "C":
        return "added"
    elif kind == "D":
        return "deleted"
    else:
        return "modified"


def review_input_files(request: ReviewRequest) -> list[CollectedFile]:
    checks_text = json.dumps(
        [check.model_dump(mode="json") for check in request.checks],
        sort_keys=True,
        indent=2,
    )
    bodies = {
        "pull_request.md": request.pull_request_text,
        "comments.md": request.comment_text,
        "evidence.md": request.evidence_text,
        "checks.json": checks_text,
    }
    collected: list[CollectedFile] = []
    for file_path, content in bodies.items():
        collected.append(
            CollectedFile(
                file_path=file_path,
                change="review_input",
                truncated=False,
                content=content,
            )
        )
    return collected


def added_documentation_text(
    repository: Path,
    merge_base: str,
    head_commit_sha: str,
    reviewed_paths: list[str],
) -> str:
    if not reviewed_paths:
        return ""
    else:
        diff = require_git(
            repository,
            ["diff", "-U0", merge_base, head_commit_sha, "--", *reviewed_paths],
        )
        added: list[str] = []
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
            else:
                pass
        return "\n".join(added)


def files_canonical(files: list[CollectedFile]) -> str:
    rows = [
        f"{collected.file_path}\t{collected.change}\t{collected.truncated}\t{collected.missing}\t{content_sha256(collected.content)}"
        for collected in sorted(files, key=lambda collected: collected.file_path)
    ]
    return "\n".join(rows)


def checks_canonical(checks: list[CheckRecord]) -> str:
    rows = [
        json.dumps(check.model_dump(mode="json"), sort_keys=True)
        for check in sorted(checks, key=lambda check: check.name)
    ]
    return "\n".join(rows)


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()
