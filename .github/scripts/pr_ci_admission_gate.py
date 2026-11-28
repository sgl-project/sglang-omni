# SPDX-License-Identifier: Apache-2.0
"""Fail-closed PR CI admission gate inspecting paths and Git tree modes via GitHub APIs."""

from __future__ import annotations

import json
import os
import posixpath
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

MAX_API_RESPONSE_BYTES = 8_388_608
API_TIMEOUT_SECONDS = 30
MAX_API_RETRIES = 2
MAX_RETRY_SLEEP_SECONDS = 10

ALLOWED_ROOT_FILES = frozenset(
    {
        "readme.md",
        "contributing.md",
    }
)

# note (0xtoward): sglang_omni_router compiles on H100 setup and is excluded for manual review.
ALLOWED_PREFIXES = (
    "sglang_omni/",
    "sglang_omni_mlx/",
    "tests/",
    "docs/",
    "examples/",
    "benchmarks/",
    "playground/",
    "voxt/",
)

PROTECTED_PREFIXES = (
    "sglang_omni_mlx/native/scripts/",
    "voxt/scripts/",
)

PROTECTED_FILENAMES = frozenset(
    {
        "conftest.py",
        "setup.py",
        "setup.cfg",
        "pyproject.toml",
        "cmakelists.txt",
        "makefile",
        "gnumakefile",
        "build.rs",
        "cargo.toml",
        "cargo.lock",
        "rust-toolchain.toml",
        "go.mod",
        "go.sum",
        "package.swift",
        "package.resolved",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pytest.ini",
        "tox.ini",
        "uv.lock",
    }
)

ALLOWED_EXTENSIONLESS_FILENAMES = frozenset(
    {
        "copying",
        "license",
        "notice",
    }
)

ALLOWED_EXACT_PATHS = frozenset(
    {
        "voxt/third_party/mygo/voxtnew-vendored",
    }
)

ALLOWED_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".md",
        ".rst",
        ".txt",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".tsv",
        ".svg",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".gif",
        ".wav",
        ".mp3",
        ".flac",
        ".pcm",
        ".css",
        ".html",
        ".xml",
        ".sql",
        ".proto",
        ".go",
        ".swift",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hpp",
        ".hh",
        ".cu",
        ".cuh",
        ".metal",
        ".mm",
        ".m",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".golden",
        ".patch",
        ".tmpl",
        ".hlsl",
        ".glsl",
    }
)

ALLOWED_GIT_MODES = frozenset({"100644", "100755"})
VALID_FILE_STATUSES = frozenset(
    {"added", "modified", "removed", "renamed", "copied", "changed"}
)
TRANSIENT_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

PYPROJECT_PATTERN = re.compile(r"^pyproject_[a-z0-9_]+\.toml$")
REQUIREMENTS_PATTERN = re.compile(r"^requirements(?:[-_][a-z0-9_.-]+)?\.txt$")


class MetadataFetchError(RuntimeError):
    """Sanitized API error retaining only HTTP status code and request identifier."""

    def __init__(
        self, status_code: int | None = None, request_id: str | None = None
    ) -> None:
        self.status_code = status_code
        self.request_id = request_id
        summary = (
            f"status={status_code}" if status_code is not None else "transport_error"
        )
        if request_id:
            summary = f"{summary}, request_id={request_id}"
        else:
            pass
        super().__init__(summary)


@dataclass(frozen=True)
class TreeNode:
    """Git tree object entry pinned to one snapshot."""

    mode: str
    entry_type: str
    sha: str


JsonFetcher = Callable[[str], object]


def retry_delay_seconds(exc: urllib.error.HTTPError, attempt: int) -> float | None:
    """Return a bounded backoff delay for transient HTTP failures, else None."""
    if exc.code not in TRANSIENT_HTTP_CODES:
        return None
    else:
        raw_retry = exc.headers.get("Retry-After") if exc.headers else None
        if raw_retry and raw_retry.isdigit():
            seconds = int(raw_retry)
            if seconds > MAX_RETRY_SLEEP_SECONDS:
                return None
            else:
                return float(seconds)
        else:
            return float(min(1 << attempt, MAX_RETRY_SLEEP_SECONDS))


def github_get_json(token: str, repo: str, path: str) -> object:
    """Issue a size-bounded authenticated GitHub REST GET request with transient retries."""
    url = f"https://api.github.com/repos/{repo}{path}"
    for attempt in range(MAX_API_RETRIES + 1):
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "omni-pr-ci-admission-gate/1.0",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=API_TIMEOUT_SECONDS
            ) as response:
                payload = response.read(MAX_API_RESPONSE_BYTES + 1)
                if len(payload) > MAX_API_RESPONSE_BYTES:
                    raise MetadataFetchError(
                        status_code=getattr(response, "status", 200)
                    )
                else:
                    return json.loads(payload.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            request_id = exc.headers.get("x-github-request-id") if exc.headers else None
            delay = (
                retry_delay_seconds(exc, attempt) if attempt < MAX_API_RETRIES else None
            )
            if delay is not None:
                time.sleep(delay)
                continue
            else:
                raise MetadataFetchError(
                    status_code=exc.code, request_id=request_id
                ) from exc
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            if attempt < MAX_API_RETRIES:
                time.sleep(float(1 << attempt))
                continue
            else:
                raise MetadataFetchError() from exc
    raise MetadataFetchError()


def normalize_repo_path(raw_path: str) -> str | None:
    """Validate and normalize a repository-relative POSIX path while keeping original case."""
    if not raw_path or "\\" in raw_path or raw_path.startswith("/"):
        return None
    elif any(ord(ch) < 32 or ord(ch) == 127 for ch in raw_path):
        return None
    else:
        norm = raw_path
        while norm.startswith("./"):
            norm = norm[2:]
        if not norm:
            return None
        else:
            parts = norm.split("/")
            if any(part in ("", ".", "..") for part in parts):
                return None
            else:
                return norm


def classify_touched_path(raw_path: str) -> str | None:
    """Return a block reason category when a touched path needs manual review, else None."""
    norm = normalize_repo_path(raw_path)
    if norm is None:
        return "invalid_path"
    else:
        lowered = norm.casefold()
        if lowered in ALLOWED_ROOT_FILES:
            return None
        elif not lowered.startswith(ALLOWED_PREFIXES):
            return "outside_allowlist"
        else:
            parts = lowered.split("/")
            if any(segment.startswith(".") for segment in parts):
                return "protected_path"
            elif lowered.startswith(PROTECTED_PREFIXES):
                return "protected_path"
            else:
                basename = parts[-1]
                suffix = posixpath.splitext(basename)[1]
                if (
                    basename in PROTECTED_FILENAMES
                    or PYPROJECT_PATTERN.match(basename)
                    or REQUIREMENTS_PATTERN.match(basename)
                ):
                    return "protected_path"
                elif (
                    lowered in ALLOWED_EXACT_PATHS
                    or basename in ALLOWED_EXTENSIONLESS_FILENAMES
                ):
                    return None
                elif suffix not in ALLOWED_SUFFIXES:
                    return "disallowed_extension"
                else:
                    return None


def is_same_repository_pr(pull_request: dict[str, object], repo: str) -> bool:
    """Return True only when the PR head explicitly comes from the base repository."""
    head = pull_request.get("head")
    if not isinstance(head, dict):
        return False
    else:
        head_repo = head.get("repo")
        if not isinstance(head_repo, dict):
            return False
        else:
            return (
                head_repo.get("fork") is False
                and isinstance(head_repo.get("full_name"), str)
                and head_repo.get("full_name") == repo
            )


def parse_tree_entries(raw_tree: object) -> dict[str, TreeNode]:
    """Parse entries from a GitHub Git tree API response."""
    if not isinstance(raw_tree, dict) or not isinstance(raw_tree.get("tree"), list):
        raise MetadataFetchError()
    else:
        entries: dict[str, TreeNode] = {}
        for item in raw_tree["tree"]:
            if not isinstance(item, dict):
                raise MetadataFetchError()
            else:
                item_path = item.get("path")
                mode = item.get("mode")
                entry_type = item.get("type")
                sha = item.get("sha")
                if (
                    isinstance(item_path, str)
                    and isinstance(mode, str)
                    and isinstance(entry_type, str)
                    and isinstance(sha, str)
                ):
                    entries[item_path] = TreeNode(
                        mode=mode, entry_type=entry_type, sha=sha
                    )
                else:
                    raise MetadataFetchError()
        return entries


def resolve_subtree_entry(
    root_tree_sha: str,
    target_path: str,
    tree_cache: dict[str, dict[str, TreeNode]],
    fetch_json: JsonFetcher,
) -> TreeNode | None:
    """Resolve one file path via non-recursive subtrees when the recursive tree truncates."""
    current_tree_sha = root_tree_sha
    segments = target_path.split("/")
    for index, segment in enumerate(segments):
        if current_tree_sha not in tree_cache:
            raw_subtree = fetch_json(
                f"/git/trees/{urllib.parse.quote(current_tree_sha, safe='')}"
            )
            if (
                not isinstance(raw_subtree, dict)
                or raw_subtree.get("truncated") is True
            ):
                raise MetadataFetchError()
            else:
                tree_cache[current_tree_sha] = parse_tree_entries(raw_subtree)
        else:
            pass
        node = tree_cache[current_tree_sha].get(segment)
        if node is None:
            return None
        elif index == len(segments) - 1:
            return node
        elif node.entry_type != "tree":
            return None
        else:
            current_tree_sha = node.sha
    return None


def resolve_tree_map(
    tree_sha: str,
    target_paths: set[str],
    fetch_json: JsonFetcher,
) -> dict[str, TreeNode]:
    """Resolve exact Git tree entries for target_paths from one pinned tree SHA."""
    if not target_paths:
        return {}
    else:
        raw_tree = fetch_json(
            f"/git/trees/{urllib.parse.quote(tree_sha, safe='')}?recursive=1"
        )
        if not isinstance(raw_tree, dict):
            raise MetadataFetchError()
        elif raw_tree.get("truncated") is not True:
            all_entries = parse_tree_entries(raw_tree)
            return {
                path: all_entries[path] for path in target_paths if path in all_entries
            }
        else:
            subtree_cache: dict[str, dict[str, TreeNode]] = {}
            resolved: dict[str, TreeNode] = {}
            for path in target_paths:
                node = resolve_subtree_entry(tree_sha, path, subtree_cache, fetch_json)
                if node is not None:
                    resolved[path] = node
                else:
                    pass
            return resolved


def extract_live_pull_state(pull_data: object) -> tuple[str, str, str, int]:
    """Validate and extract (base_ref, base_sha, head_sha, changed_files) from /pulls/{n}."""
    if not isinstance(pull_data, dict):
        raise MetadataFetchError()
    else:
        base_obj = pull_data.get("base")
        head_obj = pull_data.get("head")
        changed_files = pull_data.get("changed_files")
        if (
            not isinstance(base_obj, dict)
            or not isinstance(base_obj.get("ref"), str)
            or not base_obj["ref"]
            or not isinstance(base_obj.get("sha"), str)
            or not isinstance(head_obj, dict)
            or not isinstance(head_obj.get("sha"), str)
            or not isinstance(changed_files, int)
            or changed_files < 0
        ):
            raise MetadataFetchError()
        else:
            return base_obj["ref"], base_obj["sha"], head_obj["sha"], changed_files


def write_github_output(outcome: str) -> None:
    """Write the gate outcome to GITHUB_OUTPUT when running in GitHub Actions."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    else:
        with open(output_path, "a", encoding="utf-8") as output_file:
            output_file.write(f"outcome={outcome}\n")


def extract_tree_sha(commit_obj: object) -> str:
    """Validate and extract tree SHA from a commit object."""
    if (
        isinstance(commit_obj, dict)
        and isinstance(commit_obj.get("tree"), dict)
        and isinstance(commit_obj["tree"].get("sha"), str)
    ):
        return commit_obj["tree"]["sha"]
    else:
        raise MetadataFetchError()


def inspect_tree_mode(
    node: TreeNode, categories: set[str], *, check_blob_type: bool
) -> bool:
    """Validate Git tree mode against symlink, submodule, and non-file anomalies."""
    if node.mode == "120000":
        categories.add("symlink_change")
        return False
    elif node.mode == "160000":
        categories.add("submodule_change")
        return False
    elif check_blob_type and (
        node.entry_type != "blob" or node.mode not in ALLOWED_GIT_MODES
    ):
        categories.add("unsupported_blob")
        return False
    else:
        return True


def evaluate_pull_request(
    pull_number: int,
    base_ref: str,
    head_sha: str,
    same_repository: bool,
    fetch_json: JsonFetcher,
) -> tuple[str, list[str]]:
    """Evaluate a pull request strictly against pinned Compare and Git Tree snapshots."""
    if same_repository:
        return "passed", []
    else:
        pass

    live_base_ref, live_base_sha, live_head_before, changed_files_count = (
        extract_live_pull_state(fetch_json(f"/pulls/{pull_number}"))
    )
    if live_head_before != head_sha:
        return "superseded", ["head_sha_mismatch"]
    elif live_base_ref != base_ref:
        return "superseded", ["base_ref_changed"]
    else:
        pass

    compare_data = fetch_json(
        f"/compare/{urllib.parse.quote(live_base_sha, safe='')}...{urllib.parse.quote(head_sha, safe='')}?per_page=1"
    )
    if not isinstance(compare_data, dict) or not isinstance(
        compare_data.get("files"), list
    ):
        raise MetadataFetchError()
    else:
        pass

    files_raw = compare_data["files"]
    if len(files_raw) != changed_files_count:
        return "blocked", ["snapshot_mismatch"]
    else:
        pass

    categories: set[str] = set()
    head_targets: set[str] = set()
    base_targets: set[str] = set()
    seen_casefold_paths: set[str] = set()
    parsed_entries: list[tuple[dict[str, object], str, str, str | None]] = []

    for entry in files_raw:
        if not isinstance(entry, dict):
            return "blocked", ["snapshot_mismatch"]
        else:
            pass
        status = entry.get("status")
        raw_filename = entry.get("filename")
        raw_previous = entry.get("previous_filename")
        if (
            not isinstance(status, str)
            or status not in VALID_FILE_STATUSES
            or not isinstance(raw_filename, str)
        ):
            categories.add("snapshot_mismatch")
            continue
        else:
            pass

        norm_filename = normalize_repo_path(raw_filename)
        if norm_filename is None:
            categories.add("invalid_path")
            continue
        else:
            pass
        path_reason = classify_touched_path(norm_filename)
        if path_reason is not None:
            categories.add(path_reason)
        else:
            pass

        norm_previous: str | None = None
        if status in ("renamed", "copied") or (
            isinstance(raw_previous, str) and raw_previous
        ):
            if not isinstance(raw_previous, str) or not raw_previous:
                categories.add("snapshot_mismatch")
                continue
            else:
                norm_previous = normalize_repo_path(raw_previous)
                if norm_previous is None:
                    categories.add("invalid_path")
                    continue
                else:
                    prev_reason = classify_touched_path(norm_previous)
                    if prev_reason is not None:
                        categories.add(prev_reason)
                    else:
                        pass
        else:
            pass

        folded_filename = norm_filename.casefold()
        if status != "removed":
            if folded_filename in seen_casefold_paths:
                categories.add("case_collision")
            else:
                seen_casefold_paths.add(folded_filename)
            head_targets.add(norm_filename)
        else:
            base_targets.add(norm_filename)

        if norm_previous is not None:
            base_targets.add(norm_previous)
        elif status == "modified":
            base_targets.add(norm_filename)
        else:
            pass

        parsed_entries.append((entry, status, norm_filename, norm_previous))

    if categories:
        return "blocked", sorted(categories)
    else:
        pass

    head_tree_sha = extract_tree_sha(
        fetch_json(f"/git/commits/{urllib.parse.quote(head_sha, safe='')}")
    )
    merge_base_commit = compare_data.get("merge_base_commit")
    if isinstance(merge_base_commit, dict) and isinstance(
        merge_base_commit.get("commit"), dict
    ):
        base_tree_sha = extract_tree_sha(merge_base_commit["commit"])
    else:
        base_tree_sha = extract_tree_sha(
            fetch_json(f"/git/commits/{urllib.parse.quote(live_base_sha, safe='')}")
        )

    head_tree_map = resolve_tree_map(head_tree_sha, head_targets, fetch_json)
    base_tree_map = resolve_tree_map(base_tree_sha, base_targets, fetch_json)

    for entry, status, norm_filename, norm_previous in parsed_entries:
        if status == "removed":
            base_node = base_tree_map.get(norm_filename)
            if base_node is not None:
                inspect_tree_mode(base_node, categories, check_blob_type=False)
            else:
                pass
            continue
        else:
            pass

        head_node = head_tree_map.get(norm_filename)
        if head_node is None:
            categories.add("snapshot_mismatch")
            continue
        else:
            pass
        if not inspect_tree_mode(head_node, categories, check_blob_type=True):
            continue
        else:
            pass

        entry_sha = entry.get("sha")
        if isinstance(entry_sha, str) and entry_sha and entry_sha != head_node.sha:
            categories.add("snapshot_mismatch")
            continue
        else:
            pass

        base_lookup = norm_previous if norm_previous is not None else norm_filename
        base_node = base_tree_map.get(base_lookup)
        if base_node is not None and not inspect_tree_mode(
            base_node, categories, check_blob_type=False
        ):
            continue
        else:
            pass

        if head_node.mode == "100755" and (
            status == "added" or base_node is None or base_node.mode != "100755"
        ):
            categories.add("executable_mode_change")
        else:
            pass

    live_base_ref_after, _, live_head_after, _ = extract_live_pull_state(
        fetch_json(f"/pulls/{pull_number}")
    )
    if live_head_after != head_sha:
        return "superseded", ["head_sha_mismatch"]
    elif live_base_ref_after != base_ref:
        return "superseded", ["base_ref_changed"]
    elif categories:
        return "blocked", sorted(categories)
    else:
        return "passed", []


def main() -> int:
    """CLI entrypoint for the pull_request_target admission gate."""
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_name != "pull_request_target":
        write_github_output("error")
        print(
            "::error::CI Admission Gate requires pull_request_target.", file=sys.stderr
        )
        return 1
    else:
        pass

    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not token or not repo or not event_path:
        write_github_output("error")
        print(
            "::error::CI Admission Gate missing required GitHub environment variables.",
            file=sys.stderr,
        )
        return 1
    else:
        pass

    try:
        with open(event_path, encoding="utf-8") as event_file:
            event = json.load(event_file)
        pull_request = event["pull_request"]
        pull_number = int(pull_request["number"])
        base_ref = str(pull_request["base"]["ref"])
        head_sha = str(pull_request["head"]["sha"])

        def fetch_repo_json(api_path: str) -> object:
            return github_get_json(token, repo, api_path)

        same_repository = is_same_repository_pr(pull_request, repo)
        outcome, _ = evaluate_pull_request(
            pull_number=pull_number,
            base_ref=base_ref,
            head_sha=head_sha,
            same_repository=same_repository,
            fetch_json=fetch_repo_json,
        )
    except MetadataFetchError as exc:
        write_github_output("error")
        print(
            f"::error::CI Admission Gate failed closed due to a metadata verification error ({exc}).",
            file=sys.stderr,
        )
        return 1
    except Exception:
        write_github_output("error")
        print(
            "::error::CI Admission Gate failed closed due to an internal error.",
            file=sys.stderr,
        )
        return 1
    else:
        write_github_output(outcome)
        if outcome == "passed":
            print("CI Admission Gate passed.")
            return 0
        elif outcome == "superseded":
            print(
                "CI Admission Gate superseded by a newer pull request update.",
                file=sys.stderr,
            )
            return 1
        else:
            print(
                "::error::CI Admission Gate requires manual maintainer review.",
                file=sys.stderr,
            )
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
