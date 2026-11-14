"""Track refactor net line deletion while excluding test files."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

TEST_DIRECTORY_NAMES = frozenset(
    {
        "test",
        "tests",
        "unit_test",
        "unit_tests",
        "integration_test",
        "integration_tests",
    }
)
TEST_FILE_NAMES = frozenset({"conftest.py"})
TEST_FILE_SUFFIXES = (
    "_test.py",
    "_tests.py",
    ".test.js",
    ".test.jsx",
    ".test.ts",
    ".test.tsx",
    ".spec.js",
    ".spec.jsx",
    ".spec.ts",
    ".spec.tsx",
)
GITHUB_BLOB_BASE_URL = "https://github.com/sgl-project/sglang-omni/blob/main/"
TTS_REFACTOR_DESIGN_DOC_URL = (
    "https://osgbw74w8zwb.sg.larksuite.com/docx/L3XId4GHYoJqSKxwXYqlXaBrgNd"
)
TTS_REFACTOR_SHARED_SURFACES = (
    (
        "PipelineStateBase",
        "sglang_omni/scheduling/pipeline_state.py",
        "Per-request state serialization, tensor-safe value handling, and usage accounting.",
    ),
    (
        "TtsEngineBuilder",
        "sglang_omni/scheduling/engine_factory.py",
        "Template for TTS SGLang AR engine startup, model-local hooks, and scheduler construction.",
    ),
    (
        "ReferenceEncodeService",
        "sglang_omni/scheduling/reference_encoder.py",
        "Bounded cache and same-key single-flight for uploaded/reference audio artifacts.",
    ),
    (
        "StageOutputCache",
        "sglang_omni/scheduling/stage_cache.py",
        "Small bounded LRU cache for non-AR stage outputs and encoded artifacts.",
    ),
    (
        "BatchVocoderBase",
        "sglang_omni/scheduling/vocoder_base.py",
        "SimpleScheduler-backed base for batched non-streaming vocoder stages.",
    ),
    (
        "ModelCapabilities",
        "sglang_omni/models/model_capabilities.py",
        "Static model capability declarations for routing, docs, and review checks.",
    ),
    (
        "SpeakerArtifactCache",
        "sglang_omni/scheduling/speaker_cache.py",
        "Process-wide LRU cache for uploaded-speaker feature artifacts.",
    ),
    (
        "ReferenceEncodeService docs",
        "docs/developer_reference/reference_encode_service.md",
        "Usage rules, cache-key contract, and migration notes for reference encoding.",
    ),
)


@dataclass(frozen=True)
class FileStat:
    paths: tuple[str, ...]
    added: int | None
    deleted: int | None
    is_test: bool

    @property
    def display_path(self) -> str:
        if len(self.paths) == 1:
            return self.paths[0]
        return f"{self.paths[0]} -> {self.paths[-1]}"


@dataclass
class Totals:
    files: int = 0
    binary_files: int = 0
    added: int = 0
    deleted: int = 0

    @property
    def net_deleted(self) -> int:
        return self.deleted - self.added

    @property
    def net_change(self) -> int:
        return self.added - self.deleted

    def add(self, file_stat: FileStat) -> None:
        self.files += 1
        if file_stat.added is None or file_stat.deleted is None:
            self.binary_files += 1
            return
        self.added += file_stat.added
        self.deleted += file_stat.deleted

    def merge(self, other: Totals) -> None:
        self.files += other.files
        self.binary_files += other.binary_files
        self.added += other.added
        self.deleted += other.deleted

    def to_dict(self) -> dict[str, int]:
        return {
            "files": self.files,
            "binary_files": self.binary_files,
            "added": self.added,
            "deleted": self.deleted,
            "net_deleted": self.net_deleted,
        }


@dataclass(frozen=True)
class DiffSpec:
    label: str
    git_args: tuple[str, ...]


@dataclass(frozen=True)
class CommitSummary:
    revision: str
    subject: str
    totals: dict[str, Totals]


@dataclass(frozen=True)
class Report:
    repo: str
    diff_label: str
    mode: str
    totals: dict[str, Totals]
    files: tuple[FileStat, ...]
    pathspecs: tuple[str, ...] = ()
    commits: tuple[CommitSummary, ...] = ()

    @property
    def target_met(self) -> bool:
        return self.totals["non_test"].net_deleted > 0


def is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = tuple(part for part in PurePosixPath(normalized).parts if part != ".")
    if not parts:
        return False

    if any(part.lower() in TEST_DIRECTORY_NAMES for part in parts[:-1]):
        return True

    name = parts[-1].lower()
    if name in TEST_FILE_NAMES:
        return True
    if name.startswith("test_"):
        return True
    return any(name.endswith(suffix) for suffix in TEST_FILE_SUFFIXES)


def parse_numstat_z(data: bytes) -> tuple[FileStat, ...]:
    tokens = data.split(b"\0")
    if tokens and tokens[-1] == b"":
        tokens.pop()

    results: list[FileStat] = []
    index = 0
    while index < len(tokens):
        header = _decode_git_path(tokens[index])
        fields = header.split("\t", 2)
        if len(fields) != 3:
            raise ValueError(f"invalid --numstat -z record header: {header!r}")

        added = _parse_numstat_count(fields[0])
        deleted = _parse_numstat_count(fields[1])
        path_field = fields[2]

        if path_field:
            paths = (path_field,)
            index += 1
        else:
            if index + 2 >= len(tokens):
                raise ValueError("rename --numstat -z record is missing paths")
            paths = (
                _decode_git_path(tokens[index + 1]),
                _decode_git_path(tokens[index + 2]),
            )
            index += 3

        results.append(
            FileStat(
                paths=paths,
                added=added,
                deleted=deleted,
                is_test=any(is_test_path(path) for path in paths),
            )
        )

    return tuple(results)


def build_report(
    repo: Path, diff_spec: DiffSpec, mode: str, pathspecs: Sequence[str] = ()
) -> Report:
    path_args = ("--", *pathspecs) if pathspecs else ()
    raw = _run_git(
        repo,
        (
            "diff",
            "--numstat",
            "-z",
            "--find-renames",
            *diff_spec.git_args,
            *path_args,
        ),
    )
    files = parse_numstat_z(raw)

    all_totals = Totals()
    test_totals = Totals()
    non_test_totals = Totals()
    for file_stat in files:
        all_totals.add(file_stat)
        if file_stat.is_test:
            test_totals.add(file_stat)
        else:
            non_test_totals.add(file_stat)

    return Report(
        repo=str(repo),
        diff_label=diff_spec.label,
        mode=mode,
        totals={
            "non_test": non_test_totals,
            "test": test_totals,
            "all": all_totals,
        },
        files=files,
        pathspecs=tuple(pathspecs),
    )


def select_commits(
    repo: Path,
    base: str,
    head: str,
    title_prefix: str | None,
    explicit_commits: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    raw = _run_git(
        repo,
        ("log", "--reverse", "--format=%H%x09%s", f"{base}..{head}"),
    )
    history = tuple(
        line.split("\t", 1)
        for line in raw.decode("utf-8", "replace").splitlines()
        if line
    )
    explicit_revisions = {
        _run_git(repo, ("rev-parse", commit)).decode().strip()
        for commit in explicit_commits
    }
    history_revisions = {revision for revision, _ in history}
    missing = explicit_revisions - history_revisions
    if missing:
        raise ValueError(
            "explicit commits are outside the selected history: "
            + ", ".join(sorted(missing))
        )

    selected = tuple(
        (revision, subject)
        for revision, subject in history
        if revision in explicit_revisions
        or (title_prefix is not None and subject.startswith(title_prefix))
    )
    if not selected:
        raise ValueError("commit-sum mode selected no commits")
    return selected


def load_commit_file(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported commit file schema: {path}")
    commits = payload.get("commits")
    if not isinstance(commits, list) or not commits:
        raise ValueError(f"commit file has no commits: {path}")

    revisions = []
    for entry in commits:
        if not isinstance(entry, dict) or not isinstance(entry.get("revision"), str):
            raise TypeError(f"invalid commit entry in {path}")
        revisions.append(entry["revision"])
    if len(revisions) != len(set(revisions)):
        raise ValueError(f"commit file contains duplicate revisions: {path}")
    return tuple(revisions)


def build_commit_sum_report(
    repo: Path,
    commits: Sequence[tuple[str, str]],
    base: str,
    head: str,
    pathspecs: Sequence[str] = (),
) -> Report:
    totals = {
        "non_test": Totals(),
        "test": Totals(),
        "all": Totals(),
    }
    files: list[FileStat] = []
    commit_summaries: list[CommitSummary] = []
    for revision, subject in commits:
        commit_report = build_report(
            repo,
            DiffSpec(
                label=f"{revision}^..{revision}",
                git_args=(f"{revision}^", revision),
            ),
            "commit",
            pathspecs,
        )
        files.extend(commit_report.files)
        for category, commit_totals in commit_report.totals.items():
            totals[category].merge(commit_totals)
        commit_summaries.append(
            CommitSummary(
                revision=revision,
                subject=subject,
                totals=commit_report.totals,
            )
        )

    return Report(
        repo=str(repo),
        diff_label=f"{len(commits)} allowlisted commits in {base}..{head}",
        mode="commit-sum",
        totals=totals,
        files=tuple(files),
        pathspecs=tuple(pathspecs),
        commits=tuple(commit_summaries),
    )


def format_text(
    report: Report, list_test_files: bool, list_non_test_files: bool
) -> str:
    lines = [
        "Refactor net deletion tracking",
        f"Repo: {report.repo}",
        f"Diff: {report.diff_label}",
        f"Mode: {report.mode}",
        f"Paths: {_format_pathspecs(report.pathspecs)}",
        "",
        _format_table(report),
        "",
        f"Target: non-test net deleted > 0 ({_format_target(report)})",
        "",
        "Test files are excluded from the progress target.",
    ]

    if list_test_files:
        label = (
            "Excluded test file touches:" if report.commits else "Excluded test files:"
        )
        lines.extend(["", label])
        lines.extend(_format_file_list(report.files, is_test=True))

    if list_non_test_files:
        label = (
            "Counted non-test file touches:"
            if report.commits
            else "Counted non-test files:"
        )
        lines.extend(["", label])
        lines.extend(_format_file_list(report.files, is_test=False))

    if report.commits:
        lines.extend(["", "Counted commits:"])
        lines.extend(_format_commit_lines(report.commits))

    return "\n".join(lines)


def format_markdown(
    report: Report, list_test_files: bool, list_non_test_files: bool
) -> str:
    file_label = "Touches" if report.commits else "Files"
    lines = [
        "### Refactor Net Deletion Tracking",
        "",
        f"- Diff: `{report.diff_label}`",
        f"- Mode: `{report.mode}`",
        f"- Paths: `{_format_pathspecs(report.pathspecs)}`",
        f"- Target: non-test net deleted > 0 (**{_format_target(report)}**)",
        "",
        f"| Category | {file_label} | Binary | Added | Deleted | Net deleted |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, label in (("non_test", "Non-test"), ("test", "Test"), ("all", "All")):
        totals = report.totals[key]
        lines.append(
            "| "
            f"{label} | {totals.files} | {totals.binary_files} | "
            f"{totals.added} | {totals.deleted} | {totals.net_deleted} |"
        )

    if list_test_files:
        label = (
            "Excluded test file touches" if report.commits else "Excluded test files"
        )
        lines.extend(["", f"<details><summary>{label}</summary>", ""])
        lines.extend(f"- `{path}`" for path in _matching_paths(report.files, True))
        lines.extend(["", "</details>"])

    if list_non_test_files:
        label = (
            "Counted non-test file touches"
            if report.commits
            else "Counted non-test files"
        )
        lines.extend(["", f"<details><summary>{label}</summary>", ""])
        lines.extend(f"- `{path}`" for path in _matching_paths(report.files, False))
        lines.extend(["", "</details>"])

    if report.commits:
        lines.extend(
            [
                "",
                "| Commit | Subject | Non-test added | Non-test deleted | Net deleted |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for commit in report.commits:
            totals = commit.totals["non_test"]
            lines.append(
                f"| `{commit.revision[:8]}` | {commit.subject} | "
                f"{totals.added} | {totals.deleted} | {totals.net_deleted} |"
            )

    return "\n".join(lines)


def format_json(report: Report) -> str:
    payload = {
        "repo": report.repo,
        "diff": report.diff_label,
        "mode": report.mode,
        "pathspecs": report.pathspecs,
        "target": {
            "name": "non_test_net_deleted_gt_zero",
            "met": report.target_met,
        },
        "totals": {key: value.to_dict() for key, value in report.totals.items()},
        "commits": [
            {
                "revision": commit.revision,
                "subject": commit.subject,
                "totals": {
                    key: value.to_dict() for key, value in commit.totals.items()
                },
            }
            for commit in report.commits
        ],
        "files": [
            {
                **asdict(file_stat),
                "display_path": file_stat.display_path,
            }
            for file_stat in report.files
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def format_html(
    report: Report,
    title: str,
    issue_url: str | None,
    refresh_seconds: int,
    list_test_files: bool,
    list_non_test_files: bool,
    scope_notes: Sequence[str] = (),
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    status_class = "met" if report.target_met else "miss"
    status_label = "On track" if report.target_met else "Needs deletion"
    issue_link = ""
    if issue_url:
        issue_link = (
            f'<a class="link" href="{html.escape(issue_url, quote=True)}">'
            "Roadmap issue</a>"
        )

    refresh_meta = ""
    if refresh_seconds > 0:
        refresh_meta = f'<meta http-equiv="refresh" content="{refresh_seconds}">'

    scope_notes_html = "".join(
        f'<div class="scope-note">{html.escape(note)}</div>' for note in scope_notes
    )

    file_sections = []
    if list_non_test_files:
        section_title = (
            "Counted non-test file touches"
            if report.commits
            else "Counted non-test files"
        )
        file_sections.append(
            _format_html_file_section(section_title, report.files, False)
        )
    if list_test_files:
        section_title = (
            "Excluded test file touches" if report.commits else "Excluded test files"
        )
        file_sections.append(
            _format_html_file_section(section_title, report.files, True)
        )
    commit_section = _format_html_commit_section(report.commits)
    file_metric_label = "Touches" if report.commits else "Files"

    rendered = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {refresh_meta}
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --border: #d9dee8;
      --green: #157f3b;
      --green-bg: #e8f6ee;
      --red: #b42318;
      --red-bg: #fdecec;
      --amber: #b54708;
      --blue: #175cd3;
      --shadow: 0 1px 2px rgba(16, 24, 40, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
      font-weight: 720;
    }}
    .subline {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .status {{
      min-width: 156px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: var(--shadow);
      padding: 12px 14px;
      text-align: right;
    }}
    .status .label {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    .status.met .label {{ color: var(--green); background: var(--green-bg); }}
    .status.miss .label {{ color: var(--red); background: var(--red-bg); }}
    .status .target {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .card h2 {{
      margin: 0 0 12px;
      font-size: 14px;
      line-height: 1.25;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
    }}
    .net {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .net strong {{
      font-size: 34px;
      line-height: 1;
      font-weight: 760;
    }}
    .net span {{
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .metric-row {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }}
    .metric .name {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }}
    .metric .value {{
      font-size: 16px;
      font-weight: 700;
    }}
    .decrease {{ color: var(--green); }}
    .increase {{ color: var(--red); }}
    .zero {{ color: var(--amber); }}
    .context {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }}
    .context code {{
      color: var(--text);
      background: #eef1f6;
      padding: 1px 5px;
      border-radius: 4px;
    }}
    .scope-note {{
      margin-top: 8px;
      color: var(--text);
    }}
    .link {{ color: var(--blue); text-decoration: none; font-weight: 700; }}
    .link:hover {{ text-decoration: underline; }}
    .resource-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .resource {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
      min-width: 0;
    }}
    .resource a {{
      color: var(--blue);
      text-decoration: none;
      font-size: 13px;
      font-weight: 760;
      overflow-wrap: anywhere;
    }}
    .resource a:hover {{ text-decoration: underline; }}
    .resource code {{
      display: block;
      margin-top: 6px;
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }}
    .resource p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .files {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .file-list {{
      margin: 0;
      padding: 0;
      list-style: none;
      max-height: 360px;
      overflow: auto;
      border-top: 1px solid var(--border);
    }}
    .file-list li {{
      padding: 8px 0;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .commit-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .commit-table-wrap {{
      overflow-x: auto;
    }}
    .commit-table th, .commit-table td {{
      border-bottom: 1px solid var(--border);
      padding: 8px;
      text-align: right;
      vertical-align: top;
    }}
    .commit-table th:first-child, .commit-table td:first-child,
    .commit-table th:nth-child(2), .commit-table td:nth-child(2) {{
      text-align: left;
    }}
    .commit-table td:nth-child(2) {{
      max-width: 520px;
    }}
    @media (max-width: 840px) {{
      header {{ display: block; }}
      .status {{ text-align: left; margin-top: 12px; }}
      .grid, .files, .resource-grid {{ grid-template-columns: 1fr; }}
      .metric-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{html.escape(title)}</h1>
        <div class="subline">
          <span>Generated {html.escape(generated_at)}</span>
          <span>Diff <code>{html.escape(report.diff_label)}</code></span>
          <span>Mode <code>{html.escape(report.mode)}</code></span>
          <span>Paths <code>{html.escape(_format_pathspecs(report.pathspecs))}</code></span>
          {issue_link}
        </div>
      </div>
      <div class="status {status_class}">
        <span class="label">{status_label}</span>
        <div class="target">non-test net change &lt; 0</div>
      </div>
    </header>

    <section class="grid">
      {_format_html_total_card("Non-test", report.totals["non_test"], file_metric_label)}
      {_format_html_total_card("Test", report.totals["test"], file_metric_label)}
      {_format_html_total_card("All", report.totals["all"], file_metric_label)}
    </section>

    <section class="context">
      Net change is displayed as <code>added lines - deleted lines</code>.
      Negative values mean the repository has fewer lines. Files classified as
      tests are reported separately and do not offset the implementation
      deletion target.
      {scope_notes_html}
    </section>

    {commit_section}

    {_format_html_resources()}

    {"".join(file_sections)}
  </main>
</body>
</html>"""
    return "\n".join(line.rstrip() for line in rendered.splitlines())


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track refactor net line deletion while excluding test files from "
            "the progress target."
        )
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository path. Defaults to the current directory.",
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base ref for branch comparisons. Defaults to origin/main.",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head ref for branch comparisons. Defaults to HEAD.",
    )
    parser.add_argument(
        "--range",
        dest="diff_range",
        help="Explicit git diff range, such as origin/main...HEAD.",
    )
    parser.add_argument(
        "--mode",
        choices=("merge-base", "direct", "worktree", "staged", "commit-sum"),
        default="merge-base",
        help=(
            "Diff mode: merge-base uses base...head, direct uses base head, "
            "worktree compares the working tree with merge-base(base, head), "
            "staged compares the index with merge-base(base, head), and "
            "commit-sum adds the first-parent diffs of allowlisted commits."
        ),
    )
    parser.add_argument(
        "--commit-title-prefix",
        help="In commit-sum mode, include commits whose subjects start with this.",
    )
    parser.add_argument(
        "--commit",
        action="append",
        default=[],
        help="In commit-sum mode, include this commit explicitly. May be repeated.",
    )
    parser.add_argument(
        "--commit-file",
        type=Path,
        help="In commit-sum mode, load explicit commits from this JSON file.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json", "html"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Limit the diff to this git pathspec. May be repeated.",
    )
    parser.add_argument(
        "--output",
        help="Write output to this path instead of stdout.",
    )
    parser.add_argument(
        "--title",
        default="TTS Refactor Progress",
        help="Dashboard title used by HTML output.",
    )
    parser.add_argument(
        "--issue-url",
        default="https://github.com/sgl-project/sglang-omni/issues/985",
        help="Roadmap issue URL used by HTML output. Pass an empty string to omit.",
    )
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=0,
        help="Add an HTML meta-refresh interval in seconds. Defaults to disabled.",
    )
    parser.add_argument(
        "--scope-note",
        action="append",
        default=[],
        help="Add a scope note to HTML output. May be repeated.",
    )
    parser.add_argument(
        "--list-test-files",
        action="store_true",
        help="List changed test files excluded from the progress target.",
    )
    parser.add_argument(
        "--list-non-test-files",
        action="store_true",
        help="List changed non-test files counted toward the progress target.",
    )
    parser.add_argument(
        "--fail-on-nonpositive",
        action="store_true",
        help="Exit with status 1 when non-test net deleted is not positive.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = Path(args.repo).resolve()

    if args.mode == "commit-sum":
        if args.diff_range:
            raise ValueError("--range cannot be used with commit-sum mode")
        explicit_commits = list(args.commit)
        if args.commit_file:
            commit_file = args.commit_file
            if not commit_file.is_absolute():
                commit_file = repo / commit_file
            explicit_commits.extend(load_commit_file(commit_file))
        commits = select_commits(
            repo,
            args.base,
            args.head,
            args.commit_title_prefix,
            explicit_commits,
        )
        report = build_commit_sum_report(
            repo,
            commits,
            args.base,
            args.head,
            args.path,
        )
    else:
        if args.commit_title_prefix or args.commit or args.commit_file:
            raise ValueError("commit selectors require commit-sum mode")
        diff_spec = resolve_diff_spec(repo, args)
        report = build_report(repo, diff_spec, args.mode, args.path)

    if args.format == "json":
        output = format_json(report)
    elif args.format == "markdown":
        output = format_markdown(report, args.list_test_files, args.list_non_test_files)
    elif args.format == "html":
        issue_url = args.issue_url or None
        output = format_html(
            report,
            args.title,
            issue_url,
            args.refresh_seconds,
            args.list_test_files,
            args.list_non_test_files,
            args.scope_note,
        )
    else:
        output = format_text(report, args.list_test_files, args.list_non_test_files)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    if args.fail_on_nonpositive and not report.target_met:
        return 1
    return 0


def resolve_diff_spec(repo: Path, args: argparse.Namespace) -> DiffSpec:
    if args.diff_range:
        return DiffSpec(label=args.diff_range, git_args=(args.diff_range,))

    if args.mode == "merge-base":
        label = f"{args.base}...{args.head}"
        return DiffSpec(label=label, git_args=(label,))

    if args.mode == "direct":
        label = f"{args.base}..{args.head}"
        return DiffSpec(label=label, git_args=(args.base, args.head))

    merge_base = _run_git(repo, ("merge-base", args.base, args.head)).decode().strip()
    if args.mode == "worktree":
        return DiffSpec(
            label=f"working tree vs merge-base({args.base}, {args.head}) {merge_base}",
            git_args=(merge_base,),
        )
    if args.mode == "staged":
        return DiffSpec(
            label=f"staged vs merge-base({args.base}, {args.head}) {merge_base}",
            git_args=("--cached", merge_base),
        )

    raise ValueError(f"unsupported mode: {args.mode}")


def _run_git(repo: Path, args: Sequence[str]) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(repo), *args),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {stderr}")
    return completed.stdout


def _decode_git_path(value: bytes) -> str:
    return value.decode("utf-8", "surrogateescape")


def _parse_numstat_count(value: str) -> int | None:
    if value == "-":
        return None
    return int(value)


def _format_table(report: Report) -> str:
    file_label = "Touches" if report.commits else "Files"
    rows = [("Category", file_label, "Binary", "Added", "Deleted", "Net deleted")]
    for key, label in (("non_test", "non-test"), ("test", "test"), ("all", "all")):
        totals = report.totals[key]
        rows.append(
            (
                label,
                str(totals.files),
                str(totals.binary_files),
                str(totals.added),
                str(totals.deleted),
                str(totals.net_deleted),
            )
        )

    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.rjust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    )


def _format_target(report: Report) -> str:
    return "met" if report.target_met else "not met"


def _format_pathspecs(pathspecs: Sequence[str]) -> str:
    if not pathspecs:
        return "(entire repository)"
    return ", ".join(pathspecs)


def _format_html_total_card(
    label: str, totals: Totals, file_metric_label: str = "Files"
) -> str:
    net_class = "decrease" if totals.net_change < 0 else "increase"
    if totals.net_change == 0:
        net_class = "zero"

    return f"""
      <article class="card">
        <h2>{html.escape(label)}</h2>
        <div class="net">
          <strong class="{net_class}">{totals.net_change:+d}</strong>
          <span>net change</span>
        </div>
        <div class="metric-row">
          {_format_html_metric("Added", totals.added)}
          {_format_html_metric("Deleted", totals.deleted)}
          {_format_html_metric(file_metric_label, totals.files)}
          {_format_html_metric("Binary", totals.binary_files)}
        </div>
      </article>"""


def _format_html_metric(label: str, value: int) -> str:
    return (
        '<div class="metric">'
        f'<div class="name">{html.escape(label)}</div>'
        f'<div class="value">{value}</div>'
        "</div>"
    )


def _format_html_file_section(
    title: str, files: Sequence[FileStat], is_test: bool
) -> str:
    paths = _matching_paths(files, is_test)
    if paths:
        items = "\n".join(f"<li>{html.escape(path)}</li>" for path in paths)
    else:
        items = '<li class="empty">none</li>'
    return f"""
    <section class="card" style="margin-top: 12px;">
      <h2>{html.escape(title)}</h2>
      <ul class="file-list">
        {items}
      </ul>
    </section>"""


def _format_html_commit_section(commits: Sequence[CommitSummary]) -> str:
    if not commits:
        return ""
    rows = []
    for commit in commits:
        non_test = commit.totals["non_test"]
        test = commit.totals["test"]
        revision = html.escape(commit.revision[:8])
        rows.append(
            "<tr>"
            f'<td><a class="link" href="'
            f"https://github.com/sgl-project/sglang-omni/commit/{commit.revision}"
            f'">{revision}</a></td>'
            f"<td>{html.escape(commit.subject)}</td>"
            f"<td>{non_test.added}</td>"
            f"<td>{non_test.deleted}</td>"
            f"<td>{non_test.net_change:+d}</td>"
            f"<td>{test.net_change:+d}</td>"
            "</tr>"
        )
    return f"""
    <section class="card" style="margin-top: 12px;">
      <h2>Counted refactor commits</h2>
      <div class="commit-table-wrap">
        <table class="commit-table">
          <thead>
            <tr>
              <th>Commit</th>
              <th>Subject</th>
              <th>Non-test added</th>
              <th>Non-test deleted</th>
              <th>Non-test net</th>
              <th>Test net</th>
            </tr>
          </thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
      </div>
    </section>"""


def _format_html_resources() -> str:
    design_url = html.escape(TTS_REFACTOR_DESIGN_DOC_URL, quote=True)
    items = [
        (
            "TTS refactor design doc",
            TTS_REFACTOR_DESIGN_DOC_URL,
            "External RFC/design source for the TTS refactor roadmap and shared-surface decisions.",
            "",
        )
    ]
    for title, path, description in TTS_REFACTOR_SHARED_SURFACES:
        items.append(
            (
                title,
                f"{GITHUB_BLOB_BASE_URL}{path}",
                description,
                path,
            )
        )

    rendered = "\n".join(
        _format_html_resource_item(title, url, description, path)
        for title, url, description, path in items
    )
    return f"""
    <section class="card" style="margin-top: 12px;">
      <h2>Reusable shared surfaces</h2>
      <div class="context" style="box-shadow: none; margin-bottom: 10px;">
        Start here before adding new model-local code. Reuse these shared files
        when a new refactor touches state, engine construction, reference
        encoding, vocoder mechanics, capability metadata, or stage caches.
        <a class="link" href="{design_url}">Open the design doc</a>.
      </div>
      <div class="resource-grid">
        {rendered}
      </div>
    </section>"""


def _format_html_resource_item(
    title: str, url: str, description: str, path: str
) -> str:
    path_line = f"<code>{html.escape(path)}</code>" if path else ""
    return f"""
        <article class="resource">
          <a href="{html.escape(url, quote=True)}">{html.escape(title)}</a>
          {path_line}
          <p>{html.escape(description)}</p>
        </article>"""


def _matching_paths(files: Sequence[FileStat], is_test: bool) -> tuple[str, ...]:
    return tuple(
        file_stat.display_path for file_stat in files if file_stat.is_test is is_test
    )


def _format_file_list(files: Sequence[FileStat], is_test: bool) -> list[str]:
    paths = _matching_paths(files, is_test)
    if not paths:
        return ["  (none)"]
    return [f"  {path}" for path in paths]


def _format_commit_lines(commits: Sequence[CommitSummary]) -> list[str]:
    return [
        "  "
        f"{commit.revision[:8]} {commit.subject}: "
        f"{commit.totals['non_test'].added} added, "
        f"{commit.totals['non_test'].deleted} deleted, "
        f"{commit.totals['non_test'].net_deleted:+d} net deleted"
        for commit in commits
    ]


if __name__ == "__main__":
    raise SystemExit(main())
