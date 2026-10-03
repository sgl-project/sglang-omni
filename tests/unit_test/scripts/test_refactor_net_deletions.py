from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "refactor_net_deletions.py"
)
SPEC = importlib.util.spec_from_file_location("refactor_net_deletions", SCRIPT_PATH)
assert SPEC is not None
refactor_net_deletions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = refactor_net_deletions
SPEC.loader.exec_module(refactor_net_deletions)


def test_detects_repo_test_paths() -> None:
    assert refactor_net_deletions.is_test_path("tests/utils.py")
    assert refactor_net_deletions.is_test_path("tests/data/query_to_draw.wav")
    assert refactor_net_deletions.is_test_path(
        "tests/unit_test/serve/test_openai_api.py"
    )
    assert refactor_net_deletions.is_test_path(
        "sglang_omni/models/foo/test_request_builders.py"
    )
    assert refactor_net_deletions.is_test_path("sglang_omni/models/foo/request_test.py")
    assert refactor_net_deletions.is_test_path("web/client/audio.spec.ts")
    assert refactor_net_deletions.is_test_path("conftest.py")


def test_does_not_treat_non_test_names_as_tests() -> None:
    assert not refactor_net_deletions.is_test_path("docs/developer_reference/main.md")
    assert not refactor_net_deletions.is_test_path(".github/workflows/test.yaml")
    assert not refactor_net_deletions.is_test_path(
        "sglang_omni/models/moss_tts/stages.py"
    )
    assert not refactor_net_deletions.is_test_path(
        "scripts/ci/utils/slash_command_handler.py"
    )


def test_parse_numstat_z_handles_regular_and_renamed_files() -> None:
    raw = (
        b"3\t10\tsglang_omni/foo.py\0"
        b"5\t1\t\0tests/unit_test/old_test.py\0tests/unit_test/new_test.py\0"
        b"-\t-\ttests/data/audio.wav\0"
    )

    stats = refactor_net_deletions.parse_numstat_z(raw)

    assert len(stats) == 3
    assert stats[0].paths == ("sglang_omni/foo.py",)
    assert stats[0].added == 3
    assert stats[0].deleted == 10
    assert not stats[0].is_test

    assert stats[1].paths == (
        "tests/unit_test/old_test.py",
        "tests/unit_test/new_test.py",
    )
    assert stats[1].added == 5
    assert stats[1].deleted == 1
    assert stats[1].is_test

    assert stats[2].added is None
    assert stats[2].deleted is None
    assert stats[2].is_test


def test_build_report_splits_test_and_non_test_totals() -> None:
    files = (
        refactor_net_deletions.FileStat(
            paths=("sglang_omni/foo.py",), added=5, deleted=20, is_test=False
        ),
        refactor_net_deletions.FileStat(
            paths=("tests/unit_test/test_foo.py",), added=30, deleted=2, is_test=True
        ),
    )
    totals = {
        "non_test": refactor_net_deletions.Totals(),
        "test": refactor_net_deletions.Totals(),
        "all": refactor_net_deletions.Totals(),
    }
    for file_stat in files:
        totals["all"].add(file_stat)
        totals["test" if file_stat.is_test else "non_test"].add(file_stat)

    report = refactor_net_deletions.Report(
        repo="/repo",
        diff_label="origin/main...HEAD",
        mode="merge-base",
        totals=totals,
        files=files,
    )

    assert report.totals["non_test"].net_deleted == 15
    assert report.totals["test"].net_deleted == -28
    assert report.totals["all"].net_deleted == -13
    assert report.target_met


def test_format_html_dashboard_escapes_paths_and_lists_test_files() -> None:
    files = (
        refactor_net_deletions.FileStat(
            paths=("sglang_omni/foo<bar>.py",), added=1, deleted=4, is_test=False
        ),
        refactor_net_deletions.FileStat(
            paths=("tests/unit_test/test_foo.py",), added=5, deleted=0, is_test=True
        ),
    )
    totals = {
        "non_test": refactor_net_deletions.Totals(),
        "test": refactor_net_deletions.Totals(),
        "all": refactor_net_deletions.Totals(),
    }
    for file_stat in files:
        totals["all"].add(file_stat)
        totals["test" if file_stat.is_test else "non_test"].add(file_stat)
    report = refactor_net_deletions.Report(
        repo="/repo",
        diff_label="baseline...origin/main",
        mode="merge-base",
        totals=totals,
        files=files,
    )

    rendered = refactor_net_deletions.format_html(
        report,
        title="TTS <Progress>",
        issue_url="https://github.com/sgl-project/sglang-omni/issues/985",
        refresh_seconds=300,
        list_test_files=True,
        list_non_test_files=True,
        scope_notes=("Excluded integration <#1112> contributes 0 scoped files.",),
    )

    assert "<title>TTS &lt;Progress&gt;</title>" in rendered
    assert '<meta http-equiv="refresh" content="300">' in rendered
    assert "https://github.com/sgl-project/sglang-omni/issues/985" in rendered
    assert "TTS refactor design doc" in rendered
    assert (
        "https://osgbw74w8zwb.sg.larksuite.com/docx/L3XId4GHYoJqSKxwXYqlXaBrgNd"
        in rendered
    )
    assert "Reusable shared surfaces" in rendered
    assert "Excluded integration &lt;#1112&gt; contributes 0 scoped files." in rendered
    assert "sglang_omni/scheduling/reference_encoder.py" in rendered
    assert "sglang_omni/foo&lt;bar&gt;.py" in rendered
    assert "tests/unit_test/test_foo.py" in rendered
    assert "non-test net change &lt; 0" in rendered
    assert "added lines - deleted lines" in rendered
    assert '<strong class="decrease">-3</strong>' in rendered
    assert '<strong class="increase">+5</strong>' in rendered


def test_commit_sum_excludes_unrelated_history(tmp_path: Path) -> None:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ("git", *args),
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    source = tmp_path / "module.py"
    source.write_text("a\nb\nc\n", encoding="utf-8")
    git("add", "module.py")
    git("commit", "-qm", "baseline")
    baseline = git("rev-parse", "HEAD")

    source.write_text("a\nb\nc\nd\n", encoding="utf-8")
    git("commit", "-qam", "T1 migration")
    t1 = git("rev-parse", "HEAD")

    source.write_text("a\nd\n", encoding="utf-8")
    git("commit", "-qam", "[TTS Refactor] remove duplicate code")

    source.write_text("a\nd\nfeature\n", encoding="utf-8")
    git("commit", "-qam", "[Feature] unrelated addition")

    commits = refactor_net_deletions.select_commits(
        tmp_path,
        baseline,
        "HEAD",
        "[TTS Refactor]",
        (t1,),
    )
    report = refactor_net_deletions.build_commit_sum_report(
        tmp_path,
        commits,
        baseline,
        "HEAD",
    )

    assert [subject for _, subject in commits] == [
        "T1 migration",
        "[TTS Refactor] remove duplicate code",
    ]
    assert report.totals["non_test"].added == 1
    assert report.totals["non_test"].deleted == 2
    assert report.totals["non_test"].net_deleted == 1
    assert len(report.commits) == 2
    rendered = refactor_net_deletions.format_html(
        report,
        title="Commit sum",
        issue_url=None,
        refresh_seconds=0,
        list_test_files=False,
        list_non_test_files=False,
    )
    assert "Counted refactor commits" in rendered
    assert "[TTS Refactor] remove duplicate code" in rendered
    assert "[Feature] unrelated addition" not in rendered


def test_tts_commit_allowlist_excludes_model_integrations() -> None:
    revisions = refactor_net_deletions.load_commit_file(
        SCRIPT_PATH.with_name("tts_refactor_commits.json")
    )

    assert len(revisions) == 20
    assert "6cafa0c491c8ff8826bc649e1fd815808b7fbef0" not in revisions
    assert "6ef910be36c68de565f2fed76639fa0d4772eecc" not in revisions
