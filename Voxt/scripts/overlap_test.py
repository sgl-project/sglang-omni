"""Tests for the Voxt pull request overlap counter."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from Voxt.scripts import overlap


class OverlapTest(unittest.TestCase):
    def test_aggregation_counts_product_code_and_excludes_support_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            (repository_root / "Voxt").mkdir()
            (repository_root / "Voxt/page.go").write_text(
                'package main\nimport "github.com/egoist/mygo/ui"\n',
                encoding="utf-8",
            )
            (repository_root / "Voxt/app.go").write_text(
                "package main\n", encoding="utf-8"
            )
            file_paths = [
                "Voxt/page.go",
                "Voxt/app.go",
                "Voxt/internal/session/session.go",
                "Voxt/page_test.go",
                "Voxt/internal/session/session_test.go",
                "Voxt/internal/i18n/locales/zh-Hans.json",
                "Voxt/docs/architecture.md",
                "Voxt/helper/Sources/main.swift",
                "Voxt/testdata/home.json",
            ]

            def fake_command_runner(
                command: list[str], working_directory: Path
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(working_directory, repository_root)
                if command == [
                    "gh",
                    "pr",
                    "view",
                    "123",
                    "--json",
                    "changedFiles,headRefOid,baseRefOid",
                ]:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout=json.dumps(
                            {
                                "changedFiles": len(file_paths),
                                "headRefOid": "head",
                                "baseRefOid": "base",
                                "files": [{"path": path} for path in file_paths],
                            }
                        ),
                        stderr="",
                    )
                elif command[:2] == ["gh", "api"] and "pulls/123/files" in command[2]:
                    entries = [
                        {"filename": path, "status": "modified"} for path in file_paths
                    ]
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps(entries), stderr=""
                    )
                elif command[:2] == ["gh", "api"]:
                    revision = command[2].rsplit("?ref=", 1)[1]
                    file_path = (
                        command[2]
                        .removeprefix("repos/:owner/:repo/contents/")
                        .split("?ref=", 1)[0]
                    )
                    if revision == "head" and file_path == "Voxt/page.go":
                        return subprocess.CompletedProcess(
                            command,
                            0,
                            stdout='package main\nimport "github.com/egoist/mygo/ui"\n',
                            stderr="",
                        )
                    elif revision == "head" and file_path == "Voxt/app.go":
                        return subprocess.CompletedProcess(
                            command, 0, stdout="package main\n", stderr=""
                        )
                    else:
                        return subprocess.CompletedProcess(
                            command, 128, stdout="", stderr="not found"
                        )
                else:
                    raise AssertionError(f"Unexpected command: {command}")

            category_counts = overlap.count_pull_request_files(
                123, repository_root, fake_command_runner
            )

            self.assertEqual(
                category_counts,
                {"view": 1, "logic": 2, "other": 6, "unknown": 0},
            )

    def test_pr_revision_controls_classification_and_deleted_files_use_base(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            (repository_root / "Voxt").mkdir()
            (repository_root / "Voxt/page.go").write_text(
                "package main\n", encoding="utf-8"
            )

            def fake_command_runner(
                command: list[str], working_directory: Path
            ) -> subprocess.CompletedProcess[str]:
                revision = command[2].rsplit("?ref=", 1)[1]
                file_path = (
                    command[2]
                    .removeprefix("repos/:owner/:repo/contents/")
                    .split("?ref=", 1)[0]
                )
                if (revision, file_path) == ("head", "Voxt/page.go"):
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout='package main\nimport "github.com/egoist/mygo/ui"\n',
                        stderr="",
                    )
                elif (revision, file_path) == ("base", "Voxt/deleted.go"):
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        stdout='package main\nimport "github.com/egoist/mygo/ui"\n',
                        stderr="",
                    )
                else:
                    return subprocess.CompletedProcess(
                        command, 128, stdout="", stderr="not found"
                    )

            self.assertEqual(
                overlap.classify_file_at_revision(
                    repository_root, "Voxt/page.go", "head", fake_command_runner
                ),
                "view",
            )
            self.assertEqual(
                overlap.classify_file_at_revision(
                    repository_root, "Voxt/deleted.go", "head", fake_command_runner
                ),
                "unknown",
            )
            self.assertEqual(
                overlap.classify_file_at_revision(
                    repository_root, "Voxt/deleted.go", "base", fake_command_runner
                ),
                "view",
            )

    def test_removed_file_uses_base_but_missing_head_does_not_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            payload = {
                "changedFiles": 2,
                "headRefOid": "head",
                "baseRefOid": "base",
                "files": [
                    {"path": "Voxt/removed.go", "status": "removed"},
                    {"path": "Voxt/modified.go", "status": "modified"},
                ],
            }

            def fake_command_runner(
                command: list[str], working_directory: Path
            ) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["gh", "pr", "view"]:
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps(payload), stderr=""
                    )
                elif command[:2] == ["gh", "api"] and "pulls/123/files" in command[2]:
                    entries = [
                        {"filename": "Voxt/removed.go", "status": "removed"},
                        {"filename": "Voxt/modified.go", "status": "modified"},
                    ]
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps(entries), stderr=""
                    )
                elif command[:2] == ["gh", "api"]:
                    if command[2].endswith("removed.go?ref=base"):
                        return subprocess.CompletedProcess(
                            command,
                            0,
                            stdout='package main\nimport "github.com/egoist/mygo/ui"\n',
                            stderr="",
                        )
                    else:
                        return subprocess.CompletedProcess(
                            command, 1, stdout="", stderr="not found"
                        )
                else:
                    raise AssertionError(f"Unexpected command: {command}")

            category_counts = overlap.count_pull_request_files(
                123, repository_root, fake_command_runner
            )

            self.assertEqual(
                category_counts,
                {"view": 1, "logic": 0, "other": 0, "unknown": 1},
            )

    def test_incomplete_pr_file_list_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            payload = {"changedFiles": 3870, "headRefOid": "head", "baseRefOid": "base"}
            pages = 0

            def fake_command_runner(
                command: list[str], working_directory: Path
            ) -> subprocess.CompletedProcess[str]:
                nonlocal pages
                if command[:3] == ["gh", "pr", "view"]:
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps(payload), stderr=""
                    )
                elif command[:2] == ["gh", "api"] and "pulls/3054/files" in command[2]:
                    pages += 1
                    entries = []
                    if pages <= 30:
                        entries = [
                            {"filename": "Voxt/other.go", "status": "modified"}
                        ] * 100
                    return subprocess.CompletedProcess(
                        command, 0, stdout=json.dumps(entries), stderr=""
                    )
                else:
                    raise AssertionError(f"Unexpected command: {command}")

            with self.assertRaisesRegex(RuntimeError, "refusing an incomplete count"):
                overlap.count_pull_request_files(
                    3054, repository_root, fake_command_runner
                )
            self.assertEqual(pages, 31)

    def test_deleted_root_files_use_origin_main_or_report_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository_root = Path(temporary_directory)
            command_results = {
                (
                    "git",
                    "show",
                    "origin/main:Voxt/deleted_view.go",
                ): subprocess.CompletedProcess(
                    ["git", "show"],
                    0,
                    stdout='package main\nimport "github.com/egoist/mygo/ui"\n',
                    stderr="",
                ),
                (
                    "git",
                    "show",
                    "origin/main:Voxt/deleted_unknown.go",
                ): subprocess.CompletedProcess(
                    ["git", "show"], 128, stdout="", stderr="not found"
                ),
            }

            def fake_command_runner(
                command: list[str], working_directory: Path
            ) -> subprocess.CompletedProcess[str]:
                self.assertEqual(working_directory, repository_root)
                if tuple(command) in command_results:
                    return command_results[tuple(command)]
                else:
                    raise AssertionError(f"Unexpected command: {command}")

            self.assertEqual(
                overlap.classify_file(
                    repository_root, "Voxt/deleted_view.go", fake_command_runner
                ),
                "view",
            )
            self.assertEqual(
                overlap.classify_file(
                    repository_root, "Voxt/deleted_unknown.go", fake_command_runner
                ),
                "unknown",
            )


if __name__ == "__main__":
    unittest.main()
