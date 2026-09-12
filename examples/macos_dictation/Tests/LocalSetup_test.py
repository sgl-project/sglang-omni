# SPDX-License-Identifier: Apache-2.0
"""Offline checks: no installers, model downloads, real servers or app launches."""

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


class LocalSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="omni-local-setup-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = self.root / "repo with spaces/examples/macos_dictation"
        self.project.mkdir(parents=True)
        for name in ("local_runtime.sh", "install_local.sh", "start_local.sh"):
            shutil.copy(PROJECT / name, self.project / name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}"}
        self.env.pop("SGLANG_OMNI_VENV", None)
        self.helper = self.project / "local_runtime.sh"

    def run_shell(self, code, *args, success=True):
        result = subprocess.run(
            ["/bin/bash", "-c", code, "test", str(self.helper), *map(str, args)],
            env=self.env,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def executable(self, name, code):
        path = self.bin / name
        path.write_text(f"#!{sys.executable}\n" + code)
        path.chmod(0o755)
        return path

    def test_dry_runs_never_install_download_or_create_files(self):
        for command in (
            "brew",
            "uv",
            "git",
            "hf",
            "ollama",
            "curl",
            "open",
            "xcrun",
            "ln",
            "mkdir",
        ):
            self.executable(command, "raise SystemExit(97)\n")
        for script in ("install_local.sh", "start_local.sh"):
            output = self.run_shell('bash "$2" --dry-run', self.project / script).stdout
            self.assertIn("Qwen3-ASR-0.6B-4bit", output)
            self.assertIn("minicpm5-2b", output)
            self.assertFalse((self.project / ".build").exists())
        output = self.run_shell(
            'bash "$2" --dry-run --skip-runtime --asr-only',
            self.project / "install_local.sh",
        ).stdout
        self.assertNotIn("install.sh", output)
        self.assertNotIn("ollama", output.lower())

    def test_unknown_options_fail_before_work(self):
        for script in ("install_local.sh", "start_local.sh"):
            self.run_shell('bash "$2" --typo', self.project / script, success=False)

    def test_printed_commands_preserve_unicode_spaces_and_quotes(self):
        arguments = ["Omni 听写.app", "a'b", "$(touch should-not-exist)", "line\nbreak"]
        result = self.run_shell('source "$1"; shift; show_command "$@"', *arguments)
        self.assertEqual(shlex.split(result.stdout), arguments)

    def test_default_service_contract_matches_client(self):
        result = self.run_shell(
            'source "$1"; printf "%s\\n" "$asr_model" "$ollama_model" "$asr_url" "$ollama_url"'
        )
        client = (
            PROJECT / "Sources/DictationCore/ServiceConfiguration.swift"
        ).read_text()
        for value in result.stdout.splitlines():
            self.assertIn(f'"{value}"', client)

    def test_app_link_is_repeatable_and_preserves_conflicts(self):
        app = self.root / "app with spaces.app"
        app.mkdir()
        link = self.root / "shortcuts/Omni $(touch should-not-exist).app"
        self.run_shell(
            'set -eu; source "$1"; dictation_app="$2"; link_app "$3"; link_app "$3"',
            app,
            link,
        )
        self.assertEqual(link.resolve(), app.resolve())
        self.assertFalse((self.root / "should-not-exist").exists())
        conflict = self.root / "existing.app"
        conflict.write_text("keep this")
        self.run_shell(
            'set -eu; source "$1"; dictation_app="$2"; link_app "$3"',
            app,
            conflict,
            success=False,
        )
        self.assertEqual(conflict.read_text(), "keep this")

    def test_cached_snapshot_startup_is_offline(self):
        package = self.root / "fake-library"
        package.mkdir()
        (package / "huggingface_hub.py").write_text(
            "def snapshot_download(*, repo_id, local_files_only):\n"
            "    assert repo_id == 'mlx-community/Qwen3-ASR-0.6B-4bit'\n"
            "    assert local_files_only is True\n"
            "    return '/cached/model with spaces'\n"
        )
        self.env["PYTHONPATH"] = str(package)
        result = self.run_shell(
            'source "$1"; dictation_python="$2"; asr_snapshot offline', sys.executable
        )
        self.assertEqual(result.stdout.strip(), "/cached/model with spaces")

    def test_cleanup_stops_only_owned_children(self):
        other = subprocess.Popen(["sleep", "30"])
        self.addCleanup(lambda: other.poll() is None and other.terminate())
        try:
            self.run_shell(
                'set -eu; source "$1"; sleep 30 & owned=$!; owned_pids=("$owned"); '
                'cleanup_owned; ! kill -0 "$owned" 2>/dev/null; kill -0 "$2"',
                other.pid,
            )
            self.assertIsNone(other.poll())
        finally:
            other.terminate()
            other.wait(timeout=5)

    def launch_fixture(self, *, wrong_asr=False, missing_llm=False):
        # Stub only OS/HTTP/interpreter boundaries. The launcher's readiness,
        # reuse, model selection and failure paths run unchanged.
        venv = self.root / "venv"
        (venv / "bin").mkdir(parents=True)
        python = venv / "bin/python"
        python.write_text(
            f"#!{sys.executable}\nimport os, sys\n"
            "if len(sys.argv) > 2 and sys.argv[1] == '-c' and 'sys.base_prefix' in sys.argv[2]:\n"
            "    raise SystemExit(0)\n"
            f"os.execv({sys.executable!r}, [{sys.executable!r}] + sys.argv[1:])\n"
        )
        python.chmod(0o755)
        self.env["SGLANG_OMNI_VENV"] = str(venv)
        self.executable(
            "uname", "import sys\nprint('Darwin' if sys.argv[1] == '-s' else 'arm64')\n"
        )
        self.executable("sw_vers", "print('14.0')\n")
        self.executable("lsof", "raise SystemExit(0)\n")
        marker = self.root / "opened"
        self.executable(
            "open", f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        )
        replies = {
            "http://127.0.0.1:8000/health": {"status": "healthy"},
            "http://127.0.0.1:8000/v1/models": {
                "data": [{"id": "wrong" if wrong_asr else "Qwen/Qwen3-ASR-0.6B"}]
            },
            "http://127.0.0.1:11434/api/tags": {
                "models": (
                    [] if missing_llm else [{"name": "openbmb/minicpm5-2b:q4_K_M"}]
                )
            },
        }
        self.executable(
            "curl", f"import json, sys\nprint(json.dumps({replies!r}[sys.argv[-1]]))\n"
        )
        (self.project / ".build/client/OmniDictation.app").mkdir(parents=True)
        return marker

    def test_existing_services_are_reused(self):
        marker = self.launch_fixture()
        result = self.run_shell('bash "$2"', self.project / "start_local.sh")
        self.assertTrue(marker.exists())
        self.assertIn("所有服务原本已运行", result.stdout)

    def test_wrong_asr_model_does_not_open_app(self):
        marker = self.launch_fixture(wrong_asr=True)
        result = self.run_shell(
            'bash "$2"', self.project / "start_local.sh", success=False
        )
        self.assertFalse(marker.exists())
        self.assertIn("8000", result.stderr)

    def test_missing_correction_model_fails_or_can_be_skipped(self):
        marker = self.launch_fixture(missing_llm=True)
        self.run_shell('bash "$2"', self.project / "start_local.sh", success=False)
        self.assertFalse(marker.exists())
        self.run_shell('bash "$2" --asr-only', self.project / "start_local.sh")
        self.assertTrue(marker.exists())

    def test_malformed_service_response_is_not_healthy(self):
        self.executable("curl", "print('not json')\n")
        self.run_shell(
            'source "$1"; dictation_python="$2"; asr_ready',
            sys.executable,
            success=False,
        )
        self.run_shell(
            'source "$1"; dictation_python="$2"; ollama_ready',
            sys.executable,
            success=False,
        )


if __name__ == "__main__":
    unittest.main()
